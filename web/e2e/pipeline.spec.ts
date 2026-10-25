import { readFileSync } from 'node:fs';
import { strFromU8, unzipSync } from 'fflate';
import { expect, test, type Page } from '@playwright/test';

interface SegmentSize {
  width: number;
  height: number;
}

interface ChapterManifest {
  title: string;
  segments: SegmentSize[];
}

function readManifest(path: string): ChapterManifest {
  const archive = unzipSync(new Uint8Array(readFileSync(path)));
  const raw = archive['manifest.json'];
  if (!raw) throw new Error('package has no manifest.json');
  const parsed: unknown = JSON.parse(strFromU8(raw));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('manifest is not an object');
  }
  const record = parsed as { title?: unknown; segments?: unknown };
  if (
    typeof record.title !== 'string' ||
    !Array.isArray(record.segments) ||
    record.segments.length < 1
  ) {
    throw new Error('manifest is missing a title or segments');
  }
  const segments = record.segments.map((entry: unknown, index: number) => {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new Error(`segment ${index} is invalid`);
    }
    const segment = entry as { width?: unknown; height?: unknown };
    if (typeof segment.width !== 'number' || typeof segment.height !== 'number') {
      throw new Error(`segment ${index} has no size`);
    }
    if (segment.width <= 0 || segment.height <= 0)
      throw new Error(`segment ${index} has an empty size`);
    return { width: segment.width, height: segment.height };
  });
  return { title: record.title, segments };
}

async function dismissHint(page: Page) {
  const later = page.getByRole('button', { name: 'Not now' });
  try {
    await later.waitFor({ state: 'visible', timeout: 2500 });
    await later.click();
  } catch {
    /* hint is not shown */
  }
}

/** True once the stored position would reopen the scroller where it is now. */
async function progressMatchesScroll(page: Page): Promise<boolean> {
  return page.evaluate(async () => {
    const root = document.querySelector('.scroller');
    if (!(root instanceof HTMLElement)) return false;
    const max = root.scrollHeight - root.clientHeight;
    if (max - root.scrollTop > 2) return false;
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('tate');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const rows = await new Promise<Array<{ segmentIndex: number; offsetFraction: number }>>(
      (resolve, reject) => {
        const request = db.transaction('progress', 'readonly').objectStore('progress').getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      },
    );
    db.close();
    const progress = rows[0];
    if (!progress) return false;
    const node = root.querySelector<HTMLElement>(`[data-index="${progress.segmentIndex}"]`);
    if (!node) return false;
    const top =
      node.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop;
    const restored = top + node.offsetHeight * progress.offsetFraction;
    return Math.abs(restored - root.scrollTop) < 48;
  });
}

test.describe('processor pipeline', () => {
  test.skip(!process.env.TATE_E2E_PACKAGE, 'TATE_E2E_PACKAGE is not set');

  test('imports a processed chapter and restores the ending', async ({ page }) => {
    const chapterPath = process.env.TATE_E2E_PACKAGE;
    if (!chapterPath) throw new Error('TATE_E2E_PACKAGE is not set');
    const manifest = readManifest(chapterPath);
    const count = manifest.segments.length;
    const countLabel = count === 1 ? '1 panel' : `${count} panels`;

    await page.goto('./');
    await dismissHint(page);
    await page.locator('input[type=file]').setInputFiles(chapterPath);
    await expect(page.getByRole('heading', { name: manifest.title })).toBeVisible();
    await expect(page.locator('.card .meta')).toContainText(countLabel);

    await page.getByRole('button', { name: `Open ${manifest.title}` }).click();
    const scroller = page.locator('.scroller');
    await expect(scroller).toBeVisible();

    for (const [index, segment] of manifest.segments.entries()) {
      const article = page.locator(`.segment[data-index="${index}"]`);
      await article.scrollIntoViewIfNeeded();
      const image = article.locator('img');
      await expect
        .poll(() =>
          image.evaluate((element) =>
            element instanceof HTMLImageElement ? element.naturalWidth : 0,
          ),
        )
        .toBeGreaterThan(0);
      const actual = await image.evaluate((element) => {
        if (!(element instanceof HTMLImageElement) || element.naturalHeight === 0) return 0;
        return element.naturalWidth / element.naturalHeight;
      });
      expect(actual).toBeCloseTo(segment.width / segment.height, 2);
    }

    await scroller.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    const ending = page.getByRole('heading', { name: 'End of chapter' });
    await expect(ending).toBeInViewport();
    await expect.poll(() => progressMatchesScroll(page)).toBe(true);
    const before = await scroller.evaluate((element) => element.scrollTop);
    expect(before).toBeGreaterThan(200);

    await page.reload();
    await expect(scroller).toBeVisible();
    await dismissHint(page);
    await expect
      .poll(async () => scroller.evaluate((element) => element.scrollTop), { timeout: 8000 })
      .toBeGreaterThan(before * 0.85);
    const after = await scroller.evaluate((element) => element.scrollTop);
    expect(Math.abs(after - before)).toBeLessThan(48);
    await expect(ending).toBeInViewport();
  });
});
