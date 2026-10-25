import { expect, test, type Page } from '@playwright/test';
import {
  createCbzPackage,
  createChapterPackage,
  createCorruptPackage,
  createInvalidManifestPackage,
  createOversizedPackage,
} from './fixture';

interface StoredProgress {
  segmentIndex: number;
  offsetFraction: number;
}

interface StoreCounts {
  chapters: number;
  segments: number;
  progress: number;
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

async function openLibrary(page: Page) {
  await page.goto('./');
  await dismissHint(page);
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
}

async function importFiles(page: Page, paths: string | string[]) {
  await page.locator('input[type=file]').setInputFiles(paths);
}

function chapterButton(page: Page, title: string) {
  return page.getByRole('button', { name: new RegExp(`^(Open|Continue) ${title}$`) });
}

async function openChapter(page: Page, title: string) {
  await chapterButton(page, title).click();
  await expect(page.locator('.scroller')).toBeVisible();
}

async function progressByTitle(page: Page): Promise<Record<string, StoredProgress>> {
  return page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('tate');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const all = <T>(store: string) =>
      new Promise<T[]>((resolve, reject) => {
        const request = db.transaction(store, 'readonly').objectStore(store).getAll();
        request.onsuccess = () => resolve(request.result as T[]);
        request.onerror = () => reject(request.error);
      });
    const chapters = await all<{ id: string; title: string }>('chapters');
    const progress = await all<{ chapterId: string } & StoredProgress>('progress');
    db.close();
    const titles = new Map(chapters.map((chapter) => [chapter.id, chapter.title]));
    const saved: Record<string, StoredProgress> = {};
    for (const row of progress) {
      const title = titles.get(row.chapterId);
      if (!title) continue;
      saved[title] = { segmentIndex: row.segmentIndex, offsetFraction: row.offsetFraction };
    }
    return saved;
  });
}

async function storeCounts(page: Page): Promise<StoreCounts> {
  return page.evaluate(async () => {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open('tate');
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    const count = (store: string) =>
      new Promise<number>((resolve, reject) => {
        const request = db.transaction(store, 'readonly').objectStore(store).count();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    const counts = {
      chapters: await count('chapters'),
      segments: await count('segments'),
      progress: await count('progress'),
    };
    db.close();
    return counts;
  });
}

async function scrollToFraction(page: Page, fraction: number) {
  const scroller = page.locator('.scroller');
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollHeight))
    .toBeGreaterThan(200);
  await scroller.evaluate((element, value) => {
    const max = element.scrollHeight - element.clientHeight;
    element.scrollTop = Math.round(max * value);
  }, fraction);
  return scroller.evaluate((element) => element.scrollTop);
}

async function leaveReader(page: Page) {
  const back = page.getByRole('button', { name: 'Back', exact: true });
  if (!(await back.isVisible())) {
    const scroller = page.locator('.scroller');
    const box = await scroller.boundingBox();
    if (!box) throw new Error('scroller has no box');
    await scroller.click({ position: { x: box.width / 2, y: box.height / 2 } });
  }
  await expect(back).toBeVisible();
  await back.click();
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
}

async function textOverflow(page: Page): Promise<number> {
  return page.evaluate(() => {
    const nodes = [document.documentElement, document.body, ...document.querySelectorAll('body *')];
    for (const node of nodes) {
      if (!(node instanceof HTMLElement) || node.dataset.baseSize) continue;
      const computed = Number.parseFloat(getComputedStyle(node).fontSize);
      const parent = node.parentElement;
      let base = computed;
      // Text that only inherits should follow the parent's original size, not the enlarged one.
      if (parent instanceof HTMLElement && parent.dataset.baseSize) {
        const parentComputed = Number.parseFloat(getComputedStyle(parent).fontSize);
        if (Math.abs(computed - parentComputed) < 0.5) base = Number(parent.dataset.baseSize);
      }
      node.dataset.baseSize = String(base);
      if (base > 0) node.style.fontSize = `${base * 2}px`;
    }
    const width = document.documentElement.clientWidth;
    let overflow = document.documentElement.scrollWidth - width;
    for (const node of document.querySelectorAll(
      '.library, .library *, .hint, .hint *, .banners, .banners *',
    )) {
      if (!(node instanceof HTMLElement)) continue;
      const box = node.getBoundingClientRect();
      if (box.width < 1 || box.height < 1) continue;
      overflow = Math.max(overflow, box.right - width, -box.left);
    }
    return overflow;
  });
}

test('imports a tate package and a plain cbz from the file picker', async ({ page }) => {
  const tate = await createChapterPackage({ title: 'Packed Chapter', fileName: 'packed.tate' });
  const cbz = await createCbzPackage('Plain Pages');
  await openLibrary(page);
  await importFiles(page, [tate, cbz]);
  await expect(page.getByRole('heading', { name: 'Packed Chapter' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Plain Pages' })).toBeVisible();
  await expect(page.getByText('Unprocessed')).toBeVisible();
  await expect(page.locator('.card', { hasText: 'Plain Pages' })).toContainText('3 pages');
  await expect(page.locator('.card', { hasText: 'Packed Chapter' })).toContainText('panels');
  await openChapter(page, 'Plain Pages');
  await expect
    .poll(() =>
      page.locator('img[alt="Panel 1"]').evaluate((element) => {
        return element instanceof HTMLImageElement ? element.naturalWidth : 0;
      }),
    )
    .toBeGreaterThan(0);
});

test('reports a readable error for a bad package and keeps the library usable', async ({
  page,
}) => {
  const corrupt = await createCorruptPackage();
  const oversized = await createOversizedPackage();
  const invalid = await createInvalidManifestPackage();
  const good = await createChapterPackage({ title: 'Usable Chapter', fileName: 'usable.tate' });
  await openLibrary(page);
  await importFiles(page, [corrupt, oversized, invalid]);
  await expect(
    page.getByRole('alert').filter({ hasText: 'could not be read as an archive' }),
  ).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'too large' })).toBeVisible();
  await expect(page.getByRole('alert').filter({ hasText: 'manifest is invalid' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'No chapters yet' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Import chapter' }).first()).toBeEnabled();
  await importFiles(page, good);
  await expect(page.getByRole('heading', { name: 'Usable Chapter' })).toBeVisible();
  await expect(chapterButton(page, 'Usable Chapter')).toBeEnabled();
});

test('scrolls a long chapter with only a few images decoded', async ({ page }) => {
  test.setTimeout(90_000);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error') errors.push(message.text());
  });
  // Above the reader's nearby window, and far below the full chapter.
  const decodedImageLimit = 24;
  const chapter = await createChapterPackage({
    title: 'Long Strip',
    fileName: 'long.tate',
    count: 300,
    width: 1000,
    height: 50,
    gap: 0,
  });
  await openLibrary(page);
  await importFiles(page, chapter);
  await expect(page.getByRole('heading', { name: 'Long Strip' })).toBeVisible({ timeout: 30_000 });
  await openChapter(page, 'Long Strip');
  const scroller = page.locator('.scroller');
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollHeight))
    .toBeGreaterThan(1000);
  await page.evaluate(() => {
    const target = document.querySelector('.scroller');
    if (!target) return;
    let peak = document.querySelectorAll('.frame img[src]').length;
    const note = () => {
      peak = Math.max(peak, document.querySelectorAll('.frame img[src]').length);
      (window as Window & { __tateImagePeak?: number }).__tateImagePeak = peak;
    };
    note();
    const observer = new MutationObserver(note);
    observer.observe(target, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ['src'],
    });
  });
  for (const fraction of [0.25, 0.5, 0.75, 1]) {
    await scroller.evaluate((element, value) => {
      const max = element.scrollHeight - element.clientHeight;
      element.scrollTop = Math.round(max * value);
    }, fraction);
    await expect
      .poll(() => page.locator('.frame img[src]').count())
      .toBeLessThanOrEqual(decodedImageLimit);
  }
  await expect(page.getByRole('heading', { name: 'End of chapter' })).toBeInViewport();
  await expect
    .poll(() =>
      page
        .locator('.frame img')
        .evaluateAll((images) =>
          images.some(
            (image) =>
              image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0,
          ),
        ),
    )
    .toBe(true);
  await expect.poll(() => page.locator('.frame img[src]').count()).toBeGreaterThan(0);
  const peak = await page.evaluate(
    () => (window as Window & { __tateImagePeak?: number }).__tateImagePeak ?? 0,
  );
  expect(peak).toBeGreaterThan(0);
  expect(peak).toBeLessThanOrEqual(decodedImageLimit);
  expect(errors).toEqual([]);
});

test('restores the reading position after reload and after reopening', async ({ page }) => {
  const chapter = await createChapterPackage({
    title: 'Resume Chapter',
    fileName: 'resume.tate',
    count: 8,
  });
  await openLibrary(page);
  await importFiles(page, chapter);
  await openChapter(page, 'Resume Chapter');
  const scroller = page.locator('.scroller');
  const before = await scrollToFraction(page, 0.55);
  expect(before).toBeGreaterThan(120);
  await expect
    .poll(async () => {
      const saved = await progressByTitle(page);
      const row = saved['Resume Chapter'];
      return (row?.segmentIndex ?? 0) > 0 || (row?.offsetFraction ?? 0) > 0.05;
    })
    .toBe(true);

  await page.reload();
  await expect(scroller).toBeVisible();
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(before * 0.8);
  const afterReload = await scroller.evaluate((element) => element.scrollTop);
  expect(Math.abs(afterReload - before)).toBeLessThan(48);

  await leaveReader(page);
  await openChapter(page, 'Resume Chapter');
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(before * 0.8);
  const afterReopen = await scroller.evaluate((element) => element.scrollTop);
  expect(Math.abs(afterReopen - before)).toBeLessThan(48);
});

test('opens an imported chapter while offline', async ({ page, context }) => {
  const chapter = await createChapterPackage({
    title: 'Offline Chapter',
    fileName: 'offline.tate',
  });
  await openLibrary(page);
  await importFiles(page, chapter);
  await expect(page.getByRole('heading', { name: 'Offline Chapter' })).toBeVisible();
  await expect
    .poll(() => page.evaluate(() => navigator.serviceWorker.controller !== null), {
      timeout: 15_000,
    })
    .toBe(true);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Offline Chapter' })).toBeVisible();
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Offline Chapter' })).toBeVisible();
  await openChapter(page, 'Offline Chapter');
  await expect
    .poll(() =>
      page.locator('img[alt="Panel 1"]').evaluate((element) => {
        return element instanceof HTMLImageElement ? element.naturalWidth : 0;
      }),
    )
    .toBeGreaterThan(0);
});

test('drops a deleted chapter from the library and from storage', async ({ page }) => {
  const chapter = await createChapterPackage({
    title: 'Disposable Chapter',
    fileName: 'disposable.tate',
    count: 3,
  });
  await openLibrary(page);
  await importFiles(page, chapter);
  await expect(page.getByRole('heading', { name: 'Disposable Chapter' })).toBeVisible();
  await expect.poll(() => storeCounts(page)).toMatchObject({ chapters: 1, segments: 3 });
  await page.getByRole('button', { name: 'Delete Disposable Chapter' }).click();
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'No chapters yet' })).toBeVisible();
  await expect.poll(() => storeCounts(page)).toEqual({ chapters: 0, segments: 0, progress: 0 });
});

test('keeps reading progress separate for two chapters', async ({ page }) => {
  const north = await createChapterPackage({
    title: 'North Chapter',
    fileName: 'north.tate',
    source: 'north.cbz',
    count: 8,
  });
  const south = await createChapterPackage({
    title: 'South Chapter',
    fileName: 'south.tate',
    source: 'south.cbz',
    count: 8,
  });
  await openLibrary(page);
  await importFiles(page, [north, south]);
  await openChapter(page, 'North Chapter');
  const scroller = page.locator('.scroller');
  const northTop = await scrollToFraction(page, 0.7);
  await expect
    .poll(async () => (await progressByTitle(page))['North Chapter']?.segmentIndex ?? 0)
    .toBeGreaterThan(0);
  await leaveReader(page);

  await openChapter(page, 'South Chapter');
  await expect
    .poll(async () => (await progressByTitle(page))['South Chapter'] !== undefined)
    .toBe(true);
  const southStart = await scroller.evaluate((element) => element.scrollTop);
  expect(southStart).toBeLessThan(24);
  const southTop = await scrollToFraction(page, 0.2);
  await expect
    .poll(async () => {
      const saved = await progressByTitle(page);
      const northRow = saved['North Chapter'];
      const southRow = saved['South Chapter'];
      if (!northRow || !southRow) return 0;
      const northPos = northRow.segmentIndex + northRow.offsetFraction;
      const southPos = southRow.segmentIndex + southRow.offsetFraction;
      return Math.abs(northPos - southPos);
    })
    .toBeGreaterThan(0.2);
  await leaveReader(page);

  await openChapter(page, 'North Chapter');
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(northTop * 0.8);
  expect(
    Math.abs((await scroller.evaluate((element) => element.scrollTop)) - northTop),
  ).toBeLessThan(48);
  await leaveReader(page);
  await openChapter(page, 'South Chapter');
  await expect
    .poll(() => scroller.evaluate((element) => element.scrollTop))
    .toBeGreaterThan(southTop * 0.8);
  expect(
    Math.abs((await scroller.evaluate((element) => element.scrollTop)) - southTop),
  ).toBeLessThan(48);
});

test('fits the library on a narrow screen with large text', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
  const hint = page.getByRole('button', { name: 'Not now' });
  await hint.waitFor({ state: 'visible', timeout: 2500 }).catch(() => undefined);
  await expect(page.locator('.storage')).toBeVisible();
  expect(await textOverflow(page)).toBeLessThanOrEqual(1);

  await dismissHint(page);
  const chapter = await createChapterPackage({ title: 'Narrow Chapter', fileName: 'narrow.tate' });
  await importFiles(page, chapter);
  await expect(page.getByRole('heading', { name: 'Narrow Chapter' })).toBeVisible();
  expect(await textOverflow(page)).toBeLessThanOrEqual(1);
  const button = page.getByRole('button', { name: 'Import chapter' }).first();
  const box = await button.boundingBox();
  expect(box).toBeTruthy();
  if (!box) return;
  expect(box.x).toBeGreaterThanOrEqual(-1);
  expect(box.x + box.width).toBeLessThanOrEqual(321);
});

test('toggles the reader chrome with a tap', async ({ page }) => {
  const chapter = await createChapterPackage({ title: 'Tap Chapter', fileName: 'tap.tate' });
  await openLibrary(page);
  await importFiles(page, chapter);
  await openChapter(page, 'Tap Chapter');
  const scroller = page.locator('.scroller');
  const overlay = page.locator('.overlay');
  const box = await scroller.boundingBox();
  if (!box) throw new Error('scroller has no box');
  const before = await overlay.getAttribute('data-overlay');
  await scroller.tap({ position: { x: box.width / 2, y: box.height / 2 } });
  const toggled = before === 'visible' ? 'hidden' : 'visible';
  await expect(overlay).toHaveAttribute('data-overlay', toggled);
  await scroller.tap({ position: { x: box.width / 2, y: box.height / 2 } });
  await expect(overlay).toHaveAttribute('data-overlay', before === 'hidden' ? 'hidden' : 'visible');
});

test('returns to the library when Escape is pressed', async ({ page }) => {
  const chapter = await createChapterPackage({ title: 'Escape Chapter', fileName: 'escape.tate' });
  await openLibrary(page);
  await importFiles(page, chapter);
  await openChapter(page, 'Escape Chapter');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Escape Chapter' })).toBeVisible();
});
