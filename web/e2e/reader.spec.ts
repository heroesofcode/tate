import { expect, test, type Page } from '@playwright/test';
import { createFixtureTate } from './fixture';

async function dismissHint(page: Page) {
  const later = page.getByRole('button', { name: 'Not now' });
  try {
    await later.waitFor({ state: 'visible', timeout: 2500 });
    await later.click();
  } catch {
    /* hint is not shown */
  }
}

async function savedProgress(page: Page) {
  return page.evaluate(async () => {
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
    return rows[0];
  });
}

test('imports a chapter, restores progress, and deletes it', async ({ page }) => {
  const fixture = await createFixtureTate();
  await page.goto('./');
  await dismissHint(page);
  await page.locator('input[type=file]').setInputFiles(fixture);
  await expect(page.getByRole('heading', { name: 'Fixture Chapter' })).toBeVisible();
  await page.getByRole('button', { name: 'Open Fixture Chapter' }).click();
  const scroller = page.locator('.scroller');
  await expect(scroller).toBeVisible();
  await page.waitForFunction(() => {
    const image = document.querySelector('img[alt="Panel 1"]') as HTMLImageElement | null;
    return !!image && image.naturalWidth > 0;
  });
  await scroller.evaluate((element) => {
    element.scrollTop = Math.round(element.scrollHeight * 0.45);
  });
  await expect
    .poll(async () => {
      const progress = await savedProgress(page);
      return (progress?.segmentIndex ?? 0) > 0 || (progress?.offsetFraction ?? 0) > 0.2;
    })
    .toBe(true);
  const before = await scroller.evaluate((element) => element.scrollTop);
  expect(before).toBeGreaterThan(200);
  await page.reload();
  await expect(scroller).toBeVisible();
  await expect
    .poll(async () => scroller.evaluate((element) => element.scrollTop), { timeout: 8000 })
    .toBeGreaterThan(before * 0.8);
  const after = await scroller.evaluate((element) => element.scrollTop);
  expect(Math.abs(after - before)).toBeLessThan(48);

  const back = page.getByRole('button', { name: 'Back', exact: true });
  if (!(await back.isVisible())) await scroller.click({ position: { x: 40, y: 220 } });
  await expect(back).toBeVisible();
  await back.click();
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
  await page.getByRole('button', { name: 'Delete Fixture Chapter' }).click();
  await page.getByRole('button', { name: 'Delete', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'No chapters yet' })).toBeVisible();
});

test('reloads offline after the shell is cached', async ({ page, context }) => {
  await page.goto('./');
  await expect
    .poll(() => page.evaluate(() => navigator.serviceWorker.controller !== null), {
      timeout: 15000,
    })
    .toBe(true);
  await page.reload();
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole('heading', { level: 1, name: 'tate' })).toBeVisible();
});
