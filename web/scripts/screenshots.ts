import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium, devices } from '@playwright/test';
import { createFixtureTate } from '../e2e/fixture';

const out = process.env.SHOT_DIR ?? 'screenshots';
const base = process.env.SHOT_URL ?? 'http://127.0.0.1:4173/tate/';

await mkdir(out, { recursive: true });
const browser = await chromium.launch();
const iphone = devices['iPhone 13'];
const context = await browser.newContext({
  ...iphone,
  viewport: { width: 390, height: 844 },
  deviceScaleFactor: 2,
  locale: 'en-US',
  isMobile: true,
  hasTouch: true,
});
const page = await context.newPage();
await page.goto(base);
await page.getByRole('heading', { name: 'Install tate' }).waitFor();
await page.screenshot({ path: join(out, 'install-hint.png') });
await page.getByRole('button', { name: 'Not now' }).click();
await page.getByRole('heading', { name: 'No chapters yet' }).waitFor();
await page.screenshot({ path: join(out, 'empty-library.png') });

const fixture = await createFixtureTate();
await page.locator('input[type=file]').setInputFiles(fixture);
await page.getByRole('heading', { name: 'Fixture Chapter' }).waitFor();
const dismiss = page.getByRole('button', { name: 'Dismiss' });
if (await dismiss.isVisible()) await dismiss.click();
await page.screenshot({ path: join(out, 'library-chapter.png') });
await page.getByRole('button', { name: 'Open Fixture Chapter' }).click();
await page.waitForFunction(() => {
  const image = document.querySelector('img[alt="Panel 1"]') as HTMLImageElement | null;
  return !!image && image.naturalWidth > 0;
});
await page.locator('.scroller').evaluate((element) => {
  element.scrollTop = Math.round(element.scrollHeight * 0.35);
});
await page.locator('[data-overlay="hidden"]').waitFor({ timeout: 6000 });
await page.waitForFunction(() => {
  const image = document.querySelector('img[alt="Panel 3"]') as HTMLImageElement | null;
  return !!image && image.naturalWidth > 0;
});
await page.screenshot({ path: join(out, 'reader-overlay-hidden.png') });
await page.locator('.scroller').click({ position: { x: 48, y: 320 } });
await page.locator('[data-overlay="visible"]').waitFor();
await page.screenshot({ path: join(out, 'reader-overlay-shown.png') });
await browser.close();
