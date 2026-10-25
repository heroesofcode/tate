import { defineConfig, devices } from '@playwright/test';

const iphone = devices['iPhone 13'];

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: 'http://127.0.0.1:4173/tate/',
    locale: 'en-US',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'pnpm build && pnpm preview --host 127.0.0.1 --port 4173 --strictPort',
    url: 'http://127.0.0.1:4173/tate/',
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
  projects: [
    {
      name: 'pixel7',
      use: { ...devices['Pixel 7'], browserName: 'chromium', locale: 'en-US' },
    },
    {
      name: 'iphone',
      use: {
        ...iphone,
        browserName: 'chromium',
        defaultBrowserType: 'chromium',
        locale: 'en-US',
      },
    },
  ],
});
