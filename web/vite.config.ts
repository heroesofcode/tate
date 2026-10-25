import { svelte } from '@sveltejs/vite-plugin-svelte';
import { defineConfig } from 'vitest/config';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig(({ command, isPreview }) => {
  // Dev and unit tests use "/". Production builds and `vite preview` use "/tate/"
  // so the preview server matches the paths baked into the built HTML.
  const raw = process.env.BASE_PATH ?? (command === 'serve' && !isPreview ? '/' : '/tate/');
  const base = raw.endsWith('/') ? raw : `${raw}/`;

  return {
    base,
    plugins: [
      svelte(),
      VitePWA({
        strategies: 'injectManifest',
        srcDir: 'src',
        filename: 'sw.ts',
        registerType: 'prompt',
        injectRegister: false,
        manifest: {
          name: 'tate',
          short_name: 'tate',
          description: 'Read manga as a vertical strip on your phone.',
          display: 'standalone',
          orientation: 'portrait',
          theme_color: '#0c0c0f',
          background_color: '#0c0c0f',
          lang: 'en',
          scope: base,
          start_url: base,
          id: base,
          icons: [
            { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png', purpose: 'any' },
            { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'any' },
            {
              src: 'icons/icon-maskable-192.png',
              sizes: '192x192',
              type: 'image/png',
              purpose: 'maskable',
            },
            {
              src: 'icons/icon-maskable-512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'maskable',
            },
          ],
          share_target: {
            action: `${base}share`,
            method: 'POST',
            enctype: 'multipart/form-data',
            params: {
              files: [
                {
                  name: 'files',
                  accept: [
                    'application/vnd.tate+zip',
                    'application/zip',
                    'application/x-cbz',
                    'application/octet-stream',
                  ],
                },
              ],
            },
          },
          file_handlers: [
            {
              action: base,
              accept: {
                'application/vnd.tate+zip': ['.tate'],
                'application/zip': ['.zip'],
                'application/x-cbz': ['.cbz'],
              },
            },
          ],
        },
        injectManifest: {
          globPatterns: ['**/*.{js,css,html,svg,png,ico,webp,woff2}'],
          maximumFileSizeToCacheInBytes: 4 * 1024 * 1024,
        },
        devOptions: { enabled: false },
      }),
    ],
    preview: {
      host: '127.0.0.1',
      port: 4173,
      strictPort: true,
    },
    test: {
      environment: 'node',
      include: ['src/**/*.test.ts'],
    },
  };
});
