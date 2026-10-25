# tate reader

Mobile web app that reads a chapter as one vertical strip. It is a Vite + Svelte 5 PWA, published at <https://heroesofcode.github.io/tate/>.

## Develop

Node.js 22 or newer and pnpm 10.

```bash
cd web
pnpm install
pnpm dev
```

The dev server uses base path `/`. Production builds use `/tate/` unless `BASE_PATH` is set (`BASE_PATH` should start with `/` and may omit the trailing slash).

```bash
BASE_PATH=/tate/ pnpm build
pnpm preview
```

Regenerate icons from `scripts/icon.svg`:

```bash
pnpm icons
```

## Test

```bash
pnpm lint
pnpm format:check
pnpm check
pnpm test
pnpm test:e2e
```

Playwright drives a production build (`vite preview`) in Chromium at phone sizes. The synthetic `.tate` fixture is drawn at test time; no chapter images are stored in the repo.

## Architecture

Hash routes (`#/` and `#/read/:id`) keep GitHub Pages on a single `index.html`. There is no client-side history fallback.

- `src/lib/manifest.ts` checks the chapter contract: format, version, fields, sequential `strip/0001.webp` names, and sane sizes. Unknown JSON fields are ignored.
- `src/lib/package.ts` reads `.tate` packages and plain `.cbz` / `.zip` page archives. Archive entry names are never used as filesystem paths. A `.cbz` becomes one segment per image, in natural order, and is marked unprocessed.
- `src/lib/storage.ts` is the IndexedDB layer (`idb`).
- `src/components/Reader.svelte` scrolls natively, reserves each segment’s aspect ratio, and keeps only nearby images decoded.
- `src/sw.ts` precaches the shell and accepts a Web Share Target `POST` at `{base}share`.

UI copy is English or Brazilian Portuguese, chosen from `navigator.language`, in `src/lib/i18n.ts`.

## Storage model

Database `tate`, schema version 1. `migrate()` creates stores when `oldVersion < 1`.

| Store      | Contents                                                                                                                |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `chapters` | Title, source, colors, segment layout, cover blob, byte size. Indexed by `identity` (`title`, `source`, segment count). |
| `segments` | Image blobs keyed by `chapterId:index`, indexed by chapter.                                                             |
| `progress` | `segmentIndex` and `offsetFraction` for each chapter.                                                                   |

Re-importing the same identity asks before replacing the blobs. Progress for that chapter is kept. Delete removes all three records.

Segment bytes are loaded one at a time while reading. Object URLs are created near the viewport and revoked once the segment scrolls away, so a long chapter does not keep every bitmap alive.

## Offline and install

`vite-plugin-pwa` uses Workbox `injectManifest`. The first visit activates the service worker so a later reload works offline. A later visit shows a reload prompt when a new worker is waiting.

The web manifest includes maskable and any-purpose icons, `share_target` (multipart POST), and `file_handlers`. Shared files are parked in the Cache API and imported when the app opens. Chromium can also deliver files through `launchQueue`.

iOS Safari does not implement the Web Share Target POST API or `file_handlers`. On iPhone, install from Share → Add to Home Screen and import with the file picker. Android can use Install app when the browser fires `beforeinstallprompt`.

## Known limitations

- Share target and file handlers need a Chromium browser. iOS uses the in-app picker.
- Pinch zoom is the browser’s own zoom. The reader does not disable it and does not implement a separate zoom mode.
- Archives above the size caps in `src/lib/limits.ts` are rejected instead of being streamed to disk.
- Unprocessed pages keep their own aspect ratio. They are not split or reflowed into panels.
