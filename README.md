# tate

Read manga the way you read webtoons. tate (縦, "vertical") turns a manga chapter into a single vertical strip of panels, in Japanese reading order, sized for a phone screen, and reads it in an installable offline web app.

The reader lives in [`web/`](web/README.md) and is published at <https://heroesofcode.github.io/tate/>.

## Getting a chapter

1. Build a `.tate` package with the processor (see `processor/` in this repository). A package is a zip with `manifest.json`, `cover.webp`, and `strip/0001.webp` onward.
2. Open the reader and import that file, or share it to tate from the system share sheet where the browser supports it.
3. Without a processed package, zip the page images into a `.cbz` or `.zip` and import that. Pages stay in order and are marked as unprocessed.

## Processor

`processor/` builds a `.tate` chapter from a CBZ, a PDF, or a folder of images. Install and usage: [processor/README.md](processor/README.md). The package shared with the reader is specified in [docs/format.md](docs/format.md).

## License

MIT
