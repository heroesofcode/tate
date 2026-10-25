# Chapter package format

Contract shared by the processor and the reader. Version 1.

A processed chapter is a ZIP file with the extension `.tate` (media type
`application/vnd.tate+zip`). Image entries are stored (they are already
compressed). The manifest may be deflated.

```
manifest.json
cover.webp            # first panel or first page, scaled to 360 px wide
strip/0001.webp       # segments in reading order, 4-digit zero-padded, 1-based
strip/0002.webp
...
```

## manifest.json

```json
{
  "format": "tate",
  "version": 1,
  "title": "Chapter title (defaults to the input file stem)",
  "source": "input file name only, never a path",
  "createdAt": "2026-10-25T04:00:00Z",
  "reading": "rtl",
  "width": 1080,
  "background": "#ffffff",
  "detector": "yolo | cv | page",
  "pageCount": 20,
  "cover": "cover.webp",
  "segments": [
    {
      "file": "strip/0001.webp",
      "width": 1080,
      "height": 760,
      "page": 0,
      "gapBefore": 0,
      "box": [0, 0, 100, 100]
    }
  ]
}
```

`box` is `[x, y, w, h]`.

`reading` is `rtl` or `ltr`. Segments are already stored in that order. A reader scrolls the strip from top to bottom and does not reorder them.

## Rules

- Every segment image is exactly `width` pixels wide. A panel narrower than the
  strip is centered on the background color. The image's real pixel size matches
  `width` and `height` in the manifest, so a reader can reserve space before the
  image loads.
- `gapBefore` is the vertical space, in strip pixels at `width` scale, that the
  reader draws above the segment in the background color. The gap is small
  between panels of the same page and larger between pages. The first segment
  is `0`.
- `page` is the 0-based page index in the source file. When only a range of
  pages is processed it keeps the source numbering, so it can be greater than
  or equal to `pageCount`. `box` is the panel rectangle in
  source-page pixels (optional, for debugging). Readers ignore unknown fields.
- Readers reject `format != "tate"` or `version > 1` with a clear message.
- WebP segments are at most 4096 px tall. A panel taller than that is split into
  consecutive segments with `gapBefore: 0`.
