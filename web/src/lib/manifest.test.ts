import { describe, expect, it } from 'vitest';
import { PackageError } from './errors';
import { MAX_SEGMENT_HEIGHT, MAX_SEGMENTS, MAX_WIDTH } from './limits';
import { validateManifest } from './manifest';

function segment(index: number, overrides: Record<string, unknown> = {}) {
  return {
    file: `strip/${String(index + 1).padStart(4, '0')}.webp`,
    width: 1080,
    height: 800,
    page: index,
    gapBefore: index === 0 ? 0 : 12,
    ...overrides,
  };
}

function manifest(overrides: Record<string, unknown> = {}) {
  return {
    format: 'tate',
    version: 1,
    title: 'Chapter One',
    source: 'chapter.cbz',
    createdAt: '2026-10-25T04:00:00Z',
    reading: 'rtl',
    width: 1080,
    background: '#ffffff',
    detector: 'page',
    pageCount: 1,
    cover: 'cover.webp',
    extra: 'ignored',
    segments: [segment(0, { note: 'ignored' })],
    ...overrides,
  };
}

function failure(value: unknown): PackageError {
  try {
    validateManifest(value);
  } catch (error) {
    if (error instanceof PackageError) return error;
    throw error;
  }
  throw new Error('manifest was accepted');
}

describe('validateManifest', () => {
  it('keeps a valid chapter and drops unknown fields', () => {
    const parsed = validateManifest(
      manifest({
        title: '  Hello  ',
        source: '  a.cbz  ',
        reading: 'ltr',
        background: '#abc',
        detector: 'yolo',
        segments: [
          segment(0, { note: 'ok', box: [0, 1, 2, 3] }),
          segment(1, { gapBefore: 0.5, box: [0, 0, 10, 10], comment: 'no' }),
        ],
      }),
    );
    expect(parsed).not.toHaveProperty('extra');
    expect(parsed.title).toBe('Hello');
    expect(parsed.source).toBe('a.cbz');
    expect(parsed.reading).toBe('ltr');
    expect(parsed.background).toBe('#abc');
    expect(parsed.detector).toBe('yolo');
    expect(parsed.segments).toEqual([
      { file: 'strip/0001.webp', width: 1080, height: 800, page: 0, gapBefore: 0 },
      { file: 'strip/0002.webp', width: 1080, height: 800, page: 1, gapBefore: 0.5 },
    ]);
  });

  it('accepts values that sit on the size limits', () => {
    const count = MAX_SEGMENTS;
    const segments = Array.from({ length: count }, (_, index) =>
      segment(index, index === 1 ? { gapBefore: 100000, page: 100000 } : {}),
    );
    segments[0] = segment(0, { width: MAX_WIDTH, height: MAX_SEGMENT_HEIGHT, page: 0 });
    for (const entry of segments) entry.width = MAX_WIDTH;
    const parsed = validateManifest(
      manifest({
        width: MAX_WIDTH,
        pageCount: 10000,
        background: '#aabbccdd',
        detector: 'cv',
        segments,
      }),
    );
    expect(parsed.segments).toHaveLength(count);
    expect(parsed.segments[0]).toMatchObject({ width: MAX_WIDTH, height: MAX_SEGMENT_HEIGHT });
    expect(parsed.segments[1]).toMatchObject({ page: 100000, gapBefore: 100000 });
    expect(parsed.pageCount).toBe(10000);
  });

  it('rejects a non-object and a package that is not tate', () => {
    expect(failure(null)).toMatchObject({ code: 'bad_manifest', detail: 'root' });
    expect(failure([])).toMatchObject({ code: 'bad_manifest', detail: 'root' });
    expect(failure('tate')).toMatchObject({ code: 'bad_manifest', detail: 'root' });
    expect(failure(manifest({ format: 'cbz' })).code).toBe('not_tate');
    expect(failure(manifest({ version: 2 }))).toMatchObject({ code: 'version', detail: '2' });
    expect(failure(manifest({ version: 99 }))).toMatchObject({ code: 'version', detail: '99' });
  });

  it('rejects missing fields', () => {
    const keys = [
      'title',
      'source',
      'createdAt',
      'reading',
      'width',
      'background',
      'detector',
      'pageCount',
      'cover',
      'segments',
      'version',
    ] as const;
    for (const key of keys) {
      const value = Object.fromEntries(Object.entries(manifest()).filter(([name]) => name !== key));
      expect(failure(value).code, key).toBe('bad_manifest');
    }
  });

  it('rejects the wrong types', () => {
    const cases: Array<[Record<string, unknown>, string]> = [
      [{ version: '1' }, 'version'],
      [{ version: 1.5 }, 'version'],
      [{ version: 0 }, 'version'],
      [{ version: -3 }, 'version'],
      [{ title: 1 }, 'title'],
      [{ title: '' }, 'title'],
      [{ title: '   ' }, 'title'],
      [{ source: 1 }, 'source'],
      [{ source: 'a/b.cbz' }, 'source'],
      [{ source: '..\\x' }, 'source'],
      [{ source: 'foo..bar' }, 'source'],
      [{ createdAt: 1 }, 'createdAt'],
      [{ createdAt: 'yesterday' }, 'createdAt'],
      [{ reading: true }, 'reading'],
      [{ reading: 'ttb' }, 'reading'],
      [{ width: '1080' }, 'width'],
      [{ width: 1.5 }, 'width'],
      [{ width: Number.NaN }, 'width'],
      [{ width: 0 }, 'width'],
      [{ background: 123 }, 'background'],
      [{ background: 'white' }, 'background'],
      [{ background: '#gg0000' }, 'background'],
      [{ background: '#ffff' }, 'background'],
      [{ detector: 1 }, 'detector'],
      [{ detector: 'magic' }, 'detector'],
      [{ pageCount: '2' }, 'pageCount'],
      [{ pageCount: 0 }, 'pageCount'],
      [{ pageCount: 1.2 }, 'pageCount'],
      [{ cover: 1 }, 'cover'],
      [{ cover: 'cover.png' }, 'cover'],
      [{ segments: {} }, 'segments'],
      [{ segments: 'strip' }, 'segments'],
      [{ segments: [] }, 'segments'],
      [{ segments: [null] }, 'segments[0]'],
      [{ segments: [segment(0, { file: 'strip/0002.webp' })] }, 'segments[0].file'],
      [{ segments: [segment(0, { width: 100 })] }, 'segments[0].width'],
      [{ segments: [segment(0, { height: '800' })] }, 'segments[0].height'],
      [{ segments: [segment(0, { height: 1.2 })] }, 'segments[0].height'],
      [{ segments: [segment(0, { page: null })] }, 'segments[0].page'],
      [{ segments: [segment(0, { page: -1 })] }, 'segments[0].page'],
      [{ segments: [segment(0, { page: 1.5 })] }, 'segments[0].page'],
      [{ segments: [segment(0, { gapBefore: '0' })] }, 'segments[0].gapBefore'],
      [{ segments: [segment(0, { gapBefore: Number.NaN })] }, 'segments[0].gapBefore'],
      [{ segments: [segment(0, { gapBefore: -1 })] }, 'segments[0].gapBefore'],
      [{ segments: [segment(0, { gapBefore: 4 })] }, 'segments[0].gapBefore'],
      [{ segments: [segment(0, { box: [1, 2, 3] })] }, 'segments[0].box'],
      [{ segments: [segment(0, { box: [1, 2, 3, Number.NaN] })] }, 'segments[0].box'],
      [{ segments: [segment(0, { box: 'nope' })] }, 'segments[0].box'],
    ];
    for (const [overrides, detail] of cases) {
      expect(failure(manifest(overrides)), detail).toMatchObject({ code: 'bad_manifest', detail });
    }
  });

  it('rejects huge numbers', () => {
    const cases: Array<[Record<string, unknown>, string]> = [
      [{ width: MAX_WIDTH + 1 }, 'width'],
      [{ width: Number.MAX_SAFE_INTEGER }, 'width'],
      [{ pageCount: 10001 }, 'pageCount'],
      [{ pageCount: 1e9 }, 'pageCount'],
      [{ title: 'a'.repeat(301) }, 'title'],
      [{ source: 'b'.repeat(301) }, 'source'],
      [{ createdAt: `${'2026-10-25T04:00:00.000Z'}${'0'.repeat(40)}` }, 'createdAt'],
      [{ segments: [segment(0, { height: MAX_SEGMENT_HEIGHT + 1 })] }, 'segments[0].height'],
      [{ segments: [segment(0, { height: 1e9 })] }, 'segments[0].height'],
      [{ segments: [segment(0, { page: 100001 })] }, 'segments[0].page'],
      [{ segments: [segment(0, { page: 1e9 })] }, 'segments[0].page'],
      [{ segments: [segment(0, { gapBefore: 100001 })] }, 'segments[0].gapBefore'],
      [
        { segments: [segment(0, { gapBefore: Number.POSITIVE_INFINITY })] },
        'segments[0].gapBefore',
      ],
      [{ segments: Array.from({ length: MAX_SEGMENTS + 1 }, () => ({})) }, 'segments'],
    ];
    for (const [overrides, detail] of cases) {
      expect(failure(manifest(overrides)), detail).toMatchObject({ code: 'bad_manifest', detail });
    }
  });
});
