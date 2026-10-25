import { strToU8, zipSync, type Zippable } from 'fflate';
import { describe, expect, it } from 'vitest';
import { PackageError } from './errors';
import { LIMITS } from './limits';
import { importPackage } from './package';
import { unzipArchive } from './unzip';

function png(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

function manifest(overrides: Record<string, unknown> = {}): Record<string, unknown> {
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
    pageCount: 2,
    cover: 'cover.webp',
    extra: 'ignored',
    segments: [
      { file: 'strip/0001.webp', width: 1080, height: 800, page: 0, gapBefore: 0, note: 'ok' },
      {
        file: 'strip/0002.webp',
        width: 1080,
        height: 900,
        page: 1,
        gapBefore: 24,
        box: [0, 0, 10, 10],
      },
    ],
    ...overrides,
  };
}

function copyBytes(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy;
}

function archive(files: Record<string, Uint8Array | string>): Uint8Array {
  const entries: Zippable = {};
  for (const [key, value] of Object.entries(files)) {
    const bytes = typeof value === 'string' ? strToU8(value) : value;
    entries[key] = [copyBytes(bytes), { level: key.endsWith('.webp') ? 0 : 6 }];
  }
  return zipSync(entries);
}

function file(bytes: Uint8Array, name: string): File {
  return new File([copyBytes(bytes)], name, { type: 'application/zip' });
}

function tateFile(name = 'chapter.tate', overrides: Record<string, unknown> = {}): File {
  return file(
    archive({
      'manifest.json': JSON.stringify(manifest(overrides)),
      'cover.webp': new Uint8Array([1, 2, 3, 4]),
      'strip/0001.webp': new Uint8Array([5, 6, 7]),
      'strip/0002.webp': new Uint8Array([8, 9]),
      'notes.txt': 'ignore me',
    }),
    name,
  );
}

describe('tate packages', () => {
  it('imports a valid chapter and ignores unknown fields', async () => {
    const chapter = await importPackage(tateFile());
    expect(chapter.title).toBe('Chapter One');
    expect(chapter.unprocessed).toBe(false);
    expect(chapter.segments).toHaveLength(2);
    expect(chapter.segments[1]?.gapBefore).toBe(24);
    expect(chapter.cover.type).toBe('image/webp');
  });

  it('imports a .zip that is actually a tate package', async () => {
    const chapter = await importPackage(tateFile('renamed.zip'));
    expect(chapter.unprocessed).toBe(false);
    expect(chapter.title).toBe('Chapter One');
  });

  it('rejects a newer format version', async () => {
    await expect(importPackage(tateFile('chapter.tate', { version: 2 }))).rejects.toMatchObject({
      code: 'version',
      detail: '2',
    });
  });

  it('rejects the wrong format, bad manifests, and missing files', async () => {
    await expect(
      importPackage(tateFile('chapter.tate', { format: 'manga' })),
    ).rejects.toMatchObject({
      code: 'not_tate',
    });
    await expect(importPackage(tateFile('chapter.tate', { version: 0 }))).rejects.toMatchObject({
      code: 'bad_manifest',
    });
    await expect(
      importPackage(tateFile('chapter.tate', { source: 'a/b.cbz' })),
    ).rejects.toMatchObject({
      code: 'bad_manifest',
      detail: 'source',
    });
    await expect(importPackage(tateFile('chapter.tate', { title: '  ' }))).rejects.toMatchObject({
      code: 'bad_manifest',
    });
    const tall = manifest();
    (tall.segments as Array<Record<string, unknown>>)[0] = {
      file: 'strip/0001.webp',
      width: 1080,
      height: 4097,
      page: 0,
      gapBefore: 0,
    };
    await expect(
      importPackage(
        file(
          archive({
            'manifest.json': JSON.stringify(tall),
            'cover.webp': new Uint8Array([1]),
            'strip/0001.webp': new Uint8Array([2]),
            'strip/0002.webp': new Uint8Array([3]),
          }),
          'chapter.tate',
        ),
      ),
    ).rejects.toMatchObject({ code: 'bad_manifest' });
    await expect(
      importPackage(file(archive({ 'cover.webp': new Uint8Array([1]) }), 'chapter.tate')),
    ).rejects.toMatchObject({
      code: 'missing_manifest',
    });
    await expect(
      importPackage(file(archive({ 'manifest.json': '{' }), 'chapter.tate')),
    ).rejects.toMatchObject({
      code: 'bad_manifest',
    });
    const missing = manifest();
    await expect(
      importPackage(
        file(
          archive({
            'manifest.json': JSON.stringify(missing),
            'cover.webp': new Uint8Array([1]),
            'strip/0001.webp': new Uint8Array([2]),
          }),
          'chapter.tate',
        ),
      ),
    ).rejects.toMatchObject({ code: 'missing_file', detail: 'strip/0002.webp' });
  });

  it('rejects archives that are not zip data', async () => {
    await expect(
      importPackage(file(new Uint8Array([1, 2, 3, 4]), 'nope.tate')),
    ).rejects.toBeInstanceOf(PackageError);
  });
});

describe('unsafe archives', () => {
  it('rejects paths that escape the package', async () => {
    const slipped = archive({
      'manifest.json': JSON.stringify(manifest()),
      '../evil.txt': 'nope',
    });
    await expect(importPackage(file(slipped, 'chapter.tate'))).rejects.toMatchObject({
      code: 'unsafe_path',
    });
    const absolute = zipSync({ '/etc/passwd': strToU8('root') });
    expect(() => unzipArchive(absolute)).toThrow(PackageError);
  });

  it('rejects an oversized archive before reading it', async () => {
    let reads = 0;
    const source = {
      name: 'huge.tate',
      size: LIMITS.maxArchiveBytes + 1,
      arrayBuffer() {
        reads += 1;
        return Promise.resolve(new ArrayBuffer(4));
      },
    };
    await expect(importPackage(source as Blob & { name: string })).rejects.toMatchObject({
      code: 'too_large',
    });
    expect(reads).toBe(0);
  });

  it('still reads an archive that sits on the size cap', async () => {
    let reads = 0;
    const bytes = new Uint8Array([1, 2, 3, 4]);
    const source = {
      name: 'edge.tate',
      size: LIMITS.maxArchiveBytes,
      arrayBuffer() {
        reads += 1;
        return Promise.resolve(bytes.buffer);
      },
    };
    await expect(importPackage(source as Blob & { name: string })).rejects.toSatisfy(
      (error: PackageError) => error instanceof PackageError && error.code !== 'too_large',
    );
    expect(reads).toBe(1);
  });

  it('rejects archives that declare or expand past the size limit', () => {
    const big = new Uint8Array(4000);
    const zipped = zipSync({ 'big.bin': big, 'also.bin': new Uint8Array(4000) });
    const limits = { ...LIMITS, maxFileBytes: 1000, maxTotalBytes: 5000, maxEntries: 10 };
    expect(() => unzipArchive(zipped, limits)).toThrow(PackageError);
    const many = zipSync({
      'a.txt': strToU8('a'),
      'b.txt': strToU8('b'),
      'c.txt': strToU8('c'),
    });
    expect(() => unzipArchive(many, { ...LIMITS, maxEntries: 2 })).toThrow(PackageError);
  });
});

describe('cbz import', () => {
  it('turns page images into segments in natural order', async () => {
    const packed = archive({
      'pages/page10.png': png(20, 30),
      'pages/page2.png': png(10, 40),
      'pages/page1.png': png(8, 8),
      '__MACOSX/pages/page1.png': png(1, 1),
      'notes.txt': 'nope',
      '.hidden.png': png(2, 2),
    });
    const chapter = await importPackage(file(packed, 'My Chapter.cbz'));
    expect(chapter.unprocessed).toBe(true);
    expect(chapter.title).toBe('My Chapter');
    expect(chapter.source).toBe('My Chapter.cbz');
    expect(chapter.segments.map((segment) => segment.width)).toEqual([8, 10, 20]);
    expect(chapter.segments.map((segment) => segment.height)).toEqual([8, 40, 30]);
    expect(chapter.segments[0]?.gapBefore).toBe(0);
    expect(chapter.segments[1]?.gapBefore).toBe(36);
    expect(chapter.detector).toBe('page');
  });
});
