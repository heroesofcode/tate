import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { strToU8, zipSync, type Zippable } from 'fflate';
import sharp from 'sharp';

export async function createFixtureTate(): Promise<string> {
  const colors = ['#f4efe6', '#1c1c22', '#ff4d2e', '#f6f1e7', '#203040', '#f0a202'];
  const images: Uint8Array[] = [];
  for (let index = 0; index < colors.length; index += 1) {
    const ink = index % 2 === 0 ? '#1c1c22' : '#f6f1e7';
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="900">
      <rect width="100%" height="100%" fill="${colors[index] ?? '#ffffff'}"/>
      <text x="50%" y="50%" font-size="220" text-anchor="middle" dominant-baseline="middle" fill="${ink}" font-family="sans-serif">${index + 1}</text>
    </svg>`;
    const webp = await sharp(Buffer.from(svg)).webp({ quality: 60 }).toBuffer();
    images.push(new Uint8Array(webp));
  }
  const cover = await sharp(Buffer.from(images[0] ?? new Uint8Array()))
    .resize({ width: 360 })
    .webp()
    .toBuffer();
  const manifest = {
    format: 'tate',
    version: 1,
    title: 'Fixture Chapter',
    source: 'fixture.cbz',
    createdAt: '2026-10-25T04:00:00Z',
    reading: 'rtl',
    width: 1080,
    background: '#f4efe6',
    detector: 'page',
    pageCount: images.length,
    cover: 'cover.webp',
    segments: images.map((_, index) => ({
      file: `strip/${String(index + 1).padStart(4, '0')}.webp`,
      width: 1080,
      height: 900,
      page: index,
      gapBefore: index === 0 ? 0 : index % 2 === 0 ? 64 : 16,
    })),
  };
  const entries: Zippable = {
    'manifest.json': [strToU8(JSON.stringify(manifest)), { level: 6 }],
    'cover.webp': [new Uint8Array(cover), { level: 0 }],
  };
  images.forEach((image, index) => {
    const copy = new Uint8Array(image.byteLength);
    copy.set(image);
    entries[`strip/${String(index + 1).padStart(4, '0')}.webp`] = [copy, { level: 0 }];
  });
  const dir = await mkdtemp(join(tmpdir(), 'tate-fixture-'));
  const path = join(dir, 'fixture.tate');
  await writeFile(path, zipSync(entries));
  return path;
}

async function writePackage(name: string, bytes: Uint8Array): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'tate-fixture-'));
  const path = join(dir, name);
  await writeFile(path, bytes);
  return path;
}

let cachedPanel: Promise<Uint8Array> | null = null;

function panelImage(): Promise<Uint8Array> {
  cachedPanel ??= sharp(
    Buffer.from(
      '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="#203040"/></svg>',
    ),
  )
    .webp({ quality: 20 })
    .toBuffer()
    .then((buffer) => new Uint8Array(buffer));
  return cachedPanel;
}

function copyBytes(bytes: Uint8Array): Uint8Array {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy;
}

function pad(index: number): string {
  return String(index).padStart(4, '0');
}

export async function createChapterPackage(options: {
  title: string;
  fileName?: string;
  source?: string;
  count?: number;
  width?: number;
  height?: number;
  gap?: number;
}): Promise<string> {
  const count = options.count ?? 4;
  const width = options.width ?? 1080;
  const height = options.height ?? 900;
  const gap = options.gap ?? 16;
  const image = await panelImage();
  const manifest = {
    format: 'tate',
    version: 1,
    title: options.title,
    source: options.source ?? `${options.title}.cbz`,
    createdAt: '2026-10-25T04:00:00Z',
    reading: 'rtl',
    width,
    background: '#f4efe6',
    detector: 'page',
    pageCount: count,
    cover: 'cover.webp',
    segments: Array.from({ length: count }, (_, index) => ({
      file: `strip/${pad(index + 1)}.webp`,
      width,
      height,
      page: index,
      gapBefore: index === 0 ? 0 : gap,
    })),
  };
  const entries: Zippable = {
    'manifest.json': [strToU8(JSON.stringify(manifest)), { level: 6 }],
    'cover.webp': [copyBytes(image), { level: 0 }],
  };
  for (let index = 0; index < count; index += 1) {
    entries[`strip/${pad(index + 1)}.webp`] = [copyBytes(image), { level: 0 }];
  }
  return writePackage(options.fileName ?? 'chapter.tate', zipSync(entries));
}

export async function createCbzPackage(title: string): Promise<string> {
  const image = await panelImage();
  const entries: Zippable = {};
  for (const name of ['page1.webp', 'page2.webp', 'page10.webp']) {
    entries[`pages/${name}`] = [copyBytes(image), { level: 0 }];
  }
  return writePackage(`${title}.cbz`, zipSync(entries));
}

export async function createCorruptPackage(): Promise<string> {
  return writePackage('broken.tate', new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
}

export async function createInvalidManifestPackage(): Promise<string> {
  const manifest = {
    format: 'tate',
    version: 1,
    title: 42,
    source: 'bad.cbz',
    createdAt: 'not-a-date',
    reading: 'sideways',
    width: 1e15,
    background: 'nope',
    detector: 'page',
    pageCount: 1e9,
    cover: 'cover.png',
    segments: [],
  };
  return writePackage(
    'invalid.tate',
    zipSync({ 'manifest.json': strToU8(JSON.stringify(manifest)) }),
  );
}

/**
 * Zip whose header advertises a file bigger than the import cap.
 * The archive itself stays small because that file is not stored.
 */
export function oversizedZipBytes(): Uint8Array {
  const name = new TextEncoder().encode('huge.webp');
  const declared = 33 * 1024 * 1024;
  const local = new Uint8Array(30 + name.length);
  const view = new DataView(local.buffer);
  view.setUint32(0, 0x04034b50, true);
  view.setUint16(4, 20, true);
  view.setUint16(8, 0, true);
  view.setUint32(18, 0, true);
  view.setUint32(22, declared, true);
  view.setUint16(26, name.length, true);
  local.set(name, 30);

  const central = new Uint8Array(46 + name.length);
  const centralView = new DataView(central.buffer);
  centralView.setUint32(0, 0x02014b50, true);
  centralView.setUint16(4, 20, true);
  centralView.setUint16(6, 20, true);
  centralView.setUint32(20, 0, true);
  centralView.setUint32(24, declared, true);
  centralView.setUint16(28, name.length, true);
  centralView.setUint32(42, 0, true);
  central.set(name, 46);

  const end = new Uint8Array(22);
  const endView = new DataView(end.buffer);
  endView.setUint32(0, 0x06054b50, true);
  endView.setUint16(8, 1, true);
  endView.setUint16(10, 1, true);
  endView.setUint32(12, central.length, true);
  endView.setUint32(16, local.length, true);

  const bytes = new Uint8Array(local.length + central.length + end.length);
  bytes.set(local, 0);
  bytes.set(central, local.length);
  bytes.set(end, local.length + central.length);
  return bytes;
}

export async function createOversizedPackage(): Promise<string> {
  return writePackage('huge.tate', oversizedZipBytes());
}
