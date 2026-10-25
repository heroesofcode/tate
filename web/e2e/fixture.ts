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
