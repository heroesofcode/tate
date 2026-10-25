import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const svg = await readFile(join(root, 'scripts', 'icon.svg'));
const iconsDir = join(root, 'public', 'icons');
await mkdir(iconsDir, { recursive: true });

async function render(size, scale) {
  const inner = Math.round(size * scale);
  const margin = Math.round((size - inner) / 2);
  const layer = await sharp(svg).resize(inner, inner).png().toBuffer();
  return sharp({
    create: { width: size, height: size, channels: 4, background: '#0c0c0f' },
  })
    .composite([{ input: layer, left: margin, top: margin }])
    .png()
    .toBuffer();
}

const any192 = await render(192, 1);
const any512 = await render(512, 1);
const maskable192 = await render(192, 0.78);
const maskable512 = await render(512, 0.78);
const apple = await render(180, 1);

await writeFile(join(iconsDir, 'icon-192.png'), any192);
await writeFile(join(iconsDir, 'icon-512.png'), any512);
await writeFile(join(iconsDir, 'icon-maskable-192.png'), maskable192);
await writeFile(join(iconsDir, 'icon-maskable-512.png'), maskable512);
await writeFile(join(iconsDir, 'apple-touch-icon.png'), apple);
await writeFile(join(root, 'public', 'favicon.svg'), svg);
