export interface ImageSize {
  width: number;
  height: number;
}

function ascii(bytes: Uint8Array, start: number, length: number): string {
  let out = '';
  for (let i = 0; i < length; i += 1) out += String.fromCharCode(bytes[start + i] ?? 0);
  return out;
}

function viewOf(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function sane(width: number, height: number): ImageSize | null {
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
  if (width < 1 || height < 1 || width > 20000 || height > 20000) return null;
  return { width, height };
}

function parsePng(bytes: Uint8Array): ImageSize | null {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 24) return null;
  for (let i = 0; i < signature.length; i += 1) if (bytes[i] !== signature[i]) return null;
  if (ascii(bytes, 12, 4) !== 'IHDR') return null;
  const view = viewOf(bytes);
  return sane(view.getUint32(16), view.getUint32(20));
}

function parseGif(bytes: Uint8Array): ImageSize | null {
  if (bytes.length < 10) return null;
  const header = ascii(bytes, 0, 6);
  if (header !== 'GIF87a' && header !== 'GIF89a') return null;
  const view = viewOf(bytes);
  return sane(view.getUint16(6, true), view.getUint16(8, true));
}

function parseJpeg(bytes: Uint8Array): ImageSize | null {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) return null;
  const view = viewOf(bytes);
  let offset = 2;
  while (offset + 9 < bytes.length) {
    if (bytes[offset] !== 0xff) return null;
    const marker = bytes[offset + 1] ?? 0;
    const startOfFrame =
      marker === 0xc0 ||
      marker === 0xc1 ||
      marker === 0xc2 ||
      marker === 0xc3 ||
      marker === 0xc9 ||
      marker === 0xca;
    if (startOfFrame) return sane(view.getUint16(offset + 7), view.getUint16(offset + 5));
    if (
      marker === 0xd8 ||
      marker === 0xd9 ||
      marker === 0x01 ||
      (marker >= 0xd0 && marker <= 0xd7)
    ) {
      offset += 2;
      continue;
    }
    if (offset + 4 > bytes.length) return null;
    const size = view.getUint16(offset + 2);
    if (size < 2) return null;
    offset += 2 + size;
  }
  return null;
}

function parseWebp(bytes: Uint8Array): ImageSize | null {
  if (bytes.length < 30) return null;
  if (ascii(bytes, 0, 4) !== 'RIFF' || ascii(bytes, 8, 4) !== 'WEBP') return null;
  const chunk = ascii(bytes, 12, 4);
  const view = viewOf(bytes);
  if (chunk === 'VP8X') {
    const width = 1 + (bytes[24] ?? 0) + ((bytes[25] ?? 0) << 8) + ((bytes[26] ?? 0) << 16);
    const height = 1 + (bytes[27] ?? 0) + ((bytes[28] ?? 0) << 8) + ((bytes[29] ?? 0) << 16);
    return sane(width, height);
  }
  if (chunk === 'VP8 ') {
    for (let i = 20; i < bytes.length - 7; i += 1) {
      if (bytes[i] === 0x9d && bytes[i + 1] === 0x01 && bytes[i + 2] === 0x2a) {
        return sane(view.getUint16(i + 3, true) & 0x3fff, view.getUint16(i + 5, true) & 0x3fff);
      }
    }
  }
  if (chunk === 'VP8L' && bytes[20] === 0x2f) {
    const b0 = bytes[21] ?? 0;
    const b1 = bytes[22] ?? 0;
    const b2 = bytes[23] ?? 0;
    const b3 = bytes[24] ?? 0;
    const width = 1 + (((b1 & 0x3f) << 8) | b0);
    const height = 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6));
    return sane(width, height);
  }
  return null;
}

export function parseImageSize(bytes: Uint8Array): ImageSize | null {
  return parsePng(bytes) ?? parseGif(bytes) ?? parseJpeg(bytes) ?? parseWebp(bytes);
}

export async function readImageSize(blob: Blob): Promise<ImageSize> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const parsed = parseImageSize(bytes);
  if (parsed) return parsed;
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(blob);
      const size = { width: bitmap.width, height: bitmap.height };
      bitmap.close();
      if (size.width > 0 && size.height > 0) return size;
    } catch {
      /* fall through */
    }
  }
  return { width: 1080, height: 1600 };
}
