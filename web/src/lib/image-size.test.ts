import { strToU8 } from 'fflate';
import { describe, expect, it } from 'vitest';
import { parseImageSize } from './image-size';

function png(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(24);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes;
}

function gif(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(10);
  bytes.set([0x47, 0x49, 0x46, 0x38, 0x39, 0x61]);
  const view = new DataView(bytes.buffer);
  view.setUint16(6, width, true);
  view.setUint16(8, height, true);
  return bytes;
}

function webp(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(30);
  bytes.set(strToU8('RIFF'), 0);
  bytes.set(strToU8('WEBP'), 8);
  bytes.set(strToU8('VP8X'), 12);
  const w = width - 1;
  const h = height - 1;
  bytes[24] = w & 255;
  bytes[25] = (w >> 8) & 255;
  bytes[26] = (w >> 16) & 255;
  bytes[27] = h & 255;
  bytes[28] = (h >> 8) & 255;
  bytes[29] = (h >> 16) & 255;
  return bytes;
}

function jpeg(width: number, height: number): Uint8Array {
  const bytes = new Uint8Array(20);
  bytes[0] = 0xff;
  bytes[1] = 0xd8;
  bytes[2] = 0xff;
  bytes[3] = 0xc0;
  const view = new DataView(bytes.buffer);
  view.setUint16(7, height);
  view.setUint16(9, width);
  return bytes;
}

describe('image sizes', () => {
  it('reads png, gif, jpeg, and webp headers', () => {
    expect(parseImageSize(png(8, 13))).toEqual({ width: 8, height: 13 });
    expect(parseImageSize(gif(320, 480))).toEqual({ width: 320, height: 480 });
    expect(parseImageSize(jpeg(1080, 900))).toEqual({ width: 1080, height: 900 });
    expect(parseImageSize(webp(360, 240))).toEqual({ width: 360, height: 240 });
    expect(parseImageSize(new Uint8Array([1, 2, 3]))).toBeNull();
  });
});
