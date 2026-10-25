import { describe, expect, it } from 'vitest';
import { clamp, percentFromProgress, readingFraction, scrollTopFor } from './progress';

describe('reading progress', () => {
  it('maps a segment index and offset onto the chapter', () => {
    expect(readingFraction(0, 0, 10)).toBe(0);
    expect(readingFraction(1, 0.5, 10)).toBeCloseTo(0.15);
    expect(readingFraction(9, 1, 10)).toBe(1);
    expect(readingFraction(-4, -1, 4)).toBe(0);
    expect(readingFraction(8, 4, 4)).toBe(1);
    expect(readingFraction(0, 0, 0)).toBe(0);
    expect(percentFromProgress(1, 0.5, 10)).toBe(15);
  });

  it('turns a saved offset into a scroll position', () => {
    expect(scrollTopFor(100, 200, 0.25)).toBe(150);
    expect(scrollTopFor(100, 200, 1.5)).toBe(400);
    expect(scrollTopFor(Number.NaN, Number.NaN, Number.NaN)).toBe(0);
    expect(scrollTopFor(50, 20, -1)).toBe(50);
    expect(scrollTopFor(Number.POSITIVE_INFINITY, 20, 0.5)).toBe(10);
    expect(scrollTopFor(50, Number.POSITIVE_INFINITY, 0.5)).toBe(50);
    expect(scrollTopFor(10, 10, Number.POSITIVE_INFINITY)).toBe(10);
    expect(scrollTopFor(0, 100, 2)).toBe(200);
  });

  it('clamps non-finite positions and rounds the percent', () => {
    expect(clamp(Number.NaN, 2, 4)).toBe(2);
    expect(clamp(Number.POSITIVE_INFINITY, 0, 1)).toBe(0);
    expect(clamp(9, 2, 4)).toBe(4);
    expect(clamp(3, 2, 4)).toBe(3);
    expect(readingFraction(Number.NaN, Number.NaN, 5)).toBe(0);
    expect(readingFraction(Number.POSITIVE_INFINITY, 0.2, 5)).toBeCloseTo(0.04);
    expect(readingFraction(1.9, 0, 10)).toBeCloseTo(0.1);
    expect(readingFraction(0, 1, 1)).toBe(1);
    expect(readingFraction(3, 0.5, -2)).toBe(0);
    expect(percentFromProgress(0, 0.005, 1)).toBe(1);
    expect(percentFromProgress(0, 0.004, 1)).toBe(0);
    expect(percentFromProgress(1, 0.26, 10)).toBe(13);
    expect(percentFromProgress(Number.NaN, Number.POSITIVE_INFINITY, 4)).toBe(0);
  });
});
