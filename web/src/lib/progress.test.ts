import { describe, expect, it } from 'vitest';
import { percentFromProgress, readingFraction, scrollTopFor } from './progress';

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
  });
});
