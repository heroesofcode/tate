import { describe, expect, it } from 'vitest';
import { naturalCompare } from './natural-sort';

describe('naturalCompare', () => {
  it('orders embedded numbers the way a reader would', () => {
    const names = ['page10.png', 'page2.png', 'page1.png'];
    expect([...names].sort(naturalCompare)).toEqual(['page1.png', 'page2.png', 'page10.png']);
    expect(naturalCompare('dir/page2.jpg', 'dir/page10.jpg')).toBeLessThan(0);
  });
});
