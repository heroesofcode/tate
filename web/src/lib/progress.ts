export function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** Portion of the chapter represented by a segment index plus an offset inside it. */
export function readingFraction(
  segmentIndex: number,
  offsetFraction: number,
  segmentCount: number,
): number {
  if (segmentCount <= 0) return 0;
  const index = clamp(Math.floor(segmentIndex), 0, segmentCount - 1);
  const offset = clamp(offsetFraction, 0, 1);
  return clamp((index + offset) / segmentCount, 0, 1);
}

export function percentFromProgress(
  segmentIndex: number,
  offsetFraction: number,
  segmentCount: number,
): number {
  return Math.round(readingFraction(segmentIndex, offsetFraction, segmentCount) * 100);
}

/** Pixel offset to restore a saved segment position. Offset may pass 1 into the end card. */
export function scrollTopFor(
  segmentTop: number,
  segmentHeight: number,
  offsetFraction: number,
): number {
  const top = Number.isFinite(segmentTop) ? segmentTop : 0;
  const height = Number.isFinite(segmentHeight) ? segmentHeight : 0;
  const offset = Number.isFinite(offsetFraction) ? Math.max(0, offsetFraction) : 0;
  return top + height * offset;
}
