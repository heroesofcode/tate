export interface UnzipLimits {
  maxArchiveBytes: number;
  maxFileBytes: number;
  maxTotalBytes: number;
  maxEntries: number;
}

export const LIMITS: UnzipLimits = {
  maxArchiveBytes: 512 * 1024 * 1024,
  maxFileBytes: 32 * 1024 * 1024,
  maxTotalBytes: 700 * 1024 * 1024,
  maxEntries: 4000,
};

export const MAX_SEGMENTS = 2000;
export const MAX_SEGMENT_HEIGHT = 4096;
export const MAX_WIDTH = 8192;
