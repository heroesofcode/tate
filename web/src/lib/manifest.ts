import { PackageError } from './errors';
import { MAX_SEGMENT_HEIGHT, MAX_SEGMENTS, MAX_WIDTH } from './limits';

export interface TateSegment {
  file: string;
  width: number;
  height: number;
  page: number;
  gapBefore: number;
}

export interface TateManifest {
  format: 'tate';
  version: 1;
  title: string;
  source: string;
  createdAt: string;
  reading: 'rtl' | 'ltr';
  width: number;
  background: string;
  detector: 'yolo' | 'cv' | 'page';
  pageCount: number;
  cover: string;
  segments: TateSegment[];
}

const DETECTORS = new Set(['yolo', 'cv', 'page']);
const READINGS = new Set(['rtl', 'ltr']);
const BACKGROUND = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;

function asRecord(value: unknown, detail: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new PackageError('bad_manifest', detail);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown, detail: string, max: number): string {
  if (typeof value !== 'string') throw new PackageError('bad_manifest', detail);
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > max) throw new PackageError('bad_manifest', detail);
  return trimmed;
}

function integer(value: unknown, detail: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min || value > max) {
    throw new PackageError('bad_manifest', detail);
  }
  return value;
}

export function validateManifest(value: unknown): TateManifest {
  const data = asRecord(value, 'root');
  if (data.format !== 'tate') throw new PackageError('not_tate');
  if (typeof data.version !== 'number' || !Number.isInteger(data.version)) {
    throw new PackageError('bad_manifest', 'version');
  }
  if (data.version > 1) throw new PackageError('version', String(data.version));
  if (data.version < 1) throw new PackageError('bad_manifest', 'version');

  const title = text(data.title, 'title', 300);
  const source = text(data.source, 'source', 300);
  if (/[/\\]/.test(source) || source.includes('..'))
    throw new PackageError('bad_manifest', 'source');

  const createdAt = text(data.createdAt, 'createdAt', 40);
  if (Number.isNaN(Date.parse(createdAt))) throw new PackageError('bad_manifest', 'createdAt');

  if (typeof data.reading !== 'string' || !READINGS.has(data.reading)) {
    throw new PackageError('bad_manifest', 'reading');
  }
  const width = integer(data.width, 'width', 1, MAX_WIDTH);
  if (typeof data.background !== 'string' || !BACKGROUND.test(data.background)) {
    throw new PackageError('bad_manifest', 'background');
  }
  if (typeof data.detector !== 'string' || !DETECTORS.has(data.detector)) {
    throw new PackageError('bad_manifest', 'detector');
  }
  const pageCount = integer(data.pageCount, 'pageCount', 1, 10000);
  if (data.cover !== 'cover.webp') throw new PackageError('bad_manifest', 'cover');
  if (
    !Array.isArray(data.segments) ||
    data.segments.length < 1 ||
    data.segments.length > MAX_SEGMENTS
  ) {
    throw new PackageError('bad_manifest', 'segments');
  }

  const segments = data.segments.map((entry, index) => {
    const segment = asRecord(entry, `segments[${index}]`);
    const file = `strip/${String(index + 1).padStart(4, '0')}.webp`;
    if (segment.file !== file) throw new PackageError('bad_manifest', `segments[${index}].file`);
    if (segment.width !== width) throw new PackageError('bad_manifest', `segments[${index}].width`);
    const height = integer(segment.height, `segments[${index}].height`, 1, MAX_SEGMENT_HEIGHT);
    const page = integer(segment.page, `segments[${index}].page`, 0, pageCount - 1);
    if (
      typeof segment.gapBefore !== 'number' ||
      !Number.isFinite(segment.gapBefore) ||
      segment.gapBefore < 0 ||
      segment.gapBefore > 100000
    ) {
      throw new PackageError('bad_manifest', `segments[${index}].gapBefore`);
    }
    if (index === 0 && segment.gapBefore !== 0) {
      throw new PackageError('bad_manifest', 'segments[0].gapBefore');
    }
    if (segment.box !== undefined) {
      if (
        !Array.isArray(segment.box) ||
        segment.box.length !== 4 ||
        segment.box.some((part) => typeof part !== 'number' || !Number.isFinite(part))
      ) {
        throw new PackageError('bad_manifest', `segments[${index}].box`);
      }
    }
    return { file, width, height, page, gapBefore: segment.gapBefore };
  });

  return {
    format: 'tate',
    version: 1,
    title,
    source,
    createdAt,
    reading: data.reading as 'rtl' | 'ltr',
    width,
    background: data.background,
    detector: data.detector as 'yolo' | 'cv' | 'page',
    pageCount,
    cover: 'cover.webp',
    segments,
  };
}
