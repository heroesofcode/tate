import { strFromU8 } from 'fflate';
import { PackageError } from './errors';
import { readImageSize } from './image-size';
import { LIMITS, MAX_SEGMENTS } from './limits';
import { validateManifest } from './manifest';
import { naturalCompare } from './natural-sort';
import { basename, fileStem, isImagePath, mimeForPath } from './paths';
import { unzipArchive } from './unzip';

export interface ImportedSegment {
  index: number;
  width: number;
  height: number;
  gapBefore: number;
  page: number;
  blob: Blob;
}

export interface ImportedChapter {
  title: string;
  source: string;
  createdAt: string;
  reading: 'rtl' | 'ltr';
  width: number;
  background: string;
  detector: string;
  pageCount: number;
  unprocessed: boolean;
  cover: Blob;
  segments: ImportedSegment[];
}

function blobFrom(bytes: Uint8Array, type: string): Blob {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Blob([copy], { type });
}

function peekFormat(bytes: Uint8Array): 'tate' | 'other' | 'invalid' {
  try {
    const value = JSON.parse(strFromU8(bytes)) as { format?: unknown };
    if (value && typeof value === 'object' && !Array.isArray(value) && value.format === 'tate') {
      return 'tate';
    }
    return 'other';
  } catch {
    return 'invalid';
  }
}

function importTate(entries: Map<string, Uint8Array>): ImportedChapter {
  const manifestBytes = entries.get('manifest.json');
  if (!manifestBytes) throw new PackageError('missing_manifest');
  if (manifestBytes.byteLength > 2_000_000) throw new PackageError('bad_manifest', 'json');
  let parsed: unknown;
  try {
    let text = strFromU8(manifestBytes);
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    parsed = JSON.parse(text) as unknown;
  } catch {
    throw new PackageError('bad_manifest', 'json');
  }
  const manifest = validateManifest(parsed);
  const coverBytes = entries.get(manifest.cover);
  if (!coverBytes?.byteLength) throw new PackageError('missing_file', manifest.cover);
  const segments = manifest.segments.map((segment, index) => {
    const bytes = entries.get(segment.file);
    if (!bytes?.byteLength) throw new PackageError('missing_file', segment.file);
    return {
      index,
      width: segment.width,
      height: segment.height,
      gapBefore: segment.gapBefore,
      page: segment.page,
      blob: blobFrom(bytes, 'image/webp'),
    };
  });
  return {
    title: manifest.title,
    source: manifest.source,
    createdAt: manifest.createdAt,
    reading: manifest.reading,
    width: manifest.width,
    background: manifest.background,
    detector: manifest.detector,
    pageCount: manifest.pageCount,
    unprocessed: false,
    cover: blobFrom(coverBytes, 'image/webp'),
    segments,
  };
}

async function importCbz(
  entries: Map<string, Uint8Array>,
  filename: string,
): Promise<ImportedChapter> {
  const images = [...entries.entries()].filter(([path, bytes]) => {
    if (path.startsWith('__MACOSX/')) return false;
    return isImagePath(path) && bytes.byteLength > 0;
  });
  images.sort((a, b) => naturalCompare(a[0], b[0]));
  if (images.length === 0) throw new PackageError('no_images');
  if (images.length > MAX_SEGMENTS) throw new PackageError('too_many_entries');

  const segments: ImportedSegment[] = [];
  for (let index = 0; index < images.length; index += 1) {
    const [path, bytes] = images[index] ?? [];
    if (!path || !bytes) continue;
    const size = await readImageSize(blobFrom(bytes, mimeForPath(path)));
    segments.push({
      index,
      width: size.width,
      height: size.height,
      gapBefore: index === 0 ? 0 : 36,
      page: index,
      blob: blobFrom(bytes, mimeForPath(path)),
    });
  }
  const width = Math.max(...segments.map((segment) => segment.width));
  const source = basename(filename) || 'chapter.cbz';
  const title = fileStem(source) || 'Chapter';
  return {
    title,
    source,
    createdAt: new Date().toISOString(),
    reading: 'rtl',
    width,
    background: '#0e0e11',
    detector: 'page',
    pageCount: segments.length,
    unprocessed: true,
    cover: segments[0]?.blob ?? new Blob(),
    segments,
  };
}

export async function importPackage(file: Blob & { name?: string }): Promise<ImportedChapter> {
  if (file.size > LIMITS.maxArchiveBytes) throw new PackageError('too_large');
  const name = file.name || 'chapter.tate';
  const forceTate = name.toLowerCase().endsWith('.tate');
  const bytes = new Uint8Array(await file.arrayBuffer());
  let entries: Map<string, Uint8Array>;
  try {
    entries = unzipArchive(bytes);
  } catch (error) {
    if (error instanceof PackageError) throw error;
    throw new PackageError('bad_zip');
  }
  const manifestBytes = entries.get('manifest.json');
  if (forceTate && !manifestBytes) throw new PackageError('missing_manifest');
  if (manifestBytes && (forceTate || peekFormat(manifestBytes) === 'tate'))
    return importTate(entries);
  return importCbz(entries, name);
}
