import { Unzip, UnzipInflate, UnzipPassThrough } from 'fflate';
import { PackageError } from './errors';
import { LIMITS, type UnzipLimits } from './limits';
import { isSafeZipPath } from './paths';

export function unzipArchive(
  data: Uint8Array,
  limits: UnzipLimits = LIMITS,
): Map<string, Uint8Array> {
  if (data.byteLength > limits.maxArchiveBytes) throw new PackageError('too_large');

  const out = new Map<string, Uint8Array>();
  const unzipper = new Unzip();
  unzipper.register(UnzipInflate);
  unzipper.register(UnzipPassThrough);
  let total = 0;
  let declaredTotal = 0;
  let entries = 0;
  let failure: Error | null = null;

  const fail = (file: { terminate?: () => void }, error: Error) => {
    failure = error;
    try {
      file.terminate?.();
    } catch {
      /* stream already closed */
    }
  };

  unzipper.onfile = (file) => {
    if (failure) {
      fail(file, failure);
      return;
    }
    entries += 1;
    try {
      if (entries > limits.maxEntries) {
        fail(file, new PackageError('too_many_entries'));
        return;
      }
      if (!isSafeZipPath(file.name)) {
        fail(file, new PackageError('unsafe_path', file.name));
        return;
      }
      if (file.name.endsWith('/')) return;
      const declared = file.originalSize ?? 0;
      if (declared > limits.maxFileBytes) {
        fail(file, new PackageError('file_too_large', file.name));
        return;
      }
      declaredTotal += declared;
      if (declaredTotal > limits.maxTotalBytes) {
        fail(file, new PackageError('too_large'));
        return;
      }
      const chunks: Uint8Array[] = [];
      let size = 0;
      file.ondata = (err, chunk, final) => {
        if (failure) return;
        if (err) {
          failure = new PackageError('bad_zip');
          return;
        }
        size += chunk.byteLength;
        total += chunk.byteLength;
        if (size > limits.maxFileBytes) {
          fail(file, new PackageError('file_too_large', file.name));
          return;
        }
        if (total > limits.maxTotalBytes) {
          fail(file, new PackageError('too_large'));
          return;
        }
        if (chunk.byteLength > 0) chunks.push(chunk);
        if (final) {
          if (out.has(file.name)) {
            failure = new PackageError('bad_zip');
            return;
          }
          const merged = new Uint8Array(size);
          let offset = 0;
          for (const part of chunks) {
            merged.set(part, offset);
            offset += part.byteLength;
          }
          out.set(file.name, merged);
        }
      };
      file.start();
    } catch (error) {
      failure = error instanceof PackageError ? error : new PackageError('bad_zip');
    }
  };

  try {
    unzipper.push(data, true);
  } catch (error) {
    if (error instanceof PackageError) throw error;
    throw new PackageError('bad_zip');
  }
  if (failure) throw failure;
  return out;
}
