export type PackageErrorCode =
  | 'too_large'
  | 'unsafe_path'
  | 'not_tate'
  | 'version'
  | 'bad_manifest'
  | 'missing_file'
  | 'missing_manifest'
  | 'no_images'
  | 'file_too_large'
  | 'too_many_entries'
  | 'bad_zip';

export class PackageError extends Error {
  readonly code: PackageErrorCode;
  readonly detail?: string;

  constructor(code: PackageErrorCode, detail?: string) {
    super(detail ? `${code}:${detail}` : code);
    this.name = 'PackageError';
    this.code = code;
    this.detail = detail;
  }
}
