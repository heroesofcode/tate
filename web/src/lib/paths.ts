function hasControlChar(name: string): boolean {
  for (let index = 0; index < name.length; index += 1) {
    if (name.charCodeAt(index) <= 0x1f) return true;
  }
  return false;
}

export function isSafeZipPath(name: string): boolean {
  if (!name || name.length > 300) return false;
  if (hasControlChar(name) || name.includes('\\')) return false;
  if (name.startsWith('/') || /^[a-zA-Z]:/.test(name)) return false;
  const trimmed = name.endsWith('/') ? name.slice(0, -1) : name;
  if (!trimmed) return false;
  return trimmed.split('/').every((part) => part !== '' && part !== '.' && part !== '..');
}

export function basename(name: string): string {
  const parts = name.split(/[/\\]/);
  return parts[parts.length - 1] || name;
}

export function fileStem(name: string): string {
  const base = basename(name);
  return base.replace(/\.(tate|cbz|zip)$/i, '') || base;
}

const IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif', '.bmp']);

export function isImagePath(name: string): boolean {
  const base = basename(name).toLowerCase();
  if (!base || base.startsWith('.') || base === 'thumbs.db') return false;
  const dot = base.lastIndexOf('.');
  if (dot < 0) return false;
  return IMAGE_EXT.has(base.slice(dot));
}

export function mimeForPath(name: string): string {
  const base = basename(name).toLowerCase();
  const dot = base.lastIndexOf('.');
  const ext = dot >= 0 ? base.slice(dot) : '';
  switch (ext) {
    case '.webp':
      return 'image/webp';
    case '.jpg':
    case '.jpeg':
      return 'image/jpeg';
    case '.png':
      return 'image/png';
    case '.gif':
      return 'image/gif';
    case '.avif':
      return 'image/avif';
    case '.bmp':
      return 'image/bmp';
    default:
      return 'application/octet-stream';
  }
}
