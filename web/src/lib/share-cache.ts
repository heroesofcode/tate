export const SHARE_CACHE = 'tate-share-inbox';

export function inboxRequestUrl(scope: string, index: number, fileName: string): string {
  const base = scope.endsWith('/') ? scope : `${scope}/`;
  return new URL(`share-inbox/${index}/${encodeURIComponent(fileName)}`, base).href;
}

export function fileNameFromInboxUrl(url: string): string {
  const encoded = new URL(url).pathname.split('/').pop() ?? 'chapter.tate';
  try {
    return decodeURIComponent(encoded) || 'chapter.tate';
  } catch {
    return 'chapter.tate';
  }
}
