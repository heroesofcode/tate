import { fileNameFromInboxUrl, SHARE_CACHE } from './share-cache';

export async function takeSharedFiles(): Promise<File[]> {
  if (typeof caches === 'undefined') return [];
  const cache = await caches.open(SHARE_CACHE);
  const requests = await cache.keys();
  const files: File[] = [];
  for (const request of requests) {
    const response = await cache.match(request);
    if (!response) continue;
    const name = fileNameFromInboxUrl(request.url);
    const blob = await response.blob();
    files.push(new File([blob], name, { type: blob.type || 'application/octet-stream' }));
    await cache.delete(request);
  }
  return files;
}
