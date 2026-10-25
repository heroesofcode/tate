export type Route = { name: 'library' } | { name: 'reader'; id: string };

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#/, '') || '/';
  const match = /^\/read\/([^/?#]+)$/.exec(path);
  if (!match?.[1]) return { name: 'library' };
  try {
    return { name: 'reader', id: decodeURIComponent(match[1]) };
  } catch {
    return { name: 'library' };
  }
}

export function libraryHash(): string {
  return '#/';
}

export function readerHash(id: string): string {
  return `#/read/${encodeURIComponent(id)}`;
}
