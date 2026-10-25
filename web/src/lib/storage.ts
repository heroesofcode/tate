import { openDB, deleteDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { ImportedChapter } from './package';

export const DB_NAME = 'tate';
export const DB_VERSION = 1;

export interface SegmentLayout {
  width: number;
  height: number;
  gapBefore: number;
  page: number;
}

export interface ChapterRecord {
  id: string;
  identity: string;
  title: string;
  source: string;
  createdAt: string;
  importedAt: number;
  lastReadAt: number | null;
  reading: 'rtl' | 'ltr';
  width: number;
  background: string;
  detector: string;
  pageCount: number;
  segmentCount: number;
  unprocessed: boolean;
  cover: Blob;
  bytes: number;
  segments: SegmentLayout[];
}

export interface SegmentRecord {
  key: string;
  chapterId: string;
  index: number;
  blob: Blob;
}

export interface ProgressRecord {
  chapterId: string;
  segmentIndex: number;
  offsetFraction: number;
  updatedAt: number;
}

interface TateDB extends DBSchema {
  chapters: {
    key: string;
    value: ChapterRecord;
    indexes: { byIdentity: string };
  };
  segments: {
    key: string;
    value: SegmentRecord;
    indexes: { byChapter: string };
  };
  progress: {
    key: string;
    value: ProgressRecord;
  };
}

export function identityOf(title: string, source: string, segmentCount: number): string {
  return `${title}\u0000${source}\u0000${segmentCount}`;
}

export function migrate(db: IDBPDatabase<TateDB>, oldVersion: number): void {
  if (oldVersion < 1) {
    const chapters = db.createObjectStore('chapters', { keyPath: 'id' });
    chapters.createIndex('byIdentity', 'identity');
    const segments = db.createObjectStore('segments', { keyPath: 'key' });
    segments.createIndex('byChapter', 'chapterId');
    db.createObjectStore('progress', { keyPath: 'chapterId' });
  }
}

let database: Promise<IDBPDatabase<TateDB>> | null = null;

export function openDatabase(): Promise<IDBPDatabase<TateDB>> {
  database ??= openDB<TateDB>(DB_NAME, DB_VERSION, {
    upgrade(db, oldVersion) {
      migrate(db, oldVersion);
    },
  });
  return database;
}

export async function closeDatabase(): Promise<void> {
  if (!database) return;
  const db = await database;
  db.close();
  database = null;
}

export async function deleteDatabase(): Promise<void> {
  await closeDatabase();
  await deleteDB(DB_NAME);
}

export async function findByIdentity(identity: string): Promise<ChapterRecord | undefined> {
  const db = await openDatabase();
  const found = await db.getAllFromIndex('chapters', 'byIdentity', identity);
  return found[0];
}

export async function listChapters(): Promise<ChapterRecord[]> {
  const db = await openDatabase();
  const chapters = await db.getAll('chapters');
  chapters.sort(
    (a, b) => Math.max(b.lastReadAt ?? 0, b.importedAt) - Math.max(a.lastReadAt ?? 0, a.importedAt),
  );
  return chapters;
}

export async function listProgress(): Promise<ProgressRecord[]> {
  const db = await openDatabase();
  return db.getAll('progress');
}

export async function getChapter(id: string): Promise<ChapterRecord | undefined> {
  const db = await openDatabase();
  return db.get('chapters', id);
}

export async function getProgress(chapterId: string): Promise<ProgressRecord | undefined> {
  const db = await openDatabase();
  return db.get('progress', chapterId);
}

export async function getSegmentBlob(chapterId: string, index: number): Promise<Blob | undefined> {
  const db = await openDatabase();
  const record = await db.get('segments', `${chapterId}:${index}`);
  return record?.blob;
}

export async function saveChapter(chapter: ImportedChapter, replaceId?: string): Promise<string> {
  const db = await openDatabase();
  const id = replaceId ?? crypto.randomUUID();
  const existing = replaceId ? await db.get('chapters', replaceId) : undefined;
  const tx = db.transaction(['chapters', 'segments'], 'readwrite');
  if (replaceId) {
    const keys = await tx.objectStore('segments').index('byChapter').getAllKeys(replaceId);
    for (const key of keys) await tx.objectStore('segments').delete(key);
  }
  const now = Date.now();
  let bytes = chapter.cover.size;
  for (const segment of chapter.segments) {
    bytes += segment.blob.size;
    await tx.objectStore('segments').put({
      key: `${id}:${segment.index}`,
      chapterId: id,
      index: segment.index,
      blob: segment.blob,
    });
  }
  const record: ChapterRecord = {
    id,
    identity: identityOf(chapter.title, chapter.source, chapter.segments.length),
    title: chapter.title,
    source: chapter.source,
    createdAt: chapter.createdAt,
    importedAt: now,
    lastReadAt: existing?.lastReadAt ?? null,
    reading: chapter.reading,
    width: chapter.width,
    background: chapter.background,
    detector: chapter.detector,
    pageCount: chapter.pageCount,
    segmentCount: chapter.segments.length,
    unprocessed: chapter.unprocessed,
    cover: chapter.cover,
    bytes,
    segments: chapter.segments.map((segment) => ({
      width: segment.width,
      height: segment.height,
      gapBefore: segment.gapBefore,
      page: segment.page,
    })),
  };
  await tx.objectStore('chapters').put(record);
  await tx.done;
  return id;
}

export async function saveProgress(
  chapterId: string,
  segmentIndex: number,
  offsetFraction: number,
): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(['progress', 'chapters'], 'readwrite');
  const now = Date.now();
  await tx.objectStore('progress').put({ chapterId, segmentIndex, offsetFraction, updatedAt: now });
  const chapter = await tx.objectStore('chapters').get(chapterId);
  if (chapter) {
    chapter.lastReadAt = now;
    await tx.objectStore('chapters').put(chapter);
  }
  await tx.done;
}

export async function deleteChapter(id: string): Promise<void> {
  const db = await openDatabase();
  const tx = db.transaction(['chapters', 'segments', 'progress'], 'readwrite');
  const keys = await tx.objectStore('segments').index('byChapter').getAllKeys(id);
  for (const key of keys) await tx.objectStore('segments').delete(key);
  await tx.objectStore('chapters').delete(id);
  await tx.objectStore('progress').delete(id);
  await tx.done;
}
