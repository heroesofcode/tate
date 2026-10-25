import 'fake-indexeddb/auto';
import { afterEach, describe, expect, it } from 'vitest';
import type { ImportedChapter } from './package';
import {
  DB_VERSION,
  deleteChapter,
  deleteDatabase,
  findByIdentity,
  getChapter,
  getProgress,
  getSegmentBlob,
  identityOf,
  listChapters,
  closeDatabase,
  openDatabase,
  saveChapter,
  saveProgress,
} from './storage';

function sample(title = 'One', source = 'a.cbz', count = 2): ImportedChapter {
  return {
    title,
    source,
    createdAt: '2026-10-25T04:00:00Z',
    reading: 'rtl',
    width: 100,
    background: '#111111',
    detector: 'page',
    pageCount: count,
    unprocessed: false,
    cover: new Blob([`cover-${title}`], { type: 'image/webp' }),
    segments: Array.from({ length: count }, (_, index) => ({
      index,
      width: 100,
      height: 200,
      gapBefore: index === 0 ? 0 : 10,
      page: index,
      blob: new Blob([`seg-${title}-${index}`], { type: 'image/webp' }),
    })),
  };
}

async function pause() {
  await new Promise((resolve) => setTimeout(resolve, 5));
}

describe('storage', () => {
  afterEach(async () => {
    await deleteDatabase();
  });

  it('creates the versioned stores and keeps data across reopen', async () => {
    const db = await openDatabase();
    expect(db.version).toBe(DB_VERSION);
    expect([...db.objectStoreNames]).toEqual(
      expect.arrayContaining(['chapters', 'segments', 'progress']),
    );
    const id = await saveChapter(sample());
    await closeDatabase();
    const again = await getChapter(id);
    expect(again?.title).toBe('One');
    expect(again?.segments).toHaveLength(2);
  });

  it('lists the most recently read chapter first', async () => {
    const older = await saveChapter(sample('Older', 'old.cbz'));
    await pause();
    await saveChapter(sample('Newer', 'new.cbz'));
    expect((await listChapters()).map((chapter) => chapter.title)).toEqual(['Newer', 'Older']);
    await saveProgress(older, 1, 0.4);
    const listed = await listChapters();
    expect(listed[0]?.title).toBe('Older');
    expect(listed[0]?.lastReadAt).toBeTypeOf('number');
    const progress = await getProgress(older);
    expect(progress).toMatchObject({ segmentIndex: 1, offsetFraction: 0.4 });
  });

  it('replaces the same chapter instead of duplicating and keeps progress', async () => {
    const id = await saveChapter(sample('Same', 'same.cbz', 2));
    await saveProgress(id, 1, 0.25);
    const found = await findByIdentity(identityOf('Same', 'same.cbz', 2));
    expect(found?.id).toBe(id);
    const replacement = sample('Same', 'same.cbz', 2);
    replacement.cover = new Blob(['new-cover'], { type: 'image/webp' });
    const id2 = await saveChapter(replacement, id);
    expect(id2).toBe(id);
    const chapters = await listChapters();
    expect(chapters).toHaveLength(1);
    expect(await chapters[0]?.cover.text()).toBe('new-cover');
    expect((await getProgress(id))?.segmentIndex).toBe(1);
    expect(await (await getSegmentBlob(id, 1))?.text()).toBe('seg-Same-1');
  });

  it('deletes metadata, blobs, and progress together', async () => {
    const id = await saveChapter(sample());
    await saveProgress(id, 0, 0.2);
    await deleteChapter(id);
    expect(await listChapters()).toEqual([]);
    expect(await getSegmentBlob(id, 0)).toBeUndefined();
    expect(await getProgress(id)).toBeUndefined();
  });
});
