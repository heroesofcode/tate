<script lang="ts">
  import { onMount } from 'svelte';
  import { formatBytes, formatRelativeTime } from '../lib/format';
  import { intlLocale, t } from '../lib/i18n';
  import { percentFromProgress } from '../lib/progress';
  import { readerHash } from '../lib/route';
  import { session } from '../lib/session.svelte';
  import { deleteChapter, listChapters, listProgress } from '../lib/storage';
  import Mark from './Mark.svelte';

  interface LibraryItem {
    id: string;
    title: string;
    segmentCount: number;
    unprocessed: boolean;
    coverUrl: string;
    percent: number;
    hasProgress: boolean;
    lastReadAt: number | null;
    importedAt: number;
  }

  let items = $state<LibraryItem[]>([]);
  let loading = $state(true);
  let view = $state<'grid' | 'list'>(readView());
  let usage = $state(0);
  let persisted = $state(false);
  let storageReady = $state(false);
  let dragDepth = $state(0);
  let fileInput = $state<HTMLInputElement | null>(null);
  let request = 0;
  let asked = false;
  let covers: string[] = [];

  function readView(): 'grid' | 'list' {
    try {
      return localStorage.getItem('tate-view') === 'list' ? 'list' : 'grid';
    } catch {
      return 'grid';
    }
  }

  function setView(next: 'grid' | 'list') {
    view = next;
    try {
      localStorage.setItem('tate-view', next);
    } catch {
      /* ignore */
    }
  }

  function releaseCovers() {
    for (const url of covers) URL.revokeObjectURL(url);
    covers = [];
  }

  async function refresh(token: number) {
    const chapters = await listChapters();
    const progress = await listProgress();
    if (token !== request) return;
    const byId = new Map(progress.map((row) => [row.chapterId, row]));
    releaseCovers();
    items = chapters.map((chapter) => {
      const row = byId.get(chapter.id);
      const coverUrl = URL.createObjectURL(chapter.cover);
      covers.push(coverUrl);
      const percent = row
        ? percentFromProgress(row.segmentIndex, row.offsetFraction, chapter.segmentCount)
        : 0;
      return {
        id: chapter.id,
        title: chapter.title,
        segmentCount: chapter.segmentCount,
        unprocessed: chapter.unprocessed,
        coverUrl,
        percent,
        hasProgress: !!row && (row.segmentIndex > 0 || row.offsetFraction > 0.01),
        lastReadAt: chapter.lastReadAt,
        importedAt: chapter.importedAt,
      };
    });
    loading = false;
    if (!asked && navigator.storage?.persist) {
      asked = true;
      try {
        await navigator.storage.persist();
      } catch {
        /* the browser may deny persistence */
      }
    }
    if (token !== request) return;
    const estimate = (await navigator.storage?.estimate?.()) ?? {};
    const kept = (await navigator.storage?.persisted?.()) ?? false;
    if (token !== request) return;
    usage = estimate.usage ?? 0;
    persisted = kept;
    storageReady = true;
  }

  $effect(() => {
    const token = ++request;
    void session.revision;
    void refresh(token);
    return () => {
      request += 1;
    };
  });

  onMount(() => releaseCovers);

  function countLabel(item: LibraryItem): string {
    if (item.unprocessed) {
      return item.segmentCount === 1 ? t('pageOne') : t('pages', { count: item.segmentCount });
    }
    return item.segmentCount === 1 ? t('segmentOne') : t('segments', { count: item.segmentCount });
  }

  function timeLabel(item: LibraryItem): string {
    const when = item.lastReadAt ?? item.importedAt;
    const relative = formatRelativeTime(when, Date.now(), intlLocale());
    return item.lastReadAt ? t('lastRead', { time: relative }) : t('added', { time: relative });
  }

  function open(id: string) {
    location.hash = readerHash(id);
  }

  function pickFiles() {
    fileInput?.click();
  }

  function onInput(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const files = [...(input.files ?? [])];
    input.value = '';
    void session.importFiles(files);
  }

  function hasFiles(event: DragEvent): boolean {
    return [...(event.dataTransfer?.types ?? [])].includes('Files');
  }

  function onDragEnter(event: DragEvent) {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
  }

  function onDragOver(event: DragEvent) {
    if (!hasFiles(event)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  }

  function onDragLeave() {
    dragDepth = Math.max(0, dragDepth - 1);
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    dragDepth = 0;
    const files = [...(event.dataTransfer?.files ?? [])];
    if (files.length) void session.importFiles(files);
  }

  async function keep() {
    try {
      persisted = (await navigator.storage?.persist?.()) ?? false;
    } catch {
      persisted = false;
    }
  }

  async function askDelete(item: LibraryItem) {
    const accepted = session.confirm
      ? await session.confirm({
          title: t('deleteTitle'),
          body: t('deleteBody', { title: item.title }),
          confirm: t('delete'),
          danger: true,
        })
      : false;
    if (!accepted) return;
    await deleteChapter(item.id);
    session.bump();
  }
</script>

<div
  class="library"
  role="region"
  aria-label={t('libraryLabel')}
  ondragenter={onDragEnter}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  ondrop={onDrop}
>
  <header class="topbar">
    <div class="brand">
      <Mark />
      <h1>tate</h1>
    </div>
    <button type="button" class="import-btn" aria-label={t('importChapter')} onclick={pickFiles}>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M12 5v14M5 12h14"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
        />
      </svg>
    </button>
  </header>

  {#if storageReady}
    <div class="storage">
      <p>
        {t('storageUsed', { used: formatBytes(usage) })}
        <span aria-hidden="true">·</span>
        {persisted ? t('storagePersistent') : t('storageTemporary')}
      </p>
      {#if !persisted}
        <button type="button" onclick={keep}>{t('keepStorage')}</button>
      {/if}
    </div>
  {/if}

  {#if session.active}
    <div class="import-progress" role="status">
      <p>{t('importing', { current: session.current, total: session.total })}</p>
      <p class="file">{session.name}</p>
      <div class="track">
        <span style={`width:${(session.current / Math.max(session.total, 1)) * 100}%`}></span>
      </div>
    </div>
  {/if}

  {#if loading}
    <p class="loading">{t('loading')}</p>
  {:else if items.length === 0}
    <section class="empty">
      <svg viewBox="0 0 160 200" aria-hidden="true">
        <rect
          x="48"
          y="16"
          width="64"
          height="168"
          rx="10"
          fill="#17171c"
          stroke="rgba(246,241,231,0.18)"
        />
        <rect x="58" y="30" width="44" height="36" rx="4" fill="#ff4d2e" />
        <rect x="58" y="72" width="44" height="52" rx="4" fill="#f6f1e7" />
        <rect x="58" y="130" width="44" height="28" rx="4" fill="#f0a202" />
      </svg>
      <h2>{t('emptyTitle')}</h2>
      <p>{t('emptyBody')}</p>
      <a
        class="how"
        href="https://github.com/heroesofcode/tate#readme"
        target="_blank"
        rel="noreferrer"
      >
        {t('howTo')}
      </a>
      <button type="button" class="btn solid wide" onclick={pickFiles}>{t('importChapter')}</button>
    </section>
  {:else}
    <div class="toolbar">
      <button
        type="button"
        class:on={view === 'grid'}
        aria-pressed={view === 'grid'}
        aria-label={t('viewGrid')}
        onclick={() => setView('grid')}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3" y="3" width="7" height="7" rx="1.5" fill="currentColor" />
          <rect x="14" y="3" width="7" height="7" rx="1.5" fill="currentColor" />
          <rect x="3" y="14" width="7" height="7" rx="1.5" fill="currentColor" />
          <rect x="14" y="14" width="7" height="7" rx="1.5" fill="currentColor" />
        </svg>
      </button>
      <button
        type="button"
        class:on={view === 'list'}
        aria-pressed={view === 'list'}
        aria-label={t('viewList')}
        onclick={() => setView('list')}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <rect x="3" y="5" width="18" height="3" rx="1" fill="currentColor" />
          <rect x="3" y="11" width="18" height="3" rx="1" fill="currentColor" />
          <rect x="3" y="17" width="18" height="3" rx="1" fill="currentColor" />
        </svg>
      </button>
    </div>
    <ul class="cards" class:list={view === 'list'}>
      {#each items as item (item.id)}
        <li class="card">
          <button
            type="button"
            class="open"
            aria-label={item.hasProgress && item.percent < 100
              ? t('continueChapter', { title: item.title })
              : t('openChapter', { title: item.title })}
            onclick={() => open(item.id)}
          >
            <div class="cover">
              <img src={item.coverUrl} alt="" />
              {#if item.unprocessed}<span class="badge">{t('unprocessed')}</span>{/if}
              <div
                class="bar"
                role="progressbar"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={item.percent}
                aria-label={t('progressLabel')}
              >
                <span style={`width:${item.percent}%`}></span>
              </div>
            </div>
            <div class="body">
              <h2>{item.title}</h2>
              <p class="meta">{countLabel(item)} · {timeLabel(item)}</p>
              {#if item.hasProgress && item.percent < 100}
                <p class="continue-label">{t('continue')}</p>
              {/if}
            </div>
          </button>
          <button
            type="button"
            class="delete"
            aria-label={t('deleteChapter', { title: item.title })}
            onclick={() => askDelete(item)}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M5 7h14M9 7V5h6v2M8 7l1 12h6l1-12"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </svg>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

{#if dragDepth > 0}
  <div class="drop">{t('dropHint')}</div>
{/if}

<input
  bind:this={fileInput}
  class="file-input"
  type="file"
  accept=".tate,.zip,.cbz,application/zip,application/x-cbz,application/vnd.tate+zip"
  multiple
  aria-label={t('importChapter')}
  onchange={onInput}
/>
