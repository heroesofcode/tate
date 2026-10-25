<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { isLightBackground } from '../lib/format';
  import { t } from '../lib/i18n';
  import { percentFromProgress, scrollTopFor } from '../lib/progress';
  import { libraryHash } from '../lib/route';
  import {
    getChapter,
    getProgress,
    getSegmentBlob,
    saveProgress,
    type ChapterRecord,
  } from '../lib/storage';

  let { id }: { id: string } = $props();

  let chapter = $state<ChapterRecord | null>(null);
  let missing = $state(false);
  let urls = $state<Array<string | null>>([]);
  let segmentIndex = $state(0);
  let offsetFraction = $state(0);
  let overlay = $state(true);
  let scroller = $state<HTMLElement | null>(null);
  let hideTimer = 0;
  let restored = false;
  const live: Record<number, string> = {};
  const inflight: Record<number, number> = {};
  let generation = 0;
  let wake: WakeLockSentinel | null = null;

  const light = $derived(chapter ? isLightBackground(chapter.background) : false);
  const percent = $derived(
    chapter ? percentFromProgress(segmentIndex, offsetFraction, chapter.segmentCount) : 0,
  );

  function goLibrary() {
    location.hash = libraryHash();
  }

  function setThemeColor(color: string) {
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', color);
  }

  function armHide() {
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => {
      overlay = false;
    }, 3200);
  }

  function showOverlay() {
    overlay = true;
    armHide();
  }

  function toggleOverlay() {
    overlay = !overlay;
    if (overlay) armHide();
    else window.clearTimeout(hideTimer);
  }

  function setUrl(index: number, url: string | null) {
    const next = urls.slice();
    next[index] = url;
    urls = next;
  }

  function revoke(index: number) {
    const url = live[index];
    if (!url) return;
    URL.revokeObjectURL(url);
    live[index] = '';
    setUrl(index, null);
  }

  function revokeAll() {
    for (const key of Object.keys(live)) revoke(Number(key));
  }

  async function activate(index: number, gen: number) {
    if (live[index] || !chapter) return;
    inflight[index] = gen;
    const blob = await getSegmentBlob(chapter.id, index);
    if (inflight[index] !== gen || !blob) return;
    const url = URL.createObjectURL(blob);
    if (inflight[index] !== gen) {
      URL.revokeObjectURL(url);
      return;
    }
    live[index] = url;
    setUrl(index, url);
  }

  function nodeTop(root: HTMLElement, node: HTMLElement): number {
    return node.getBoundingClientRect().top - root.getBoundingClientRect().top + root.scrollTop;
  }

  function desired(root: HTMLElement): number[] {
    const top = root.scrollTop;
    const height = root.clientHeight || 1;
    const nodes = [...root.querySelectorAll<HTMLElement>('[data-index]')];
    const ranked = nodes.map((node) => {
      const index = Number(node.dataset.index);
      const start = nodeTop(root, node);
      const end = start + node.offsetHeight;
      const dist = end < top ? top - end : start > top + height ? start - (top + height) : 0;
      return { index, dist };
    });
    ranked.sort((a, b) => a.dist - b.dist);
    const near = ranked.filter((item) => item.dist <= height * 1.25);
    return (near.length ? near : ranked).slice(0, 18).map((item) => item.index);
  }

  function syncWindow() {
    const root = scroller;
    if (!root) return;
    const gen = ++generation;
    const want = new Set(desired(root));
    for (const key of Object.keys(live)) {
      const index = Number(key);
      if (!want.has(index)) revoke(index);
    }
    for (const index of want) {
      if (!live[index]) void activate(index, gen);
    }
  }

  function readPosition(): { segmentIndex: number; offsetFraction: number } {
    const root = scroller;
    if (!root) return { segmentIndex, offsetFraction };
    const nodes = [...root.querySelectorAll<HTMLElement>('[data-index]')];
    const first = nodes[0];
    if (!first) return { segmentIndex: 0, offsetFraction: 0 };
    const top = root.scrollTop;
    const firstTop = nodeTop(root, first);
    if (top <= firstTop) return { segmentIndex: 0, offsetFraction: 0 };
    let current = first;
    for (const node of nodes) {
      if (nodeTop(root, node) <= top + 0.5) current = node;
      else break;
    }
    const height = current.offsetHeight || 1;
    const index = Number(current.dataset.index);
    const offset = (top - nodeTop(root, current)) / height;
    return { segmentIndex: index, offsetFraction: Math.max(0, offset) };
  }

  async function restore() {
    if (!chapter || !scroller) return;
    const saved = await getProgress(chapter.id);
    await tick();
    if (!saved || !scroller) return;
    const node = scroller.querySelector<HTMLElement>(`[data-index="${saved.segmentIndex}"]`);
    if (!node) return;
    scroller.scrollTo({
      top: scrollTopFor(nodeTop(scroller, node), node.offsetHeight, saved.offsetFraction),
      behavior: 'auto',
    });
    segmentIndex = saved.segmentIndex;
    offsetFraction = saved.offsetFraction;
  }

  async function acquireWake() {
    if (!('wakeLock' in navigator)) return;
    try {
      wake = await navigator.wakeLock.request('screen');
      wake.addEventListener('release', () => {
        wake = null;
      });
    } catch {
      wake = null;
    }
  }

  let pointerCount = 0;
  let pointerMax = 0;
  let startX = 0;
  let startY = 0;
  let startT = 0;
  let ignoreTap = false;

  function onPointerDown(event: PointerEvent) {
    pointerCount += 1;
    pointerMax = Math.max(pointerMax, pointerCount);
    if (pointerCount === 1) {
      startX = event.clientX;
      startY = event.clientY;
      startT = event.timeStamp;
      const target = event.target;
      ignoreTap = target instanceof Element && !!target.closest('button, a');
    }
  }

  function onPointerUp(event: PointerEvent) {
    pointerCount = Math.max(0, pointerCount - 1);
    if (pointerCount > 0) return;
    const multi = pointerMax > 1;
    pointerMax = 0;
    if (multi || ignoreTap) return;
    if (Math.abs(event.clientX - startX) > 12 || Math.abs(event.clientY - startY) > 12) return;
    if (event.timeStamp - startT > 500) return;
    toggleOverlay();
  }

  function onPointerCancel() {
    pointerCount = 0;
    pointerMax = 0;
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.preventDefault();
      goLibrary();
      return;
    }
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleOverlay();
    }
  }

  onMount(() => {
    let stop = () => {};
    let cancelled = false;
    void (async () => {
      const record = await getChapter(id);
      if (cancelled) return;
      if (!record) {
        missing = true;
        return;
      }
      chapter = record;
      urls = record.segments.map(() => null);
      await tick();
      if (cancelled || !scroller) return;
      const root = scroller;
      setThemeColor(record.background);
      void acquireWake();
      showOverlay();
      let frame = 0;
      let saveTimer = 0;
      const schedule = (force: boolean) => {
        const pos = readPosition();
        segmentIndex = pos.segmentIndex;
        offsetFraction = pos.offsetFraction;
        if (!restored && !force) return;
        window.clearTimeout(saveTimer);
        const write = () => {
          void saveProgress(record.id, pos.segmentIndex, pos.offsetFraction);
        };
        if (force) write();
        else saveTimer = window.setTimeout(write, 280);
      };
      const onScroll = () => {
        if (frame) return;
        frame = requestAnimationFrame(() => {
          frame = 0;
          syncWindow();
          schedule(false);
        });
      };
      root.addEventListener('scroll', onScroll, { passive: true });
      const onResize = () => syncWindow();
      window.addEventListener('resize', onResize);
      const onVis = () => {
        if (document.visibilityState === 'hidden') schedule(true);
        else void acquireWake();
      };
      document.addEventListener('visibilitychange', onVis);
      const onHide = () => schedule(true);
      window.addEventListener('pagehide', onHide);
      void (async () => {
        await restore();
        if (cancelled) return;
        syncWindow();
        restored = true;
        schedule(true);
      })();
      stop = () => {
        root.removeEventListener('scroll', onScroll);
        window.removeEventListener('resize', onResize);
        document.removeEventListener('visibilitychange', onVis);
        window.removeEventListener('pagehide', onHide);
        if (frame) cancelAnimationFrame(frame);
        window.clearTimeout(saveTimer);
        window.clearTimeout(hideTimer);
        if (restored) schedule(true);
        revokeAll();
        setThemeColor('#0c0c0f');
        void wake?.release();
      };
    })();
    return () => {
      cancelled = true;
      stop();
    };
  });
</script>

{#if missing}
  <section class="missing">
    <p>{t('missingChapter')}</p>
    <button type="button" class="btn solid" onclick={goLibrary}>{t('backToLibrary')}</button>
  </section>
{:else if chapter}
  <div class="reader" class:light style={`background:${chapter.background}`}>
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div
      class="scroller"
      bind:this={scroller}
      role="region"
      aria-label={t('readerRegion')}
      tabindex="0"
      onpointerdown={onPointerDown}
      onpointerup={onPointerUp}
      onpointercancel={onPointerCancel}
      onkeydown={onKeydown}
    >
      <div class="sheet">
        <div class="safe-top"></div>
        {#each chapter.segments as segment, index (index)}
          {@const src = urls[index]}
          <article
            class="segment"
            data-index={index}
            style={`padding-top:${(segment.gapBefore / chapter.width) * 100}%;background:${chapter.background}`}
          >
            <div class="frame" style={`aspect-ratio:${segment.width} / ${segment.height}`}>
              {#if src}
                <img
                  {src}
                  alt={`Panel ${index + 1}`}
                  width={segment.width}
                  height={segment.height}
                  decoding="async"
                  draggable="false"
                />
              {/if}
            </div>
          </article>
        {/each}
        <section class="end-card">
          <h2>{t('endTitle')}</h2>
          <p>{t('endBody')}</p>
          <button type="button" class="pill" onclick={goLibrary}>{t('backToLibrary')}</button>
        </section>
        <div class="safe-bottom"></div>
      </div>
    </div>

    <div
      class="overlay"
      class:visible={overlay}
      data-overlay={overlay ? 'visible' : 'hidden'}
      aria-hidden={!overlay}
    >
      <div class="overlay-top">
        <button type="button" class="back" aria-label={t('back')} onclick={goLibrary}>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path
              d="M15 5l-7 7 7 7"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
        </button>
        <p class="reader-title">{chapter.title}</p>
      </div>
      <div class="overlay-bottom">
        <p class="percent">{percent}%</p>
        <div
          class="reader-progress"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={percent}
          aria-label={t('progressLabel')}
        >
          <span style={`width:${percent}%`}></span>
        </div>
      </div>
    </div>
  </div>
{/if}
