<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { registerSW } from 'virtual:pwa-register';
  import InstallHint from './components/InstallHint.svelte';
  import Library from './components/Library.svelte';
  import Reader from './components/Reader.svelte';
  import { activeLocale, t } from './lib/i18n';
  import { watchLaunchQueue } from './lib/install';
  import { parseHash, type Route } from './lib/route';
  import { session, type ConfirmRequest } from './lib/session.svelte';
  import { takeSharedFiles } from './lib/share-inbox';
  import { toasts } from './lib/toast.svelte';

  let route = $state<Route>(parseHash(location.hash));
  let updateReady = $state(false);
  let applyUpdate: (reload?: boolean) => Promise<void> = async () => {};
  let dialogEl = $state<HTMLDialogElement | null>(null);
  let dialogCopy = $state<ConfirmRequest>({
    title: 'tate',
    body: '',
    confirm: 'OK',
    danger: false,
  });
  let pending: ((value: boolean) => void) | null = null;

  session.confirm = async (request) => {
    dialogCopy = request;
    await tick();
    dialogEl?.showModal();
    return new Promise((resolve) => {
      pending = resolve;
    });
  };

  function answer(value: boolean) {
    if (!pending) return;
    const resolve = pending;
    pending = null;
    dialogEl?.close();
    resolve(value);
  }

  function onDialogClose() {
    if (!pending) return;
    const resolve = pending;
    pending = null;
    resolve(false);
  }

  onMount(() => {
    document.documentElement.lang = activeLocale() === 'pt' ? 'pt-BR' : 'en';
    const onHash = () => {
      route = parseHash(location.hash);
    };
    window.addEventListener('hashchange', onHash);
    applyUpdate = registerSW({
      onNeedRefresh() {
        updateReady = true;
      },
      onRegisteredSW(swUrl, registration) {
        void swUrl;
        if (registration?.waiting && !navigator.serviceWorker.controller) {
          registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        }
      },
    });
    void (async () => {
      const shared = await takeSharedFiles();
      if (shared.length) await session.importFiles(shared);
      const url = new URL(location.href);
      if (url.searchParams.has('shared')) {
        url.searchParams.delete('shared');
        history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
      }
    })();
    watchLaunchQueue((files) => {
      void session.importFiles(files);
    });
    return () => window.removeEventListener('hashchange', onHash);
  });
</script>

{#if route.name === 'reader'}
  {#key route.id}
    <Reader id={route.id} />
  {/key}
{:else}
  <Library />
  <InstallHint />
{/if}

<div class="banners">
  {#if updateReady}
    <div class="banner" role="status">
      <p>{t('updateTitle')}</p>
      <button type="button" onclick={() => applyUpdate(true)}>{t('updateAction')}</button>
    </div>
  {/if}
  {#each toasts.items as toast (toast.id)}
    <div
      class="toast"
      class:error={toast.tone === 'error'}
      role={toast.tone === 'error' ? 'alert' : 'status'}
    >
      <p>{toast.message}</p>
      <button type="button" aria-label={t('dismiss')} onclick={() => toasts.dismiss(toast.id)}
        >×</button
      >
    </div>
  {/each}
</div>

<dialog bind:this={dialogEl} aria-labelledby="dialog-title" onclose={onDialogClose}>
  <h2 id="dialog-title">{dialogCopy.title}</h2>
  <p>{dialogCopy.body}</p>
  <div class="dialog-actions">
    <button type="button" class="btn ghost" onclick={() => answer(false)}>{t('cancel')}</button>
    <button
      type="button"
      class="btn"
      class:danger={dialogCopy.danger}
      class:solid={!dialogCopy.danger}
      onclick={() => answer(true)}
    >
      {dialogCopy.confirm}
    </button>
  </div>
</dialog>
