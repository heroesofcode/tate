<script lang="ts">
  import { onMount } from 'svelte';
  import { t } from '../lib/i18n';
  import { installPlatform, isInstalledDisplay, type InstallPlatform } from '../lib/install';

  const STORAGE_KEY = 'tate-install-dismissed';

  interface BeforeInstallPromptEvent extends Event {
    prompt(): Promise<void>;
    userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
  }

  let visible = $state(false);
  let platform = $state<InstallPlatform>('other');
  let deferred = $state<BeforeInstallPromptEvent | null>(null);

  onMount(() => {
    platform = installPlatform(navigator.userAgent);
    if (platform === 'other' || isInstalledDisplay()) return;
    try {
      if (localStorage.getItem(STORAGE_KEY) === '1') return;
    } catch {
      /* private mode */
    }
    visible = true;
    const onPrompt = (event: Event) => {
      event.preventDefault();
      deferred = event as BeforeInstallPromptEvent;
    };
    window.addEventListener('beforeinstallprompt', onPrompt);
    return () => window.removeEventListener('beforeinstallprompt', onPrompt);
  });

  function dismiss() {
    visible = false;
    try {
      localStorage.setItem(STORAGE_KEY, '1');
    } catch {
      /* ignore */
    }
  }

  async function install() {
    if (!deferred) return;
    await deferred.prompt();
    const choice = await deferred.userChoice;
    deferred = null;
    if (choice.outcome === 'accepted') dismiss();
  }
</script>

{#if visible}
  <aside class="hint" aria-label={t('installTitle')}>
    <div>
      <h2>{t('installTitle')}</h2>
      <p>{platform === 'ios' ? t('installIos') : t('installAndroid')}</p>
    </div>
    <div class="hint-actions">
      {#if deferred}
        <button type="button" class="btn solid" onclick={install}>{t('installAction')}</button>
      {/if}
      <button type="button" class="btn ghost" onclick={dismiss}>{t('notNow')}</button>
    </div>
  </aside>
{/if}
