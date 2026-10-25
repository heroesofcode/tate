<script lang="ts">
  import { onMount } from 'svelte';
  import { t } from '../lib/i18n';
  import {
    canOfferInstall,
    dismissInstallOffer,
    installPlatform,
    installPromptController,
    type BeforeInstallPromptEvent,
    type InstallPlatform,
  } from '../lib/install';

  let visible = $state(false);
  let platform = $state<InstallPlatform>('other');
  let deferred = $state<BeforeInstallPromptEvent | null>(null);

  onMount(() => {
    platform = installPlatform(navigator.userAgent);
    if (!canOfferInstall()) return;
    visible = true;
    return installPromptController().subscribe((event) => {
      deferred = event;
    });
  });

  function dismiss() {
    visible = false;
    dismissInstallOffer();
  }

  async function install() {
    const outcome = await installPromptController().prompt();
    if (outcome === 'accepted') dismiss();
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
