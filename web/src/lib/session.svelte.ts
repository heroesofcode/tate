import { packageErrorText, t } from './i18n';
import { importPackage } from './package';
import { findByIdentity, identityOf, saveChapter } from './storage';
import { toasts } from './toast.svelte';

export interface ConfirmRequest {
  title: string;
  body: string;
  confirm: string;
  danger: boolean;
}

type ConfirmFn = (request: ConfirmRequest) => Promise<boolean>;

class Session {
  active = $state(false);
  current = $state(0);
  total = $state(0);
  name = $state('');
  revision = $state(0);
  confirm: ConfirmFn | null = null;

  bump() {
    this.revision += 1;
  }

  async importFiles(files: File[]) {
    if (!files.length || this.active) return;
    this.active = true;
    this.total = files.length;
    if (location.hash.startsWith('#/read/')) location.hash = '#/';
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      if (!file) continue;
      this.current = index + 1;
      this.name = file.name;
      try {
        const chapter = await importPackage(file);
        const identity = identityOf(chapter.title, chapter.source, chapter.segments.length);
        const existing = await findByIdentity(identity);
        let replaceId: string | undefined;
        if (existing) {
          const accepted = this.confirm
            ? await this.confirm({
                title: t('replaceTitle'),
                body: t('replaceBody'),
                confirm: t('replace'),
                danger: false,
              })
            : false;
          if (!accepted) continue;
          replaceId = existing.id;
        }
        await saveChapter(chapter, replaceId);
        toasts.push(t('importDone', { title: chapter.title }));
      } catch (error) {
        toasts.push(
          t('importFailed', { name: file.name, message: packageErrorText(error) }),
          'error',
        );
      }
    }
    this.active = false;
    this.bump();
  }
}

export const session = new Session();
