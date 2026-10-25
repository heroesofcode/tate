export type InstallPlatform = 'ios' | 'android' | 'other';

export function installPlatform(userAgent: string): InstallPlatform {
  const ua = userAgent.toLowerCase();
  if (/iphone|ipad|ipod/.test(ua)) return 'ios';
  if (ua.includes('android')) return 'android';
  return 'other';
}

export function isInstalledDisplay(): boolean {
  const nav = navigator as Navigator & { standalone?: boolean };
  return window.matchMedia('(display-mode: standalone)').matches || nav.standalone === true;
}

interface FileSystemFileHandleLike {
  getFile(): Promise<File>;
}

interface LaunchParamsLike {
  files?: FileSystemFileHandleLike[];
}

interface LaunchQueueLike {
  setConsumer(consumer: (params: LaunchParamsLike) => void | Promise<void>): void;
}

const INSTALL_DISMISSED_KEY = 'tate-install-dismissed';

export interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export function canOfferInstall(): boolean {
  if (installPlatform(navigator.userAgent) === 'other' || isInstalledDisplay()) return false;
  try {
    if (localStorage.getItem(INSTALL_DISMISSED_KEY) === '1') return false;
  } catch {
    /* private mode */
  }
  return true;
}

export function dismissInstallOffer(): void {
  try {
    localStorage.setItem(INSTALL_DISMISSED_KEY, '1');
  } catch {
    /* ignore */
  }
}

type PromptListener = (event: BeforeInstallPromptEvent | null) => void;

export function createInstallPromptController(
  target: EventTarget,
  eligible: () => boolean = () => true,
) {
  let deferred: BeforeInstallPromptEvent | null = null;
  const listeners = new Set<PromptListener>();

  const notify = () => {
    for (const listener of listeners) listener(deferred);
  };

  const onPrompt = (event: Event) => {
    if (!eligible()) return;
    event.preventDefault();
    deferred = event as BeforeInstallPromptEvent;
    notify();
  };

  target.addEventListener('beforeinstallprompt', onPrompt);

  return {
    current: () => deferred,
    subscribe(listener: PromptListener) {
      listeners.add(listener);
      listener(deferred);
      return () => listeners.delete(listener);
    },
    async prompt() {
      const pending = deferred;
      if (!pending) return null;
      await pending.prompt();
      const choice = await pending.userChoice;
      deferred = null;
      notify();
      return choice.outcome;
    },
  };
}

const browserInstallPrompt =
  typeof window === 'undefined' ? null : createInstallPromptController(window, canOfferInstall);

export function installPromptController() {
  if (!browserInstallPrompt) throw new Error('install prompt is unavailable');
  return browserInstallPrompt;
}

export function watchLaunchQueue(onFiles: (files: File[]) => void): void {
  const queue = (window as Window & { launchQueue?: LaunchQueueLike }).launchQueue;
  if (!queue) return;
  queue.setConsumer(async (params) => {
    if (!params.files?.length) return;
    const files = await Promise.all(params.files.map((handle) => handle.getFile()));
    onFiles(files);
  });
}
