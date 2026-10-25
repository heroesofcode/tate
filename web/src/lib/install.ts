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

export function watchLaunchQueue(onFiles: (files: File[]) => void): void {
  const queue = (window as Window & { launchQueue?: LaunchQueueLike }).launchQueue;
  if (!queue) return;
  queue.setConsumer(async (params) => {
    if (!params.files?.length) return;
    const files = await Promise.all(params.files.map((handle) => handle.getFile()));
    onFiles(files);
  });
}
