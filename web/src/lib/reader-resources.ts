interface HeldLock {
  release(): Promise<void>;
  addEventListener(type: 'release', listener: () => void): void;
}

function forget(lock: HeldLock) {
  try {
    void lock.release().catch(() => {});
  } catch {
    /* already released */
  }
}

export function createScreenWake(request: () => Promise<HeldLock>) {
  let closed = false;
  let ticket = 0;
  let held: HeldLock | null = null;

  return {
    async acquire() {
      if (closed) return;
      const mine = ++ticket;
      let lock: HeldLock;
      try {
        lock = await request();
      } catch {
        return;
      }
      if (closed || mine !== ticket) {
        forget(lock);
        return;
      }
      const previous = held;
      held = lock;
      lock.addEventListener('release', () => {
        if (held === lock) held = null;
      });
      if (previous && previous !== lock) forget(previous);
    },
    close() {
      if (closed) return;
      closed = true;
      ticket += 1;
      const lock = held;
      held = null;
      if (lock) forget(lock);
    },
  };
}

export function createUrlSlots(urls?: {
  create: (blob: Blob) => string;
  revoke: (url: string) => void;
}) {
  const create = urls?.create ?? ((blob: Blob) => URL.createObjectURL(blob));
  const revoke = urls?.revoke ?? ((url: string) => URL.revokeObjectURL(url));
  let closed = false;
  const live = new Map<number, string>();
  const inflight = new Map<number, number>();

  function revokeIndex(index: number): boolean {
    const url = live.get(index);
    if (!url) return false;
    live.delete(index);
    revoke(url);
    return true;
  }

  return {
    has(index: number) {
      return live.has(index);
    },
    indices() {
      return [...live.keys()];
    },
    revoke: revokeIndex,
    close() {
      closed = true;
      inflight.clear();
      for (const index of [...live.keys()]) revokeIndex(index);
    },
    async claim(index: number, generation: number, load: () => Promise<Blob | undefined>) {
      if (closed || live.has(index)) return null;
      inflight.set(index, generation);
      const blob = await load();
      if (closed || inflight.get(index) !== generation || !blob) return null;
      const url = create(blob);
      if (closed || inflight.get(index) !== generation) {
        revoke(url);
        return null;
      }
      live.set(index, url);
      return url;
    },
  };
}
