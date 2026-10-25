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
  let ticket = 0;
  const live = new Map<number, string>();
  const inflight = new Map<number, number>();

  function revokeIndex(index: number): boolean {
    inflight.delete(index);
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
    /** True while a load for this index is still running. */
    pending(index: number) {
      return inflight.has(index);
    },
    /** Every index that is shown or still loading. */
    indices() {
      return [...new Set([...live.keys(), ...inflight.keys()])];
    },
    /** Drops the shown url and cancels any load still running for the index. */
    revoke: revokeIndex,
    close() {
      closed = true;
      inflight.clear();
      for (const index of [...live.keys()]) revokeIndex(index);
    },
    /**
     * Loads a segment unless it is already shown or loading. A load is only
     * dropped when the index is revoked or the slots close, so scrolling does
     * not restart work that is still wanted.
     */
    async claim(index: number, load: () => Promise<Blob | undefined>) {
      if (closed || live.has(index) || inflight.has(index)) return null;
      const mine = ++ticket;
      inflight.set(index, mine);
      let blob: Blob | undefined;
      try {
        blob = await load();
      } catch {
        blob = undefined;
      }
      if (closed || inflight.get(index) !== mine) return null;
      if (!blob) {
        inflight.delete(index);
        return null;
      }
      const url = create(blob);
      if (closed || inflight.get(index) !== mine) {
        revoke(url);
        return null;
      }
      inflight.delete(index);
      live.set(index, url);
      return url;
    },
  };
}
