import { describe, expect, it } from 'vitest';
import { createScreenWake, createUrlSlots } from './reader-resources';

function deferred<T>() {
  let resolve: (value: T) => void = () => {};
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function urlSlots() {
  const created: string[] = [];
  const revoked: string[] = [];
  let next = 0;
  const slots = createUrlSlots({
    create: () => {
      next += 1;
      const url = `blob:${next}`;
      created.push(url);
      return url;
    },
    revoke: (url) => {
      revoked.push(url);
    },
  });
  return { slots, created, revoked };
}

interface FakeLock {
  released: boolean;
  release(): Promise<void>;
  addEventListener(type: 'release', listener: () => void): void;
}

function fakeLock(): FakeLock {
  const listeners: Array<() => void> = [];
  return {
    released: false,
    release() {
      this.released = true;
      for (const listener of listeners) listener();
      return Promise.resolve();
    },
    addEventListener(_type: 'release', listener: () => void) {
      listeners.push(listener);
    },
  };
}

describe('segment object urls', () => {
  it('revokes urls already shown when the reader closes', async () => {
    const { slots, revoked } = urlSlots();
    const pending = deferred<Blob | undefined>();
    const claim = slots.claim(1, 1, () => pending.promise);
    pending.resolve(new Blob(['panel']));
    expect(await claim).toBe('blob:1');
    slots.close();
    expect(slots.has(1)).toBe(false);
    expect(revoked).toEqual(['blob:1']);
  });

  it('does not retain an object url finished after the reader closes', async () => {
    const { slots, created, revoked } = urlSlots();
    const pending = deferred<Blob | undefined>();
    const claim = slots.claim(1, 1, () => pending.promise);
    slots.close();
    pending.resolve(new Blob(['panel']));
    expect(await claim).toBeNull();
    expect(slots.has(1)).toBe(false);
    expect(revoked).toEqual(created);
  });

  it('revokes a url when the reader closes while it is being created', async () => {
    const revoked: string[] = [];
    let slots: ReturnType<typeof createUrlSlots> | null = null;
    slots = createUrlSlots({
      create: () => {
        slots?.close();
        return 'blob:late';
      },
      revoke: (url) => {
        revoked.push(url);
      },
    });
    expect(await slots.claim(0, 1, async () => new Blob(['panel']))).toBeNull();
    expect(slots.has(0)).toBe(false);
    expect(revoked).toEqual(['blob:late']);
  });

  it('drops a load superseded by a newer generation', async () => {
    const { slots, created, revoked } = urlSlots();
    const first = deferred<Blob | undefined>();
    const second = deferred<Blob | undefined>();
    const older = slots.claim(0, 1, () => first.promise);
    const newer = slots.claim(0, 2, () => second.promise);
    first.resolve(new Blob(['old']));
    expect(await older).toBeNull();
    second.resolve(new Blob(['new']));
    expect(await newer).toBe('blob:1');
    expect(created).toEqual(['blob:1']);
    expect(revoked).toEqual([]);
    expect(slots.has(0)).toBe(true);
  });
});

describe('screen wake lock', () => {
  it('releases a lock that resolves after the reader closes', async () => {
    const pending = deferred<FakeLock>();
    const wake = createScreenWake(() => pending.promise);
    const acquiring = wake.acquire();
    wake.close();
    const lock = fakeLock();
    pending.resolve(lock);
    await acquiring;
    expect(lock.released).toBe(true);
  });

  it('releases a lock already held when the reader closes', async () => {
    const lock = fakeLock();
    const wake = createScreenWake(async () => lock);
    await wake.acquire();
    wake.close();
    expect(lock.released).toBe(true);
    wake.close();
    expect(lock.released).toBe(true);
  });

  it('does not request a lock after close', async () => {
    let calls = 0;
    const wake = createScreenWake(async () => {
      calls += 1;
      return fakeLock();
    });
    wake.close();
    await wake.acquire();
    expect(calls).toBe(0);
  });

  it('releases an older in-flight lock when a newer request wins', async () => {
    const first = deferred<FakeLock>();
    const second = deferred<FakeLock>();
    const requests = [first.promise, second.promise];
    const wake = createScreenWake(() => {
      const next = requests.shift();
      if (!next) throw new Error('unexpected request');
      return next;
    });
    const olderAcquire = wake.acquire();
    const newerAcquire = wake.acquire();
    const older = fakeLock();
    const newer = fakeLock();
    first.resolve(older);
    second.resolve(newer);
    await Promise.all([olderAcquire, newerAcquire]);
    expect(older.released).toBe(true);
    expect(newer.released).toBe(false);
    wake.close();
    expect(newer.released).toBe(true);
  });
});
