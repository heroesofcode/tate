import { describe, expect, it } from 'vitest';
import { createInstallPromptController, type BeforeInstallPromptEvent } from './install';

function promptEvent(outcome: 'accepted' | 'dismissed' = 'accepted'): BeforeInstallPromptEvent {
  const event = new Event('beforeinstallprompt', { cancelable: true });
  return Object.assign(event, {
    prompt: () => Promise.resolve(),
    userChoice: Promise.resolve({ outcome }),
  });
}

describe('install prompt', () => {
  it('keeps the deferred prompt after the last subscriber leaves', () => {
    const target = new EventTarget();
    const prompts = createInstallPromptController(target);
    const stop = prompts.subscribe(() => {});
    const event = promptEvent();
    target.dispatchEvent(event);
    stop();
    expect(event.defaultPrevented).toBe(true);
    expect(prompts.current()).toBe(event);
  });

  it('still captures a prompt that arrives while nobody is subscribed', () => {
    const target = new EventTarget();
    const prompts = createInstallPromptController(target);
    const event = promptEvent();
    target.dispatchEvent(event);
    let seen: BeforeInstallPromptEvent | null = null;
    const stop = prompts.subscribe((value) => {
      seen = value;
    });
    expect(seen).toBe(event);
    stop();
    const again = promptEvent('dismissed');
    target.dispatchEvent(again);
    expect(prompts.current()).toBe(again);
  });

  it('leaves the browser prompt alone when install is not offered', () => {
    const target = new EventTarget();
    const prompts = createInstallPromptController(target, () => false);
    const event = promptEvent();
    target.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
    expect(prompts.current()).toBeNull();
  });

  it('clears the stored prompt after it is used', async () => {
    const target = new EventTarget();
    const prompts = createInstallPromptController(target);
    let seen: BeforeInstallPromptEvent | null = promptEvent('dismissed');
    target.dispatchEvent(seen);
    prompts.subscribe((value) => {
      seen = value;
    });
    await expect(prompts.prompt()).resolves.toBe('dismissed');
    expect(seen).toBeNull();
    expect(prompts.current()).toBeNull();
  });
});
