import { describe, expect, it } from 'vitest';
import { detectLocale, lookupMessage, messages, type MessageKey } from './i18n';

describe('i18n', () => {
  it('uses Brazilian Portuguese for pt and English otherwise', () => {
    expect(detectLocale('pt-BR')).toBe('pt');
    expect(detectLocale('pt-PT')).toBe('pt');
    expect(detectLocale('PT')).toBe('pt');
    expect(detectLocale('en-US')).toBe('en');
    expect(detectLocale('ja-JP')).toBe('en');
    expect(detectLocale('')).toBe('en');
  });

  it('falls back to English and then to the key', () => {
    const keys = Object.keys(messages.en) as MessageKey[];
    for (const key of keys) expect(messages.pt[key].length).toBeGreaterThan(0);
    expect(messages.pt.emptyTitle).toMatch(/Nenhum/);
    expect(lookupMessage({ en: { hi: 'Hello' }, pt: {} }, 'pt', 'hi')).toBe('Hello');
    expect(lookupMessage({ en: {}, pt: {} }, 'fr', 'missing')).toBe('missing');
  });
});
