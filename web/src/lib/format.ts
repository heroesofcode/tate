export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = index === 0 || value >= 10 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[index]}`;
}

export function formatRelativeTime(timestamp: number, now: number, locale: string): string {
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });
  const seconds = Math.round((timestamp - now) / 1000);
  const divisions: [Intl.RelativeTimeFormatUnit, number][] = [
    ['year', 60 * 60 * 24 * 365],
    ['month', 60 * 60 * 24 * 30],
    ['week', 60 * 60 * 24 * 7],
    ['day', 60 * 60 * 24],
    ['hour', 60 * 60],
    ['minute', 60],
    ['second', 1],
  ];
  const abs = Math.abs(seconds);
  for (const [unit, size] of divisions) {
    if (abs >= size || unit === 'second') return rtf.format(Math.round(seconds / size), unit);
  }
  return rtf.format(0, 'second');
}

export function isLightBackground(color: string): boolean {
  const hex = color.trim().replace('#', '');
  const full =
    hex.length === 3
      ? hex
          .split('')
          .map((channel) => channel + channel)
          .join('')
      : hex.slice(0, 6);
  if (!/^[0-9a-fA-F]{6}$/.test(full)) return false;
  const r = Number.parseInt(full.slice(0, 2), 16);
  const g = Number.parseInt(full.slice(2, 4), 16);
  const b = Number.parseInt(full.slice(4, 6), 16);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 170;
}
