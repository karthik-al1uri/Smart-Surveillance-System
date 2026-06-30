import { describe, it, expect } from 'vitest';
import { formatSeverity, formatBytes, formatDuration, severityLabel } from '../utils/formatters';

describe('formatters', () => {
  it('formatSeverity converts score to percentage string', () => {
    expect(formatSeverity(0.82)).toBe('82%');
    expect(formatSeverity(1.0)).toBe('100%');
    expect(formatSeverity(0)).toBe('0%');
  });

  it('formatBytes formats bytes correctly', () => {
    expect(formatBytes(500)).toBe('500 B');
    expect(formatBytes(1500)).toBe('1.5 KB');
    expect(formatBytes(1500000)).toBe('1.4 MB');
  });

  it('formatDuration formats seconds correctly', () => {
    expect(formatDuration(30)).toBe('30.0s');
    expect(formatDuration(90)).toBe('1m 30s');
  });

  it('severityLabel maps score to correct label', () => {
    expect(severityLabel(0.95)).toBe('critical');
    expect(severityLabel(0.75)).toBe('high');
    expect(severityLabel(0.5)).toBe('medium');
    expect(severityLabel(0.2)).toBe('low');
  });
});
