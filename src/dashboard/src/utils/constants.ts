export const API_BASE = '/api/v1';

export const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  medium: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  high: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  critical: 'bg-red-200 text-red-900 dark:bg-red-800 dark:text-red-100',
};

export const CATEGORY_COLORS: Record<string, string> = {
  violent: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  weapon: 'bg-red-200 text-red-900 dark:bg-red-800 dark:text-red-100',
  suspicious: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
  urgent: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
  normal: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
};

export const CHART_COLORS: Record<string, string> = {
  violent: '#EF4444',
  weapon: '#DC2626',
  suspicious: '#F59E0B',
  urgent: '#F97316',
  normal: '#6B7280',
};

export const STATUS_COLORS: Record<string, string> = {
  pending: 'bg-yellow-100 text-yellow-800',
  delivered: 'bg-blue-100 text-blue-800',
  acknowledged: 'bg-green-100 text-green-800',
  dismissed: 'bg-gray-100 text-gray-800',
  escalated: 'bg-red-100 text-red-800',
  failed: 'bg-red-100 text-red-800',
};

export const ACTION_LABELS = [
  'fighting', 'falling', 'loitering', 'tailgating', 'running',
  'knife', 'gun', 'crowd', 'normal', 'walking', 'standing',
];

export const WS_URL = `ws://${window.location.hostname}:8765`;
