import client from './client';
import type { SystemHealth, StorageStats } from '../types';

export async function getSystemHealth(): Promise<SystemHealth> {
  const r = await client.get<SystemHealth>('/system/health');
  return r.data;
}

export async function getSystemStats(): Promise<Record<string, unknown>> {
  const r = await client.get('/system/stats');
  return r.data;
}

export async function getStorageStats(): Promise<StorageStats> {
  const r = await client.get<StorageStats>('/clips/storage');
  return r.data;
}

export async function getConfig(): Promise<Record<string, unknown>> {
  const r = await client.get('/config');
  return r.data;
}

export async function updateScoringConfig(updates: Record<string, unknown>): Promise<void> {
  await client.put('/config/scoring', updates);
}

export async function updateNotificationConfig(updates: Record<string, unknown>): Promise<void> {
  await client.put('/config/notifications', updates);
}
