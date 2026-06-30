import client from './client';
import type { Alert, AlertListResponse, AlertStats } from '../types';

export interface AlertFilters {
  status?: string;
  priority?: string;
  limit?: number;
  offset?: number;
}

export async function listAlerts(filters: AlertFilters = {}): Promise<AlertListResponse> {
  const r = await client.get<AlertListResponse>('/alerts', { params: filters });
  return r.data;
}

export async function getActiveAlerts(): Promise<Alert[]> {
  const r = await client.get<Alert[]>('/alerts/active');
  return r.data;
}

export async function getAlert(id: string): Promise<Alert> {
  const r = await client.get<Alert>(`/alerts/${id}`);
  return r.data;
}

export async function acknowledgeAlert(id: string, operator?: string): Promise<void> {
  await client.post(`/alerts/${id}/acknowledge`, { operator });
}

export async function dismissAlert(id: string, reason: string, operator?: string): Promise<void> {
  await client.post(`/alerts/${id}/dismiss`, { reason, operator });
}

export async function getAlertStats(): Promise<AlertStats> {
  const r = await client.get<AlertStats>('/alerts/stats');
  return r.data;
}
