import client from './client';
import type { Camera, Zone } from '../types';

export async function listCameras(): Promise<Camera[]> {
  const r = await client.get<Camera[]>('/cameras');
  return r.data;
}

export async function getCamera(id: string): Promise<Camera> {
  const r = await client.get<Camera>(`/cameras/${id}`);
  return r.data;
}

export async function createCamera(data: Partial<Camera>): Promise<Camera> {
  const r = await client.post<Camera>('/cameras', data);
  return r.data;
}

export async function updateCamera(id: string, data: Partial<Camera>): Promise<Camera> {
  const r = await client.put<Camera>(`/cameras/${id}`, data);
  return r.data;
}

export async function deleteCamera(id: string): Promise<void> {
  await client.delete(`/cameras/${id}`);
}

export async function updateZones(id: string, zones: Zone[]): Promise<Camera> {
  const r = await client.put<Camera>(`/cameras/${id}/zones`, { zones });
  return r.data;
}

export async function getZones(id: string): Promise<{ zones: Zone[] }> {
  const r = await client.get<{ zones: Zone[] }>(`/cameras/${id}/zones`);
  return r.data;
}
