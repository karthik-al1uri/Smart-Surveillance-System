import { API_BASE } from '../utils/constants';

export function getClipUrl(eventId: string): string {
  return `${API_BASE}/clips/${eventId}`;
}

export function getSnapshotUrl(cameraId: string): string {
  return `${API_BASE}/cameras/${cameraId}/snapshot`;
}
