import client from './client';
import type { Event, EventListResponse, EventStats } from '../types';

export interface EventFilters {
  camera_id?: string;
  category?: string;
  start_time?: string;
  end_time?: string;
  limit?: number;
  offset?: number;
}

export async function listEvents(filters: EventFilters = {}): Promise<EventListResponse> {
  const r = await client.get<EventListResponse>('/events', { params: filters });
  return r.data;
}

export async function getEvent(id: string): Promise<Event> {
  const r = await client.get<Event>(`/events/${id}`);
  return r.data;
}

export async function getEventStats(camera_id?: string): Promise<EventStats> {
  const r = await client.get<EventStats>('/events/stats', { params: { camera_id } });
  return r.data;
}

export async function submitFeedback(
  eventId: string,
  data: { is_correct: boolean; corrected_label?: string; notes?: string; operator?: string }
): Promise<void> {
  await client.post(`/events/${eventId}/feedback`, data);
}
