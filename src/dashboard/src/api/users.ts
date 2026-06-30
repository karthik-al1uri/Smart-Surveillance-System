import client from './client';
import type { User } from '../types';

export async function listUsers(): Promise<User[]> {
  const r = await client.get<User[]>('/users');
  return r.data;
}

export async function createUser(data: {
  username: string;
  password: string;
  role: string;
  full_name?: string;
}): Promise<User> {
  const r = await client.post<User>('/users', data);
  return r.data;
}

export async function updateUser(id: string, data: Record<string, unknown>): Promise<User> {
  const r = await client.put<User>(`/users/${id}`, data);
  return r.data;
}

export async function deleteUser(id: string): Promise<void> {
  await client.delete(`/users/${id}`);
}
