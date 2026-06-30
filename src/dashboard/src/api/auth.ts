import client from './client';
import type { TokenResponse, User } from '../types';

export async function login(username: string, password: string): Promise<TokenResponse> {
  const response = await client.post<TokenResponse>('/auth/login', { username, password });
  localStorage.setItem('access_token', response.data.access_token);
  return response.data;
}

export function logout(): void {
  localStorage.removeItem('access_token');
}

export async function getCurrentUser(): Promise<User> {
  const response = await client.get<User>('/auth/me');
  return response.data;
}

export async function refreshToken(): Promise<TokenResponse> {
  const response = await client.post<TokenResponse>('/auth/refresh');
  localStorage.setItem('access_token', response.data.access_token);
  return response.data;
}
