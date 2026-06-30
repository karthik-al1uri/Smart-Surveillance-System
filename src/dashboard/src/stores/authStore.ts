import { create } from 'zustand';
import * as authApi from '../api/auth';
import type { User } from '../types';

interface AuthState {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: localStorage.getItem('access_token'),
  user: null,
  isAuthenticated: !!localStorage.getItem('access_token'),
  isLoading: false,
  error: null,

  login: async (username, password) => {
    set({ isLoading: true, error: null });
    try {
      await authApi.login(username, password);
      const user = await authApi.getCurrentUser();
      set({ token: localStorage.getItem('access_token'), user, isAuthenticated: true, isLoading: false });
    } catch {
      set({ error: 'Invalid username or password', isLoading: false });
      throw new Error('Login failed');
    }
  },

  logout: () => {
    authApi.logout();
    set({ token: null, user: null, isAuthenticated: false });
  },

  checkAuth: async () => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      set({ isAuthenticated: false });
      return;
    }
    try {
      const user = await authApi.getCurrentUser();
      set({ token, user, isAuthenticated: true });
    } catch {
      authApi.logout();
      set({ token: null, user: null, isAuthenticated: false });
    }
  },
}));
