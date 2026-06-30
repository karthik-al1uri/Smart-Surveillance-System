import { create } from 'zustand';
import type { Alert } from '../types';

interface AlertStoreState {
  liveAlerts: Alert[];
  unreadCount: number;
  addAlert: (alert: Alert) => void;
  markRead: (alertId: string) => void;
  markAllRead: () => void;
  clearAll: () => void;
  setAlerts: (alerts: Alert[]) => void;
}

export const useAlertStore = create<AlertStoreState>((set) => ({
  liveAlerts: [],
  unreadCount: 0,

  addAlert: (alert) =>
    set((state) => ({
      liveAlerts: [alert, ...state.liveAlerts].slice(0, 100),
      unreadCount: state.unreadCount + 1,
    })),

  markRead: (alertId) =>
    set((state) => ({
      unreadCount: Math.max(0, state.unreadCount - 1),
      liveAlerts: state.liveAlerts.map((a) =>
        a.id === alertId ? { ...a, status: 'acknowledged' as const } : a
      ),
    })),

  markAllRead: () => set({ unreadCount: 0 }),

  clearAll: () => set({ liveAlerts: [], unreadCount: 0 }),

  setAlerts: (alerts) =>
    set({
      liveAlerts: alerts,
      unreadCount: alerts.filter((a) => a.status === 'pending' || a.status === 'delivered').length,
    }),
}));
