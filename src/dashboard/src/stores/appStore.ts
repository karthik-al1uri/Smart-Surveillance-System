import { create } from 'zustand';

type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
}

interface AppState {
  sidebarOpen: boolean;
  darkMode: boolean;
  toasts: Toast[];
  toggleSidebar: () => void;
  toggleDarkMode: () => void;
  addToast: (type: ToastType, message: string) => void;
  removeToast: (id: string) => void;
}

const savedDark = localStorage.getItem('darkMode');
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
const initialDark = savedDark !== null ? savedDark === 'true' : prefersDark;

if (initialDark) {
  document.documentElement.classList.add('dark');
}

export const useAppStore = create<AppState>((set) => ({
  sidebarOpen: true,
  darkMode: initialDark,
  toasts: [],

  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  toggleDarkMode: () =>
    set((s) => {
      const next = !s.darkMode;
      localStorage.setItem('darkMode', String(next));
      if (next) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      return { darkMode: next };
    }),

  addToast: (type, message) => {
    const id = Math.random().toString(36).slice(2);
    set((s) => ({ toasts: [...s.toasts, { id, type, message }] }));
    setTimeout(() => {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    }, 5000);
  },

  removeToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));
