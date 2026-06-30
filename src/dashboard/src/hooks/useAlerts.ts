import { useEffect, useCallback } from 'react';
import { useAlertStore } from '../stores/alertStore';
import { useAppStore } from '../stores/appStore';
import { useWebSocket } from './useWebSocket';
import { getActiveAlerts } from '../api/alerts';
import { WS_URL } from '../utils/constants';
import type { Alert } from '../types';

export function useAlerts() {
  const { addAlert, setAlerts } = useAlertStore();
  const { addToast } = useAppStore();

  // Seed from REST on mount
  useEffect(() => {
    getActiveAlerts()
      .then((alerts) => setAlerts(alerts))
      .catch(() => {/* backend may not be running */});
  }, [setAlerts]);

  const handleMessage = useCallback(
    (data: unknown) => {
      const alert = data as Alert;
      if (alert && alert.id) {
        addAlert(alert);
        addToast('warning', `New alert: ${alert.title ?? alert.priority}`);
      }
    },
    [addAlert, addToast]
  );

  useWebSocket(WS_URL, handleMessage);
}
