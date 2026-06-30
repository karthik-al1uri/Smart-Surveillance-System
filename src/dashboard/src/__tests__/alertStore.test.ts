import { describe, it, expect, beforeEach } from 'vitest';
import { useAlertStore } from '../stores/alertStore';
import type { Alert } from '../types';

const makeAlert = (id: string, status: Alert['status'] = 'pending'): Alert => ({
  id,
  event_id: `ev_${id}`,
  priority: 'high',
  status,
  title: `Alert ${id}`,
});

describe('alertStore', () => {
  beforeEach(() => {
    useAlertStore.getState().clearAll();
  });

  it('starts with empty alerts and zero unread', () => {
    const { liveAlerts, unreadCount } = useAlertStore.getState();
    expect(liveAlerts).toHaveLength(0);
    expect(unreadCount).toBe(0);
  });

  it('addAlert increases unreadCount', () => {
    useAlertStore.getState().addAlert(makeAlert('a1'));
    expect(useAlertStore.getState().unreadCount).toBe(1);
    expect(useAlertStore.getState().liveAlerts).toHaveLength(1);
  });

  it('addAlert prepends newest first', () => {
    useAlertStore.getState().addAlert(makeAlert('a1'));
    useAlertStore.getState().addAlert(makeAlert('a2'));
    expect(useAlertStore.getState().liveAlerts[0].id).toBe('a2');
  });

  it('markRead decreases unreadCount', () => {
    useAlertStore.getState().addAlert(makeAlert('a1'));
    useAlertStore.getState().markRead('a1');
    expect(useAlertStore.getState().unreadCount).toBe(0);
  });

  it('clearAll resets everything', () => {
    useAlertStore.getState().addAlert(makeAlert('a1'));
    useAlertStore.getState().clearAll();
    expect(useAlertStore.getState().liveAlerts).toHaveLength(0);
    expect(useAlertStore.getState().unreadCount).toBe(0);
  });

  it('setAlerts computes unreadCount from pending/delivered', () => {
    useAlertStore.getState().setAlerts([
      makeAlert('a1', 'pending'),
      makeAlert('a2', 'delivered'),
      makeAlert('a3', 'acknowledged'),
    ]);
    expect(useAlertStore.getState().unreadCount).toBe(2);
  });
});
