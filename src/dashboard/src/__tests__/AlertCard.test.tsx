import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AlertCard } from '../components/features/AlertCard';
import type { Alert } from '../types';

vi.mock('../api/alerts', () => ({
  acknowledgeAlert: vi.fn().mockResolvedValue(undefined),
  dismissAlert: vi.fn().mockResolvedValue(undefined),
}));

const mockAlert: Alert = {
  id: 'alert_1',
  event_id: 'ev_1',
  priority: 'critical',
  title: 'Fighting detected — Camera Front',
  description: 'Violent activity detected.',
  status: 'pending',
  created_at: new Date(Date.now() - 60000).toISOString(),
};

function renderCard(alert: Alert = mockAlert) {
  return render(
    <MemoryRouter>
      <AlertCard alert={alert} />
    </MemoryRouter>
  );
}

describe('AlertCard', () => {
  it('renders the alert title', () => {
    renderCard();
    expect(screen.getByText('Fighting detected — Camera Front')).toBeInTheDocument();
  });

  it('renders priority badge', () => {
    renderCard();
    expect(screen.getByText('critical')).toBeInTheDocument();
  });

  it('shows Acknowledge button for active alerts', () => {
    renderCard();
    expect(screen.getByText('Acknowledge')).toBeInTheDocument();
  });

  it('shows Dismiss button for active alerts', () => {
    renderCard();
    expect(screen.getByText('Dismiss')).toBeInTheDocument();
  });

  it('does not show action buttons for acknowledged alerts', () => {
    renderCard({ ...mockAlert, status: 'acknowledged' });
    expect(screen.queryByText('Acknowledge')).not.toBeInTheDocument();
  });

  it('shows View Event link', () => {
    renderCard();
    expect(screen.getByText(/View Event/)).toBeInTheDocument();
  });
});
