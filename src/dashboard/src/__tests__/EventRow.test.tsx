import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { EventRow } from '../components/features/EventRow';
import type { Event } from '../types';

const mockEvent: Event = {
  id: 'ev_1',
  camera_id: 'cam_01',
  timestamp: new Date().toISOString(),
  event_category: 'violent',
  event_label: 'fighting',
  severity_score: 0.82,
  acknowledged: false,
};

function renderRow(event: Event = mockEvent) {
  return render(
    <MemoryRouter>
      <table><tbody><EventRow event={event} /></tbody></table>
    </MemoryRouter>
  );
}

describe('EventRow', () => {
  it('renders camera id', () => {
    renderRow();
    expect(screen.getByText('cam_01')).toBeInTheDocument();
  });

  it('renders event label', () => {
    renderRow();
    expect(screen.getByText('fighting')).toBeInTheDocument();
  });

  it('renders category badge', () => {
    renderRow();
    expect(screen.getByText('violent')).toBeInTheDocument();
  });

  it('renders severity percentage', () => {
    renderRow();
    expect(screen.getByText('82%')).toBeInTheDocument();
  });

  it('shows no clip indicator when no clip_path', () => {
    renderRow();
    const dashElements = screen.getAllByText('—');
    expect(dashElements.length).toBeGreaterThan(0);
  });
});
