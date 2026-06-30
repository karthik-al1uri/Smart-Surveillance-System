import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Badge } from '../components/common/Badge';

describe('Badge', () => {
  it('renders children text', () => {
    render(<Badge>critical</Badge>);
    expect(screen.getByText('critical')).toBeInTheDocument();
  });

  it('applies priority color class for critical', () => {
    const { container } = render(<Badge variant="priority" level="critical">critical</Badge>);
    expect(container.firstChild).toHaveClass('bg-red-200');
  });

  it('applies priority color class for low', () => {
    const { container } = render(<Badge variant="priority" level="low">low</Badge>);
    expect(container.firstChild).toHaveClass('bg-blue-100');
  });

  it('applies category color for violent', () => {
    const { container } = render(<Badge variant="category" level="violent">violent</Badge>);
    expect(container.firstChild).toHaveClass('bg-red-100');
  });

  it('falls back to default gray for unknown level', () => {
    const { container } = render(<Badge variant="priority" level="unknown">unknown</Badge>);
    expect(container.firstChild).toHaveClass('bg-gray-100');
  });
});
