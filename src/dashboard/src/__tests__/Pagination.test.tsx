import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Pagination } from '../components/common/Pagination';

describe('Pagination', () => {
  it('returns null when only one page', () => {
    const { container } = render(<Pagination total={10} limit={25} offset={0} onChange={() => {}} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders page numbers for multiple pages', () => {
    render(<Pagination total={100} limit={25} offset={0} onChange={() => {}} />);
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
  });

  it('calls onChange with correct offset when next clicked', async () => {
    const onChange = vi.fn();
    render(<Pagination total={100} limit={25} offset={0} onChange={onChange} />);
    await userEvent.click(screen.getByText('→'));
    expect(onChange).toHaveBeenCalledWith(25);
  });

  it('calls onChange when a page number is clicked', async () => {
    const onChange = vi.fn();
    render(<Pagination total={100} limit={25} offset={0} onChange={onChange} />);
    await userEvent.click(screen.getByText('2'));
    expect(onChange).toHaveBeenCalledWith(25);
  });

  it('disables prev button on first page', () => {
    render(<Pagination total={100} limit={25} offset={0} onChange={() => {}} />);
    expect(screen.getByText('←')).toBeDisabled();
  });

  it('disables next button on last page', () => {
    render(<Pagination total={100} limit={25} offset={75} onChange={() => {}} />);
    expect(screen.getByText('→')).toBeDisabled();
  });
});
