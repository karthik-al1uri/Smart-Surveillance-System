import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { LoginPage } from '../pages/LoginPage';
import { useAuthStore } from '../stores/authStore';

vi.mock('../api/auth', () => ({
  login: vi.fn().mockResolvedValue({ access_token: 'test-token', token_type: 'bearer' }),
  getCurrentUser: vi.fn().mockResolvedValue({ id: '1', username: 'admin', role: 'admin', enabled: true }),
  logout: vi.fn(),
}));

function renderLoginPage() {
  return render(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>
  );
}

describe('LoginPage', () => {
  beforeEach(() => {
    useAuthStore.setState({ isAuthenticated: false, user: null, token: null, error: null, isLoading: false });
    localStorage.clear();
  });

  it('renders username and password fields', () => {
    renderLoginPage();
    expect(screen.getByPlaceholderText('admin')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('••••••••')).toBeInTheDocument();
  });

  it('renders sign in button', () => {
    renderLoginPage();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows branding text', () => {
    renderLoginPage();
    expect(screen.getByText('Smart Surveillance')).toBeInTheDocument();
  });

  it('calls login on form submit', async () => {
    const { login } = await import('../api/auth');
    renderLoginPage();
    await userEvent.type(screen.getByPlaceholderText('admin'), 'admin');
    await userEvent.type(screen.getByPlaceholderText('••••••••'), 'admin');
    await userEvent.click(screen.getByRole('button', { name: /sign in/i }));
    await waitFor(() => {
      expect(login).toHaveBeenCalledWith('admin', 'admin');
    });
  });

  it('shows error message on failed login', async () => {
    const { login } = await import('../api/auth');
    (login as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error('Unauthorized'));
    useAuthStore.setState({ error: 'Invalid username or password', isLoading: false });
    renderLoginPage();
    expect(screen.getByText('Invalid username or password')).toBeInTheDocument();
  });
});
