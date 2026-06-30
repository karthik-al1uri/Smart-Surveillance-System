import { useEffect, useState } from 'react';
import { listUsers, createUser, deleteUser } from '../api/users';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { useAppStore } from '../stores/appStore';
import { useAuthStore } from '../stores/authStore';
import { Navigate } from 'react-router-dom';
import type { User } from '../types';

const ROLES = ['admin', 'operator', 'viewer'] as const;

export function UserManagementPage() {
  const { user: currentUser } = useAuthStore();
  const { addToast } = useAppStore();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [newUser, setNewUser] = useState({ username: '', password: '', full_name: '', role: 'operator' });

  if (currentUser?.role !== 'admin') {
    return <Navigate to="/" replace />;
  }

  useEffect(() => {
    listUsers().then(setUsers).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const u = await createUser(newUser);
      setUsers([...users, u]);
      setNewUser({ username: '', password: '', full_name: '', role: 'operator' });
      addToast('success', `User ${u.username} created`);
    } catch {
      addToast('error', 'Failed to create user');
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteUser(deleteTarget);
      setUsers(users.filter((u) => u.id !== deleteTarget));
      addToast('success', 'User deleted');
    } catch {
      addToast('error', 'Failed to delete user');
    } finally {
      setDeleteTarget(null);
    }
  };

  if (loading) return <LoadingSpinner message="Loading users…" />;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">User Management</h1>

      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-900/50">
            <tr>
              {['Username', 'Full Name', 'Role', 'Status', 'Actions'].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
            {users.map((u) => (
              <tr key={u.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">{u.username}</td>
                <td className="px-4 py-3 text-sm text-gray-500">{u.full_name ?? '—'}</td>
                <td className="px-4 py-3"><Badge level={u.role} variant="default">{u.role}</Badge></td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${u.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                    {u.enabled ? 'Active' : 'Disabled'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {u.id !== currentUser?.id && (
                    <button
                      onClick={() => setDeleteTarget(u.id)}
                      className="text-xs text-red-500 hover:underline"
                    >
                      Delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
        <h2 className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-4">Add User</h2>
        <form onSubmit={handleAdd} className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { key: 'username', label: 'Username', type: 'text' },
            { key: 'password', label: 'Password', type: 'password' },
            { key: 'full_name', label: 'Full Name', type: 'text' },
          ].map(({ key, label, type }) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</label>
              <input
                type={type}
                value={newUser[key as keyof typeof newUser]}
                onChange={(e) => setNewUser({ ...newUser, [key]: e.target.value })}
                className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-700"
                required={key !== 'full_name'}
              />
            </div>
          ))}
          <div>
            <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">Role</label>
            <select
              value={newUser.role}
              onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
              className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-700"
            >
              {ROLES.map((r) => <option key={r} value={r} className="capitalize">{r}</option>)}
            </select>
          </div>
          <div className="sm:col-span-2">
            <button type="submit" className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700">
              Add User
            </button>
          </div>
        </form>
      </div>

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete User"
        message="Are you sure you want to delete this user? This cannot be undone."
        confirmLabel="Delete"
        danger
        onConfirm={handleDelete}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
