import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronDownIcon, ArrowRightIcon } from '@heroicons/react/20/solid';
import { Badge } from '../common/Badge';
import { acknowledgeAlert, dismissAlert } from '../../api/alerts';
import { useAppStore } from '../../stores/appStore';
import { useAuthStore } from '../../stores/authStore';
import { formatTimeAgo } from '../../utils/formatters';
import type { Alert } from '../../types';

const DISMISS_REASONS = ['false_positive', 'duplicate', 'resolved', 'test'];

interface AlertCardProps {
  alert: Alert;
  onStatusChange?: () => void;
}

export function AlertCard({ alert, onStatusChange }: AlertCardProps) {
  const navigate = useNavigate();
  const { addToast } = useAppStore();
  const { user } = useAuthStore();
  const [showDismissMenu, setShowDismissMenu] = useState(false);
  const [loading, setLoading] = useState(false);

  const isActive = alert.status === 'pending' || alert.status === 'delivered';

  const handleAck = async () => {
    setLoading(true);
    try {
      await acknowledgeAlert(alert.id, user?.username);
      addToast('success', 'Alert acknowledged');
      onStatusChange?.();
    } catch {
      addToast('error', 'Failed to acknowledge alert');
    } finally {
      setLoading(false);
    }
  };

  const handleDismiss = async (reason: string) => {
    setShowDismissMenu(false);
    setLoading(true);
    try {
      await dismissAlert(alert.id, reason, user?.username);
      addToast('success', 'Alert dismissed');
      onStatusChange?.();
    } catch {
      addToast('error', 'Failed to dismiss alert');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`rounded-xl border p-4 bg-white dark:bg-gray-800 shadow-sm ${
      isActive ? 'border-l-4 border-l-red-500' : 'border-gray-200 dark:border-gray-700'
    }`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-3 min-w-0">
          {isActive && (
            <span className="mt-1 w-2 h-2 rounded-full bg-red-500 animate-pulse flex-shrink-0" />
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge variant="priority" level={alert.priority}>{alert.priority}</Badge>
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {alert.title ?? 'Untitled Alert'}
              </span>
            </div>
            {alert.description && (
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 line-clamp-2">
                {alert.description}
              </p>
            )}
            <div className="mt-1 flex items-center gap-2 text-xs text-gray-400">
              <span>{alert.created_at ? formatTimeAgo(alert.created_at) : '—'}</span>
              <span>·</span>
              <Badge variant="status" level={alert.status}>{alert.status}</Badge>
            </div>
          </div>
        </div>
      </div>

      {isActive && (
        <div className="mt-3 flex items-center gap-2">
          <button
            onClick={handleAck}
            disabled={loading}
            className="px-3 py-1.5 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            Acknowledge
          </button>

          <div className="relative">
            <button
              onClick={() => setShowDismissMenu((v) => !v)}
              disabled={loading}
              className="flex items-center gap-1 px-3 py-1.5 text-xs bg-gray-200 dark:bg-gray-700 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50"
            >
              Dismiss <ChevronDownIcon className="w-3 h-3" />
            </button>
            {showDismissMenu && (
              <div className="absolute left-0 top-full mt-1 w-40 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg shadow-lg z-10">
                {DISMISS_REASONS.map((r) => (
                  <button
                    key={r}
                    onClick={() => handleDismiss(r)}
                    className="block w-full text-left px-3 py-2 text-xs hover:bg-gray-50 dark:hover:bg-gray-600 capitalize"
                  >
                    {r.replace('_', ' ')}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            onClick={() => navigate(`/events/${alert.event_id}`)}
            className="flex items-center gap-1 px-3 py-1.5 text-xs text-indigo-600 dark:text-indigo-400 hover:underline ml-auto"
          >
            View Event <ArrowRightIcon className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
}
