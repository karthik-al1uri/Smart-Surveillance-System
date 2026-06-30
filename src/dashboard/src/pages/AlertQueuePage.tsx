import { useEffect, useState, useCallback } from 'react';
import { listAlerts } from '../api/alerts';
import { AlertCard } from '../components/features/AlertCard';
import { Pagination } from '../components/common/Pagination';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EmptyState } from '../components/common/EmptyState';
import type { AlertListResponse } from '../types';

const STATUS_TABS = ['all', 'pending', 'delivered', 'acknowledged', 'dismissed'];
const PRIORITIES = ['critical', 'high', 'medium', 'low'];
const LIMIT = 20;

export function AlertQueuePage() {
  const [result, setResult] = useState<AlertListResponse>({ alerts: [], total: 0, limit: LIMIT, offset: 0 });
  const [loading, setLoading] = useState(true);
  const [activeStatus, setActiveStatus] = useState('all');
  const [activePriority, setActivePriority] = useState('');
  const [offset, setOffset] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await listAlerts({
        status: activeStatus === 'all' ? undefined : activeStatus,
        priority: activePriority || undefined,
        limit: LIMIT,
        offset,
      });
      setResult(r);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [activeStatus, activePriority, offset]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { setOffset(0); }, [activeStatus, activePriority]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Alert Queue</h1>

      {/* Status tabs */}
      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit flex-wrap">
        {STATUS_TABS.map((s) => (
          <button
            key={s}
            onClick={() => setActiveStatus(s)}
            className={`px-3 py-1.5 text-sm rounded-md capitalize transition-colors ${
              activeStatus === s
                ? 'bg-white dark:bg-gray-700 shadow text-gray-900 dark:text-gray-100 font-medium'
                : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      {/* Priority filter */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-sm text-gray-500">Priority:</span>
        <button
          onClick={() => setActivePriority('')}
          className={`px-3 py-1 text-xs rounded-full border ${
            !activePriority ? 'bg-indigo-600 text-white border-indigo-600' : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
          }`}
        >
          All
        </button>
        {PRIORITIES.map((p) => (
          <button
            key={p}
            onClick={() => setActivePriority(p === activePriority ? '' : p)}
            className={`px-3 py-1 text-xs rounded-full border capitalize ${
              activePriority === p
                ? 'bg-indigo-600 text-white border-indigo-600'
                : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
            }`}
          >
            {p}
          </button>
        ))}
      </div>

      {loading ? (
        <LoadingSpinner message="Loading alerts…" />
      ) : result.alerts.length === 0 ? (
        <EmptyState icon="🔔" title="No alerts" description="No alerts match your current filters." />
      ) : (
        <div className="space-y-3">
          {result.alerts.map((a) => (
            <AlertCard key={a.id} alert={a} onStatusChange={load} />
          ))}
        </div>
      )}

      <Pagination total={result.total} limit={LIMIT} offset={offset} onChange={setOffset} />
    </div>
  );
}
