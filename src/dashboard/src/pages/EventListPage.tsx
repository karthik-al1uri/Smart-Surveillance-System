import { useEffect, useState, useCallback } from 'react';
import { listEvents } from '../api/events';
import { listCameras } from '../api/cameras';
import { EventRow } from '../components/features/EventRow';
import { Pagination } from '../components/common/Pagination';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EmptyState } from '../components/common/EmptyState';
import type { EventListResponse, Camera } from '../types';

const CATEGORIES = ['violent', 'weapon', 'suspicious', 'urgent', 'normal'];
const LIMIT = 25;

export function EventListPage() {
  const [result, setResult] = useState<EventListResponse>({ events: [], total: 0, limit: LIMIT, offset: 0 });
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [cameraId, setCameraId] = useState('');
  const [category, setCategory] = useState('');
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    listCameras().then(setCameras).catch(() => {});
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await listEvents({
        camera_id: cameraId || undefined,
        category: category || undefined,
        limit: LIMIT,
        offset,
      });
      setResult(r);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [cameraId, category, offset]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { setOffset(0); }, [cameraId, category]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Event History</h1>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 items-center">
        <select
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
          className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-800"
        >
          <option value="">All Cameras</option>
          {cameras.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-800"
        >
          <option value="">All Categories</option>
          {CATEGORIES.map((c) => <option key={c} value={c} className="capitalize">{c}</option>)}
        </select>
        <span className="text-sm text-gray-400 ml-auto">{result.total} events</span>
      </div>

      {loading ? (
        <LoadingSpinner message="Loading events…" />
      ) : result.events.length === 0 ? (
        <EmptyState icon="📋" title="No events found" description="Try adjusting your filters." />
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-900/50">
              <tr>
                {['Time', 'Camera', 'Category', 'Label', 'Severity', 'Clip'].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wide">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {result.events.map((ev) => <EventRow key={ev.id} event={ev} />)}
            </tbody>
          </table>
        </div>
      )}

      <Pagination total={result.total} limit={LIMIT} offset={offset} onChange={setOffset} />
    </div>
  );
}
