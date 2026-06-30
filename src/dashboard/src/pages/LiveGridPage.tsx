import { useEffect, useState } from 'react';
import { listCameras } from '../api/cameras';
import { getActiveAlerts } from '../api/alerts';
import { CameraCard } from '../components/features/CameraCard';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { EmptyState } from '../components/common/EmptyState';
import type { Camera, Alert } from '../types';

export function LiveGridPage() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [activeAlerts, setActiveAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const [cams, alerts] = await Promise.all([listCameras(), getActiveAlerts()]);
      setCameras(cams);
      setActiveAlerts(alerts);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  const alertCameraIds = new Set(activeAlerts.map((a) => a.event_id));

  if (loading) return <LoadingSpinner message="Loading cameras…" />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Live View</h1>
        <p className="text-sm text-gray-400">Snapshots refresh every 5s · {cameras.length} cameras</p>
      </div>
      <p className="text-xs text-gray-400 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded px-3 py-2">
        ℹ️ Live view uses polling snapshots. For real-time streaming, WebRTC/HLS integration is planned for Phase 12+.
      </p>
      {cameras.length === 0 ? (
        <EmptyState icon="📷" title="No cameras configured" description="Add cameras in Settings to see them here." />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {cameras.map((cam) => (
            <CameraCard key={cam.id} camera={cam} hasAlert={alertCameraIds.has(cam.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
