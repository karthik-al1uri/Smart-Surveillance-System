import { useEffect, useState } from 'react';
import { listCameras, createCamera, deleteCamera } from '../api/cameras';
import { getSystemHealth, getStorageStats } from '../api/config';
import { ConfirmDialog } from '../components/common/ConfirmDialog';
import { StatusDot } from '../components/common/StatusDot';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { useAppStore } from '../stores/appStore';
import { useNavigate } from 'react-router-dom';
import { formatBytes } from '../utils/formatters';
import type { Camera, SystemHealth, StorageStats } from '../types';

const TABS = ['Cameras', 'System'] as const;
type Tab = typeof TABS[number];

export function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('Cameras');
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [storage, setStorage] = useState<StorageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const { addToast } = useAppStore();
  const navigate = useNavigate();

  const [newCam, setNewCam] = useState({ id: '', name: '', stream_url: '', location: '' });

  useEffect(() => {
    Promise.all([listCameras(), getSystemHealth(), getStorageStats()])
      .then(([cams, h, s]) => { setCameras(cams); setHealth(h); setStorage(s); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleAddCamera = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const created = await createCamera({ ...newCam, enabled: true });
      setCameras([...cameras, created]);
      setNewCam({ id: '', name: '', stream_url: '', location: '' });
      addToast('success', 'Camera added');
    } catch {
      addToast('error', 'Failed to add camera');
    }
  };

  const handleDeleteCamera = async () => {
    if (!deleteTarget) return;
    try {
      await deleteCamera(deleteTarget);
      setCameras(cameras.filter((c) => c.id !== deleteTarget));
      addToast('success', 'Camera deleted');
    } catch {
      addToast('error', 'Failed to delete camera');
    } finally {
      setDeleteTarget(null);
    }
  };

  if (loading) return <LoadingSpinner message="Loading settings…" />;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>

      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg w-fit">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-4 py-1.5 text-sm rounded-md transition-colors ${
              activeTab === t
                ? 'bg-white dark:bg-gray-700 shadow font-medium text-gray-900 dark:text-gray-100'
                : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {activeTab === 'Cameras' && (
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 overflow-hidden shadow-sm">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-900/50">
                <tr>
                  {['ID', 'Name', 'Location', 'Status', 'Actions'].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                {cameras.map((cam) => (
                  <tr key={cam.id} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                    <td className="px-4 py-3 text-xs text-gray-400 font-mono">{cam.id}</td>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">{cam.name}</td>
                    <td className="px-4 py-3 text-sm text-gray-500">{cam.location ?? '—'}</td>
                    <td className="px-4 py-3">
                      <StatusDot status={cam.enabled ? 'online' : 'offline'} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-2">
                        <button
                          onClick={() => navigate(`/zones/${cam.id}`)}
                          className="text-xs text-indigo-600 hover:underline"
                        >
                          Edit Zones
                        </button>
                        <button
                          onClick={() => setDeleteTarget(cam.id)}
                          className="text-xs text-red-500 hover:underline"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <h2 className="font-semibold text-gray-900 dark:text-gray-100 mb-4 text-sm">Add Camera</h2>
            <form onSubmit={handleAddCamera} className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { key: 'id', label: 'Camera ID', placeholder: 'cam_06' },
                { key: 'name', label: 'Name', placeholder: 'Parking Lot' },
                { key: 'stream_url', label: 'Stream URL', placeholder: 'rtsp://192.168.1.106/stream' },
                { key: 'location', label: 'Location', placeholder: 'Building B' },
              ].map(({ key, label, placeholder }) => (
                <div key={key}>
                  <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</label>
                  <input
                    value={newCam[key as keyof typeof newCam]}
                    onChange={(e) => setNewCam({ ...newCam, [key]: e.target.value })}
                    placeholder={placeholder}
                    className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-700"
                    required={key !== 'location'}
                  />
                </div>
              ))}
              <div className="sm:col-span-2">
                <button type="submit" className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700">
                  Add Camera
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {activeTab === 'System' && (
        <div className="space-y-4">
          {health && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
              <h2 className="font-semibold text-gray-900 dark:text-gray-100 mb-3 text-sm">System Health</h2>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(health.components ?? {}).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between text-sm py-1">
                    <span className="text-gray-500 capitalize">{k}</span>
                    <StatusDot status={v === 'ok' ? 'online' : 'offline'} label={String(v)} />
                  </div>
                ))}
              </div>
            </div>
          )}
          {storage && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
              <h2 className="font-semibold text-gray-900 dark:text-gray-100 mb-3 text-sm">Storage</h2>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <span className="text-gray-500">Total Clips</span><span className="font-medium">{storage.total_clips}</span>
                <span className="text-gray-500">Total Size</span><span className="font-medium">{formatBytes(storage.total_size_bytes)}</span>
              </div>
            </div>
          )}
        </div>
      )}

      <ConfirmDialog
        open={!!deleteTarget}
        title="Delete Camera"
        message={`Are you sure you want to delete camera "${deleteTarget}"? This cannot be undone.`}
        confirmLabel="Delete"
        danger
        onConfirm={handleDeleteCamera}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  );
}
