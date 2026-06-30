import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCamera, updateZones } from '../api/cameras';
import { getSnapshotUrl } from '../api/clips';
import { ZoneCanvas } from '../components/features/ZoneCanvas';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { useAppStore } from '../stores/appStore';
import { TrashIcon } from '@heroicons/react/24/outline';
import type { Camera, Zone } from '../types';

const ZONE_TYPES = ['restricted', 'monitored', 'safe', 'perimeter'] as const;

export function ZoneEditorPage() {
  const { cameraId } = useParams<{ cameraId: string }>();
  const navigate = useNavigate();
  const { addToast } = useAppStore();
  const [camera, setCamera] = useState<Camera | null>(null);
  const [zones, setZones] = useState<Zone[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!cameraId) return;
    getCamera(cameraId)
      .then((c) => { setCamera(c); setZones(c.zones_config ?? []); })
      .catch(() => addToast('error', 'Camera not found'))
      .finally(() => setLoading(false));
  }, [cameraId]);

  const handleSave = async () => {
    if (!cameraId) return;
    setSaving(true);
    try {
      await updateZones(cameraId, zones);
      addToast('success', 'Zones saved');
    } catch {
      addToast('error', 'Failed to save zones');
    } finally {
      setSaving(false);
    }
  };

  const updateZoneField = (zoneId: string, field: keyof Zone, value: unknown) => {
    setZones(zones.map((z) => z.zone_id === zoneId ? { ...z, [field]: value } : z));
  };

  const deleteZone = (zoneId: string) => {
    setZones(zones.filter((z) => z.zone_id !== zoneId));
  };

  if (loading) return <LoadingSpinner message="Loading camera…" />;
  if (!camera) return <div className="text-gray-500 text-center py-16">Camera not found.</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="text-indigo-600 hover:underline text-sm">← Back</button>
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Zone Editor — {camera.name}</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Canvas */}
        <div className="lg:col-span-2">
          <ZoneCanvas
            zones={zones}
            onChange={setZones}
            snapshotUrl={`${getSnapshotUrl(camera.id)}?t=${Date.now()}`}
          />
        </div>

        {/* Zone list */}
        <div className="space-y-3">
          <h2 className="font-semibold text-gray-900 dark:text-gray-100">Zones ({zones.length})</h2>
          {zones.length === 0 ? (
            <p className="text-sm text-gray-400">No zones yet. Draw on the canvas to add.</p>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {zones.map((zone) => (
                <div key={zone.zone_id} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <input
                      value={zone.name}
                      onChange={(e) => updateZoneField(zone.zone_id, 'name', e.target.value)}
                      className="text-sm font-medium bg-transparent border-b border-gray-300 dark:border-gray-600 focus:outline-none w-32"
                    />
                    <button onClick={() => deleteZone(zone.zone_id)} className="text-red-400 hover:text-red-600">
                      <TrashIcon className="w-4 h-4" />
                    </button>
                  </div>
                  <select
                    value={zone.zone_type}
                    onChange={(e) => updateZoneField(zone.zone_id, 'zone_type', e.target.value)}
                    className="w-full text-xs border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700"
                  >
                    {ZONE_TYPES.map((t) => <option key={t} value={t} className="capitalize">{t}</option>)}
                  </select>
                  <p className="text-xs text-gray-400">{zone.polygon.length} vertices</p>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleSave}
            disabled={saving}
            className="w-full py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-60"
          >
            {saving ? 'Saving…' : 'Save Zones'}
          </button>
        </div>
      </div>
    </div>
  );
}
