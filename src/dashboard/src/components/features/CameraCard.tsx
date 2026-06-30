import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { PencilSquareIcon } from '@heroicons/react/24/outline';
import { StatusDot } from '../common/StatusDot';
import { getSnapshotUrl } from '../../api/clips';
import type { Camera } from '../../types';

interface CameraCardProps {
  camera: Camera;
  hasAlert?: boolean;
}

export function CameraCard({ camera, hasAlert }: CameraCardProps) {
  const navigate = useNavigate();
  const [snapshotError, setSnapshotError] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // Refresh snapshot every 5 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (imgRef.current && !snapshotError) {
        imgRef.current.src = `${getSnapshotUrl(camera.id)}?t=${Date.now()}`;
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [camera.id, snapshotError]);

  return (
    <div
      className={`rounded-xl border bg-white dark:bg-gray-800 overflow-hidden shadow-sm ${
        hasAlert ? 'border-2 border-red-500' : 'border-gray-200 dark:border-gray-700'
      }`}
    >
      <div className="relative bg-gray-900 aspect-video flex items-center justify-center">
        {!snapshotError ? (
          <img
            ref={imgRef}
            src={`${getSnapshotUrl(camera.id)}?t=${Date.now()}`}
            alt={camera.name}
            className="w-full h-full object-cover"
            onError={() => setSnapshotError(true)}
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-gray-500">
            <span className="text-3xl">📷</span>
            <span className="text-xs">No snapshot available</span>
          </div>
        )}
        {hasAlert && (
          <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-0.5 rounded-full animate-pulse">
            ALERT
          </div>
        )}
      </div>

      <div className="p-3 flex items-center justify-between">
        <div>
          <p className="font-medium text-sm text-gray-900 dark:text-gray-100">{camera.name}</p>
          {camera.location && (
            <p className="text-xs text-gray-400">{camera.location}</p>
          )}
          <StatusDot status={camera.enabled ? 'online' : 'offline'} />
        </div>
        <button
          onClick={() => navigate(`/zones/${camera.id}`)}
          className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400"
          title="Edit zones"
        >
          <PencilSquareIcon className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
