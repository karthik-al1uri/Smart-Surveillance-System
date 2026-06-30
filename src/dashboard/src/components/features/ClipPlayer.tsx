import { useRef, useState } from 'react';
import { getClipUrl } from '../../api/clips';

interface ClipPlayerProps {
  eventId: string;
  hasClip: boolean;
}

export function ClipPlayer({ eventId, hasClip }: ClipPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState(false);

  const stepFrame = (direction: 1 | -1) => {
    if (videoRef.current) {
      videoRef.current.currentTime += direction * (1 / 30);
    }
  };

  if (!hasClip) {
    return (
      <div className="aspect-video bg-gray-100 dark:bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-400">
          <div className="text-4xl mb-2">🎬</div>
          <p className="text-sm">Clip not available</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="aspect-video bg-gray-100 dark:bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-gray-400">
          <div className="text-4xl mb-2">⚠️</div>
          <p className="text-sm">Clip file not found on disk</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        src={getClipUrl(eventId)}
        controls
        className="w-full rounded-lg bg-black aspect-video"
        onError={() => setError(true)}
      />
      <div className="flex items-center gap-2 text-xs text-gray-500">
        <button
          onClick={() => stepFrame(-1)}
          className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          ← Frame
        </button>
        <button
          onClick={() => stepFrame(1)}
          className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          Frame →
        </button>
        <select
          onChange={(e) => { if (videoRef.current) videoRef.current.playbackRate = Number(e.target.value); }}
          className="ml-auto px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-transparent text-xs"
          defaultValue="1"
        >
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1">1×</option>
          <option value="2">2×</option>
        </select>
      </div>
    </div>
  );
}
