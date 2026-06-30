import { useRef, useEffect, useState, useCallback } from 'react';
import type { Zone } from '../../types';

const ZONE_COLORS: Record<string, string> = {
  restricted: 'rgba(239,68,68,0.35)',
  monitored: 'rgba(245,158,11,0.35)',
  safe: 'rgba(34,197,94,0.35)',
  perimeter: 'rgba(99,102,241,0.35)',
};
const ZONE_STROKE: Record<string, string> = {
  restricted: '#ef4444',
  monitored: '#f59e0b',
  safe: '#22c55e',
  perimeter: '#6366f1',
};

interface ZoneCanvasProps {
  zones: Zone[];
  onChange: (zones: Zone[]) => void;
  snapshotUrl?: string;
  width?: number;
  height?: number;
}

type Mode = 'draw' | 'select';

export function ZoneCanvas({ zones, onChange, snapshotUrl, width = 640, height = 360 }: ZoneCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<Mode>('draw');
  const [inProgress, setInProgress] = useState<number[][]>([]);
  const [mousePos, setMousePos] = useState<[number, number] | null>(null);
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);

  // Load background snapshot
  useEffect(() => {
    if (!snapshotUrl) return;
    const img = new Image();
    img.src = snapshotUrl;
    img.onload = () => setBgImage(img);
  }, [snapshotUrl]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (bgImage) {
      ctx.drawImage(bgImage, 0, 0, width, height);
    } else {
      ctx.fillStyle = '#1f2937';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#6b7280';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Camera frame (no snapshot available)', width / 2, height / 2);
    }

    // Draw existing zones
    zones.forEach((zone) => {
      if (zone.polygon.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(zone.polygon[0][0], zone.polygon[0][1]);
      zone.polygon.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
      ctx.closePath();
      ctx.fillStyle = ZONE_COLORS[zone.zone_type] ?? 'rgba(99,102,241,0.35)';
      ctx.fill();
      ctx.strokeStyle = ZONE_STROKE[zone.zone_type] ?? '#6366f1';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label
      const cx = zone.polygon.reduce((s, p) => s + p[0], 0) / zone.polygon.length;
      const cy = zone.polygon.reduce((s, p) => s + p[1], 0) / zone.polygon.length;
      ctx.fillStyle = 'white';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(zone.name, cx, cy);
    });

    // Draw in-progress polygon
    if (inProgress.length > 0) {
      ctx.beginPath();
      ctx.moveTo(inProgress[0][0], inProgress[0][1]);
      inProgress.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
      if (mousePos) ctx.lineTo(mousePos[0], mousePos[1]);
      ctx.strokeStyle = '#818cf8';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      inProgress.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#6366f1';
        ctx.fill();
      });
    }
  }, [zones, inProgress, mousePos, bgImage, width, height]);

  useEffect(() => { draw(); }, [draw]);

  const getPos = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [
      Math.round((e.clientX - rect.left) * (width / rect.width)),
      Math.round((e.clientY - rect.top) * (height / rect.height)),
    ];
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'draw') return;
    const [x, y] = getPos(e);
    const first = inProgress[0];
    // Close polygon if clicking near first vertex
    if (first && inProgress.length >= 3 && Math.hypot(x - first[0], y - first[1]) < 15) {
      const newZone: Zone = {
        zone_id: `zone_${Date.now()}`,
        name: `Zone ${zones.length + 1}`,
        zone_type: 'restricted',
        polygon: inProgress,
      };
      onChange([...zones, newZone]);
      setInProgress([]);
    } else {
      setInProgress([...inProgress, [x, y]]);
    }
  };

  const handleDblClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'draw' || inProgress.length < 3) return;
    e.preventDefault();
    const newZone: Zone = {
      zone_id: `zone_${Date.now()}`,
      name: `Zone ${zones.length + 1}`,
      zone_type: 'restricted',
      polygon: inProgress,
    };
    onChange([...zones, newZone]);
    setInProgress([]);
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setInProgress(inProgress.slice(0, -1));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        {(['draw', 'select'] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => { setMode(m); setInProgress([]); }}
            className={`px-3 py-1.5 text-sm rounded-lg capitalize ${
              mode === m ? 'bg-indigo-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            {m === 'draw' ? '✏️ Draw' : '↖ Select'}
          </button>
        ))}
        <button
          onClick={() => { setInProgress([]); onChange([]); }}
          className="px-3 py-1.5 text-sm rounded-lg bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 ml-auto"
        >
          Clear All
        </button>
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleClick}
        onDoubleClick={handleDblClick}
        onContextMenu={handleContextMenu}
        onMouseMove={(e) => setMousePos(getPos(e))}
        onMouseLeave={() => setMousePos(null)}
        className="rounded-lg cursor-crosshair w-full border border-gray-300 dark:border-gray-600"
        style={{ maxWidth: width }}
      />
      <p className="text-xs text-gray-400">
        {mode === 'draw'
          ? 'Click to place vertices · Double-click or click first vertex to close · Right-click to undo last point'
          : 'Click a zone to select it'}
      </p>
    </div>
  );
}
