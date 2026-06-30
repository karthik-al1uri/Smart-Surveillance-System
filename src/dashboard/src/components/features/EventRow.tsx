import { useNavigate } from 'react-router-dom';
import { Badge } from '../common/Badge';
import { formatTimeAgo, formatSeverity } from '../../utils/formatters';
import type { Event } from '../../types';

interface EventRowProps {
  event: Event;
}

export function EventRow({ event }: EventRowProps) {
  const navigate = useNavigate();
  const pct = Math.round(event.severity_score * 100);

  return (
    <tr
      className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
      onClick={() => navigate(`/events/${event.id}`)}
    >
      <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400 whitespace-nowrap">
        {event.timestamp ? formatTimeAgo(event.timestamp) : '—'}
      </td>
      <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">
        {event.camera_id}
      </td>
      <td className="px-4 py-3">
        <Badge variant="category" level={event.event_category}>{event.event_category}</Badge>
      </td>
      <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">{event.event_label}</td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-1.5">
            <div
              className={`h-1.5 rounded-full ${
                pct >= 90 ? 'bg-red-600' : pct >= 70 ? 'bg-red-400' : pct >= 40 ? 'bg-yellow-400' : 'bg-blue-400'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">{formatSeverity(event.severity_score)}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-sm">
        {event.clip_path ? (
          <span className="text-green-600 dark:text-green-400 text-xs">📹 Clip</span>
        ) : (
          <span className="text-gray-400 text-xs">—</span>
        )}
      </td>
    </tr>
  );
}
