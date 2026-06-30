import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, ResponsiveContainer,
  type PieLabelRenderProps,
} from 'recharts';
import { getEventStats } from '../api/events';
import { getAlertStats, listAlerts } from '../api/alerts';
import { AlertCard } from '../components/features/AlertCard';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { CHART_COLORS } from '../utils/constants';
import type { EventStats, AlertStats, Alert } from '../types';

export function DashboardPage() {
  const navigate = useNavigate();
  const [eventStats, setEventStats] = useState<EventStats | null>(null);
  const [alertStats, setAlertStats] = useState<AlertStats | null>(null);
  const [recentAlerts, setRecentAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    try {
      const [es, as_, ra] = await Promise.all([
        getEventStats(),
        getAlertStats(),
        listAlerts({ limit: 5 }),
      ]);
      setEventStats(es);
      setAlertStats(as_);
      setRecentAlerts(ra.alerts);
    } catch {
      /* backend may not be running in dev */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  const activeAlerts = alertStats
    ? (alertStats.by_status['pending'] ?? 0) + (alertStats.by_status['delivered'] ?? 0)
    : 0;
  const violentEvents = eventStats?.by_category['violent'] ?? 0;

  const pieData = eventStats
    ? Object.entries(eventStats.by_category).map(([name, value]) => ({ name, value }))
    : [];

  // Fake hourly data for chart (replace with real time-series endpoint in future)
  const hourlyData = Array.from({ length: 12 }, (_, i) => ({
    hour: `${(i * 2).toString().padStart(2, '0')}:00`,
    violent: Math.floor(Math.random() * 4),
    suspicious: Math.floor(Math.random() * 6),
    urgent: Math.floor(Math.random() * 3),
  }));

  if (loading) return <LoadingSpinner message="Loading dashboard…" />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Dashboard</h1>
        <span className="text-sm text-gray-400">Last 24 hours</span>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Total Events', value: eventStats?.total ?? 0, color: 'text-indigo-600', icon: '📋' },
          { label: 'Active Alerts', value: activeAlerts, color: activeAlerts > 0 ? 'text-red-600' : 'text-green-600', icon: '🔔' },
          { label: 'Violent Events', value: violentEvents, color: 'text-red-500', icon: '⚠️' },
          { label: 'Total Alerts', value: alertStats?.total ?? 0, color: 'text-gray-600', icon: '📊' },
        ].map((card) => (
          <div key={card.label} className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-2xl">{card.icon}</span>
              <span className={`text-2xl font-bold ${card.color}`}>{card.value}</span>
            </div>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{card.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Events over time */}
        <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Events Over Time (24h)</h2>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="violent" stroke={CHART_COLORS.violent} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="suspicious" stroke={CHART_COLORS.suspicious} dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="urgent" stroke={CHART_COLORS.urgent} dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent alerts */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Recent Alerts</h2>
            <button onClick={() => navigate('/alerts')} className="text-xs text-indigo-600 hover:underline">
              View all →
            </button>
          </div>
          <div className="flex-1 space-y-2 overflow-auto">
            {recentAlerts.length === 0 ? (
              <p className="text-sm text-gray-400 text-center py-8">No alerts</p>
            ) : (
              recentAlerts.map((a) => <AlertCard key={a.id} alert={a} onStatusChange={load} />)
            )}
          </div>
        </div>
      </div>

      {/* Category pie */}
      {pieData.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Events by Category</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }: PieLabelRenderProps) => `${String(name ?? '')} ${(((percent as number | undefined) ?? 0) * 100).toFixed(0)}%`}>
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={CHART_COLORS[entry.name] ?? '#8884d8'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
