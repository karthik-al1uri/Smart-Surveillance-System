import { useEffect, useState } from 'react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  type PieLabelRenderProps,
} from 'recharts';
import { getEventStats } from '../api/events';
import { getAlertStats } from '../api/alerts';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { CHART_COLORS } from '../utils/constants';
import type { EventStats, AlertStats } from '../types';

const HOURS = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`);
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

function fakeHourly() {
  return HOURS.map((hour) => ({
    hour,
    violent: Math.floor(Math.random() * 5),
    suspicious: Math.floor(Math.random() * 8),
    urgent: Math.floor(Math.random() * 4),
    normal: Math.floor(Math.random() * 12),
  }));
}

function fakeHeatmap(): Record<string, number | string>[] {
  return DAYS.map((day) => ({
    day,
    ...Object.fromEntries(Array.from({ length: 24 }, (_, h) => [String(h), Math.floor(Math.random() * 10)])),
  }));
}

function fakeSeverityHistogram() {
  return Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}–${(i + 1) * 10}%`,
    count: Math.floor(Math.random() * 30),
  }));
}

export function AnalyticsPage() {
  const [eventStats, setEventStats] = useState<EventStats | null>(null);
  const [alertStats, setAlertStats] = useState<AlertStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getEventStats(), getAlertStats()])
      .then(([es, as_]) => { setEventStats(es); setAlertStats(as_); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSpinner message="Loading analytics…" />;

  const pieData = eventStats
    ? Object.entries(eventStats.by_category).map(([name, value]) => ({ name, value }))
    : [];
  const cameraData = eventStats
    ? Object.entries(eventStats.by_camera).map(([name, value]) => ({ name, value }))
    : [];
  const alertStatusData = alertStats
    ? Object.entries(alertStats.by_status).map(([name, value]) => ({ name, value }))
    : [];
  const hourlyData = fakeHourly();
  const severityData = fakeSeverityHistogram();

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">Analytics</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Events over time */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Events Over Time (24h)</h2>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={hourlyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="hour" tick={{ fontSize: 9 }} interval={3} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Legend />
              {Object.keys(CHART_COLORS).map((cat) => (
                <Area key={cat} type="monotone" dataKey={cat} stackId="1" stroke={CHART_COLORS[cat]} fill={CHART_COLORS[cat]} fillOpacity={0.4} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Events by category pie */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Events by Category</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={80} label={({ name, percent }: PieLabelRenderProps) => `${String(name ?? '')} ${(((percent as number | undefined) ?? 0) * 100).toFixed(0)}%`}>
                {pieData.map((entry, i) => <Cell key={i} fill={CHART_COLORS[entry.name] ?? '#8884d8'} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Events by camera */}
        {cameraData.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Events by Camera</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={cameraData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Severity distribution */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Severity Distribution</h2>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={severityData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="range" tick={{ fontSize: 9 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Alert status */}
        {alertStatusData.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Alert Status Breakdown</h2>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={alertStatusData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={80} />
                <Tooltip />
                <Bar dataKey="value" fill="#22c55e" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Event heatmap (simplified grid) */}
        <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm">
          <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Event Heatmap (Day × Hour)</h2>
          <div className="overflow-x-auto">
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="w-8" />
                  {Array.from({ length: 24 }, (_, h) => (
                    <th key={h} className="w-5 text-center text-gray-400 font-normal pb-1">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {fakeHeatmap().map((row) => (
                  <tr key={row.day}>
                    <td className="text-gray-400 pr-2 text-right">{row.day}</td>
                    {Array.from({ length: 24 }, (_, h) => {
                      const v = (row[String(h)] as number) ?? 0;
                      const opacity = Math.min(v / 10, 1);
                      return (
                        <td key={h}>
                          <div
                            className="w-4 h-4 rounded-sm"
                            style={{ backgroundColor: `rgba(99,102,241,${opacity})` }}
                            title={`${row.day} ${h}:00 — ${v} events`}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-gray-400 mt-2">Darker = more events</p>
        </div>
      </div>
    </div>
  );
}
