import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getEvent, submitFeedback } from '../api/events';
import { ClipPlayer } from '../components/features/ClipPlayer';
import { Badge } from '../components/common/Badge';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { useAppStore } from '../stores/appStore';
import { useAuthStore } from '../stores/authStore';
import { formatDateTime, formatSeverity } from '../utils/formatters';
import { ACTION_LABELS } from '../utils/constants';
import type { Event } from '../types';

export function EventDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { addToast } = useAppStore();
  const { user } = useAuthStore();
  const [event, setEvent] = useState<Event | null>(null);
  const [loading, setLoading] = useState(true);

  const [feedbackCorrect, setFeedbackCorrect] = useState<boolean | null>(null);
  const [correctedLabel, setCorrectedLabel] = useState('');
  const [notes, setNotes] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!id) return;
    getEvent(id).then(setEvent).catch(() => addToast('error', 'Event not found')).finally(() => setLoading(false));
  }, [id]);

  const handleFeedback = async () => {
    if (!id || feedbackCorrect === null) return;
    setSubmitting(true);
    try {
      await submitFeedback(id, {
        is_correct: feedbackCorrect,
        corrected_label: feedbackCorrect ? undefined : correctedLabel,
        notes: notes || undefined,
        operator: user?.username,
      });
      addToast('success', 'Feedback submitted');
      setFeedbackSubmitted(true);
    } catch {
      addToast('error', 'Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <LoadingSpinner message="Loading event…" />;
  if (!event) return <div className="text-gray-500 text-center py-16">Event not found.</div>;

  const pct = Math.round(event.severity_score * 100);

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="text-indigo-600 hover:underline text-sm">← Back</button>
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          {event.event_label} — {event.camera_id}
        </h1>
        <Badge variant="category" level={event.event_category}>{event.event_category}</Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Clip player */}
        <div>
          <ClipPlayer eventId={event.id} hasClip={!!event.clip_path} />
        </div>

        {/* Event details */}
        <div className="space-y-4">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm space-y-3">
            <h2 className="font-semibold text-gray-900 dark:text-gray-100">Event Details</h2>
            {[
              { label: 'Category', value: event.event_category },
              { label: 'Label', value: event.event_label },
              { label: 'Camera', value: event.camera_id },
              { label: 'Zone', value: event.zone_name ?? '—' },
              { label: 'Time', value: event.timestamp ? formatDateTime(event.timestamp) : '—' },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between text-sm">
                <span className="text-gray-500">{label}</span>
                <span className="font-medium text-gray-900 dark:text-gray-100 capitalize">{value}</span>
              </div>
            ))}
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-500">Severity</span>
                <span className="font-medium">{formatSeverity(event.severity_score)}</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${pct >= 90 ? 'bg-red-600' : pct >= 70 ? 'bg-red-400' : pct >= 40 ? 'bg-yellow-400' : 'bg-blue-400'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>

            {event.contributing_signals && event.contributing_signals.length > 0 && (
              <div>
                <p className="text-sm text-gray-500 mb-1">Contributing Signals</p>
                {event.contributing_signals.map((s, i) => (
                  <div key={i} className="flex justify-between text-xs text-gray-600 dark:text-gray-400">
                    <span>{s.signal_type}</span>
                    <span>{(s.value * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Feedback */}
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 shadow-sm space-y-3">
            <h2 className="font-semibold text-gray-900 dark:text-gray-100">Operator Feedback</h2>
            {feedbackSubmitted ? (
              <p className="text-sm text-green-600 dark:text-green-400">✓ Feedback submitted. Thank you.</p>
            ) : (
              <>
                <p className="text-sm text-gray-500">Was this prediction correct?</p>
                <div className="flex gap-2">
                  <button
                    onClick={() => setFeedbackCorrect(true)}
                    className={`flex-1 py-2 text-sm rounded-lg border transition-colors ${feedbackCorrect === true ? 'bg-green-600 text-white border-green-600' : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
                  >
                    ✓ Correct
                  </button>
                  <button
                    onClick={() => setFeedbackCorrect(false)}
                    className={`flex-1 py-2 text-sm rounded-lg border transition-colors ${feedbackCorrect === false ? 'bg-red-600 text-white border-red-600' : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
                  >
                    ✗ Incorrect
                  </button>
                </div>
                {feedbackCorrect === false && (
                  <div className="space-y-2">
                    <select
                      value={correctedLabel}
                      onChange={(e) => setCorrectedLabel(e.target.value)}
                      className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-700"
                    >
                      <option value="">Select correct label…</option>
                      {ACTION_LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
                    </select>
                    <textarea
                      value={notes}
                      onChange={(e) => setNotes(e.target.value)}
                      placeholder="Notes (optional)"
                      rows={2}
                      className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded-lg px-3 py-1.5 bg-white dark:bg-gray-700 resize-none"
                    />
                  </div>
                )}
                {feedbackCorrect !== null && (
                  <button
                    onClick={handleFeedback}
                    disabled={submitting}
                    className="w-full py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-60"
                  >
                    {submitting ? 'Submitting…' : 'Submit Feedback'}
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
