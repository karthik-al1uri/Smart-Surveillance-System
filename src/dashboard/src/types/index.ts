export interface Zone {
  zone_id: string;
  name: string;
  zone_type: 'restricted' | 'monitored' | 'safe' | 'perimeter';
  polygon: number[][];
  schedule_start?: string;
  schedule_end?: string;
  rules?: ZoneRule[];
}

export interface ZoneRule {
  rule_type: string;
  value?: number;
}

export interface Camera {
  id: string;
  name: string;
  stream_url: string;
  location?: string;
  enabled: boolean;
  indoor: boolean;
  zones_config: Zone[];
  created_at?: string;
}

export interface ScoringSignal {
  signal_type: string;
  value: number;
  weight: number;
}

export interface Event {
  id: string;
  camera_id: string;
  timestamp: string;
  event_category: string;
  event_label: string;
  severity_score: number;
  contributing_signals?: ScoringSignal[];
  dominant_signal?: string;
  track_id?: number;
  zone_name?: string;
  clip_path?: string;
  alert_decision?: string;
  acknowledged: boolean;
  feedback_correct?: boolean;
  feedback_label?: string;
  created_at?: string;
}

export interface EventListResponse {
  events: Event[];
  total: number;
  limit: number;
  offset: number;
}

export interface Alert {
  id: string;
  event_id: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  title?: string;
  description?: string;
  status: 'pending' | 'delivered' | 'acknowledged' | 'dismissed' | 'escalated' | 'failed';
  created_at?: string;
  delivered_at?: string;
  acknowledged_at?: string;
  acknowledged_by?: string;
  dismissed_at?: string;
  dismiss_reason?: string;
}

export interface AlertListResponse {
  alerts: Alert[];
  total: number;
  limit: number;
  offset: number;
}

export interface User {
  id: string;
  username: string;
  full_name?: string;
  role: 'admin' | 'operator' | 'viewer';
  enabled: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface EventStats {
  total: number;
  by_category: Record<string, number>;
  by_camera: Record<string, number>;
}

export interface AlertStats {
  total: number;
  by_status: Record<string, number>;
  by_priority: Record<string, number>;
}

export interface SystemHealth {
  status: string;
  components: Record<string, string>;
}

export interface StorageStats {
  total_clips: number;
  total_size_bytes: number;
  total_size_gb: number;
}
