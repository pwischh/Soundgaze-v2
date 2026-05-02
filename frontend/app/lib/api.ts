// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Method = "pca" | "tsne" | "umap";
export type Metric = "cosine" | "euclidean" | "manhattan";
export type EvalRating = { track_id: number; score: 0 | 1 };

export interface EvalPayload {
  evaluator_name: string;
  seed_track_id: number;
  method: Method;
  metric: Metric;
  k: number;
  timestamp: string;
  ratings: EvalRating[];
}

export interface TrackPoint {
  track_id: number;
  title: string;
  artist: string;
  genre: string;
  xyz: [number, number, number];
}

export interface Neighbor extends TrackPoint {
  score: number;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/**
 * Fetch all 3D track points for a given DR method.
 * Backend: GET /points?method=umap
 */
export async function fetchPoints(method: Method): Promise<TrackPoint[]> {
  const res = await fetch(`${API_BASE}/points?method=${method}`);
  if (!res.ok) throw new Error(`fetchPoints failed: ${res.status}`);
  const data = await res.json();
  return data.tracks as TrackPoint[];
}

/**
 * Fetch k nearest neighbors for a track in hybrid embedding space.
 * Backend: GET /similar?track_id=...&method=...&k=...
 */
export async function fetchSimilar(
  trackId: number,
  method: Method,
  k = 10,
): Promise<Neighbor[]> {
  const params = new URLSearchParams({
    track_id: String(trackId),
    method,
    k: String(k),
  });
  const res = await fetch(`${API_BASE}/similar?${params}`);
  if (!res.ok) throw new Error(`fetchSimilar failed: ${res.status}`);
  const data = await res.json();
  return data.neighbors as Neighbor[];
}

export interface MetricSet {
  silhouette: number;
  davies_bouldin: number;
  genre_purity_at_10: number;
  trustworthiness: number;
}

export type MetricsResponse = Partial<Record<Method, MetricSet>>;

export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) return {};
  return res.json();
}

export async function submitEval(payload: EvalPayload): Promise<void> {
  await fetch(`${API_BASE}/eval/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
