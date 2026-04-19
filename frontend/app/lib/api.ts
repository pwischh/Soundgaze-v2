// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Method = "pca" | "tsne" | "umap";
export type Metric = "cosine" | "euclidean" | "manhattan";

export interface TrackPoint {
  track_id: number;
  title: string;
  artist: string;
  genre: string;
  xyz: [number, number, number]; // method-specific coords from backend
}

export interface MetricSet {
  silhouette: number;
  davies_bouldin: number;
  genre_purity_at_10: number;
  trustworthiness: number;
}

export interface MetricsResponse {
  pca: MetricSet;
  tsne: MetricSet;
  umap: MetricSet;
}

export interface EvalRating {
  track_id: number;
  score: 0 | 1; // 1 = similar, 0 = not similar
}

export interface EvalPayload {
  evaluator_name: string;
  seed_track_id: number;
  method: Method;
  metric: Metric;
  k: number;
  timestamp: string;
  ratings: EvalRating[];
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Toggle to true to develop without the backend running
const USE_PLACEHOLDERS = true;

// ---------------------------------------------------------------------------
// Placeholder generators
// ---------------------------------------------------------------------------

const GENRES = ["Hip-Hop", "Pop", "Folk", "Experimental", "Rock", "International", "Electronic", "Instrumental"];
const ARTISTS = ["Artist A", "Artist B", "Artist C", "Artist D", "Artist E", "Artist F"];

function randXyz(seed: number): [number, number, number] {
  const s = Math.sin(seed * 127.1) * 43758.5453;
  const t = Math.sin(seed * 311.7) * 43758.5453;
  const u = Math.sin(seed * 743.3) * 43758.5453;
  return [s - Math.floor(s), t - Math.floor(t), u - Math.floor(u)];
}

function makePlaceholderTrack(id: number): TrackPoint {
  return {
    track_id: id,
    title: `Track ${id}`,
    artist: ARTISTS[id % ARTISTS.length],
    genre: GENRES[id % GENRES.length],
    xyz: randXyz(id),
  };
}

function makePlaceholderPoints(n: number): TrackPoint[] {
  return Array.from({ length: n }, (_, i) => makePlaceholderTrack(i + 2));
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/**
 * Fetch all 3D track points for a given DR method.
 * Backend: GET /points?method=umap
 */
export async function fetchPoints(method: Method): Promise<TrackPoint[]> {
  if (USE_PLACEHOLDERS) return makePlaceholderPoints(500);

  const res = await fetch(`${API_BASE}/points?method=${method}`);
  if (!res.ok) throw new Error(`fetchPoints failed: ${res.status}`);
  const data = await res.json();
  return data.tracks as TrackPoint[];
}

/**
 * Fetch k nearest neighbors for a track under a given method + metric.
 * Backend: GET /similar?track_id=...&method=...&metric=...&k=...
 */
export async function fetchSimilar(
  trackId: number,
  method: Method,
  metric: Metric = "cosine",
  k = 10,
): Promise<TrackPoint[]> {
  if (USE_PLACEHOLDERS) {
    return Array.from({ length: k }, (_, i) => makePlaceholderTrack(trackId + i + 1));
  }

  const params = new URLSearchParams({
    track_id: String(trackId),
    method,
    metric,
    k: String(k),
  });
  const res = await fetch(`${API_BASE}/similar?${params}`);
  if (!res.ok) throw new Error(`fetchSimilar failed: ${res.status}`);
  const data = await res.json();
  return data.neighbors as TrackPoint[];
}

/**
 * Submit a human evaluation rating set.
 * Backend: POST /eval/submit
 */
export async function submitEval(payload: EvalPayload): Promise<void> {
  if (USE_PLACEHOLDERS) {
    await new Promise((r) => setTimeout(r, 600));
    return;
  }

  const res = await fetch(`${API_BASE}/eval/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`submitEval failed: ${res.status}`);
}

/**
 * Fetch precomputed quantitative metrics for all three DR methods.
 * Backend: GET /metrics
 */
export async function fetchMetrics(): Promise<MetricsResponse | null> {
  if (USE_PLACEHOLDERS) {
    return {
      pca:  { silhouette: 0.12, davies_bouldin: 1.84, genre_purity_at_10: 0.31, trustworthiness: 0.887 },
      tsne: { silhouette: 0.41, davies_bouldin: 0.91, genre_purity_at_10: 0.62, trustworthiness: 0.990 },
      umap: { silhouette: 0.28, davies_bouldin: 1.22, genre_purity_at_10: 0.49, trustworthiness: 0.944 },
    };
  }

  try {
    const res = await fetch(`${API_BASE}/metrics`);
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
