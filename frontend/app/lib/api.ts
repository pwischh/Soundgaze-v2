// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Method = "pca" | "tsne" | "umap";

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
