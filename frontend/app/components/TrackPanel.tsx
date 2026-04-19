"use client";

import type { TrackPoint } from "../lib/api";

interface TrackPanelProps {
  track: TrackPoint | null;
  neighbors: TrackPoint[];
  isLoading: boolean;
  isOpen: boolean;
  onClose: () => void;
  onRate: () => void;
}

const GENRE_COLOR: Record<string, string> = {
  "Hip-Hop":       "#FF6B35",
  "Pop":           "#A855F7",
  "Folk":          "#22C55E",
  "Experimental":  "#F59E0B",
  "Rock":          "#EF4444",
  "International": "#06B6D4",
  "Electronic":    "#1DB954",
  "Instrumental":  "#8B5CF6",
};

export default function TrackPanel({
  track,
  neighbors,
  isLoading,
  isOpen,
  onClose,
  onRate,
}: TrackPanelProps) {
  if (!isOpen || !track) return null;

  const genreColor = GENRE_COLOR[track.genre] ?? "#6b7280";

  return (
    <div
      className="flex flex-col w-64 h-fit max-h-[calc(100vh-3.5rem)] overflow-hidden"
      style={{
        backgroundColor: "#121212",
        borderLeft: "2px solid #333",
        boxShadow: "-6px 0px 0px 0px rgba(0,0,0,0.5)",
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between p-3 border-b-2 border-[#333]">
        <div className="flex-1 min-w-0 pr-2">
          <p className="font-black text-xs uppercase tracking-widest text-white/40 mb-0.5">
            Track #{track.track_id}
          </p>
          <h2 className="font-black text-base uppercase tracking-wide text-white leading-tight truncate">
            {track.title}
          </h2>
          <p className="font-black text-xs text-white/60 uppercase tracking-widest truncate mt-0.5">
            {track.artist}
          </p>
          <span
            className="inline-block mt-1.5 px-2 py-0.5 font-black text-[9px] uppercase tracking-widest text-black"
            style={{ backgroundColor: genreColor }}
          >
            {track.genre}
          </span>
        </div>
        <button
          onClick={onClose}
          className="font-black text-lg text-white/40 hover:text-white leading-none transition-colors shrink-0"
          aria-label="Close panel"
        >
          ✕
        </button>
      </div>

      {/* Neighbors list */}
      <div className="flex-1 overflow-y-auto p-3">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-black text-[9px] uppercase tracking-widest text-white/50">
            Nearest Neighbors
          </h3>
          <span className="font-mono text-[8px] text-white/30 uppercase">
            {neighbors.length} results
          </span>
        </div>

        {isLoading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 border-2 border-[#333] animate-pulse" style={{ backgroundColor: "#1c1c1c" }} />
            ))}
          </div>
        ) : neighbors.length === 0 ? (
          <p className="font-mono text-xs text-white/30 uppercase tracking-widest">
            No neighbors found
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {neighbors.map((n, idx) => (
              <NeighborRow key={n.track_id} rank={idx + 1} track={n} />
            ))}
          </div>
        )}
      </div>

      {/* Rate button */}
      <div className="p-3 border-t-2 border-[#333]">
        <button
          onClick={onRate}
          disabled={neighbors.length === 0}
          className="w-full neo-btn-primary text-sm disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Rate
        </button>
      </div>
    </div>
  );
}

function NeighborRow({ rank, track }: { rank: number; track: TrackPoint }) {
  const genreColor = GENRE_COLOR[track.genre] ?? "#6b7280";
  return (
    <div
      className="flex items-center gap-2 px-2 py-1.5 border-2 border-[#333] hover:border-white/50 transition-colors"
      style={{ backgroundColor: "#1c1c1c" }}
    >
      <span className="font-black text-[10px] text-white/30 w-4 shrink-0 text-right">
        {rank}
      </span>
      <div
        className="w-1.5 h-6 shrink-0"
        style={{ backgroundColor: genreColor }}
      />
      <div className="flex-1 min-w-0">
        <p className="font-black text-[11px] uppercase tracking-wide text-white truncate">
          {track.title}
        </p>
        <p className="font-mono text-[9px] text-white/50 truncate">
          {track.artist}
        </p>
      </div>
    </div>
  );
}
