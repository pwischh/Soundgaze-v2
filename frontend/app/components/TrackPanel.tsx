"use client";

import type { TrackPoint } from "../lib/api";

interface TrackPanelProps {
  track: TrackPoint | null;
  neighbors: TrackPoint[];
  isLoading: boolean;
  isOpen: boolean;
  onClose: () => void;
  onSelectNeighbor?: (track: TrackPoint) => void;
}

export const GENRE_COLOR: Record<string, string> = {
  "Hip-Hop":       "#FF6B35",
  "Pop":           "#A855F7",
  "Folk":          "#22C55E",
  "Experimental":  "#F59E0B",
  "Rock":          "#EF4444",
  "International": "#06B6D4",
  "Electronic":    "#EC4899", // Pink
  "Instrumental":  "#3B82F6", // Distinct Blue
};

export default function TrackPanel({
  track,
  neighbors,
  isLoading,
  isOpen,
  onClose,
  onSelectNeighbor,
}: TrackPanelProps) {
  if (!isOpen || !track) return null;

  const genreColor = GENRE_COLOR[track.genre] ?? "#6b7280";

  return (
    <div
      className="flex flex-col w-96 h-fit max-h-[calc(100vh-3.5rem)] overflow-hidden border-y-2 border-l-2 border-[#333]"
      style={{
        backgroundColor: "#121212",
        boxShadow: "-6px 0px 0px 0px rgba(0,0,0,0.5)",
      }}
    >
      {/* Header */}
      <div className="flex items-start justify-between p-4 border-b-2 border-[#333]">
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
      <div className="flex-1 overflow-y-auto p-4">
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
          <div className="flex flex-col gap-2">
            {neighbors.map((n, idx) => (
              <NeighborRow
                key={n.track_id}
                rank={idx + 1}
                track={n}
                onSelect={onSelectNeighbor}
              />
            ))}
          </div>
        )}
      </div>

    </div>
  );
}

function NeighborRow({
  rank,
  track,
  onSelect,
}: {
  rank: number;
  track: TrackPoint;
  onSelect?: (track: TrackPoint) => void;
}) {
  const genreColor = GENRE_COLOR[track.genre] ?? "#6b7280";
  const score = (track as any).score;
  const showScore = score !== undefined;

  return (
    <button
      onClick={() => onSelect?.(track)}
      className="group w-full flex flex-col text-left gap-1 px-3 py-2 border-2 border-[#333] hover:border-[#1DB954] hover:bg-[#1a1a1a] transition-all cursor-pointer"
      style={{ backgroundColor: "#1c1c1c" }}
    >
      <div className="w-full flex items-center gap-3">
        <span className="font-black text-[10px] text-white/30 w-4 shrink-0 text-right">
          {rank}
        </span>
        <div
          className="w-1.5 h-6 shrink-0"
          style={{ backgroundColor: genreColor }}
        />
        <div className="flex-1 min-w-0 flex flex-col justify-center">
          <p className="font-black text-[11px] uppercase tracking-wide text-white truncate leading-tight mb-0.5">
            {track.title}
          </p>
          <div className="flex items-center justify-between font-mono text-[9px] truncate">
            <span className="text-white/50">{track.artist}</span>
            {showScore && (
              <span className="font-black text-[10px] text-black bg-[#1DB954] px-1.5 py-0.5 shrink-0 ml-2 shadow-[2px_2px_0px_0px_rgba(255,255,255,0.2)]">
                SIM: {score.toFixed(3)}
              </span>
            )}
          </div>
        </div>
      </div>
      
      {/* Extra Info on Hover */}
      <div className="hidden group-hover:flex flex-col w-full pt-2 mt-1 border-t-2 border-[#333]/50 font-mono text-[9px] text-[#1DB954] tracking-widest uppercase">
        <div className="flex justify-between w-full">
          <span>ID: {track.track_id}</span>
          <span>GENRE: {track.genre}</span>
        </div>
      </div>
    </button>
  );
}
