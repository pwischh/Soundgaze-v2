"use client";

import { useState, useCallback } from "react";
import { submitEval, type TrackPoint, type Method, type Metric, type EvalRating } from "../lib/api";

interface EvalModalProps {
  isOpen: boolean;
  onClose: () => void;
  seedTrack: TrackPoint;
  neighbors: TrackPoint[];
  method: Method;
  metric: Metric;
}

type Rating = 0 | 1;

export default function EvalModal({
  isOpen,
  onClose,
  seedTrack,
  neighbors,
  method,
  metric,
}: EvalModalProps) {
  const [evaluatorName, setEvaluatorName] = useState("");
  const [ratings, setRatings] = useState<Record<number, Rating>>({});
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const allRated = neighbors.every((n) => ratings[n.track_id] !== undefined);
  const canSubmit = evaluatorName.trim().length > 0 && allRated && !submitting;

  const setRating = useCallback((trackId: number, score: Rating) => {
    setRatings((prev) => ({ ...prev, [trackId]: score }));
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    const ratingList: EvalRating[] = neighbors.map((n) => ({
      track_id: n.track_id,
      score: ratings[n.track_id] as Rating,
    }));
    try {
      await submitEval({
        evaluator_name: evaluatorName.trim(),
        seed_track_id:  seedTrack.track_id,
        method,
        metric,
        k: neighbors.length,
        timestamp: new Date().toISOString(),
        ratings: ratingList,
      });
      setSubmitted(true);
      setTimeout(() => {
        onClose();
        setSubmitted(false);
        setRatings({});
        setEvaluatorName("");
        setSubmitting(false);
      }, 1800);
    } catch (err) {
      console.error("submitEval failed", err);
      setSubmitting(false);
    }
  }, [canSubmit, evaluatorName, ratings, neighbors, seedTrack, method, metric, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ backgroundColor: "rgba(0,0,0,0.85)" }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-lg max-h-[90vh] flex flex-col border-2 border-[#333] overflow-hidden"
        style={{ backgroundColor: "#121212", boxShadow: "8px 8px 0px 0px rgba(0,0,0,0.5)" }}
      >
        {submitted ? (
          // Success screen
          <div className="flex flex-col items-center justify-center p-12 gap-4">
            <div className="font-black text-5xl" style={{ color: "#1DB954" }}>✓</div>
            <p className="font-black text-xl uppercase tracking-widest text-white text-center">
              Evaluation Submitted
            </p>
            <p className="font-mono text-xs text-white/50 uppercase tracking-widest text-center">
              Thank you, {evaluatorName}
            </p>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="flex items-start justify-between p-5 border-b-2 border-[#333]">
              <div>
                <h2 className="font-black text-base uppercase tracking-widest text-white">
                  Rate These Recommendations
                </h2>
                <p className="font-mono text-[10px] uppercase tracking-widest text-white/40 mt-0.5">
                  Method: {method.toUpperCase()} · Metric: {metric} · {neighbors.length} tracks
                </p>
              </div>
              <button
                onClick={onClose}
                className="font-black text-lg text-white/40 hover:text-white transition-colors"
              >
                ✕
              </button>
            </div>

            {/* Seed track */}
            <div className="px-5 py-3 border-b-2 border-[#333]" style={{ backgroundColor: "#1c1c1c" }}>
              <p className="font-black text-[9px] uppercase tracking-widest text-white/40 mb-1">
                Seed Track
              </p>
              <p className="font-black text-sm uppercase tracking-wide text-white">
                {seedTrack.title}
              </p>
              <p className="font-mono text-[10px] text-white/50">
                {seedTrack.artist} · {seedTrack.genre}
              </p>
            </div>

            {/* Ratings list */}
            <div className="flex-1 overflow-y-auto">
              <div className="px-5 py-3 border-b-2 border-[#333]">
                <p className="font-black text-[9px] uppercase tracking-widest text-white/40">
                  For each track: does it feel musically similar to the seed?
                </p>
              </div>
              {neighbors.map((n, idx) => (
                <RatingRow
                  key={n.track_id}
                  rank={idx + 1}
                  track={n}
                  rating={ratings[n.track_id]}
                  onRate={(score) => setRating(n.track_id, score)}
                />
              ))}
            </div>

            {/* Footer */}
            <div className="p-5 border-t-2 border-[#333] flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <label className="font-black text-[10px] uppercase tracking-widest text-white shrink-0">
                  Your Name
                </label>
                <input
                  type="text"
                  value={evaluatorName}
                  onChange={(e) => setEvaluatorName(e.target.value)}
                  placeholder="Required to submit"
                  className="flex-1 px-3 py-1.5 border-2 border-[#333] font-mono text-xs text-white bg-[#1a1a1a] focus:outline-none focus:border-[#1DB954] transition-colors"
                />
              </div>

              <div className="flex items-center justify-between">
                <span className="font-mono text-[9px] uppercase tracking-widest text-white/40">
                  {Object.keys(ratings).length}/{neighbors.length} rated
                </span>
                <button
                  onClick={handleSubmit}
                  disabled={!canSubmit}
                  className="neo-btn-primary text-xs px-5 py-2 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none disabled:translate-x-0 disabled:translate-y-0"
                >
                  {submitting ? "Submitting..." : "Submit"}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function RatingRow({
  rank,
  track,
  rating,
  onRate,
}: {
  rank: number;
  track: TrackPoint;
  rating: Rating | undefined;
  onRate: (score: Rating) => void;
}) {
  return (
    <div className="flex items-center gap-3 px-5 py-3 border-b-2 border-[#333] hover:bg-[#1a1a1a] transition-colors">
      <span className="font-black text-[10px] text-white/30 w-4 shrink-0 text-right">
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <p className="font-black text-[11px] uppercase tracking-wide text-white truncate">
          {track.title}
        </p>
        <p className="font-mono text-[9px] text-white/50 truncate">
          {track.artist} · {track.genre}
        </p>
      </div>
      <div className="flex gap-1.5 shrink-0">
        <RatingBtn
          label="Similar"
          active={rating === 1}
          activeColor="#1DB954"
          onClick={() => onRate(1)}
        />
        <RatingBtn
          label="Not Similar"
          active={rating === 0}
          activeColor="#EF4444"
          onClick={() => onRate(0)}
        />
      </div>
    </div>
  );
}

function RatingBtn({
  label,
  active,
  activeColor,
  onClick,
}: {
  label: string;
  active: boolean;
  activeColor: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="px-2 py-1 font-black text-[9px] uppercase tracking-widest border-2 transition-all"
      style={{
        backgroundColor: active ? activeColor : "transparent",
        color: active ? "#000" : "#fff",
        borderColor: active ? activeColor : "#333",
        boxShadow: active ? "2px 2px 0px 0px rgba(255,255,255,0.2)" : "none",
      }}
    >
      {label}
    </button>
  );
}
