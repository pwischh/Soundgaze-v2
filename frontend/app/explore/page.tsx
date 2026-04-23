"use client";

import { useState, useCallback } from "react";
import Navbar from "../components/Navbar";
import PointCloudViewer from "../components/PointCloudViewer";
import MethodSwitcher from "../components/MethodSwitcher";
import TrackPanel, { GENRE_COLOR } from "../components/TrackPanel";
import { usePointCloud } from "./hooks/usePointCloud";
import { useSongSelection } from "./hooks/useSongSelection";
import type { Method, TrackPoint } from "../lib/api";

const LEGEND = [
  { color: "#FF2D2D", label: "Selected" },
  { color: "#FF6B35", label: "k-NN Neighbors" },
  { color: "#4a4a5a", label: "All Tracks" },
];

export default function ExplorePage() {
  const [method, setMethod] = useState<Method>("umap");

  const {
    points,
    isLoading,
    spawnPoints,
    clearSpawned,
    getAllPointsForMethod,
    displayedIdsRef,
  } = usePointCloud(method);

  const {
    selectedTrack,
    panelOpen,
    neighbors,
    knnIds,
    neighborsLoading,
    selectTrack,
    closePanel,
  } = useSongSelection(method, {
    spawnPoints,
    clearSpawned,
    getAllPointsForMethod,
    displayedIdsRef,
  });

  const handleMethodChange = useCallback((m: Method) => setMethod(m), []);

  const handlePointClick = useCallback(
    (point: TrackPoint) => selectTrack(point),
    [selectTrack],
  );

  return (
    <main className="relative w-screen h-screen bg-near-black overflow-hidden flex flex-col">
      <Navbar />

      <div className="relative flex-1 overflow-hidden flex">

        {/* Three.js canvas */}
        <div className="absolute inset-0 z-0">
          <PointCloudViewer
            points={points}
            knnIds={knnIds}
            selectedId={selectedTrack?.track_id ?? null}
            onPointClick={handlePointClick}
          />
        </div>

        {/* Corner green vignettes */}
        <div
          className="absolute inset-0 z-0 pointer-events-none"
          style={{
            background: `
              radial-gradient(ellipse 18% 22% at 0% 0%,    rgba(29,185,84,0.15) 0%, transparent 100%),
              radial-gradient(ellipse 18% 22% at 100% 0%,  rgba(29,185,84,0.15) 0%, transparent 100%),
              radial-gradient(ellipse 18% 22% at 0% 100%,  rgba(29,185,84,0.15) 0%, transparent 100%),
              radial-gradient(ellipse 18% 22% at 100% 100%,rgba(29,185,84,0.15) 0%, transparent 100%)
            `,
          }}
        />

        {/* Loading spinner */}
        {isLoading && (
          <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
            <div className="flex flex-col items-center gap-3">
              <div
                className="w-8 h-8 rounded-full border-2 border-transparent animate-spin"
                style={{ borderTopColor: "#1DB954", borderRightColor: "rgba(29,185,84,0.3)" }}
              />
              <span className="font-black text-[9px] uppercase tracking-widest text-white/40">
                Loading {method.toUpperCase()}
              </span>
            </div>
          </div>
        )}

        {/* Legends — left edge */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 z-10 flex flex-col gap-4">
          <div
            className="flex flex-col gap-2 px-4 py-4 border-2 border-white/20"
            style={{
              backgroundColor: "rgba(8,9,12,0.85)",
              boxShadow: "4px 4px 0px 0px rgba(255,255,255,0.15)",
            }}
          >
            {LEGEND.map(({ color, label }) => (
              <div key={label} className="flex items-center gap-3">
                <span
                  className="shrink-0 rounded-full"
                  style={{ width: 12, height: 12, backgroundColor: color }}
                />
                <span className="font-black text-xs uppercase tracking-widest text-white/70">
                  {label}
                </span>
              </div>
            ))}
          </div>

          <div
            className="flex flex-col gap-2 px-4 py-4 border-2 border-white/20"
            style={{
              backgroundColor: "rgba(8,9,12,0.85)",
              boxShadow: "4px 4px 0px 0px rgba(255,255,255,0.15)",
            }}
          >
            <h4 className="font-black text-[10px] uppercase tracking-widest text-white/40 mb-1 border-b border-white/10 pb-2">
              Genre Colors
            </h4>
            {Object.entries(GENRE_COLOR).map(([genre, color]) => (
              <div key={genre} className="flex items-center gap-3">
                <span
                  className="shrink-0"
                  style={{ width: 12, height: 12, backgroundColor: color }}
                />
                <span className="font-black text-[10px] uppercase tracking-widest text-white/70">
                  {genre}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Method switcher — bottom center */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
          <MethodSwitcher
            method={method}
            onChange={handleMethodChange}
            disabled={isLoading}
          />
        </div>

        {/* Track panel — right edge (slides in) */}
        <div
          className="absolute right-0 top-14 z-10 transition-transform duration-300 ease-in-out"
          style={{
            transform: panelOpen ? "translateX(0)" : "translateX(100%)",
          }}
        >
          <TrackPanel
            track={selectedTrack}
            neighbors={neighbors}
            isLoading={neighborsLoading}
            isOpen={panelOpen}
            onClose={closePanel}
            onSelectNeighbor={handlePointClick}
          />
        </div>

      </div>
    </main>
  );
}
