"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { fetchSimilar, type TrackPoint, type Method } from "../../lib/api";

interface SpawnAPI {
  spawnPoints: (pts: TrackPoint[]) => void;
  clearSpawned: () => void;
  getAllPointsForMethod: (m: Method) => TrackPoint[];
  displayedIdsRef: React.RefObject<Set<number>>;
}

export function useSongSelection(method: Method, spawn: SpawnAPI) {
  const [selectedTrack, setSelectedTrack] = useState<TrackPoint | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [neighbors, setNeighbors] = useState<TrackPoint[]>([]);
  const [knnIds, setKnnIds] = useState<Set<number>>(new Set());
  const [neighborsLoading, setNeighborsLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const selectedTrackRef = useRef<TrackPoint | null>(null);
  selectedTrackRef.current = selectedTrack;

  const fetchNeighbors = useCallback(
    async (track: TrackPoint, signal: AbortSignal) => {
      setNeighborsLoading(true);
      setNeighbors([]);
      setKnnIds(new Set());
      try {
        const nbrs = await fetchSimilar(track.track_id, method, 10);
        if (signal.aborted) return;

        setNeighbors(nbrs);
        setKnnIds(new Set(nbrs.map((n) => n.track_id)));

        const allById = new Map(
          spawn.getAllPointsForMethod(method).map((p) => [p.track_id, p]),
        );
        const toSpawn = nbrs
          .filter((n) => !spawn.displayedIdsRef.current?.has(n.track_id))
          .map((n) => allById.get(n.track_id))
          .filter((p): p is TrackPoint => p !== undefined);
        if (toSpawn.length > 0) spawn.spawnPoints(toSpawn);
      } catch {
        if (!signal.aborted) console.error("fetchSimilar failed");
      } finally {
        if (!signal.aborted) setNeighborsLoading(false);
      }
    },
    [method, spawn],
  );

  const selectTrack = useCallback(
    (track: TrackPoint) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      spawn.clearSpawned();
      setSelectedTrack(track);
      setPanelOpen(true);
      fetchNeighbors(track, ctrl.signal);
    },
    [fetchNeighbors, spawn],
  );

  useEffect(() => {
    const track = selectedTrackRef.current;
    if (!track) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    fetchNeighbors(track, ctrl.signal);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method]);

  const closePanel = useCallback(() => {
    abortRef.current?.abort();
    spawn.clearSpawned();
    setSelectedTrack(null);
    setPanelOpen(false);
    setNeighbors([]);
    setKnnIds(new Set());
    setNeighborsLoading(false);
  }, [spawn]);

  return {
    selectedTrack,
    panelOpen,
    neighbors,
    knnIds,
    neighborsLoading,
    selectTrack,
    closePanel,
  };
}
