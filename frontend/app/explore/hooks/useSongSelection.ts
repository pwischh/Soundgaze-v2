"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { fetchSimilar, type TrackPoint, type Method, type Metric } from "../../lib/api";

export function useSongSelection(method: Method, metric: Metric) {
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
        const nbrs = await fetchSimilar(track.track_id, method, metric, 10);
        if (signal.aborted) return;
        setNeighbors(nbrs);
        setKnnIds(new Set(nbrs.map((n) => n.track_id)));
      } catch {
        if (!signal.aborted) console.error("fetchSimilar failed");
      } finally {
        if (!signal.aborted) setNeighborsLoading(false);
      }
    },
    [method, metric],
  );

  const selectTrack = useCallback(
    (track: TrackPoint) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      setSelectedTrack(track);
      setPanelOpen(true);
      fetchNeighbors(track, ctrl.signal);
    },
    [fetchNeighbors],
  );

  // Re-fetch neighbors when method or metric changes while a track is selected
  useEffect(() => {
    const track = selectedTrackRef.current;
    if (!track) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    fetchNeighbors(track, ctrl.signal);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [method, metric]);

  const closePanel = useCallback(() => {
    abortRef.current?.abort();
    setSelectedTrack(null);
    setPanelOpen(false);
    setNeighbors([]);
    setKnnIds(new Set());
    setNeighborsLoading(false);
  }, []);

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
