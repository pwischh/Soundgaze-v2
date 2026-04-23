"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { fetchPoints, type Method, type TrackPoint } from "../../lib/api";

const ALL_METHODS: Method[] = ["umap", "tsne", "pca"];
const DISPLAY_LIMIT = 2000;

function randomSample(arr: TrackPoint[], n: number): TrackPoint[] {
  if (arr.length <= n) return arr;
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy.slice(0, n);
}

export function usePointCloud(method: Method) {
  const [points, setPoints] = useState<TrackPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const allPointsCache = useRef<Partial<Record<Method, TrackPoint[]>>>({});
  const sampleCache = useRef<Partial<Record<Method, TrackPoint[]>>>({});
  const basePoints = useRef<TrackPoint[]>([]);
  const displayedIdsRef = useRef<Set<number>>(new Set());
  const methodRef = useRef(method);
  methodRef.current = method;

  useEffect(() => {
    ALL_METHODS.forEach((m) => {
      fetchPoints(m)
        .then((tracks) => {
          allPointsCache.current[m] = tracks;
          const sample = randomSample(tracks, DISPLAY_LIMIT);
          sampleCache.current[m] = sample;
          if (methodRef.current === m) {
            basePoints.current = sample;
            displayedIdsRef.current = new Set(sample.map((p) => p.track_id));
            setPoints(sample);
            setIsLoading(false);
          }
        })
        .catch((err) => console.error(`fetchPoints(${m}) failed`, err));
    });
  }, []);

  useEffect(() => {
    const sample = sampleCache.current[method];
    if (sample) {
      basePoints.current = sample;
      displayedIdsRef.current = new Set(sample.map((p) => p.track_id));
      setPoints(sample);
      setIsLoading(false);
    } else {
      setIsLoading(true);
    }
  }, [method]);

  const spawnPoints = useCallback((toSpawn: TrackPoint[]) => {
    setPoints((prev) => {
      const existingIds = new Set(prev.map((p) => p.track_id));
      const newOnes = toSpawn.filter((p) => !existingIds.has(p.track_id));
      return newOnes.length > 0 ? [...prev, ...newOnes] : prev;
    });
  }, []);

  const clearSpawned = useCallback(() => {
    setPoints(basePoints.current);
  }, []);

  const getAllPointsForMethod = useCallback((m: Method): TrackPoint[] => {
    return allPointsCache.current[m] ?? [];
  }, []);

  return {
    points,
    isLoading,
    spawnPoints,
    clearSpawned,
    getAllPointsForMethod,
    displayedIdsRef,
  };
}
