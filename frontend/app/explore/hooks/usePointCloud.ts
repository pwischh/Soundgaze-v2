"use client";

import { useState, useEffect, useRef } from "react";
import { fetchPoints, type Method, type TrackPoint } from "../../lib/api";

export function usePointCloud(method: Method) {
  const [points, setPoints] = useState<TrackPoint[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  // Cache each method's points after first fetch to avoid re-fetching on toggle
  const cache = useRef<Partial<Record<Method, TrackPoint[]>>>({});

  useEffect(() => {
    const cached = cache.current[method];
    if (cached) {
      setPoints(cached);
      return;
    }

    setIsLoading(true);
    fetchPoints(method)
      .then((tracks) => {
        cache.current[method] = tracks;
        setPoints(tracks);
      })
      .catch((err) => console.error("fetchPoints failed", err))
      .finally(() => setIsLoading(false));
  }, [method]);

  return { points, isLoading };
}
