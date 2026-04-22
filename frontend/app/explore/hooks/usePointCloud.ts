"use client";

import { useState, useEffect, useRef } from "react";
import { fetchPoints, type Method, type TrackPoint } from "../../lib/api";

const ALL_METHODS: Method[] = ["umap", "tsne", "pca"];

export function usePointCloud(method: Method) {
  const [points, setPoints] = useState<TrackPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const cache = useRef<Partial<Record<Method, TrackPoint[]>>>({});
  const methodRef = useRef(method);
  methodRef.current = method;

  // Pre-fetch all methods in parallel on mount
  useEffect(() => {
    ALL_METHODS.forEach((m) => {
      fetchPoints(m)
        .then((tracks) => {
          cache.current[m] = tracks;
          if (methodRef.current === m) {
            setPoints(tracks);
            setIsLoading(false);
          }
        })
        .catch((err) => console.error(`fetchPoints(${m}) failed`, err));
    });
  }, []);

  // When method changes, serve from cache (will be populated by the time user switches)
  useEffect(() => {
    const cached = cache.current[method];
    if (cached) {
      setPoints(cached);
      setIsLoading(false);
    } else {
      setIsLoading(true);
    }
  }, [method]);

  return { points, isLoading };
}
