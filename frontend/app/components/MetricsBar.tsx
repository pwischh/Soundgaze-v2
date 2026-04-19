"use client";

import { useEffect, useState } from "react";
import { fetchMetrics, type Method, type MetricSet, type MetricsResponse } from "../lib/api";

interface MetricsBarProps {
  method: Method;
}

const METHOD_LABEL: Record<Method, string> = {
  pca:  "PCA",
  tsne: "t-SNE",
  umap: "UMAP",
};

export default function MetricsBar({ method }: MetricsBarProps) {
  const [allMetrics, setAllMetrics] = useState<MetricsResponse | null>(null);

  useEffect(() => {
    fetchMetrics().then(setAllMetrics).catch(() => {});
  }, []);

  const m: MetricSet | undefined = allMetrics?.[method];

  return (
    <div
      className="flex items-center gap-0 border-2 border-[#333] font-mono text-xs uppercase tracking-widest"
      style={{
        backgroundColor: "#121212",
        boxShadow: "4px 4px 0px 0px rgba(0,0,0,0.5)"
      }}
    >
      <span
        className="px-5 py-2.5 font-black text-xs uppercase tracking-widest border-r-2 border-[#333]"
        style={{ color: "#1DB954" }}
      >
        {METHOD_LABEL[method]}
      </span>

      {m ? (
        <>
          <Stat label="Silhouette"    value={m.silhouette.toFixed(3)} />
          <Stat label="DB Index"      value={m.davies_bouldin.toFixed(2)} />
          <Stat label="Genre P@10"    value={`${(m.genre_purity_at_10 * 100).toFixed(0)}%`} />
          <Stat label="Trust"         value={m.trustworthiness.toFixed(3)} />
        </>
      ) : (
        <span className="px-5 py-2.5 text-white/30">metrics loading...</span>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 px-5 py-2.5 border-r-2 border-[#333] last:border-r-0">
      <span className="text-white/40">{label}:</span>
      <span className="text-white font-black">{value}</span>
    </div>
  );
}
