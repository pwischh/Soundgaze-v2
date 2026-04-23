"use client";

import type { Method } from "../lib/api";

interface MethodSwitcherProps {
  method: Method;
  onChange: (method: Method) => void;
  disabled?: boolean;
}

const METHODS: { value: Method; label: string }[] = [
  { value: "pca",  label: "PCA" },
  { value: "tsne", label: "t-SNE" },
  { value: "umap", label: "UMAP" },
];

export default function MethodSwitcher({ method, onChange, disabled }: MethodSwitcherProps) {
  return (
    <div
      className="flex border-2 border-white/20"
      style={{ boxShadow: "0px 4px 0px 0px rgba(255,255,255,0.25)" }}
    >
      {METHODS.map(({ value, label }, i) => (
        <button
          key={value}
          onClick={() => !disabled && onChange(value)}
          disabled={disabled}
          className={[
            "px-8 py-3.5 font-black text-sm uppercase tracking-widest transition-colors",
            i > 0 ? "border-l border-white/20" : "",
            method === value
              ? "bg-spotify-green text-black"
              : "bg-spotify-black text-white hover:bg-white/10 disabled:opacity-40",
          ].join(" ")}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
