import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        "spotify-green": "#1DB954",
        "near-black":    "#0a0a0a",
        "spotify-black": "#191414",
        "off-white":     "#f5f5f5",
        canvas:  "#0a0a0a",
        surface: "#191414",
        accent:  "#1DB954",
        primary: "#FFFFFF",
        muted:   "#6b7280",
      },
      fontFamily: {
        sans: ["var(--font-syne)", "sans-serif"],
        syne: ["var(--font-syne)", "sans-serif"],
        mono: ["Courier New", "Courier", "monospace"],
      },
      keyframes: {
        marquee: {
          "0%":   { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
      animation: {
        "marquee-slow":   "marquee 30s linear infinite",
        "marquee-medium": "marquee 20s linear infinite",
        "marquee-fast":   "marquee 12s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
