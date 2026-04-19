interface MarqueeTickerProps {
  text?: string;
  variant?: "green" | "black";
  speed?: "slow" | "medium" | "fast";
  tilt?: number;
}

const DEFAULT_TEXT =
  "PCA • t-SNE • UMAP • SOUNDGAZE V2 • FMA DATASET • MUSIC RECOMMENDATION •";

const SPEED_CLASS = {
  slow:   "animate-marquee-slow",
  medium: "animate-marquee-medium",
  fast:   "animate-marquee-fast",
} as const;

export default function MarqueeTicker({
  text = DEFAULT_TEXT,
  variant = "green",
  speed = "medium",
  tilt = 0,
}: MarqueeTickerProps) {
  const isGreen = variant === "green";
  const wrapperStyle = tilt !== 0 ? { transform: `rotate(${tilt}deg)` } : undefined;

  return (
    <div
      className={`w-full overflow-hidden whitespace-nowrap border-y-4 border-black py-1.5
                  ${isGreen ? "bg-spotify-green" : "bg-black"}`}
      style={wrapperStyle}
    >
      <span
        className={`inline-block ${SPEED_CLASS[speed]}
                    font-black text-lg uppercase tracking-widest select-none
                    ${isGreen ? "text-black" : "text-white"}`}
      >
        {text}&nbsp;&nbsp;&nbsp;{text}&nbsp;&nbsp;&nbsp;
      </span>
    </div>
  );
}
