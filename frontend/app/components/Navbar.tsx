import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="relative z-20 flex items-center justify-between px-6 h-14 bg-spotify-black shrink-0">

      {/* Left: wordmark */}
      <div className="flex items-center gap-3">
        <Link
          href="/explore"
          className="font-black text-xl uppercase tracking-widest text-white hover:text-spotify-green transition-colors"
        >
          Soundgaze 2.0
        </Link>
      </div>

      {/* Right: research context label */}
      <div className="flex items-center gap-4">
        <span className="hidden md:inline-flex items-center gap-2 px-4 py-1.5 font-mono text-xs uppercase tracking-widest text-white/70 bg-[#121212] border-2 border-[#333]">
          DR Comparison · FMA Small · 8K Tracks
        </span>
      </div>

    </nav>
  );
}
