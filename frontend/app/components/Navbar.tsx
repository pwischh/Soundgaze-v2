import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="relative z-20 flex items-center justify-between px-6 h-14 bg-spotify-black shrink-0">

      {/* Left: wordmark */}
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="font-black text-xl uppercase tracking-widest text-white hover:text-spotify-green transition-colors"
        >
          Soundgaze 2.0
        </Link>
      </div>

    </nav>
  );
}
