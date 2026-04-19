import type { Metadata } from "next";
import { Syne } from "next/font/google";
import "./globals.css";

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-syne",
  display: "swap",
  weight: ["400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "Soundgaze 2.0 — DR Comparison for Music Recommendation",
  description: "Project comparing PCA, t-SNE, and UMAP on PANN audio embeddings.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={syne.variable}>
      <body>{children}</body>
    </html>
  );
}
