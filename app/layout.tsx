import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const title = "StrataMoE Lab — Trace-driven MoE memory research";
const description =
  "A deterministic simulator for comparing LRU, LFU, and shift-aware expert caching across GPU, RAM, and NVMe tiers.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const forwardedHost = requestHeaders.get("x-forwarded-host")?.split(",")[0]?.trim();
  const host = forwardedHost ?? requestHeaders.get("host") ?? "localhost:3000";
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto")?.split(",")[0]?.trim();
  const protocol = forwardedProtocol ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = new URL(`${protocol}://${host}`);
  const socialImage = new URL("/og.png", origin);

  return {
    metadataBase: origin,
    title,
    description,
    applicationName: "StrataMoE Lab",
    keywords: [
      "mixture of experts",
      "inference",
      "memory hierarchy",
      "cache policy",
      "deterministic simulation",
    ],
    openGraph: {
      type: "website",
      siteName: "StrataMoE Lab",
      title,
      description,
      images: [
        {
          url: socialImage,
          width: 1672,
          height: 941,
          alt: "StrataMoE Lab — deterministic MoE memory hierarchy simulator",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [socialImage],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
