"use client";

import dynamic from "next/dynamic";

const DentalDashboard = dynamic(
  () => import("@/components/dental-dashboard").then((mod) => mod.DentalDashboard),
  {
    ssr: false,
    loading: () => (
      <div
        className="h-screen w-screen overflow-hidden relative bg-background flex items-center justify-center"
        suppressHydrationWarning
      >
        <div
          className="flex flex-col items-center gap-3 text-muted-foreground animate-pulse"
          suppressHydrationWarning
        >
          <div className="size-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
          <span className="text-xs font-medium tracking-wide">Loading Dental AI...</span>
        </div>
      </div>
    ),
  }
);

export default function Home() {
  return (
    <main className="h-screen overflow-hidden relative bg-background" suppressHydrationWarning>
      <DentalDashboard />
    </main>
  );
}
