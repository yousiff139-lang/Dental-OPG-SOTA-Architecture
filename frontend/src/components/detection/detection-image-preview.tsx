"use client";
import { Detection, DentalSegment } from "@/lib/types";
import { useEffect, useRef, useState, useCallback } from "react";
import { DetectionBadgeList } from "./detection-badge-list";
import { DetectionOverlay } from "./detection-overlay";

interface DetectionImagePreviewProps {
  src: string;
  alt: string;
  width: number;
  height: number;
  detections?: Detection[];
  segments?: DentalSegment[];
  cvOverlayUrl?: string;
  isPanoramic?: boolean;
  scanMode?: string;
}

export function DetectionImagePreview({
  src,
  alt,
  width,
  height,
  detections = [],
  segments = [],
  cvOverlayUrl,
  isPanoramic = false,
  scanMode,
}: DetectionImagePreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const [renderedBounds, setRenderedBounds] = useState<{
    width: number;
    height: number;
    offsetX: number;
    offsetY: number;
  }>({ width: 0, height: 0, offsetX: 0, offsetY: 0 });

  // Update dimensions from the actual <img> DOM node
  const updateRenderedDimensions = useCallback(() => {
    const img = imageRef.current;
    if (!img) return;

    const rect = img.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    setRenderedBounds({
      width: rect.width,
      height: rect.height,
      offsetX: 0,
      offsetY: 0,
    });
  }, []);

  // Listen to window resize and ResizeObserver on image container
  useEffect(() => {
    updateRenderedDimensions();

    const img = imageRef.current;
    if (!img) return;

    let resizeObserver: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        updateRenderedDimensions();
      });
      resizeObserver.observe(img);
    }

    window.addEventListener("resize", updateRenderedDimensions);

    return () => {
      window.removeEventListener("resize", updateRenderedDimensions);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [updateRenderedDimensions]);

  return (
    <div className="w-full flex flex-col items-center" ref={containerRef}>
      <div className="space-y-3 sm:space-y-4 w-full">
        {/* Precise Image Container: wraps image exactly without letterbox offsets */}
        <div className="relative mx-auto flex justify-center max-w-full">
          <div className="relative inline-block max-w-full rounded-lg overflow-hidden border border-border/40 shadow-sm bg-black/10">
            {/* Standard responsive <img> to guarantee 1:1 display coordinates without Next.js letterbox shifts */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              ref={imageRef}
              src={src}
              alt={alt}
              className="block max-w-full h-auto max-h-[60vh] mx-auto select-none"
              onLoad={updateRenderedDimensions}
            />

            {/* Bounding box overlay strictly aligned to rendered image pixels */}
            {renderedBounds.width > 0 && renderedBounds.height > 0 && (
              <DetectionOverlay
                detections={detections}
                originalWidth={
                  width || imageRef.current?.naturalWidth || 1
                }
                originalHeight={
                  height || imageRef.current?.naturalHeight || 1
                }
                displayWidth={renderedBounds.width}
                displayHeight={renderedBounds.height}
                offsetX={renderedBounds.offsetX}
                offsetY={renderedBounds.offsetY}
                segments={segments}
                cvOverlayUrl={cvOverlayUrl}
                isPanoramic={isPanoramic}
              />
            )}
          </div>
        </div>

        {/* Detection badges summary */}
        <div className="space-y-3 sm:space-y-4">
          <DetectionBadgeList detections={detections} />
        </div>
      </div>
    </div>
  );
}
