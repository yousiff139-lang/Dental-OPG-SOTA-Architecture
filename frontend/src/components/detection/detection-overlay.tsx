"use client";
import React, { useState } from "react";
import {
  getDetectionColor,
  getDetectionLabel,
  getAnatomicalLocation,
} from "@/lib/constants";
import { Detection, DentalSegment } from "@/lib/types";
import { Layers, Grid3X3, Sparkles, CheckCircle2, ShieldAlert } from "lucide-react";

interface DetectionOverlayProps {
  detections: Detection[];
  originalWidth: number;
  originalHeight: number;
  displayWidth: number;
  displayHeight: number;
  offsetX?: number;
  offsetY?: number;
  segments?: DentalSegment[];
  cvOverlayUrl?: string;
  isPanoramic?: boolean;
}

export function DetectionOverlay({
  detections,
  originalWidth,
  originalHeight,
  displayWidth,
  displayHeight,
  offsetX = 0,
  offsetY = 0,
  segments = [],
  cvOverlayUrl,
  isPanoramic = false,
}: DetectionOverlayProps) {
  const [showQuadrants, setShowQuadrants] = useState<boolean>(true);
  const [showCvOverlay, setShowCvOverlay] = useState<boolean>(false);
  const [cvOpacity, setCvOpacity] = useState<number>(0.55);

  if (!originalWidth || !originalHeight || !displayWidth || !displayHeight) {
    return null;
  }

  const scaleX = displayWidth / originalWidth;
  const scaleY = displayHeight / originalHeight;

  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Interactive Controls Bar pinned to top-right of image */}
      <div className="absolute top-2 right-2 z-40 flex items-center gap-1.5 bg-black/75 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-white/15 pointer-events-auto shadow-lg text-[11px]">
        {/* Toggle Quadrants */}
        {segments && segments.length > 0 && (
          <button
            type="button"
            onClick={() => setShowQuadrants(!showQuadrants)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors font-medium ${
              showQuadrants
                ? "bg-primary/20 text-primary border border-primary/40 shadow-xs"
                : "text-muted-foreground hover:text-white hover:bg-white/10"
            }`}
            title="Toggle Dental Quadrants Grid"
          >
            <Grid3X3 className="size-3.5" />
            <span>Quadrants</span>
          </button>
        )}

        {/* Toggle CV Heatmap */}
        {cvOverlayUrl && (
          <button
            type="button"
            onClick={() => setShowCvOverlay(!showCvOverlay)}
            className={`flex items-center gap-1 px-2 py-1 rounded transition-colors font-medium ${
              showCvOverlay
                ? "bg-amber-500/25 text-amber-400 border border-amber-500/40 shadow-xs"
                : "text-muted-foreground hover:text-white hover:bg-white/10"
            }`}
            title="Toggle Image Processing Radiolucency Overlay"
          >
            <Sparkles className="size-3.5 text-amber-400" />
            <span>CV Overlay</span>
          </button>
        )}
      </div>

      {/* CV Image Processing Heatmap Overlay Layer */}
      {showCvOverlay && cvOverlayUrl && (
        <div
          className="absolute inset-0 pointer-events-none transition-opacity duration-200 z-10 mix-blend-screen"
          style={{
            opacity: cvOpacity,
            backgroundImage: `url(${cvOverlayUrl})`,
            backgroundSize: "100% 100%",
            backgroundRepeat: "no-repeat",
          }}
        />
      )}

      {/* Segment-by-Segment Quadrant Grid Lines & Labels */}
      {showQuadrants && segments && segments.length > 0 && (
        <div className="absolute inset-0 pointer-events-none z-15">
          {/* Subtle Mid-lines dividing the quadrants */}
          {isPanoramic && (
            <>
              {/* Vertical Divider */}
              <div
                className="absolute top-0 bottom-0 border-r border-dashed border-cyan-400/30"
                style={{ left: `${displayWidth * 0.5}px` }}
              />
              {/* Horizontal Divider */}
              <div
                className="absolute left-0 right-0 border-b border-dashed border-cyan-400/30"
                style={{ top: `${displayHeight * 0.5}px` }}
              />
            </>
          )}

          {/* Quadrant Info Badges */}
          {segments.map((seg) => {
            const segLeft = seg.x * scaleX;
            const segTop = seg.y * scaleY;
            const segW = seg.width * scaleX;
            const segH = seg.height * scaleY;

            // Place label in the outer corner of each quadrant
            const isRightSide = seg.id === "Q2" || seg.id === "Q3";
            const isBottomSide = seg.id === "Q3" || seg.id === "Q4";

            return (
              <div
                key={seg.id}
                className="absolute pointer-events-auto"
                style={{
                  left: `${segLeft}px`,
                  top: `${segTop}px`,
                  width: `${segW}px`,
                  height: `${segH}px`,
                }}
              >
                {/* Quadrant corner pill */}
                <div
                  className={`absolute z-20 px-1.5 py-0.5 rounded text-[10px] font-mono font-medium backdrop-blur-md bg-black/60 border border-cyan-500/30 text-cyan-300 shadow-sm transition-opacity hover:opacity-100 opacity-75 ${
                    isBottomSide ? "bottom-2" : "top-2"
                  } ${isRightSide ? "right-2" : "left-2"}`}
                  title={`${seg.name} - ${seg.description || ""} (${seg.findings_count || 0} lesions)`}
                >
                  <span className="font-bold">{seg.id}</span>
                  <span className="opacity-75 text-[9px] ml-1">
                    {seg.dental_code || seg.name.split(":")[0]}
                  </span>
                  {(seg.findings_count ?? 0) > 0 && (
                    <span className="ml-1 px-1 py-0.2 rounded-full bg-red-500/30 text-red-300 font-bold text-[9px]">
                      {seg.findings_count}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Detection Boxes with Multi-Scanner Badges */}
      {detections.map((detection) => {
        const boxColor = getDetectionColor(detection.class_id, detection.class);
        const displayClass =
          detection.class || getDetectionLabel(detection.class_id);

        const boxLeft = (detection.x - detection.width / 2) * scaleX;
        const boxTop = (detection.y - detection.height / 2) * scaleY;
        const boxWidth = detection.width * scaleX;
        const boxHeight = detection.height * scaleY;

        const left = offsetX + boxLeft;
        const top = offsetY + boxTop;
        const width = Math.max(10, boxWidth);
        const height = Math.max(10, boxHeight);

        // Place label inside the box if box is too close to top
        const isNearTop = top < 26;

        // Determine location label based on position
        const relativeY = detection.y / originalHeight;
        const relativeX = detection.x / originalWidth;
        const locationLabel =
          detection.location_label ||
          getAnatomicalLocation(relativeX, relativeY, detection.class);

        // Determine shape style based on class
        const clsLower = (detection.class || "").toLowerCase();
        const isCavity =
          detection.class_id === 0 ||
          clsLower.includes("cavity") ||
          clsLower.includes("caries");
        const isPeriapical =
          detection.class_id === 1 ||
          clsLower.includes("periapical") ||
          clsLower === "pa";

        // Inner contour padding (% of box size)
        const contourPadX = width * 0.08;
        const contourPadY = height * 0.08;

        return (
          <div
            key={detection.detection_id || `${detection.x}-${detection.y}`}
            className="absolute pointer-events-auto group transition-all duration-150 z-20 hover:z-35"
            style={{
              left: `${left}px`,
              top: `${top}px`,
              width: `${width}px`,
              height: `${height}px`,
            }}
          >
            {/* Outer box locating the finding */}
            <div
              className={`absolute inset-0 transition-all ${
                isCavity
                  ? "border-[2.5px] border-red-500 rounded-sm shadow-[0_0_12px_rgba(239,68,68,0.55)] group-hover:shadow-[0_0_20px_rgba(239,68,68,0.9)]"
                  : isPeriapical
                  ? "border-2 border-orange-500 rounded-sm shadow-[0_0_10px_rgba(249,115,22,0.5)] group-hover:shadow-[0_0_18px_rgba(249,115,22,0.8)]"
                  : "border-2 border-dashed border-blue-500 rounded-sm"
              }`}
              style={{
                borderColor: boxColor,
              }}
            />

            {/* Inner target oval */}
            <div
              className={`absolute transition-all ${
                isCavity
                  ? "rounded-full border-2 border-red-500 bg-red-500/20 shadow-[inset_0_0_8px_rgba(239,68,68,0.3)]"
                  : isPeriapical
                  ? "rounded-full border-2 border-orange-500 bg-orange-500/15 shadow-[inset_0_0_8px_rgba(249,115,22,0.25)]"
                  : "rounded-md border border-blue-400 bg-blue-500/10"
              }`}
              style={{
                left: `${contourPadX}px`,
                top: `${contourPadY}px`,
                width: `${Math.max(4, width - contourPadX * 2)}px`,
                height: `${Math.max(4, height - contourPadY * 2)}px`,
                borderColor: boxColor,
              }}
            />

            {/* Pulsing center target dot */}
            <div
              className="absolute rounded-full animate-pulse"
              style={{
                width: "7px",
                height: "7px",
                left: `${width / 2 - 3.5}px`,
                top: `${height / 2 - 3.5}px`,
                backgroundColor: boxColor,
                boxShadow: `0 0 8px ${boxColor}`,
              }}
            />

            {/* Top label badge: class + multi-stage check icons */}
            <div
              className={`absolute px-1.5 py-0.5 text-[10px] font-bold text-white truncate max-w-[280px] rounded shadow-md flex items-center gap-1 ${
                isNearTop ? "top-0.5 left-0.5" : "-top-5 left-0"
              }`}
              style={{ backgroundColor: boxColor }}
              title={`${displayClass} (${(detection.confidence * 100).toFixed(
                1
              )}%) — ${locationLabel} | Stage 1 Diolo: Yes | Stage 2 Recheck: ${
                detection.stage2_rechecked ? "Verified" : "Pending"
              } | CV IP Score: ${(detection.ip_score ?? 0).toFixed(2)}`}
            >
              <span>{isCavity ? "🦷" : isPeriapical ? "🔬" : "📍"}</span>
              <span>{displayClass}</span>
              <span className="opacity-90 font-medium text-[9px]">
                ({(detection.confidence * 100).toFixed(0)}%)
              </span>
              {/* Multi-Scanner stage badge indicators */}
              <span
                className="ml-0.5 px-1 py-0 rounded text-[8px] bg-black/40 text-emerald-300 font-mono"
                title="Cross-checked by Stage 1 Diolo & Stage 2 YOLO Recheck"
              >
                S1+S2✓
              </span>
            </div>

            {/* Bottom label: anatomical location and quadrant */}
            {locationLabel && (
              <div
                className="absolute -bottom-4 left-0 px-1 py-0 text-[9px] font-medium rounded-xs shadow-xs truncate max-w-[240px]"
                style={{
                  backgroundColor: `${boxColor}E6`,
                  color: "white",
                }}
              >
                📍 {locationLabel}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
