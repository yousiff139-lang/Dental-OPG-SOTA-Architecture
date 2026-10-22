"use client";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Separator } from "../ui/separator";
import { Skeleton } from "../ui/skeleton";
import { TabsContent, Tabs, TabsList, TabsTrigger } from "../ui/tabs";
import { AlertCircle, Clock, ShieldCheck, CheckCircle2, Sparkles, Layers, Cpu, Grid3X3 } from "lucide-react";
import { DetectionImagePreview } from "../detection/detection-image-preview";
import { DicomMetadata } from "../dicom/dicom-metadata";
import { StatusIndicator } from "./status-indicator";
import { DiagnosticReportComponent } from "./diagnostic-report";
import {
  DetectionProgress,
  ConvertedDicomData,
  DiagnosticReport,
} from "@/lib/types";
import {
  formatFileSize,
  getDetectionColor,
  getAnatomicalLocation,
} from "@/lib/constants";

interface ResultCardProps {
  fileState: DetectionProgress["files"][0];
  fileData?: ConvertedDicomData;
  onReportGenerated?: (fileId: string, report: DiagnosticReport) => void;
  onStartGeneratingReport?: () => void;
}

export function ResultCard({
  fileState,
  fileData,
  onReportGenerated,
  onStartGeneratingReport,
}: ResultCardProps) {
  const handleReportGenerated = (report: DiagnosticReport) => {
    onReportGenerated?.(fileState.fileId, report);
  };

  return (
    <TabsContent key={fileState.fileId} value={fileState.fileId}>
      <Card className="w-full">
        <CardHeader>
          <div className="flex items-center gap-2">
            <StatusIndicator status={fileState.status} />
            <CardTitle className="text-base sm:text-lg truncate">
              File: {fileState.fileName}
            </CardTitle>
          </div>
          <CardDescription>
            {fileState.status === "loading" && "Processing..."}
            {fileState.status === "success" && (
              <div className="flex flex-col">
                <span>
                  Detections: {fileState.result?.predictions.length || 0}
                </span>
                <span>Patient: {fileState.result?.metadata?.patient_name || 'N/A'}</span>
              </div>
            )}
            {fileState.status === "error" &&
              `Error: ${fileState.error?.message || "Processing failed"}`}
            {fileState.status === "pending" && "Waiting to process..."}
          </CardDescription>
          {fileData && fileState.result && (
            <DetectionImagePreview
              src={fileData.dataUrl}
              alt={`X-ray with detections - ${fileState.result.fileName}`}
              width={
                fileState.result.image_width ||
                (fileState.result.image_info?.converted_size
                  ? fileState.result.image_info.converted_size[0]
                  : fileData.width)
              }
              height={
                fileState.result.image_height ||
                (fileState.result.image_info?.converted_size
                  ? fileState.result.image_info.converted_size[1]
                  : fileData.height)
              }
              detections={fileState.result.predictions}
              segments={fileState.result.segments}
              cvOverlayUrl={fileState.result.cv_overlay_url}
              isPanoramic={fileState.result.is_panoramic}
              scanMode={fileState.result.scan_mode}
            />
          )}
        </CardHeader>
        <CardContent className="space-y-3 sm:space-y-4">
          {fileState.status === "loading" && (
            <div className="space-y-3">
              <Skeleton className="h-48 w-full rounded-md" />
              <div className="space-y-2">
                <Skeleton className="h-4 w-1/3" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-2/3" />
              </div>
            </div>
          )}

          {fileState.status === "error" && (
            <div className="p-4 border border-red-200 rounded-md bg-red-50">
              <div className="flex items-center gap-2 text-red-800">
                <AlertCircle className="h-4 w-4" />
                <span className="font-medium">Processing Failed</span>
              </div>
              <p className="text-sm text-red-600 mt-1">
                {fileState.error?.message ||
                  "An unknown error occurred during processing."}
              </p>
            </div>
          )}

          {fileState.status === "pending" && (
            <div className="p-4 border border-gray-200 rounded-md bg-gray-50">
              <div className="flex items-center gap-2 text-gray-600">
                <Clock className="h-4 w-4" />
                <span className="font-medium">Waiting to Process</span>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                This file is queued for processing.
              </p>
            </div>
          )}

          {fileState.status === "success" && fileState.result && (
            <Tabs defaultValue="analysis" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="analysis">Analysis Results</TabsTrigger>
                <TabsTrigger value="report">Diagnostic Report</TabsTrigger>
              </TabsList>

              <TabsContent value="analysis" className="space-y-4 mt-4">
                {fileData && fileState.result && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Image Metadata</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 text-xs sm:text-sm">
                      <span className="text-muted-foreground font-medium">
                        File Name:
                      </span>
                      <span className="truncate">{fileData.fileName}</span>

                      <span className="text-muted-foreground font-medium">
                        Dimensions:
                      </span>
                      <span>
                        {fileData.width}×{fileData.height}px
                      </span>

                      <span className="text-muted-foreground font-medium">
                        File Size:
                      </span>
                      <span>{formatFileSize(fileData.fileSize)}</span>

                      <span className="text-muted-foreground font-medium">
                        Detection Count:
                      </span>
                      <span>{fileState.result.predictions.length}</span>
                    </div>
                  </div>
                )}
                <Separator />

                {/* Global Radiologist Health Audit Banner */}
                {fileState.result?.is_healthy && (
                  <div className="p-3.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 flex items-start gap-3">
                    <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0 mt-0.5" />
                    <div className="space-y-1">
                      <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                        <ShieldCheck className="h-3.5 w-3.5" />
                        Verified Clean Radiograph — No Pathosis
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        {fileState.result.radiologist_assessment || "Film audited as structurally intact with sound enamel crowns and intact apical lamina dura. Zero cavities or periapical lesions."}
                      </p>
                    </div>
                  </div>
                )}

                {/* Multi-Scanning System Pipeline Tracker */}
                {fileState.result && (
                  <div className="p-3 rounded-xl border border-primary/20 bg-primary/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold flex items-center gap-1.5 text-primary">
                        <Layers className="size-3.5" />
                        Multi-Scanning System Pipeline
                      </span>
                      <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                        {fileState.result.is_panoramic ? "Panoramic Radiograph (OPG)" : "Intraoral Radiograph"}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-[11px] pt-1">
                      {/* Stage 1 */}
                      <div className="p-2 rounded-lg bg-card/60 border border-border/50 flex flex-col gap-0.5">
                        <div className="flex items-center gap-1 font-semibold text-foreground">
                          <Cpu className="size-3 text-cyan-500" />
                          <span>Stage 1: Diolo (YOLO)</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                          {fileState.result.is_panoramic ? "Multi-quadrant arch decomposition" : "Full-frame candidate proposal"}
                        </p>
                      </div>

                      {/* Stage 2 */}
                      <div className="p-2 rounded-lg bg-card/60 border border-border/50 flex flex-col gap-0.5">
                        <div className="flex items-center gap-1 font-semibold text-foreground">
                          <ShieldCheck className="size-3 text-emerald-500" />
                          <span>Stage 2: Recheck YOLO</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                          Focused patch re-scan & clinical cross-checking
                        </p>
                      </div>

                      {/* Stage 3 */}
                      <div className="p-2 rounded-lg bg-card/60 border border-border/50 flex flex-col gap-0.5">
                        <div className="flex items-center gap-1 font-semibold text-foreground">
                          <Sparkles className="size-3 text-amber-500" />
                          <span>Stage 3: CV Overlay</span>
                        </div>
                        <p className="text-[10px] text-muted-foreground">
                          Segment Black-Hat radiolucency & enamel edges
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Segment-by-Segment Quadrants Diagnostic Grid */}
                {fileState.result?.segments && fileState.result.segments.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold flex items-center gap-1.5 text-foreground">
                      <Grid3X3 className="size-3.5 text-primary" />
                      <span>Segment-by-Segment Analysis</span>
                      <span className="text-[10px] text-muted-foreground">
                        ({fileState.result.segments.length} segments analyzed)
                      </span>
                    </h4>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                      {fileState.result.segments.map((seg) => (
                        <div
                          key={seg.id}
                          className="p-2 rounded-lg border border-border/50 bg-card/40 flex flex-col gap-1"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-bold text-foreground">{seg.id}</span>
                            <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold ${
                              (seg.findings_count || 0) > 0
                                ? "bg-red-500/20 text-red-400 border border-red-500/30"
                                : "bg-muted text-muted-foreground"
                            }`}>
                              {seg.findings_count || 0} findings
                            </span>
                          </div>
                          <span className="text-[10px] text-muted-foreground truncate" title={seg.name}>
                            {seg.name.split(":")[1]?.trim() || seg.name}
                          </span>
                          {seg.metrics && (
                            <div className="text-[9px] text-muted-foreground/80 flex justify-between pt-0.5 border-t border-border/30">
                              <span>Lucency: {seg.metrics.radiolucency_activity}</span>
                              <span>Edge: {seg.metrics.edge_gradient}</span>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <Separator />

                {/* Detected Dental Conditions & Findings */}
                {fileState.result && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold flex items-center gap-1.5">
                        <span>Detected Conditions</span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-muted font-normal">
                          {fileState.result.predictions.length} found
                        </span>
                      </h4>
                    </div>

                    {fileState.result.predictions.length === 0 ? (
                      <div className="flex items-center gap-2 p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 text-xs sm:text-sm">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
                        <span>No cavities or periapical lesions detected. Dental structures appear within normal limits.</span>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 gap-2.5">
                        {fileState.result.predictions.map((p, idx) => {
                          const imgW =
                            fileState.result?.image_width ||
                            fileData?.width ||
                            2000;
                          const imgH =
                            fileState.result?.image_height ||
                            fileData?.height ||
                            2000;
                          const loc =
                            p.location_label ||
                            getAnatomicalLocation(p.x / imgW, p.y / imgH, p.class);
                          const color = getDetectionColor(p.class_id, p.class);

                          return (
                            <div
                              key={p.detection_id || idx}
                              className="p-2.5 rounded-lg border border-border/60 bg-card/60 hover:bg-card transition-all flex flex-col gap-1.5"
                            >
                              <div className="flex items-center justify-between flex-wrap gap-2">
                                <div className="flex items-center gap-2">
                                  <span
                                    className="px-2 py-0.5 text-xs font-semibold rounded-md text-white shadow-xs"
                                    style={{ backgroundColor: color }}
                                  >
                                    {p.class}
                                  </span>
                                  <span className="text-xs font-medium text-foreground">
                                    {(p.confidence * 100).toFixed(1)}% confidence
                                  </span>
                                  {p.segment && (
                                    <span className="px-1.5 py-0.5 text-[10px] font-mono rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                                      {p.segment}
                                    </span>
                                  )}
                                </div>
                                <div className="flex items-center gap-1.5">
                                  {p.stage2_rechecked && (
                                    <span className="inline-flex items-center gap-1 text-[10px] font-medium text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                                      <ShieldCheck className="h-3 w-3" />
                                      Stage 2 Rechecked
                                    </span>
                                  )}
                                  {p.ip_verified && (
                                    <span className="inline-flex items-center gap-1 text-[10px] font-medium text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                                      <Sparkles className="h-3 w-3" />
                                      CV IP: {((p.ip_score ?? 0) * 100).toFixed(0)}%
                                    </span>
                                  )}
                                </div>
                              </div>
                              <div className="text-xs text-muted-foreground flex items-center gap-1">
                                <span>📍 Location:</span>
                                <span className="font-medium text-foreground">
                                  {loc}
                                </span>
                              </div>
                              {p.gemini_reasoning && (
                                <p
                                  className="text-[11px] text-muted-foreground italic border-l-2 pl-2 mt-0.5"
                                  style={{ borderColor: color }}
                                >
                                  &ldquo;{p.gemini_reasoning}&rdquo;
                                </p>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
                <Separator />

                {/* DICOM Metadata - only show if available */}
                {fileState.result.metadata && (
                  <DicomMetadata metadata={fileState.result.metadata} />
                )}

                <Separator />

                {/* Image Info - only show if available */}
                {fileState.result.image_info && (
                  <div>
                    <h4 className="text-sm font-medium mb-2">
                      Image Information
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1 text-xs sm:text-sm">
                      <span className="text-muted-foreground font-medium">
                        Original Shape:
                      </span>
                      <span>
                        {fileState.result.image_info.original_shape.join("×")}
                      </span>

                      <span className="text-muted-foreground font-medium">
                        Converted Format:
                      </span>
                      <span>{fileState.result.image_info.converted_format}</span>

                      <span className="text-muted-foreground font-medium">
                        Converted Size:
                      </span>
                      <span>
                        {fileState.result.image_info.converted_size.join("×")}
                      </span>

                      <span className="text-muted-foreground font-medium">
                        Pixel Range:
                      </span>
                      <span>
                        {fileState.result.image_info.pixel_array_min.toFixed(1)} -{" "}
                        {fileState.result.image_info.pixel_array_max.toFixed(1)}
                      </span>
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="report" className="mt-4">
                <DiagnosticReportComponent
                  predictions={fileState.result.predictions}
                  metadata={fileState.result.metadata}
                  imageInfo={fileState.result.image_info}
                  report={fileState.diagnosticReport}
                  onReportGenerated={handleReportGenerated}
                  onStartGenerating={onStartGeneratingReport}
                  originalImageSrc={fileData?.dataUrl} // Pass the original image source
                />
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  );
}
