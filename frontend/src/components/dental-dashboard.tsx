"use client";

import { DicomDropzone } from "@/components/dicom/dicom-dropzone";
import { PredictionResults } from "@/components/results/prediction-results";
import { useDicomDetection } from "@/hooks/use-dicom-detection";
import { useDroppedFilesStore } from "@/lib/store";
import { SplineScene } from "@/components/ui/splite";
import { Spotlight } from "@/components/ui/spotlight";
import { RobotChatBubble, type RobotState } from "@/components/robot-chat-bubble";
import { ConsentModal } from "@/components/consent-modal";
import { Brain } from "lucide-react";
import { useState, useEffect, useMemo, useCallback } from "react";
import type { DiagnosticReport } from "@/lib/types";

export function DentalDashboard() {
  const { files } = useDroppedFilesStore();
  const dicomDetectionMutation = useDicomDetection();
  const [robotState, setRobotState] = useState<RobotState>("idle");
  const [isRobotLoaded, setIsRobotLoaded] = useState(false);
  const [hasConsented, setHasConsented] = useState(false);

  // Handle consent acceptance
  const handleConsent = useCallback(() => {
    setHasConsented(true);
  }, []);

  const handlePredict = async () => {
    if (files.length === 0) return;
    setRobotState("predicting");
    dicomDetectionMutation.mutate({ files });
  };

  const { detectionProgress, updateDiagnosticReport } = dicomDetectionMutation;
  const hasAnyActivity = detectionProgress.files.length > 0;

  // Count conditions found
  const conditionsFound = useMemo(() => {
    return detectionProgress.files.reduce((total, file) => {
      return total + (file.result?.predictions?.length || 0);
    }, 0);
  }, [detectionProgress.files]);

  // Track state changes
  useEffect(() => {
    // When files are uploaded
    if (files.length > 0 && robotState === "idle") {
      setRobotState("file_uploaded");
    }
    // Reset to idle when files are cleared
    if (files.length === 0 && robotState !== "idle") {
      setRobotState("idle");
    }
  }, [files.length, robotState]);

  // Track prediction completion
  useEffect(() => {
    const allComplete = detectionProgress.files.length > 0 &&
      detectionProgress.files.every(f => f.status === "success" || f.status === "error");

    if (allComplete && robotState === "predicting") {
      setRobotState("prediction_complete");
    }
  }, [detectionProgress.files, robotState]);

  // Handle report generation state
  const handleReportGenerated = (fileId: string, report: DiagnosticReport) => {
    setRobotState("report_complete");
    updateDiagnosticReport(fileId, report);
  };

  const handleStartGeneratingReport = () => {
    setRobotState("generating_report");
  };

  // Handle 3D robot load
  const handleRobotLoad = () => {
    setIsRobotLoaded(true);
  };

  return (
    <div className="h-screen overflow-hidden relative bg-background" suppressHydrationWarning>
      {/* Consent Modal - Must accept before using */}
      <ConsentModal onAccept={handleConsent} />

      {/* Spotlight Effect */}
      <Spotlight
        className="-top-40 left-0 md:left-60 md:-top-20"
        fill="white"
      />

      {/* Main Layout - Side by Side */}
      <div className="relative z-20 h-full flex flex-col lg:flex-row">
        {/* LEFT: Upload + Diagnostic Report - Both Always Visible */}
        <div className="lg:w-[45%] h-full flex flex-col p-3 lg:p-4 gap-3 overflow-hidden">
          {/* Header - Compact */}
          <div className="flex items-center gap-2 shrink-0">
            <div className="p-1.5 rounded-lg bg-primary/10 border border-primary/20">
              <Brain className="size-4 text-primary" />
            </div>
            <div>
              <h1 className="text-base font-bold leading-tight">Dental X-Ray</h1>
              <p className="text-[10px] text-primary">AI Analysis</p>
            </div>
          </div>

          {/* Upload Card - Compact but fully visible */}
          <div className="shrink-0 bg-card/50 backdrop-blur-sm border border-border rounded-xl p-3 max-h-[45%] overflow-auto">
            <h2 className="text-xs font-semibold mb-2">Upload X-Ray</h2>
            <DicomDropzone
              dicomDetectionMutation={dicomDetectionMutation}
              onPredict={handlePredict}
            />
          </div>

          {/* Diagnostic Report - Takes remaining space */}
          <div className="flex-1 min-h-0 bg-card/50 backdrop-blur-sm border border-border rounded-xl p-3 flex flex-col overflow-hidden">
            <h2 className="text-xs font-semibold mb-2 flex items-center gap-2 shrink-0">
              <Brain className="size-3 text-primary" />
              Diagnostic Report
              {hasAnyActivity && (
                <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-primary/20 text-primary">
                  Active
                </span>
              )}
            </h2>

            <div className="flex-1 overflow-auto">
              {!hasAnyActivity && (
                <div className="h-full flex flex-col items-center justify-center text-center">
                  <div className="p-2 rounded-full bg-primary/5 border border-primary/10 mb-2">
                    <Brain className="size-4 text-primary/50" />
                  </div>
                  <p className="text-[11px] text-muted-foreground max-w-xs">
                    Upload X-rays for AI-powered cavity and lesion detection.
                  </p>
                </div>
              )}

              {hasAnyActivity && (
                <PredictionResults
                  detectionProgress={detectionProgress}
                  onReportGenerated={handleReportGenerated}
                  onStartGeneratingReport={handleStartGeneratingReport}
                />
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: 3D Robot with Chat Bubble - Only load after consent */}
        <div className="lg:w-[55%] h-full relative">
          {hasConsented ? (
            <>
              {/* Robot Chat Bubble - Only shows after robot loads */}
              <RobotChatBubble
                state={robotState}
                conditionsFound={conditionsFound}
                isRobotLoaded={isRobotLoaded}
                hasConsented={hasConsented}
              />

              {/* 3D Robot */}
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full"
                onLoad={handleRobotLoad}
              />
            </>
          ) : (
            /* Placeholder before consent - subtle visual */
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center opacity-30">
                <Brain className="size-16 mx-auto mb-4 text-primary/50" />
                <p className="text-sm text-muted-foreground">
                  Accept terms to meet your AI assistant
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
