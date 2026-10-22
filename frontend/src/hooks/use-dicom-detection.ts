import { useMutation } from "@tanstack/react-query";
import { useState, useCallback } from "react";
import apiClient from "@/lib/axios";
import {
  ConvertedDicomData,
  DicomDetectionResponse,
  DicomDetectionResult,
  FileDetectionState,
  DetectionProgress,
  DiagnosticReport,
} from "@/lib/types";

interface DetectDicomFilesParams {
  files: ConvertedDicomData[];
}

export function useDicomDetection() {
  const [detectionProgress, setDetectionProgress] = useState<DetectionProgress>(
    {
      files: [],
      isAllComplete: false,
      hasAnyResults: false,
      hasAnyErrors: false,
    }
  );

  const detectDicomFiles = useCallback(
    async (params: DetectDicomFilesParams) => {
      const { files } = params;

      if (files.length === 0) {
        return [];
      }

      // Initialize all files as pending
      const initialFiles: FileDetectionState[] = files.map((file) => ({
        fileId: file.id,
        fileName: file.fileName,
        status: "pending" as const,
      }));

      setDetectionProgress({
        files: initialFiles,
        isAllComplete: false,
        hasAnyResults: false,
        hasAnyErrors: false,
      });

      // Update file status to loading
      const updateFileStatus = (
        fileId: string,
        updates: Partial<FileDetectionState>
      ) => {
        setDetectionProgress((prev) => {
          const updatedFiles = prev.files.map((file) =>
            file.fileId === fileId ? { ...file, ...updates } : file
          );

          const hasAnyResults = updatedFiles.some(
            (file) => file.status === "success"
          );
          const hasAnyErrors = updatedFiles.some(
            (file) => file.status === "error"
          );
          const isAllComplete = updatedFiles.every(
            (file) => file.status === "success" || file.status === "error"
          );

          return {
            files: updatedFiles,
            isAllComplete,
            hasAnyResults,
            hasAnyErrors,
          };
        });
      };

      // Process all files in parallel
      const detectionPromises = files.map(async (file) => {
        try {
          // Update to loading state
          updateFileStatus(file.id, { status: "loading" });

          const formData = new FormData();
          formData.append("file", file.originalFile);

          // Determine endpoint based on file type
          const extension = file.fileName.toLowerCase().split('.').pop();
          const isDicom = ['dcm', 'dicom', 'rvg'].includes(extension || '');
          const endpoint = isDicom ? "/detect-dicom" : "/detect";

          const response = await apiClient.post<DicomDetectionResponse>(
            endpoint,
            formData,
            {
              headers: {
                "Content-Type": "multipart/form-data",
              },
            }
          );

          // Normalize the response format for standard images
          // The /detect endpoint returns { predictions: [...] }
          // The /detect-dicom endpoint returns { predictions: [...], metadata: {...}, image_info: {...} }
          const result: DicomDetectionResult = {
            predictions: response.data.predictions || [],
            metadata: response.data.metadata || null,
            image_info: response.data.image_info || null,
            image_width: response.data.image_width,
            image_height: response.data.image_height,
            scan_mode: response.data.scan_mode,
            is_panoramic: response.data.is_panoramic,
            segments: response.data.segments,
            cv_overlay_url: response.data.cv_overlay_url,
            fileId: file.id,
            fileName: file.fileName,
          };

          // Update to success state
          updateFileStatus(file.id, {
            status: "success",
            result,
          });

          return result;
        } catch (error) {
          const errorObj =
            error instanceof Error ? error : new Error("Unknown error");

          // Update to error state
          updateFileStatus(file.id, {
            status: "error",
            error: errorObj,
          });

          throw errorObj;
        }
      });

      // Wait for all to complete, but don't fail if some fail
      const results = await Promise.allSettled(detectionPromises);

      // Return only successful results
      return results
        .filter(
          (result): result is PromiseFulfilledResult<DicomDetectionResult> =>
            result.status === "fulfilled"
        )
        .map((result) => result.value);
    },
    []
  );

  const mutation = useMutation({
    mutationFn: detectDicomFiles,
    onError: (error) => {
      console.error("DICOM detection failed:", error);
    },
  });

  const updateDiagnosticReport = useCallback(
    (fileId: string, report: DiagnosticReport) => {
      setDetectionProgress((prev) => {
        const updatedFiles = prev.files.map((file) =>
          file.fileId === fileId ? { ...file, diagnosticReport: report } : file
        );

        return {
          ...prev,
          files: updatedFiles,
        };
      });
    },
    []
  );

  const reset = useCallback(() => {
    setDetectionProgress({
      files: [],
      isAllComplete: false,
      hasAnyResults: false,
      hasAnyErrors: false,
    });
    mutation.reset();
  }, [mutation]);

  return {
    ...mutation,
    detectionProgress,
    updateDiagnosticReport,
    reset,
  };
}
