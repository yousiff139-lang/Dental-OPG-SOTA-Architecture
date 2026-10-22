// Application constants
export const FILE_SIZE_LIMITS = {
  MAX_SIZE: 10 * 1024 * 1024, // 10 MB
} as const;

export const SUPPORTED_FILE_EXTENSIONS = {
  "application/dicom": [".dcm", ".rvg"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
} as const;

// Keep legacy name for backwards compatibility
export const SUPPORTED_DICOM_EXTENSIONS = SUPPORTED_FILE_EXTENSIONS;

// Utility function for consistent file size formatting
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
};
// Color palette for different detection classes
export const DETECTION_COLORS = [
  "#ef4444", // red - Cavity
  "#f97316", // orange - Periapical Lesion
  "#3b82f6", // blue - Other Condition
  "#22c55e", // green
  "#eab308", // yellow
  "#8b5cf6", // violet
  "#ec4899", // pink
  "#06b6d4", // cyan
];

export function getDetectionColor(
  classId?: number | string | null,
  className?: string
): string {
  const check = (
    typeof classId === "string" ? classId : className || ""
  ).toLowerCase();

  if (check.includes("cavity") || check.includes("caries")) {
    return "#ef4444"; // Vivid Red for Cavity
  }
  if (check.includes("periapical") || check.includes("pa")) {
    return "#f97316"; // Bright Orange for Periapical Lesion
  }
  if (check.includes("impacted")) {
    return "#8b5cf6"; // Purple/Violet for Impacted / Unerupted Tooth
  }

  if (typeof classId === "number" && !isNaN(classId)) {
    if (classId === 0) return "#ef4444";
    if (classId === 1) return "#f97316";
    if (classId === 2) return "#8b5cf6";
    return DETECTION_COLORS[Math.abs(classId) % DETECTION_COLORS.length];
  }

  return "#ef4444";
}

export const DETECTION_LABELS: Record<number, string> = {
  0: "Cavity",
  1: "Periapical Lesion",
  2: "Impacted Tooth",
};

export function getDetectionLabel(
  classId?: number | string | null,
  fallback?: string
): string {
  if (typeof classId === "string" && classId) {
    return classId;
  }
  if (typeof classId === "number" && !isNaN(classId) && DETECTION_LABELS[classId]) {
    return DETECTION_LABELS[classId];
  }
  return fallback || "Cavity";
}

/**
 * Determine an approximate anatomical location based on the detection's
 * relative position within the X-ray image.
 */
export function getAnatomicalLocation(
  relativeX: number,
  relativeY: number,
  detectedClass: string
): string {
  // Vertical position
  let verticalZone: string;
  if (relativeY < 0.35) {
    verticalZone = "Crown region";
  } else if (relativeY < 0.65) {
    verticalZone = "Mid-root region";
  } else {
    verticalZone = "Apex region";
  }

  // Horizontal position
  let horizontalZone: string;
  if (relativeX < 0.33) {
    horizontalZone = "Left";
  } else if (relativeX < 0.66) {
    horizontalZone = "Central";
  } else {
    horizontalZone = "Right";
  }

  // Combine with class-specific location terms
  const cls = (detectedClass || "").toLowerCase();
  if (cls.includes("periapical")) {
    return `${horizontalZone} — Periapical (root apex)`;
  } else if (cls.includes("cavity") || cls.includes("caries")) {
    if (relativeY < 0.4) {
      return `${horizontalZone} — Coronal (tooth crown)`;
    } else {
      return `${horizontalZone} — Cervical (tooth neck)`;
    }
  }

  return `${horizontalZone} — ${verticalZone}`;
}


