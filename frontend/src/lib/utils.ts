import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { DicomImage, NativePixelDecoder } from "dcmjs-imaging";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Convert a DICOM file to a JPG/PNG data URL for display
 * @param file - The DICOM file to convert
 * @param format - Output format ('jpeg' | 'png')
 * @param quality - JPEG quality (0-1, only applies to JPEG)
 * @returns Promise<string> - Data URL of the converted image
 */
export async function convertDicomToImageUrl(
  file: File,
  format: "jpeg" | "png" = "jpeg",
  quality: number = 0.8
): Promise<{
  dataUrl: string;
  width: number;
  height: number;
}> {
  // Ensure native decoders are initialized
  try {
    await NativePixelDecoder.initializeAsync({
      webAssemblyModulePathOrUrl: "/dcmjs-native-codecs.wasm",
    });
  } catch {
    // If already initialized, this will throw but we can continue
    console.log("Native decoders already initialized or failed to initialize");
  }

  // Read file as ArrayBuffer
  const arrayBuffer = await file.arrayBuffer();

  // Create DICOM image instance
  const image = new DicomImage(arrayBuffer);

  // Render the image
  const renderingResult = image.render();

  // Create a temporary canvas
  const canvas = document.createElement("canvas");
  canvas.width = renderingResult.width;
  canvas.height = renderingResult.height;

  // Get 2D context
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to get canvas context");
  }

  // Create ImageData from rendered pixels (RGBA format)
  const imageData = new ImageData(
    new Uint8ClampedArray(renderingResult.pixels),
    renderingResult.width,
    renderingResult.height
  );

  // Draw the image data to canvas
  ctx.putImageData(imageData, 0, 0);

  // Convert canvas to data URL
  const mimeType = format === "jpeg" ? "image/jpeg" : "image/png";
  const dataUrl = canvas.toDataURL(mimeType, quality);

  return {
    dataUrl,
    width: renderingResult.width,
    height: renderingResult.height,
  };
}

/**
 * Convert DICOM file to Blob for download or further processing
 * @param file - The DICOM file to convert
 * @param format - Output format ('jpeg' | 'png')
 * @param quality - JPEG quality (0-1, only applies to JPEG)
 * @returns Promise<Blob> - Blob of the converted image
 */
export async function convertDicomToBlob(
  file: File,
  format: "jpeg" | "png" = "jpeg",
  quality: number = 0.8
): Promise<Blob> {
  const { dataUrl } = await convertDicomToImageUrl(file, format, quality);

  // Convert data URL to blob
  const response = await fetch(dataUrl);
  return response.blob();
}

/**
 * Check if a file is a DICOM/RVG format based on extension
 * @param file - The file to check
 * @returns boolean - True if file is DICOM/RVG format
 */
export function isDicomFile(file: File): boolean {
  const extension = file.name.toLowerCase().split('.').pop();
  return ['dcm', 'dicom', 'rvg'].includes(extension || '');
}

/**
 * Convert a standard image file (PNG/JPEG) to a data URL for display
 * @param file - The image file to convert
 * @returns Promise with dataUrl, width, and height
 */
export async function convertStandardImageToUrl(
  file: File
): Promise<{
  dataUrl: string;
  width: number;
  height: number;
}> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;

      // Create an image to get dimensions
      const img = new Image();
      img.onload = () => {
        resolve({
          dataUrl,
          width: img.width,
          height: img.height,
        });
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = dataUrl;
    };

    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

