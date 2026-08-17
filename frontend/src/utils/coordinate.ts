import { NormalizedPoint } from "../types";

export const MIN_SCALE = 0.1; // 10%
export const MAX_SCALE = 4.0; // 400%

export function clampZoom(zoom: number): number {
  return Math.max(MIN_SCALE, Math.min(MAX_SCALE, zoom));
}

/**
 * Converts a screen/viewport client coordinate (relative to the container)
 * into a normalized [0, 1] image coordinate, taking zoom and pan into account.
 */
export function screenToNormalizedImage(
  screenX: number,
  screenY: number,
  panX: number,
  panY: number,
  zoom: number,
  imageWidth: number,
  imageHeight: number
): NormalizedPoint {
  const clampedZoom = clampZoom(zoom);
  const imgX = (screenX - panX) / clampedZoom;
  const imgY = (screenY - panY) / clampedZoom;

  const normX = Math.max(0, Math.min(1, imgX / imageWidth));
  const normY = Math.max(0, Math.min(1, imgY / imageHeight));

  return { x: normX, y: normY };
}

/**
 * Converts a normalized [0, 1] coordinate back to screen/viewport pixels.
 */
export function normalizedImageToScreen(
  point: NormalizedPoint,
  panX: number,
  panY: number,
  zoom: number,
  imageWidth: number,
  imageHeight: number
): { x: number; y: number } {
  const clampedZoom = clampZoom(zoom);
  const imgX = point.x * imageWidth;
  const imgY = point.y * imageHeight;

  const screenX = imgX * clampedZoom + panX;
  const screenY = imgY * clampedZoom + panY;

  return { x: screenX, y: screenY };
}

/**
 * Calculate the optimal zoom scale and centered pan offsets
 * to fit an image of dimensions (imgW, imgH) into a container (contW, contH).
 */
export function calculateFitToScreen(
  imgW: number,
  imgH: number,
  contW: number,
  contH: number,
  paddingFactor = 0.90
): { zoom: number; panX: number; panY: number } {
  if (imgW <= 0 || imgH <= 0 || contW <= 0 || contH <= 0) {
    return { zoom: 1, panX: 0, panY: 0 };
  }

  const scaleX = contW / imgW;
  const scaleY = contH / imgH;
  const rawZoom = Math.min(scaleX, scaleY) * paddingFactor;
  const zoom = clampZoom(rawZoom);

  const panX = (contW - imgW * zoom) / 2;
  const panY = (contH - imgH * zoom) / 2;

  return { zoom, panX, panY };
}

