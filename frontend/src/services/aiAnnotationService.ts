import { Annotation, AnnotationShape, ReferenceAgreementMetrics } from "../types";
import { AI_ASSISTANCE_METADATA } from "../constants/theme";

export interface AiInferenceOptions {
  inferenceMode: "standard" | "tta";
  threshold?: number;
  hasReferenceMask?: boolean;
}

export interface AiInferenceResult {
  annotation: Annotation;
  inferenceTimeMs: number;
  finding: "possible-region-detected" | "no-region-detected" | "unavailable";
  maximumOutputScore: number;
  maskCoveragePercent: number;
  regionCount: number;
  referenceMetrics: ReferenceAgreementMetrics | null;
}

export const aiAnnotationService = {
  async requestAiAnnotation(
    imageId: string,
    options: AiInferenceOptions
  ): Promise<AiInferenceResult> {
    const startTime = performance.now();

    // Simulate 1.0s - 1.4s inference delay
    const delay = options.inferenceMode === "tta" ? 1400 : 1000;
    await new Promise((res) => setTimeout(res, delay));

    const threshold = options.threshold ?? AI_ASSISTANCE_METADATA.defaultThreshold;

    // Simulate model output score
    const maximumOutputScore = Number((0.76 + Math.random() * 0.18).toFixed(3));
    const isDetected = maximumOutputScore >= threshold;

    const finding: "possible-region-detected" | "no-region-detected" = isDetected
      ? "possible-region-detected"
      : "no-region-detected";

    const maskCoveragePercent = isDetected
      ? Number((2.6 + Math.random() * 3.8).toFixed(2))
      : 0;

    // Generate realistic apical/lateral pneumothorax polygon in normalized coordinates
    const shapes: AnnotationShape[] = isDetected
      ? [
          {
            id: `shp-ai-${Date.now()}`,
            type: "polygon",
            color: "#22D3EE", // Cyan AI Mask
            label: "AI Pneumothorax Suggestion",
            createdAt: new Date().toISOString(),
            points: [
              { x: 0.20 + (Math.random() * 0.04 - 0.02), y: 0.22 },
              { x: 0.31 + (Math.random() * 0.04 - 0.02), y: 0.21 },
              { x: 0.37 + (Math.random() * 0.04 - 0.02), y: 0.29 },
              { x: 0.35 + (Math.random() * 0.04 - 0.02), y: 0.40 },
              { x: 0.27 + (Math.random() * 0.04 - 0.02), y: 0.45 },
              { x: 0.18 + (Math.random() * 0.04 - 0.02), y: 0.39 },
              { x: 0.15 + (Math.random() * 0.04 - 0.02), y: 0.30 },
            ],
          },
        ]
      : [];

    // Reference metrics: ONLY provided if an expert ground-truth reference mask exists for this image
    let referenceMetrics: ReferenceAgreementMetrics | null = null;
    if (options.hasReferenceMask && isDetected) {
      referenceMetrics = {
        dice: Number((0.85 + Math.random() * 0.08).toFixed(3)),
        iou: Number((0.74 + Math.random() * 0.10).toFixed(3)),
        precision: Number((0.88 + Math.random() * 0.08).toFixed(3)),
        recall: Number((0.83 + Math.random() * 0.10).toFixed(3)),
        referenceName: "Expert Reference Mask",
      };
    }

    const totalTimeMs = Math.round(performance.now() - startTime);

    const annotation: Annotation = {
      id: `ann-ai-${Date.now()}`,
      imageId,
      source: "ai",
      shapes,
      originalAiShapes: shapes, // Saved to restore upon cancel
      status: "draft",
      finding,
      threshold,
      maximumOutputScore,
      maskCoveragePercent,
      processingTimeMs: totalTimeMs,
      referenceMetrics,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    return {
      annotation,
      inferenceTimeMs: totalTimeMs,
      finding,
      maximumOutputScore,
      maskCoveragePercent,
      regionCount: shapes.length,
      referenceMetrics,
    };
  },
};
