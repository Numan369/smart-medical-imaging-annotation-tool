import { Annotation, AnnotationShape, AuditEvent } from "../types";
import { getStoredAnnotations, getStoredAuditLogs, STORAGE_KEYS } from "../utils/storage";

export const annotationService = {
  async getAnnotationByImageId(imageId: string): Promise<Annotation | null> {
    const annotations = getStoredAnnotations();
    return annotations[imageId] || null;
  },

  async saveAnnotation(
    imageId: string,
    shapes: AnnotationShape[],
    source: Annotation["source"],
    status: Annotation["status"],
    reviewerName?: string,
    notes?: string,
    metadata?: Partial<Annotation>
  ): Promise<Annotation> {
    const annotations = getStoredAnnotations();
    const existing = annotations[imageId];

    const newAnnotation: Annotation = {
      id: existing?.id || `ann-${Date.now().toString().slice(-6)}`,
      imageId,
      source,
      shapes,
      originalAiShapes: metadata?.originalAiShapes || existing?.originalAiShapes,
      status,
      finding: metadata?.finding || existing?.finding || "possible-region-detected",
      threshold: metadata?.threshold ?? existing?.threshold ?? 0.35,
      maximumOutputScore: metadata?.maximumOutputScore ?? existing?.maximumOutputScore,
      maskCoveragePercent: metadata?.maskCoveragePercent ?? existing?.maskCoveragePercent,
      processingTimeMs: metadata?.processingTimeMs ?? existing?.processingTimeMs,
      referenceMetrics: metadata?.referenceMetrics !== undefined ? metadata.referenceMetrics : (existing?.referenceMetrics ?? null),
      reviewerName,
      reviewerNotes: notes,
      createdAt: existing?.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      ...metadata,
    };

    annotations[imageId] = newAnnotation;
    localStorage.setItem(STORAGE_KEYS.ANNOTATIONS, JSON.stringify(annotations));
    return newAnnotation;
  },

  async discardAnnotation(imageId: string): Promise<void> {
    const annotations = getStoredAnnotations();
    delete annotations[imageId];
    localStorage.setItem(STORAGE_KEYS.ANNOTATIONS, JSON.stringify(annotations));
  },

  async logAuditEvent(event: Omit<AuditEvent, "id" | "timestamp">): Promise<AuditEvent> {
    const logs = getStoredAuditLogs();
    const newEvent: AuditEvent = {
      ...event,
      id: `evt-${Date.now().toString().slice(-6)}`,
      timestamp: new Date().toISOString(),
    };

    const updated = [newEvent, ...logs];
    localStorage.setItem(STORAGE_KEYS.AUDIT_LOGS, JSON.stringify(updated));
    return newEvent;
  },

  async getAuditLogs(): Promise<AuditEvent[]> {
    return getStoredAuditLogs();
  }
};
