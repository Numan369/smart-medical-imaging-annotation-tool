import { MedicalImage, Annotation } from "../types";
import { BRAND_NAME, RESEARCH_DISCLAIMER } from "../constants/theme";

export interface ExportPayload {
  exportDate: string;
  application: string;
  disclaimer: string;
  image: {
    id: string;
    name: string;
    modality: string;
    width: number;
    height: number;
  };
  annotation: {
    id: string;
    source: string;
    status: string;
    finding?: string;
    shapesCount: number;
    shapes: Annotation["shapes"];
    maximumOutputScore?: number;
    maskCoveragePercent?: number;
    referenceAgreementMetrics?: Annotation["referenceMetrics"];
    reviewerName?: string;
    reviewerNotes?: string;
    updatedAt: string;
  };
}

export function exportAnnotationAsJson(image: MedicalImage, annotation: Annotation): void {
  const payload: ExportPayload = {
    exportDate: new Date().toISOString(),
    application: BRAND_NAME,
    disclaimer: RESEARCH_DISCLAIMER,
    image: {
      id: image.id,
      name: image.name,
      modality: image.modality,
      width: image.width,
      height: image.height,
    },
    annotation: {
      id: annotation.id,
      source: annotation.source,
      status: annotation.status,
      finding: annotation.finding,
      shapesCount: annotation.shapes.length,
      shapes: annotation.shapes,
      maximumOutputScore: annotation.maximumOutputScore,
      maskCoveragePercent: annotation.maskCoveragePercent,
      referenceAgreementMetrics: annotation.referenceMetrics,
      reviewerName: annotation.reviewerName,
      reviewerNotes: annotation.reviewerNotes,
      updatedAt: annotation.updatedAt,
    },
  };

  const jsonString = JSON.stringify(payload, null, 2);
  const blob = new Blob([jsonString], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const cleanName = image.name.replace(/\.[^/.]+$/, "");
  a.href = url;
  a.download = `${cleanName}_annotation_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
