export type Modality = "xray" | "ct" | "mri";

export type AnnotationStatus =
  | "unannotated"
  | "awaiting-review"
  | "finalized";

export type AnnotationSource =
  | "manual"
  | "ai"
  | "ai-edited";

export type WorkspaceMode =
  | "view"
  | "manual-edit"
  | "ai-processing"
  | "ai-review"
  | "ai-edit"
  | "finalized";

export type ShapeType = "polygon" | "brush-stroke" | "bbox" | "eraser-stroke";

export interface NormalizedPoint {
  x: number; // 0 to 1 relative to image width
  y: number; // 0 to 1 relative to image height
}

export interface AnnotationShape {
  id: string;
  type: ShapeType;
  points: NormalizedPoint[];
  brushRadius?: number;
  color?: string;
  label?: string;
  createdAt: string;
}

export interface ReferenceAgreementMetrics {
  dice: number;
  iou: number;
  precision: number;
  recall: number;
  referenceName?: string;
}

export interface AiSuggestionResult {
  finding: "possible-region-detected" | "no-region-detected" | "unavailable";
  regionCount: number;
  maskCoveragePercent: number;
  maximumOutputScore?: number;
  threshold?: number;
  processingTimeMs?: number;
  referenceMetrics?: ReferenceAgreementMetrics | null;
}

export interface MedicalImage {
  id: string;
  name: string;
  modality: Modality;
  fileType: "png" | "jpeg" | "dicom";
  previewUrl?: string;
  contentHash?: string;
  originalFilename?: string;
  uploadedAt: string;
  status: AnnotationStatus;
  notes?: string;
  width: number;
  height: number;
  annotator?: string;
  fileSizeFormatted?: string;
  referenceMaskUrl?: string;
  hasReferenceMask?: boolean;
  dicomMetadata?: {
    patientId?: string;
    studyDate?: string;
    viewPosition?: string;
    photometricInterpretation?: string;
    pixelSpacing?: string;
    kvp?: string;
    institution?: string;
  };
}

export interface Annotation {
  id: string;
  imageId: string;
  source: AnnotationSource;
  shapes: AnnotationShape[];
  originalAiShapes?: AnnotationShape[]; // Preserved during editing so Cancel can restore
  status: "draft" | "reviewed" | "rejected";
  finding?: "possible-region-detected" | "no-region-detected" | "unavailable";
  threshold?: number;
  maximumOutputScore?: number;
  maskCoveragePercent?: number;
  processingTimeMs?: number;
  referenceMetrics?: ReferenceAgreementMetrics | null;
  createdAt: string;
  updatedAt: string;
  reviewerName?: string;
  reviewerNotes?: string;
  rejectionReason?: string;
}

export interface AuditEvent {
  id: string;
  imageId: string;
  imageName: string;
  action: string;
  actor: string;
  timestamp: string;
  source?: AnnotationSource;
  previousStatus?: AnnotationStatus;
  newStatus?: AnnotationStatus;
  notes?: string;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  institution?: string;
}

export interface UserSettings {
  defaultModality: Modality;
  defaultOverlayOpacity: number;
  defaultAnnotationColor: string;
  enableKeyboardShortcuts: boolean;
  theme: "light" | "dark" | "system";
}

export interface FilterState {
  search: string;
  status: "all" | AnnotationStatus;
  modality: "all" | Modality;
  sortBy: "uploadedAt" | "name" | "status";
  sortOrder: "asc" | "desc";
  viewMode: "table" | "grid";
}
