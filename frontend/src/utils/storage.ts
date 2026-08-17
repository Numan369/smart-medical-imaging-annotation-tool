import { MedicalImage, Annotation, AuditEvent, UserProfile, UserSettings, AnnotationStatus } from "../types";
import { normalizeImageName } from "./crypto";

const CURRENT_SCHEMA_VERSION = "2.2.0";
const STORAGE_PREFIX = "smart_med_annotator_v2_";

export const STORAGE_KEYS = {
  USER: `${STORAGE_PREFIX}user`,
  SETTINGS: `${STORAGE_PREFIX}settings`,
  IMAGES: `${STORAGE_PREFIX}images`,
  ANNOTATIONS: `${STORAGE_PREFIX}annotations`,
  AUDIT_LOGS: `${STORAGE_PREFIX}audit_logs`,
  VERSION: `${STORAGE_PREFIX}schema_version`,
};

/**
 * Creates an SVG Data URL representing a neutral radiograph preview placeholder.
 */
export function generateSyntheticXrayDataUrl(_id: string, label: string): string {
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
  <defs>
    <radialGradient id="lungGradientL" cx="35%" cy="45%" r="35%">
      <stop offset="0%" stop-color="#0a0e14" />
      <stop offset="60%" stop-color="#141c26" />
      <stop offset="100%" stop-color="#3b4859" />
    </radialGradient>
    <radialGradient id="lungGradientR" cx="65%" cy="45%" r="35%">
      <stop offset="0%" stop-color="#0a0e14" />
      <stop offset="60%" stop-color="#141c26" />
      <stop offset="100%" stop-color="#3b4859" />
    </radialGradient>
    <linearGradient id="spine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#3d4957" />
      <stop offset="50%" stop-color="#606f82" />
      <stop offset="100%" stop-color="#3d4957" />
    </linearGradient>
  </defs>
  
  <!-- Thoracic background -->
  <rect width="512" height="512" fill="#080c10" />
  
  <!-- Ribcage outline -->
  <ellipse cx="256" cy="270" rx="195" ry="190" fill="#1b242e" stroke="#475569" stroke-width="4" />
  
  <!-- Left Lung Field -->
  <path d="M 140 130 C 170 120, 220 160, 225 290 C 228 370, 190 410, 130 405 C 85 400, 80 300, 85 220 C 90 160, 115 135, 140 130 Z" fill="url(#lungGradientL)" />
  
  <!-- Right Lung Field -->
  <path d="M 372 130 C 342 120, 292 160, 287 290 C 284 370, 322 410, 382 405 C 427 400, 432 300, 427 220 C 422 160, 397 135, 372 130 Z" fill="url(#lungGradientR)" />
  
  <!-- Mediastinum / Heart Silhouette -->
  <path d="M 230 180 C 245 180, 256 200, 256 250 C 256 310, 210 390, 275 400 C 285 390, 290 320, 275 250 C 265 200, 255 180, 240 180 Z" fill="#475569" opacity="0.85" />
  
  <!-- Clavicles -->
  <path d="M 100 135 Q 170 145 240 165" stroke="#718096" stroke-width="8" stroke-linecap="round" fill="none" opacity="0.9" />
  <path d="M 412 135 Q 342 145 272 165" stroke="#718096" stroke-width="8" stroke-linecap="round" fill="none" opacity="0.9" />
  
  <!-- Ribs (Bilateral arches) -->
  <path d="M 105 180 Q 170 210 235 220" stroke="#4a5568" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.75" />
  <path d="M 407 180 Q 342 210 277 220" stroke="#4a5568" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.75" />
  <path d="M 95 230 Q 170 260 235 270" stroke="#4a5568" stroke-width="5.5" stroke-linecap="round" fill="none" opacity="0.7" />
  <path d="M 417 230 Q 342 260 277 270" stroke="#4a5568" stroke-width="5.5" stroke-linecap="round" fill="none" opacity="0.7" />
  <path d="M 90 285 Q 170 315 235 320" stroke="#4a5568" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.65" />
  <path d="M 422 285 Q 342 315 277 320" stroke="#4a5568" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.65" />
  <path d="M 92 340 Q 170 365 235 370" stroke="#4a5568" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.6" />
  <path d="M 420 340 Q 342 365 277 370" stroke="#4a5568" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.6" />
  
  <!-- Spine column -->
  <rect x="246" y="110" width="20" height="320" fill="url(#spine)" opacity="0.6" rx="3" />
  
  <!-- Marker / Label -->
  <rect x="20" y="20" width="130" height="40" rx="4" fill="#030712" opacity="0.8" />
  <text x="30" y="44" fill="#94a3b8" font-family="monospace" font-size="12" font-weight="bold">${label}</text>
  <text x="470" y="45" fill="#64748b" font-family="sans-serif" font-size="18" font-weight="bold">R</text>
</svg>
`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export const INITIAL_USER: UserProfile = {
  id: "usr-demo-01",
  name: "Dr. Sarah Jenkins, MD",
  email: "s.jenkins@radiology.hospital.org",
  role: "Consultant Radiologist",
  institution: "St. Jude University Medical Center",
};

export const INITIAL_SETTINGS: UserSettings = {
  defaultModality: "xray",
  defaultOverlayOpacity: 0.55,
  defaultAnnotationColor: "#F59E0B",
  enableKeyboardShortcuts: true,
  theme: "light",
};

/**
 * 3 Initial Demo Images:
 * 1. PTX-014-XR (awaiting-review) -> 1
 * 2. PTX-017-XR (finalized)       -> 1
 * 3. PTX-067-XR (unannotated)     -> 1
 * Total = 3
 */
export const SEEDED_STUDIES: MedicalImage[] = [
  {
    id: "img-ptx-014",
    name: "PTX-014-XR.png",
    modality: "xray",
    fileType: "png",
    previewUrl: "/demo/ptx-014.png",
    uploadedAt: new Date(Date.now() - 3600000 * 24 * 2).toISOString(),
    status: "awaiting-review",
    notes: "Right apical visceral pleural line visible on erect CXR. Patient presenting with acute pleuritic chest pain.",
    width: 512,
    height: 512,
    annotator: "Awaiting human review",
    fileSizeFormatted: "2.4 MB",
    hasReferenceMask: false,
    dicomMetadata: {
      patientId: "PT-772910",
      studyDate: "2026-08-11",
      viewPosition: "PA ERECT",
      photometricInterpretation: "MONOCHROME2",
      pixelSpacing: "0.143 mm",
      kvp: "120 kV",
      institution: "St. Jude University Medical Center",
    }
  },
  {
    id: "img-ptx-017",
    name: "PTX-017-XR.png",
    modality: "xray",
    fileType: "png",
    previewUrl: "/demo/ptx-017.png",
    uploadedAt: new Date(Date.now() - 3600000 * 24 * 3).toISOString(),
    status: "finalized",
    notes: "Left lateral pneumothorax with peripheral lung parenchymal margin displacement. Expert finalized.",
    width: 512,
    height: 512,
    annotator: "Dr. Sarah Jenkins",
    fileSizeFormatted: "2.1 MB",
    hasReferenceMask: true,
    dicomMetadata: {
      patientId: "PT-881923",
      studyDate: "2026-08-12",
      viewPosition: "AP SUPINE",
      photometricInterpretation: "MONOCHROME2",
      pixelSpacing: "0.143 mm",
      kvp: "115 kV",
      institution: "St. Jude University Medical Center",
    }
  },
  {
    id: "img-ptx-067",
    name: "PTX-067-XR.png",
    modality: "xray",
    fileType: "png",
    previewUrl: "/demo/ptx-067.png",
    uploadedAt: new Date(Date.now() - 3600000 * 4).toISOString(),
    status: "unannotated",
    notes: "Newly uploaded image. Ready for manual segmentation or AI suggestion request.",
    width: 512,
    height: 512,
    annotator: "Unassigned",
    fileSizeFormatted: "2.8 MB",
    hasReferenceMask: false,
    dicomMetadata: {
      patientId: "PT-918234",
      studyDate: "2026-08-15",
      viewPosition: "PA ERECT",
      photometricInterpretation: "MONOCHROME2",
      pixelSpacing: "0.143 mm",
      kvp: "120 kV",
      institution: "St. Jude University Medical Center",
    }
  }
];

export const SEEDED_ANNOTATIONS: Record<string, Annotation> = {
  "img-ptx-014": {
    id: "ann-014",
    imageId: "img-ptx-014",
    source: "ai",
    status: "draft",
    finding: "possible-region-detected",
    threshold: 0.35,
    maximumOutputScore: 0.884,
    maskCoveragePercent: 4.12,
    processingTimeMs: 1140,
    referenceMetrics: null, // No reference mask exists for this image -> Reference-based metrics: Not available
    createdAt: new Date(Date.now() - 3600000 * 20).toISOString(),
    updatedAt: new Date(Date.now() - 3600000 * 20).toISOString(),
    shapes: [
      {
        id: "shp-014-1",
        type: "polygon",
        color: "#22D3EE",
        label: "AI Pneumothorax Suggestion (Right Apical)",
        createdAt: new Date(Date.now() - 3600000 * 20).toISOString(),
        points: [
          { x: 0.19, y: 0.23 },
          { x: 0.28, y: 0.22 },
          { x: 0.36, y: 0.28 },
          { x: 0.35, y: 0.38 },
          { x: 0.28, y: 0.44 },
          { x: 0.20, y: 0.41 },
          { x: 0.16, y: 0.33 },
        ]
      }
    ]
  },
  "img-ptx-017": {
    id: "ann-017",
    imageId: "img-ptx-017",
    source: "ai-edited",
    status: "reviewed",
    finding: "possible-region-detected",
    threshold: 0.35,
    maximumOutputScore: 0.912,
    maskCoveragePercent: 5.68,
    processingTimeMs: 1220,
    referenceMetrics: {
      dice: 0.894,
      iou: 0.808,
      precision: 0.915,
      recall: 0.874,
      referenceName: "Senior Expert Ground Truth"
    },
    reviewerName: "Dr. Sarah Jenkins",
    reviewerNotes: "Adjusted lateral boundary to encompass subtle apical margin.",
    createdAt: new Date(Date.now() - 3600000 * 30).toISOString(),
    updatedAt: new Date(Date.now() - 3600000 * 28).toISOString(),
    shapes: [
      {
        id: "shp-017-1",
        type: "polygon",
        color: "#16A34A",
        label: "Expert Confirmed Pneumothorax",
        createdAt: new Date(Date.now() - 3600000 * 28).toISOString(),
        points: [
          { x: 0.68, y: 0.21 },
          { x: 0.79, y: 0.23 },
          { x: 0.85, y: 0.32 },
          { x: 0.83, y: 0.45 },
          { x: 0.75, y: 0.48 },
          { x: 0.69, y: 0.40 },
          { x: 0.65, y: 0.29 },
        ]
      }
    ]
  }
};

export const SEEDED_AUDIT_LOGS: AuditEvent[] = [
  {
    id: "evt-001",
    imageId: "img-ptx-017",
    imageName: "PTX-017-XR.png",
    action: "AI suggestion edited and saved",
    actor: "Dr. Sarah Jenkins",
    timestamp: new Date(Date.now() - 3600000 * 28).toISOString(),
    source: "ai-edited",
    previousStatus: "awaiting-review",
    newStatus: "finalized",
    notes: "AI suggestion adjusted and finalized as expert annotation."
  },
  {
    id: "evt-002",
    imageId: "img-ptx-014",
    imageName: "PTX-014-XR.png",
    action: "AI suggestion generated",
    actor: "AI Assistance",
    timestamp: new Date(Date.now() - 3600000 * 20).toISOString(),
    source: "ai",
    previousStatus: "unannotated",
    newStatus: "awaiting-review",
    notes: "Possible pneumothorax region detected. Awaiting human review."
  },
  {
    id: "evt-003",
    imageId: "img-ptx-067",
    imageName: "PTX-067-XR.png",
    action: "Image added",
    actor: "Dr. Sarah Jenkins",
    timestamp: new Date(Date.now() - 3600000 * 4).toISOString(),
    previousStatus: undefined,
    newStatus: "unannotated",
    notes: "Direct image upload."
  }
];

/**
 * Normalizes any legacy or missing status string to the canonical 3-status model:
 * 'unannotated' | 'awaiting-review' | 'finalized'
 */
export function normalizeStatus(rawStatus: any): AnnotationStatus {
  if (!rawStatus) return "unannotated";
  const st = String(rawStatus).toLowerCase().trim();
  if (st === "awaiting-review" || st === "ai-suggested" || st === "needs-review" || st === "ai_suggested" || st === "needs_review") {
    return "awaiting-review";
  }
  if (st === "finalized" || st === "expert-annotated" || st === "reviewed" || st === "expert_annotated") {
    return "finalized";
  }
  return "unannotated";
}

/**
 * Safe JSON parser with fallback.
 */
export function safeParseJson<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw);
  } catch (err) {
    console.warn("Storage JSON parsing failed, falling back to default.", err);
    return fallback;
  }
}

export interface DuplicateAnalysisGroup {
  key: string;
  items: MedicalImage[];
  best: MedicalImage;
  duplicates: MedicalImage[];
}

export interface DuplicateAnalysisResult {
  groups: DuplicateAnalysisGroup[];
  totalDuplicates: number;
}

export interface DeduplicationReport {
  scannedCount: number;
  duplicateCount: number;
  removedCount: number;
  keptImages: MedicalImage[];
  removedImageIds: string[];
}

/**
 * Analyzes current stored images and detects exact duplicates based on:
 * - Content hash (if available)
 * - Normalized filename + dimensions + modality + size
 */
export function analyzeDuplicateImages(images: MedicalImage[]): DuplicateAnalysisResult {
  const map = new Map<string, MedicalImage[]>();

  for (const img of images) {
    const normName = normalizeImageName(img.name);
    const key = img.contentHash
      ? `hash:${img.contentHash}`
      : `meta:${normName}|${img.modality}|${img.width}x${img.height}|${img.fileSizeFormatted || ""}`;

    if (!map.has(key)) {
      map.set(key, []);
    }
    map.get(key)!.push(img);
  }

  const groups: DuplicateAnalysisGroup[] = [];
  let totalDuplicates = 0;

  const statusWeight = (s: AnnotationStatus) => (s === "finalized" ? 3 : s === "awaiting-review" ? 2 : 1);

  for (const [key, items] of map.entries()) {
    if (items.length > 1) {
      // Sort priority:
      // 1. Status (finalized > awaiting-review > unannotated)
      // 2. Has reference mask (true > false)
      // 3. Has notes
      // 4. Most recently updated
      const sorted = [...items].sort((a, b) => {
        const diff = statusWeight(b.status) - statusWeight(a.status);
        if (diff !== 0) return diff;
        const timeA = new Date(a.uploadedAt).getTime();
        const timeB = new Date(b.uploadedAt).getTime();
        return timeB - timeA;
      });

      const best = sorted[0];
      const duplicates = sorted.slice(1);
      groups.push({ key, items: sorted, best, duplicates });
      totalDuplicates += duplicates.length;
    }
  }

  return { groups, totalDuplicates };
}

/**
 * Safely deduplicates stored images and cleans up associated annotations and audit logs.
 */
export function deduplicateImagesAndStorage(): DeduplicationReport {
  const rawImages = localStorage.getItem(STORAGE_KEYS.IMAGES);
  const images = safeParseJson<MedicalImage[]>(rawImages, SEEDED_STUDIES);
  const { groups, totalDuplicates } = analyzeDuplicateImages(images);

  if (totalDuplicates === 0) {
    return {
      scannedCount: images.length,
      duplicateCount: 0,
      removedCount: 0,
      keptImages: images,
      removedImageIds: [],
    };
  }

  const removedIds = new Set<string>();
  const duplicateToBestMap = new Map<string, MedicalImage>();

  for (const group of groups) {
    for (const dup of group.duplicates) {
      removedIds.add(dup.id);
      duplicateToBestMap.set(dup.id, group.best);
    }
  }

  const keptImages = images.filter((img) => !removedIds.has(img.id));
  localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(keptImages));

  // Merge & clean annotations
  const annotations = getStoredAnnotations();
  let annotationsModified = false;

  for (const [dupId, best] of duplicateToBestMap.entries()) {
    const dupAnn = annotations[dupId];
    if (dupAnn) {
      if (!annotations[best.id]) {
        annotations[best.id] = { ...dupAnn, imageId: best.id };
        annotationsModified = true;
      }
      delete annotations[dupId];
      annotationsModified = true;
    }
  }

  if (annotationsModified) {
    localStorage.setItem(STORAGE_KEYS.ANNOTATIONS, JSON.stringify(annotations));
  }

  // Clean and re-point audit logs
  const rawLogs = getStoredAuditLogs();
  const cleanedLogs: AuditEvent[] = [];
  const seenSignatures = new Set<string>();

  for (const log of rawLogs) {
    const targetImageId = duplicateToBestMap.has(log.imageId)
      ? duplicateToBestMap.get(log.imageId)!.id
      : log.imageId;

    if (removedIds.has(log.imageId) && !duplicateToBestMap.has(log.imageId)) {
      continue;
    }

    const sig = `${targetImageId}|${log.action}|${log.actor}|${log.newStatus || ""}`;
    if (!seenSignatures.has(sig)) {
      seenSignatures.add(sig);
      cleanedLogs.push({
        ...log,
        imageId: targetImageId,
      });
    }
  }

  localStorage.setItem(STORAGE_KEYS.AUDIT_LOGS, JSON.stringify(cleanedLogs));

  return {
    scannedCount: images.length,
    duplicateCount: totalDuplicates,
    removedCount: removedIds.size,
    keptImages,
    removedImageIds: Array.from(removedIds),
  };
}

/**
 * Run migration once per schema version.
 */
function runStorageMigrationOnce(): void {
  try {
    const version = localStorage.getItem(STORAGE_KEYS.VERSION);
    if (version === CURRENT_SCHEMA_VERSION) {
      return;
    }

    // 1. Migrate & deduplicate stored images
    const currentRaw = localStorage.getItem(STORAGE_KEYS.IMAGES);
    if (currentRaw) {
      const currentList = safeParseJson<any[]>(currentRaw, []);
      if (Array.isArray(currentList) && currentList.length > 0) {
        const sanitized: MedicalImage[] = currentList.map((img: any, idx: number) => ({
          id: String(img.id || `img-sanitized-${idx}`),
          name: String(img.name || `Image-${idx + 1}.png`),
          modality: img.modality === "ct" || img.modality === "mri" ? img.modality : "xray",
          fileType: img.fileType === "dicom" || img.fileType === "jpeg" ? img.fileType : "png",
          previewUrl: img.previewUrl || (
            img.id === "img-ptx-014" ? "/demo/ptx-014.png" :
            img.id === "img-ptx-017" ? "/demo/ptx-017.png" :
            img.id === "img-ptx-067" ? "/demo/ptx-067.png" :
            generateSyntheticXrayDataUrl(img.id, img.name)
          ),
          contentHash: img.contentHash,
          originalFilename: img.originalFilename || img.name,
          uploadedAt: img.uploadedAt || new Date().toISOString(),
          status: normalizeStatus(img.status),
          notes: img.notes || "",
          width: Number(img.width) || 512,
          height: Number(img.height) || 512,
          annotator: img.status === "finalized"
            ? (img.annotator?.replace(/V3C/g, "").trim() || "Dr. Sarah Jenkins")
            : img.status === "awaiting-review"
            ? "Awaiting human review"
            : "Unassigned",
          fileSizeFormatted: img.fileSizeFormatted || "2.4 MB",
          hasReferenceMask: !!img.hasReferenceMask,
          dicomMetadata: img.dicomMetadata,
        }));

        localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(sanitized));
      }
    }

    // 2. Perform deduplication migration
    deduplicateImagesAndStorage();

    // Mark migration completed for current version
    localStorage.setItem(STORAGE_KEYS.VERSION, CURRENT_SCHEMA_VERSION);
  } catch (err) {
    console.error("Storage migration failed safely:", err);
  }
}

export function getStoredUser(): UserProfile {
  try {
    const item = localStorage.getItem(STORAGE_KEYS.USER);
    if (!item) {
      localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(INITIAL_USER));
      return INITIAL_USER;
    }
    return safeParseJson<UserProfile>(item, INITIAL_USER);
  } catch {
    return INITIAL_USER;
  }
}

export function getStoredSettings(): UserSettings {
  try {
    const item = localStorage.getItem(STORAGE_KEYS.SETTINGS);
    if (!item) {
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(INITIAL_SETTINGS));
      return INITIAL_SETTINGS;
    }
    return safeParseJson<UserSettings>(item, INITIAL_SETTINGS);
  } catch {
    return INITIAL_SETTINGS;
  }
}

export function getStoredImages(): MedicalImage[] {
  runStorageMigrationOnce();
  try {
    const item = localStorage.getItem(STORAGE_KEYS.IMAGES);
    if (!item) {
      localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(SEEDED_STUDIES));
      return SEEDED_STUDIES;
    }
    const parsed = safeParseJson<any[]>(item, SEEDED_STUDIES);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(SEEDED_STUDIES));
      return SEEDED_STUDIES;
    }

    // Defensive normalization
    return parsed.map((img: any, idx: number) => ({
      id: String(img.id || `img-safe-${idx}`),
      name: String(img.name || `Image-${idx + 1}.png`),
      modality: img.modality === "ct" || img.modality === "mri" ? img.modality : "xray",
      fileType: img.fileType === "dicom" || img.fileType === "jpeg" ? img.fileType : "png",
      previewUrl: img.previewUrl || (
        img.id === "img-ptx-014" ? "/demo/ptx-014.png" :
        img.id === "img-ptx-017" ? "/demo/ptx-017.png" :
        img.id === "img-ptx-067" ? "/demo/ptx-067.png" :
        generateSyntheticXrayDataUrl(img.id, img.name)
      ),
      contentHash: img.contentHash,
      originalFilename: img.originalFilename || img.name,
      uploadedAt: img.uploadedAt || new Date().toISOString(),
      status: normalizeStatus(img.status),
      notes: img.notes || "",
      width: Number(img.width) || 512,
      height: Number(img.height) || 512,
      annotator: normalizeStatus(img.status) === "finalized"
        ? (img.annotator || "Dr. Sarah Jenkins")
        : normalizeStatus(img.status) === "awaiting-review"
        ? "Awaiting human review"
        : "Unassigned",
      fileSizeFormatted: img.fileSizeFormatted || "2.4 MB",
      hasReferenceMask: !!img.hasReferenceMask,
      dicomMetadata: img.dicomMetadata,
    }));
  } catch (err) {
    console.error("Failed to read stored images, using default seeds:", err);
    return SEEDED_STUDIES;
  }
}

export function getStoredAnnotations(): Record<string, Annotation> {
  try {
    const item = localStorage.getItem(STORAGE_KEYS.ANNOTATIONS);
    if (!item) {
      localStorage.setItem(STORAGE_KEYS.ANNOTATIONS, JSON.stringify(SEEDED_ANNOTATIONS));
      return SEEDED_ANNOTATIONS;
    }
    const parsed = safeParseJson<Record<string, Annotation>>(item, SEEDED_ANNOTATIONS);
    if (!parsed || typeof parsed !== "object") {
      return SEEDED_ANNOTATIONS;
    }
    Object.keys(parsed).forEach((key) => {
      if (!parsed[key].shapes || !Array.isArray(parsed[key].shapes)) {
        parsed[key].shapes = [];
      }
    });
    return parsed;
  } catch {
    return SEEDED_ANNOTATIONS;
  }
}

export function getStoredAuditLogs(): AuditEvent[] {
  try {
    const item = localStorage.getItem(STORAGE_KEYS.AUDIT_LOGS);
    if (!item) {
      localStorage.setItem(STORAGE_KEYS.AUDIT_LOGS, JSON.stringify(SEEDED_AUDIT_LOGS));
      return SEEDED_AUDIT_LOGS;
    }
    const parsed = safeParseJson<AuditEvent[]>(item, SEEDED_AUDIT_LOGS);
    if (!Array.isArray(parsed) || parsed.length === 0) {
      return SEEDED_AUDIT_LOGS;
    }
    return parsed;
  } catch {
    return SEEDED_AUDIT_LOGS;
  }
}

export function resetDemoStorage(): void {
  try {
    localStorage.removeItem("smart_med_annotator_images");
    localStorage.removeItem("smart_med_annotator_annotations");
    localStorage.removeItem("smart_med_annotator_audit_logs");
    localStorage.removeItem("smart_med_session_token");
    
    localStorage.setItem(STORAGE_KEYS.VERSION, CURRENT_SCHEMA_VERSION);
    localStorage.setItem(STORAGE_KEYS.USER, JSON.stringify(INITIAL_USER));
    localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(INITIAL_SETTINGS));
    localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(SEEDED_STUDIES));
    localStorage.setItem(STORAGE_KEYS.ANNOTATIONS, JSON.stringify(SEEDED_ANNOTATIONS));
    localStorage.setItem(STORAGE_KEYS.AUDIT_LOGS, JSON.stringify(SEEDED_AUDIT_LOGS));
  } catch (err) {
    console.error("Storage reset failed:", err);
  }
}

