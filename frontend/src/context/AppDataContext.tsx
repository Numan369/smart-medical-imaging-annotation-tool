import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from "react";
import {
  MedicalImage,
  Annotation,
  AuditEvent,
  UserSettings,
  FilterState,
  AnnotationShape,
  AnnotationStatus,
} from "../types";
import { imageService, UploadImagePayload } from "../services/imageService";
import { annotationService } from "../services/annotationService";
import {
  getStoredSettings,
  resetDemoStorage,
  STORAGE_KEYS,
} from "../utils/storage";

interface AppDataContextType {
  images: MedicalImage[];
  isLoadingImages: boolean;
  annotations: Record<string, Annotation>;
  auditLogs: AuditEvent[];
  settings: UserSettings;
  filters: FilterState;
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  refreshImages: () => Promise<void>;
  uploadImage: (payload: UploadImagePayload) => Promise<MedicalImage>;
  deleteImage: (id: string) => Promise<void>;
  getAnnotation: (imageId: string) => Promise<Annotation | null>;
  saveAnnotation: (
    imageId: string,
    shapes: AnnotationShape[],
    source: Annotation["source"],
    status: Annotation["status"],
    reviewerName?: string,
    notes?: string,
    metadata?: Partial<Annotation>
  ) => Promise<Annotation>;
  discardAnnotation: (imageId: string, reason?: string, actorName?: string) => Promise<void>;
  updateSettings: (newSettings: Partial<UserSettings>) => void;
  resetAllDemoData: () => Promise<void>;
}

const defaultFilters: FilterState = {
  search: "",
  status: "all",
  modality: "all",
  sortBy: "uploadedAt",
  sortOrder: "desc",
  viewMode: "table",
};

const AppDataContext = createContext<AppDataContextType | undefined>(undefined);

export const AppDataProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [images, setImages] = useState<MedicalImage[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(true);
  const [annotations, setAnnotations] = useState<Record<string, Annotation>>({});
  const [auditLogs, setAuditLogs] = useState<AuditEvent[]>([]);
  const [settings, setSettings] = useState<UserSettings>(getStoredSettings());
  const [filters, setFilters] = useState<FilterState>(defaultFilters);

  const refreshImages = useCallback(async () => {
    try {
      const fetchedImages = await imageService.getImages();
      const logs = await annotationService.getAuditLogs();
      setImages(fetchedImages);
      setAuditLogs(logs);
    } catch (err) {
      console.error("Error refreshing data", err);
    } finally {
      setIsLoadingImages(false);
    }
  }, []);

  useEffect(() => {
    refreshImages();
  }, [refreshImages]);

  const uploadImage = useCallback(async (payload: UploadImagePayload): Promise<MedicalImage> => {
    const newImg = await imageService.uploadImage(payload);
    await annotationService.logAuditEvent({
      imageId: newImg.id,
      imageName: newImg.name,
      action: "Image added",
      actor: "Dr. Sarah Jenkins",
      newStatus: "unannotated",
      notes: payload.notes || "New image added to workspace",
    });
    await refreshImages();
    return newImg;
  }, [refreshImages]);

  const deleteImage = useCallback(async (id: string): Promise<void> => {
    const target = images.find((i) => i.id === id);
    await imageService.deleteImage(id);
    if (target) {
      await annotationService.logAuditEvent({
        imageId: target.id,
        imageName: target.name,
        action: "Image removed",
        actor: "Dr. Sarah Jenkins",
        notes: "Deleted by user",
      });
    }
    await refreshImages();
  }, [images, refreshImages]);

  const getAnnotation = useCallback(async (imageId: string): Promise<Annotation | null> => {
    return annotationService.getAnnotationByImageId(imageId);
  }, []);

  const saveAnnotation = useCallback(async (
    imageId: string,
    shapes: AnnotationShape[],
    source: Annotation["source"],
    status: Annotation["status"],
    reviewerName = "Dr. Sarah Jenkins",
    notes?: string,
    metadata?: Partial<Annotation>
  ): Promise<Annotation> => {
    const ann = await annotationService.saveAnnotation(
      imageId,
      shapes,
      source,
      status,
      reviewerName,
      notes,
      metadata
    );

    setAnnotations((prev) => ({ ...prev, [imageId]: ann }));

    // Status transition:
    // If accepted/reviewed manual or AI -> finalized
    // If AI draft awaiting review -> awaiting-review
    let imageStatus: AnnotationStatus = "unannotated";
    if (status === "reviewed") {
      imageStatus = "finalized";
    } else if (source === "ai" || source === "ai-edited") {
      imageStatus = "awaiting-review";
    }

    const targetAnnotator =
      status === "reviewed"
        ? reviewerName
        : imageStatus === "awaiting-review"
        ? "Awaiting human review"
        : "Unassigned";

    await imageService.updateImageStatus(imageId, imageStatus, targetAnnotator);

    const targetImg = images.find((i) => i.id === imageId);
    const actionText =
      status === "reviewed"
        ? source === "ai"
          ? "AI suggestion accepted"
          : source === "ai-edited"
          ? "AI suggestion edited and saved"
          : "Manual annotation saved"
        : source === "ai"
        ? "AI suggestion generated"
        : "Annotation draft saved";

    await annotationService.logAuditEvent({
      imageId,
      imageName: targetImg?.name || imageId,
      action: actionText,
      actor: source === "ai" && status !== "reviewed" ? "AI Assistance" : reviewerName,
      source,
      newStatus: imageStatus,
      notes,
    });

    await refreshImages();
    return ann;
  }, [images, refreshImages]);

  const discardAnnotation = useCallback(async (
    imageId: string,
    reason?: string,
    actorName = "Dr. Sarah Jenkins"
  ): Promise<void> => {
    await annotationService.discardAnnotation(imageId);
    setAnnotations((prev) => {
      const copy = { ...prev };
      delete copy[imageId];
      return copy;
    });

    await imageService.updateImageStatus(imageId, "unannotated", "Unassigned");

    const targetImg = images.find((i) => i.id === imageId);
    await annotationService.logAuditEvent({
      imageId,
      imageName: targetImg?.name || imageId,
      action: "AI suggestion rejected and discarded",
      actor: actorName,
      previousStatus: targetImg?.status,
      newStatus: "unannotated",
      notes: reason ? `Rejection reason: ${reason}` : "AI suggestion discarded by reviewer",
    });

    await refreshImages();
  }, [images, refreshImages]);

  const updateSettings = useCallback((newSettings: Partial<UserSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...newSettings };
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(updated));
      return updated;
    });
  }, []);

  const resetAllDemoData = useCallback(async () => {
    resetDemoStorage();
    setSettings(getStoredSettings());
    await refreshImages();
  }, [refreshImages]);

  const contextValue = useMemo(
    () => ({
      images,
      isLoadingImages,
      annotations,
      auditLogs,
      settings,
      filters,
      setFilters,
      refreshImages,
      uploadImage,
      deleteImage,
      getAnnotation,
      saveAnnotation,
      discardAnnotation,
      updateSettings,
      resetAllDemoData,
    }),
    [
      images,
      isLoadingImages,
      annotations,
      auditLogs,
      settings,
      filters,
      refreshImages,
      uploadImage,
      deleteImage,
      getAnnotation,
      saveAnnotation,
      discardAnnotation,
      updateSettings,
      resetAllDemoData,
    ]
  );

  return (
    <AppDataContext.Provider value={contextValue}>
      {children}
    </AppDataContext.Provider>
  );
};

export function useAppData() {
  const context = useContext(AppDataContext);
  if (!context) {
    throw new Error("useAppData must be used within an AppDataProvider");
  }
  return context;
}
