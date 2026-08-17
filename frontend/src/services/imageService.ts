import { MedicalImage, AnnotationStatus } from "../types";
import { getStoredImages, STORAGE_KEYS, generateSyntheticXrayDataUrl } from "../utils/storage";
import {
  storeImageBlob,
  getFreshObjectUrlForImage,
  deleteImageBlob,
} from "../utils/indexedDb";
import { calculateFileSha256, normalizeImageName } from "../utils/crypto";

export interface UploadImagePayload {
  file: File;
  name: string;
  modality: "xray" | "ct" | "mri";
  notes?: string;
  precalculatedHash?: string;
}

export const imageService = {
  /**
   * Retrieves all medical images with fresh object URLs hydrated from IndexedDB.
   */
  async getImages(): Promise<MedicalImage[]> {
    const rawImages = getStoredImages();

    // Hydrate preview URLs asynchronously from IndexedDB for uploaded files
    const hydrated = await Promise.all(
      rawImages.map(async (img) => {
        // Seeded demo images with permanent assets
        if (img.id === "img-ptx-014") return { ...img, previewUrl: "/demo/ptx-014.png" };
        if (img.id === "img-ptx-017") return { ...img, previewUrl: "/demo/ptx-017.png" };
        if (img.id === "img-ptx-067") return { ...img, previewUrl: "/demo/ptx-067.png" };

        // For user uploads, attempt to retrieve fresh Object URL from IndexedDB
        try {
          const freshUrl = await getFreshObjectUrlForImage(img.id);
          if (freshUrl) {
            return { ...img, previewUrl: freshUrl };
          }
        } catch (err) {
          console.warn(`Failed to hydrate fresh object URL for ${img.id}:`, err);
        }

        // If previewUrl is an expired blob: URL or undefined, provide fallback
        if (!img.previewUrl || img.previewUrl.startsWith("blob:")) {
          return {
            ...img,
            previewUrl: generateSyntheticXrayDataUrl(img.id, img.name),
          };
        }

        return img;
      })
    );

    return hydrated;
  },

  async getImageById(id: string): Promise<MedicalImage | null> {
    const images = await this.getImages();
    return images.find((img) => img.id === id) || null;
  },

  /**
   * Checks if an image with the exact same content hash already exists.
   */
  async checkExactDuplicate(file: Blob, precalculatedHash?: string): Promise<MedicalImage | null> {
    const hash = precalculatedHash || (await calculateFileSha256(file));
    const images = getStoredImages();
    return images.find((img) => img.contentHash === hash) || null;
  },

  /**
   * Checks if an image with the same normalized name already exists.
   */
  async checkNameCollision(name: string): Promise<MedicalImage | null> {
    const norm = normalizeImageName(name);
    const images = getStoredImages();
    return images.find((img) => normalizeImageName(img.name) === norm) || null;
  },

  async uploadImage(payload: UploadImagePayload): Promise<MedicalImage> {
    const isDicom = payload.file.name.toLowerCase().endsWith(".dcm");
    const newId = `img-${Date.now().toString().slice(-6)}`;

    // 1. Calculate file content hash
    const contentHash = payload.precalculatedHash || (await calculateFileSha256(payload.file));

    // 2. Persist binary blob to IndexedDB (for non-DICOM or all formats)
    await storeImageBlob(newId, payload.file, contentHash, payload.file.type || "image/png");

    // 3. Obtain a live Object URL for this session
    let previewUrl: string;
    if (isDicom) {
      previewUrl = generateSyntheticXrayDataUrl(newId, `DICOM: ${payload.name.slice(0, 10)}`);
    } else {
      const fresh = await getFreshObjectUrlForImage(newId);
      previewUrl = fresh || URL.createObjectURL(payload.file);
    }

    const fileSizeFormatted = `${(payload.file.size / (1024 * 1024)).toFixed(2)} MB`;

    const newImage: MedicalImage = {
      id: newId,
      name: payload.name || payload.file.name,
      modality: payload.modality,
      fileType: isDicom ? "dicom" : (payload.file.type.includes("png") ? "png" : "jpeg"),
      previewUrl,
      contentHash,
      originalFilename: payload.file.name,
      uploadedAt: new Date().toISOString(),
      status: "unannotated",
      notes: payload.notes || "",
      width: 512,
      height: 512,
      annotator: "Unassigned",
      fileSizeFormatted,
      hasReferenceMask: false,
      dicomMetadata: isDicom ? {
        patientId: `PT-${Math.floor(100000 + Math.random() * 900000)}`,
        studyDate: new Date().toISOString().slice(0, 10),
        viewPosition: "PA ERECT",
        photometricInterpretation: "MONOCHROME2",
        pixelSpacing: "0.143 mm",
        kvp: "120 kV",
        institution: "Uploaded Research Workstation",
      } : undefined
    };

    const images = getStoredImages();
    const updated = [newImage, ...images];
    localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(updated));

    return newImage;
  },

  async reattachImageBlob(imageId: string, file: File): Promise<string> {
    const contentHash = await calculateFileSha256(file);
    await storeImageBlob(imageId, file, contentHash, file.type);
    const freshUrl = await getFreshObjectUrlForImage(imageId);
    return freshUrl || URL.createObjectURL(file);
  },

  async updateImageStatus(id: string, status: AnnotationStatus, annotator?: string): Promise<void> {
    const images = getStoredImages();
    const index = images.findIndex((img) => img.id === id);
    if (index !== -1) {
      images[index].status = status;
      if (annotator) {
        images[index].annotator = annotator;
      }
      localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(images));
    }
  },

  async deleteImage(id: string): Promise<void> {
    await deleteImageBlob(id);
    const images = getStoredImages();
    const updated = images.filter((img) => img.id !== id);
    localStorage.setItem(STORAGE_KEYS.IMAGES, JSON.stringify(updated));
  }
};

