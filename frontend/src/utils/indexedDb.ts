/**
 * Browser-native IndexedDB storage for binary medical image Blobs.
 * Avoids storing ephemeral `blob:http` URLs in localStorage.
 */

export interface StoredImageBlob {
  imageId: string;
  blob: Blob;
  contentHash: string;
  mimeType: string;
  savedAt: string;
}

const DB_NAME = "MediMask_ImageStore";
const DB_VERSION = 1;
const STORE_NAME = "image_blobs";

// In-memory active object URLs to prevent memory leaks while enabling fast display
const activeObjectUrls = new Map<string, string>();

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) {
      reject(new Error("IndexedDB is not supported in this browser environment."));
      return;
    }

    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "imageId" });
        store.createIndex("contentHash", "contentHash", { unique: false });
      }
    };

    request.onsuccess = () => {
      resolve(request.result);
    };

    request.onerror = () => {
      reject(request.error);
    };
  });
}

export async function storeImageBlob(
  imageId: string,
  blob: Blob,
  contentHash: string,
  mimeType = "image/png"
): Promise<void> {
  try {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);

      const record: StoredImageBlob = {
        imageId,
        blob,
        contentHash,
        mimeType,
        savedAt: new Date().toISOString(),
      };

      const putReq = store.put(record);
      putReq.onsuccess = () => resolve();
      putReq.onerror = () => reject(putReq.error);
    });
  } catch (err) {
    console.error(`Failed to store image blob for ${imageId} in IndexedDB:`, err);
  }
}

export async function getImageBlob(imageId: string): Promise<StoredImageBlob | null> {
  try {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);

      const getReq = store.get(imageId);
      getReq.onsuccess = () => {
        resolve(getReq.result || null);
      };
      getReq.onerror = () => reject(getReq.error);
    });
  } catch (err) {
    console.error(`Failed to retrieve image blob for ${imageId}:`, err);
    return null;
  }
}

export async function deleteImageBlob(imageId: string): Promise<void> {
  revokeActiveObjectUrl(imageId);
  try {
    const db = await openDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const delReq = store.delete(imageId);
      delReq.onsuccess = () => resolve();
      delReq.onerror = () => reject(delReq.error);
    });
  } catch (err) {
    console.error(`Failed to delete image blob for ${imageId}:`, err);
  }
}

/**
 * Creates or retrieves a valid Object URL for the stored blob.
 */
export async function getFreshObjectUrlForImage(imageId: string): Promise<string | null> {
  if (activeObjectUrls.has(imageId)) {
    return activeObjectUrls.get(imageId)!;
  }

  const stored = await getImageBlob(imageId);
  if (!stored || !stored.blob) {
    return null;
  }

  const objectUrl = URL.createObjectURL(stored.blob);
  activeObjectUrls.set(imageId, objectUrl);
  return objectUrl;
}

/**
 * Revokes an object URL when no longer needed to prevent memory leaks.
 */
export function revokeActiveObjectUrl(imageId: string): void {
  if (activeObjectUrls.has(imageId)) {
    const url = activeObjectUrls.get(imageId);
    if (url && url.startsWith("blob:")) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // Safe ignore
      }
    }
    activeObjectUrls.delete(imageId);
  }
}

/**
 * Clear all cached object URLs on cleanup.
 */
export function revokeAllObjectUrls(): void {
  activeObjectUrls.forEach((url) => {
    if (url.startsWith("blob:")) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // Safe ignore
      }
    }
  });
  activeObjectUrls.clear();
}
