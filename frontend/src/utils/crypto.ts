/**
 * Cryptographic and string normalization utilities for duplicate image detection.
 */

/**
 * Calculates SHA-256 hash string for an uploaded File or Blob using the browser Web Crypto API.
 */
export async function calculateFileSha256(file: Blob): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
  return hashHex;
}

/**
 * Normalizes a filename or image title for robust comparison:
 * - Trims leading and trailing spaces
 * - Converts to lower case
 * - Collapses consecutive whitespace into a single space
 */
export function normalizeImageName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, " ");
}

/**
 * Generates an auto-incremented unique filename suggestion when a name collision occurs.
 * e.g., 'scan.png' -> 'scan (2).png' -> 'scan (3).png'
 */
export function generateUniqueSuggestedName(baseName: string, existingNames: string[]): string {
  const normalizedExisting = new Set(existingNames.map(normalizeImageName));

  const dotIdx = baseName.lastIndexOf(".");
  const ext = dotIdx !== -1 ? baseName.slice(dotIdx) : "";
  const nameWithoutExt = dotIdx !== -1 ? baseName.slice(0, dotIdx) : baseName;

  // Clean out any existing (N) suffix in base name
  const cleanBase = nameWithoutExt.replace(/\s*\(\d+\)$/, "").trim();

  let counter = 2;
  let candidate = `${cleanBase} (${counter})${ext}`;

  while (normalizedExisting.has(normalizeImageName(candidate))) {
    counter++;
    candidate = `${cleanBase} (${counter})${ext}`;
  }

  return candidate;
}
