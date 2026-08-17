import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppData } from "../context/AppDataContext";
import { useToast } from "../context/ToastContext";
import { Dropzone } from "../components/upload/Dropzone";
import { DicomPlaceholder } from "../components/upload/DicomPlaceholder";
import { Button } from "../components/common/Button";
import { Modal } from "../components/common/Modal";
import { PlusCircle, ArrowLeft, FileText, AlertTriangle, Copy, ExternalLink } from "lucide-react";
import { calculateFileSha256, generateUniqueSuggestedName } from "../utils/crypto";
import { imageService } from "../services/imageService";
import { MedicalImage } from "../types";

export const UploadPage: React.FC = () => {
  const { images, uploadImage } = useAppData();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [file, setFile] = useState<File | null>(null);
  const [contentHash, setContentHash] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState("");
  const [modality, setModality] = useState<"xray" | "ct" | "mri">("xray");
  const [notes, setNotes] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDicom, setIsDicom] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [isCheckingHash, setIsCheckingHash] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Field validation touched states
  const [touchedName, setTouchedName] = useState(false);

  // Duplicate Modals State
  const [exactDuplicateMatch, setExactDuplicateMatch] = useState<MedicalImage | null>(null);
  const [nameCollisionMatch, setNameCollisionMatch] = useState<MedicalImage | null>(null);
  const [suggestedRename, setSuggestedRename] = useState<string>("");

  const handleFileSelect = async (selectedFile: File) => {
    setError(null);
    const ext = selectedFile.name.toLowerCase();
    const isDcm = ext.endsWith(".dcm");

    if (selectedFile.size > 25 * 1024 * 1024) {
      setError("File size exceeds maximum 25MB limit.");
      return;
    }

    setFile(selectedFile);
    setDisplayName(selectedFile.name);
    setIsDicom(isDcm);

    if (isDcm) {
      setPreviewUrl(null);
    } else {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
    }

    // Compute hash in background
    setIsCheckingHash(true);
    try {
      const hash = await calculateFileSha256(selectedFile);
      setContentHash(hash);
    } catch (err) {
      console.warn("Could not compute SHA-256 hash:", err);
    } finally {
      setIsCheckingHash(false);
    }
  };

  const isFormValid = !!file && displayName.trim().length > 0 && !!modality && !isCheckingHash;

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouchedName(true);

    if (!file) {
      setError("Please select a medical image file (PNG, JPG, or DICOM).");
      return;
    }
    if (!displayName.trim()) {
      setError("Image name cannot be blank.");
      return;
    }

    // Level B: Exact File Duplicate Check by Hash
    let hash = contentHash;
    if (!hash) {
      setIsCheckingHash(true);
      hash = await calculateFileSha256(file);
      setContentHash(hash);
      setIsCheckingHash(false);
    }

    const exactMatch = await imageService.checkExactDuplicate(file, hash);
    if (exactMatch) {
      setExactDuplicateMatch(exactMatch);
      return;
    }

    // Level A: Unique Image Name Check
    const nameMatch = await imageService.checkNameCollision(displayName.trim());
    if (nameMatch) {
      const existingNames = images.map((i) => i.name);
      const suggested = generateUniqueSuggestedName(displayName.trim(), existingNames);
      setSuggestedRename(suggested);
      setNameCollisionMatch(nameMatch);
      return;
    }

    await performDirectUpload(displayName.trim(), hash);
  };

  const performDirectUpload = async (finalName: string, hash: string) => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(25);

    try {
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 30;
        });
      }, 120);

      const createdImage = await uploadImage({
        file,
        name: finalName,
        modality,
        notes: notes.trim(),
        precalculatedHash: hash,
      });

      clearInterval(interval);
      setUploadProgress(100);

      showToast("success", "Image Added", `${createdImage.name} is ready for annotation.`);
      navigate(`/workspace/${createdImage.id}`);
    } catch (err) {
      setError("Failed to add image. Please verify file integrity.");
      setIsUploading(false);
    }
  };

  const handleApplyRename = () => {
    if (suggestedRename) {
      setDisplayName(suggestedRename);
    }
    setNameCollisionMatch(null);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 pb-12">
      {/* Header with Back button */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate("/dashboard")}
            className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 transition-colors mb-1 font-medium"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Dashboard</span>
          </button>
          <h1 className="text-xl font-bold text-slate-900 tracking-tight">New Annotation</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Select a chest X-ray image to start manual segmentation or request an AI suggestion
          </p>
        </div>
      </div>

      <form onSubmit={handleUpload} className="grid grid-cols-1 md:grid-cols-12 gap-6">
        {/* Left: Dropzone and Visual Preview */}
        <div className="md:col-span-7 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <PlusCircle className="w-4 h-4 text-teal-700" />
              <span>1. Select Medical Image File <span className="text-red-500">*</span></span>
            </h3>

            <Dropzone
              onFileSelect={handleFileSelect}
              selectedFile={file}
              error={error}
            />

            {/* Local Preview Area */}
            {file && !isDicom && previewUrl && (
              <div className="bg-slate-950 rounded-xl p-3 border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400 font-mono px-1">
                  <span>Selected Image Preview</span>
                  <span>{file.name}</span>
                </div>
                <div className="h-56 bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center">
                  <img
                    src={previewUrl}
                    alt="Selected radiograph preview"
                    className="w-full h-full object-contain"
                  />
                </div>
              </div>
            )}

            {/* Honest DICOM Notice */}
            {file && isDicom && (
              <DicomPlaceholder
                fileName={file.name}
                fileSizeFormatted={`${(file.size / (1024 * 1024)).toFixed(2)} MB`}
                dicomMetadata={{
                  patientId: "PT-NEW",
                  studyDate: new Date().toISOString().slice(0, 10),
                  viewPosition: "PA ERECT",
                  photometricInterpretation: "MONOCHROME2",
                }}
              />
            )}
          </div>
        </div>

        {/* Right: Required & Optional Fields */}
        <div className="md:col-span-5 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-teal-700" />
              <span>2. Image Details</span>
            </h3>

            <div className="space-y-3.5 text-xs">
              {/* Image Name (Required) */}
              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Image Name / Filename <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={displayName}
                  onChange={(e) => {
                    setDisplayName(e.target.value);
                    setTouchedName(true);
                  }}
                  onBlur={() => setTouchedName(true)}
                  placeholder="e.g. PTX-088-CXR.png"
                  className={`w-full p-2 bg-slate-50 border rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white ${
                    touchedName && !displayName.trim() ? "border-red-300 bg-red-50/30" : "border-slate-300"
                  }`}
                  required
                />
                {touchedName && !displayName.trim() ? (
                  <span className="text-[11px] text-red-600 block mt-1">Image name is required.</span>
                ) : (
                  <span className="text-[11px] text-slate-400 block mt-1">
                    Auto-filled from selected file. Unique name required.
                  </span>
                )}
              </div>

              {/* Modality (Required) */}
              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Modality <span className="text-red-500">*</span>
                </label>
                <select
                  value={modality}
                  onChange={(e) => setModality(e.target.value as "xray")}
                  className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer"
                >
                  <option value="xray">Chest X-ray (CXR)</option>
                  <option value="ct" disabled>CT (Future module)</option>
                  <option value="mri" disabled>MRI (Future module)</option>
                </select>
              </div>

              {/* Notes (Optional) */}
              <div>
                <label className="block font-semibold text-slate-700 mb-1">
                  Notes (optional)
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add any relevant clinical observations or patient position notes..."
                  rows={3}
                  className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                />
              </div>
            </div>

            {/* Progress indicator during upload */}
            {isUploading && (
              <div className="space-y-1.5 pt-2">
                <div className="flex justify-between text-xs text-slate-600 font-medium">
                  <span>Saving Image into Workspace...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                  <div
                    className="bg-teal-600 h-full transition-all duration-200"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Actions */}
            <div className="pt-3 border-t border-slate-100 flex items-center justify-end gap-2">
              <Button
                type="button"
                variant="secondary"
                onClick={() => navigate("/dashboard")}
                disabled={isUploading}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="teal"
                disabled={!isFormValid || isUploading}
                isLoading={isUploading || isCheckingHash}
                leftIcon={<PlusCircle className="w-4 h-4" />}
              >
                {isCheckingHash ? "Checking image..." : "Open Annotation Workspace"}
              </Button>
            </div>
          </div>
        </div>
      </form>

      {/* Exact Duplicate Hash Modal */}
      {exactDuplicateMatch && (
        <Modal
          isOpen={true}
          onClose={() => setExactDuplicateMatch(null)}
          title="This image has already been added"
          subtitle="Exact file duplicate detected (SHA-256 hash match)"
          footer={
            <>
              <Button variant="secondary" onClick={() => setExactDuplicateMatch(null)}>
                Cancel
              </Button>
              <Button
                variant="teal"
                onClick={() => {
                  navigate(`/workspace/${exactDuplicateMatch.id}`);
                }}
                leftIcon={<ExternalLink className="w-4 h-4" />}
              >
                Open Existing Image
              </Button>
            </>
          }
        >
          <div className="space-y-3 text-xs text-slate-700">
            <div className="flex items-center gap-3 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-900">
              <AlertTriangle className="w-5 h-5 text-amber-700 flex-shrink-0" />
              <p>
                An identical image file already exists in this workspace: <strong>{exactDuplicateMatch.name}</strong>.
                Duplicate image records are prevented to keep your annotation dataset consistent.
              </p>
            </div>
            <p className="text-slate-600">
              Status: <strong className="capitalize">{exactDuplicateMatch.status.replace("-", " ")}</strong> (Annotator: {exactDuplicateMatch.annotator || "Unassigned"})
            </p>
          </div>
        </Modal>
      )}

      {/* Name Collision Modal */}
      {nameCollisionMatch && (
        <Modal
          isOpen={true}
          onClose={() => setNameCollisionMatch(null)}
          title="An image with this name already exists"
          subtitle="Image names must be unique within the workspace"
          footer={
            <>
              <Button variant="secondary" onClick={() => setNameCollisionMatch(null)}>
                Cancel
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  navigate(`/workspace/${nameCollisionMatch.id}`);
                }}
                leftIcon={<ExternalLink className="w-4 h-4" />}
              >
                Open Existing Image
              </Button>
              <Button
                variant="teal"
                onClick={handleApplyRename}
                leftIcon={<Copy className="w-4 h-4" />}
              >
                Rename New Image ({suggestedRename})
              </Button>
            </>
          }
        >
          <div className="space-y-3 text-xs text-slate-700">
            <p>
              Another image named <strong>"{nameCollisionMatch.name}"</strong> is already in your workspace.
            </p>
            <p className="text-slate-600">
              You can open the existing image, or rename this new image to <strong>{suggestedRename}</strong> before opening the workspace.
            </p>
          </div>
        </Modal>
      )}
    </div>
  );
};

