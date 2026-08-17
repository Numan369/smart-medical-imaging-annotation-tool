import React, { useRef, useState } from "react";
import { UploadCloud, FileImage, AlertCircle, CheckCircle } from "lucide-react";
import clsx from "clsx";

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
  error?: string | null;
}

export const Dropzone: React.FC<DropzoneProps> = ({
  onFileSelect,
  selectedFile,
  error,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={clsx(
          "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center",
          isDragOver
            ? "border-teal-500 bg-teal-50/50 scale-[1.01]"
            : selectedFile
            ? "border-emerald-300 bg-emerald-50/20"
            : "border-slate-300 bg-white hover:border-slate-400 hover:bg-slate-50/50"
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.dcm,image/png,image/jpeg"
          onChange={handleChange}
          className="hidden"
        />

        <div
          className={clsx(
            "w-14 h-14 rounded-full flex items-center justify-center mb-3 transition-colors",
            selectedFile
              ? "bg-emerald-100 text-emerald-700"
              : "bg-slate-100 text-slate-500"
          )}
        >
          {selectedFile ? (
            <CheckCircle className="w-7 h-7" />
          ) : (
            <UploadCloud className="w-7 h-7 text-navy-900" />
          )}
        </div>

        {selectedFile ? (
          <div className="space-y-1">
            <p className="font-semibold text-slate-900 text-sm flex items-center justify-center gap-1.5">
              <FileImage className="w-4 h-4 text-emerald-600" />
              <span>{selectedFile.name}</span>
            </p>
            <p className="text-xs text-slate-500">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to upload
            </p>
            <p className="text-[11px] text-teal-700 font-medium pt-2">Click or drag another file to replace</p>
          </div>
        ) : (
          <div className="space-y-1.5">
            <p className="text-sm font-semibold text-slate-800">
              Drag & drop medical image here, or{" "}
              <span className="text-teal-700 underline decoration-teal-500 underline-offset-2">
                browse files
              </span>
            </p>
            <p className="text-xs text-slate-500">
              Supports Chest X-ray formats: <span className="font-mono font-medium text-slate-700">.PNG, .JPG, .JPEG, .DCM</span> (up to 20MB)
            </p>
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-1.5 text-xs text-red-600 bg-red-50 p-2.5 rounded-lg border border-red-200">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
};
