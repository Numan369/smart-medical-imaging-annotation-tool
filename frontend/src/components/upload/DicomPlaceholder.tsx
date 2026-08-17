import React from "react";
import { FileArchive, AlertCircle, Cpu } from "lucide-react";

interface DicomPlaceholderProps {
  fileName: string;
  fileSizeFormatted?: string;
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

export const DicomPlaceholder: React.FC<DicomPlaceholderProps> = ({
  fileName,
  fileSizeFormatted,
  dicomMetadata,
}) => {
  return (
    <div className="bg-slate-900 border border-slate-700 text-slate-200 rounded-xl p-6 flex flex-col items-center justify-center text-center space-y-4 max-w-lg mx-auto">
      <div className="w-16 h-16 rounded-full bg-slate-800 border border-slate-700 text-teal-400 flex items-center justify-center">
        <FileArchive className="w-8 h-8" />
      </div>

      <div className="space-y-1">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-teal-900/60 text-teal-300 border border-teal-700 mb-1">
          <Cpu className="w-3 h-3" />
          <span>DICOM Native Medical Image (.dcm)</span>
        </div>
        <h4 className="text-base font-semibold text-white">{fileName}</h4>
        {fileSizeFormatted && <p className="text-xs text-slate-400">File size: {fileSizeFormatted}</p>}
      </div>

      {/* Honest Prototype Notice */}
      <div className="bg-slate-800/80 border border-slate-700 rounded-lg p-3 text-left w-full text-xs space-y-1.5">
        <div className="flex items-center gap-1.5 text-amber-400 font-semibold">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>DICOM Browser Decoder Adapter</span>
        </div>
        <p className="text-slate-300 text-[11px] leading-relaxed">
          Standard web browsers cannot decode 16-bit uncompressed DICOM pixel arrays natively.
          In this prototype, metadata and geometry are loaded, and the image is connected via the backend DICOM adapter interface for AI inference.
        </p>
      </div>

      {/* Extracted / Mock Header Tags */}
      {dicomMetadata && (
        <div className="w-full text-left bg-slate-950 p-3 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1 font-sans">
            Header Metadata Tags
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[11px]">
            <div><span className="text-slate-500">Patient ID:</span> <span className="text-slate-300">{dicomMetadata.patientId || "N/A"}</span></div>
            <div><span className="text-slate-500">Study Date:</span> <span className="text-slate-300">{dicomMetadata.studyDate || "N/A"}</span></div>
            <div><span className="text-slate-500">View:</span> <span className="text-slate-300">{dicomMetadata.viewPosition || "PA"}</span></div>
            <div><span className="text-slate-500">Photometric:</span> <span className="text-slate-300">{dicomMetadata.photometricInterpretation || "MONOCHROME2"}</span></div>
          </div>
        </div>
      )}
    </div>
  );
};
