import React, { useState } from "react";
import { Modal } from "../common/Modal";
import { Button } from "../common/Button";
import { Sparkles, Layers, AlertCircle, CheckCircle2 } from "lucide-react";
import { AI_ASSISTANCE_METADATA } from "../../constants/theme";

interface AiRequestModalProps {
  isOpen: boolean;
  onClose: () => void;
  onRequest: (mode: "standard" | "tta", threshold: number) => Promise<void>;
  isLoading: boolean;
}

export const AiRequestModal: React.FC<AiRequestModalProps> = ({
  isOpen,
  onClose,
  onRequest,
  isLoading,
}) => {
  const [inferenceMode, setInferenceMode] = useState<"standard" | "tta">("standard");
  const [threshold, setThreshold] = useState<number>(AI_ASSISTANCE_METADATA.defaultThreshold);

  const handleSubmit = async () => {
    await onRequest(inferenceMode, threshold);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Request AI Pneumothorax Suggestion"
      subtitle="Generate candidate lesion segmentation proposals"
      maxWidth="md"
      footer={
        <>
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            variant="teal"
            onClick={handleSubmit}
            isLoading={isLoading}
            leftIcon={<Sparkles className="w-4 h-4" />}
          >
            {isLoading ? "Generating AI Suggestion..." : "Generate AI Suggestion"}
          </Button>
        </>
      }
    >
      <div className="space-y-4 text-xs text-slate-700">
        {/* Suggestion Mode Selection */}
        <div className="space-y-2">
          <label className="font-semibold text-slate-900 block">Select Suggestion Sensitivity Mode:</label>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
            {/* Standard Mode */}
            <div
              onClick={() => setInferenceMode("standard")}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                inferenceMode === "standard"
                  ? "border-teal-600 bg-teal-50/60 ring-2 ring-teal-500/20"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-900 text-xs">Standard Suggestion</span>
                {inferenceMode === "standard" && <CheckCircle2 className="w-4 h-4 text-teal-700" />}
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Balanced sensitivity and specificity for general pneumothorax candidate regions.
              </p>
            </div>

            {/* High Sensitivity Mode */}
            <div
              onClick={() => setInferenceMode("tta")}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                inferenceMode === "tta"
                  ? "border-teal-600 bg-teal-50/60 ring-2 ring-teal-500/20"
                  : "border-slate-200 bg-white hover:border-slate-300"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-semibold text-slate-900 text-xs flex items-center gap-1">
                  <Layers className="w-3.5 h-3.5 text-teal-600" />
                  <span>High Sensitivity Preview</span>
                </span>
                {inferenceMode === "tta" && <CheckCircle2 className="w-4 h-4 text-teal-700" />}
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                Multi-pass analysis aimed at subtle or small apical rim pneumothoraces.
              </p>
            </div>
          </div>
        </div>

        {/* Operating Threshold Adjustment */}
        <div className="space-y-1.5 pt-1">
          <div className="flex justify-between items-center text-xs">
            <span className="font-medium text-slate-800">Operating Threshold:</span>
            <span className="font-mono text-teal-800 font-bold bg-slate-100 px-2 py-0.5 rounded">
              {threshold.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min="0.15"
            max="0.65"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-full accent-teal-600 cursor-pointer"
          />
          <span className="text-[10px] text-slate-400 block">
            Default: 0.35 (standard operating threshold)
          </span>
        </div>

        {/* Human Review Disclaimer */}
        <div className="flex items-start gap-2 bg-amber-50 p-2.5 rounded-lg border border-amber-200 text-amber-800 text-[11px] leading-relaxed">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <span>
            AI-generated suggestions require human review and must not be treated as a medical diagnosis.
          </span>
        </div>
      </div>
    </Modal>
  );
};
