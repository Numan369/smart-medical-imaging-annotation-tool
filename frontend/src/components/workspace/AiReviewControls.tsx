import React, { useState } from "react";
import { Annotation } from "../../types";
import { Button } from "../common/Button";
import { Modal } from "../common/Modal";
import {
  CheckCircle2,
  XCircle,
  Edit3,
  AlertTriangle,
  HelpCircle,
  Sparkles,
  Info,
  Check,
} from "lucide-react";

interface AiReviewControlsProps {
  annotation: Annotation;
  onAccept: () => Promise<void>;
  onReject: (reason: string) => Promise<void>;
  onEdit: () => void;
  isProcessing: boolean;
}

export const AiReviewControls: React.FC<AiReviewControlsProps> = ({
  annotation,
  onAccept,
  onReject,
  onEdit,
  isProcessing,
}) => {
  const [isAcceptModalOpen, setIsAcceptModalOpen] = useState(false);
  const [isRejectModalOpen, setIsRejectModalOpen] = useState(false);
  const [rejectionReason, setRejectionReason] = useState("Normal pleural anatomy / false positive");

  const handleConfirmAccept = async () => {
    await onAccept();
    setIsAcceptModalOpen(false);
  };

  const handleConfirmReject = async () => {
    await onReject(rejectionReason);
    setIsRejectModalOpen(false);
  };

  const hasRegions = (annotation.shapes && annotation.shapes.length > 0) || annotation.finding === "possible-region-detected";
  const isFindingDetected = annotation.finding === "possible-region-detected" || (annotation.finding === undefined && hasRegions);

  return (
    <>
      <div className="bg-white border-t border-slate-200 shadow-md z-30 flex flex-col divide-y divide-slate-100 flex-shrink-0">
        {/* Top Banner: Prominent Finding Message */}
        <div className="px-4 py-2.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 bg-slate-50/70">
          <div className="flex items-center gap-3">
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                isFindingDetected ? "bg-amber-100 text-amber-800" : "bg-teal-100 text-teal-800"
              }`}
            >
              {isFindingDetected ? (
                <AlertTriangle className="w-4 h-4 text-amber-700" />
              ) : (
                <Sparkles className="w-4 h-4 text-teal-700" />
              )}
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h4 className="font-bold text-xs text-slate-900">
                  {isFindingDetected
                    ? "Possible pneumothorax region detected"
                    : "No pneumothorax region detected by AI"}
                </h4>
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-cyan-100 text-cyan-800 border border-cyan-200">
                  Cyan Overlay
                </span>
              </div>
              <p className="text-[11px] text-slate-600">
                {isFindingDetected
                  ? "AI identified one or more regions that may represent pneumothorax. Review the segmentation before accepting it."
                  : "The AI did not generate a pneumothorax region at the selected operating threshold. Human review is still required."}
              </p>
            </div>
          </div>

          {/* Persistent Disclaimer Badge */}
          <div className="text-[11px] text-slate-500 bg-white px-2.5 py-1 rounded border border-slate-200 font-medium whitespace-nowrap">
            AI suggestions require human review
          </div>
        </div>

        {/* Middle: AI Suggestion Summary & Conditional Reference Agreement */}
        <div className="px-4 py-2 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-700 bg-white">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
            {/* Suggested Regions Count */}
            <div>
              <span className="text-slate-500">Suggested regions:</span>{" "}
              <strong className="font-mono text-slate-900">{annotation.shapes.length}</strong>
            </div>

            {/* Mask Coverage */}
            <div>
              <span className="text-slate-500">Suggested mask coverage:</span>{" "}
              <strong className="font-mono text-slate-900">
                {annotation.maskCoveragePercent !== undefined ? `${annotation.maskCoveragePercent}%` : "0.00%"}
              </strong>
            </div>

            {/* Maximum Model Output Score */}
            <div className="flex items-center gap-1">
              <span className="text-slate-500">Maximum model output score:</span>{" "}
              <strong className="font-mono text-teal-800">
                {annotation.maximumOutputScore !== undefined ? annotation.maximumOutputScore : "0.85"}
              </strong>
              <span
                className="text-slate-400 hover:text-slate-600 cursor-help"
                title="The highest pixel-level output produced by the segmentation model. It is not diagnostic confidence."
              >
                <HelpCircle className="w-3.5 h-3.5" />
              </span>
            </div>

            {/* Reference-based metrics / AI-to-Reference Agreement */}
            <div className="flex items-center gap-1 border-l border-slate-200 pl-4">
              {annotation.referenceMetrics ? (
                <div className="flex items-center gap-3">
                  <span className="text-slate-500 font-semibold">AI-to-Reference Agreement:</span>
                  <span className="font-mono">Dice: <strong className="text-emerald-700">{annotation.referenceMetrics.dice}</strong></span>
                  <span className="font-mono">IoU: <strong className="text-slate-800">{annotation.referenceMetrics.iou}</strong></span>
                  <span className="font-mono">Recall: <strong className="text-slate-800">{annotation.referenceMetrics.recall}</strong></span>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-slate-500">
                  <span>Reference-based metrics: <strong className="font-medium text-slate-600">Not available</strong></span>
                  <span
                    className="text-slate-400 hover:text-slate-600 cursor-help"
                    title="Dice, IoU, precision, and recall can only be calculated when an expert reference annotation is available."
                  >
                    <Info className="w-3.5 h-3.5" />
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom: Persistent Review Action Bar (Prominently Visible without Scrolling) */}
        <div className="px-4 py-2.5 flex items-center justify-between gap-4 bg-slate-50 border-t border-slate-200">
          <div className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
            <Check className="w-4 h-4 text-teal-700" />
            <span>Review Decision:</span>
          </div>

          <div className="flex items-center gap-2.5">
            {/* Reject Suggestion */}
            <Button
              variant="danger"
              size="sm"
              onClick={() => setIsRejectModalOpen(true)}
              disabled={isProcessing}
              leftIcon={<XCircle className="w-4 h-4" />}
            >
              Reject Suggestion
            </Button>

            {/* Edit Suggestion */}
            <Button
              variant="secondary"
              size="sm"
              onClick={onEdit}
              disabled={isProcessing}
              leftIcon={<Edit3 className="w-4 h-4 text-slate-700" />}
            >
              Edit Suggestion
            </Button>

            {/* Accept Suggestion */}
            <Button
              variant="teal"
              size="sm"
              onClick={() => setIsAcceptModalOpen(true)}
              disabled={isProcessing}
              leftIcon={<CheckCircle2 className="w-4 h-4" />}
            >
              Accept Suggestion
            </Button>
          </div>
        </div>
      </div>

      {/* Accept Suggestion Confirmation Modal */}
      <Modal
        isOpen={isAcceptModalOpen}
        onClose={() => setIsAcceptModalOpen(false)}
        title="Accept AI suggestion?"
        subtitle="Finalize this AI segmentation proposal as reviewed expert annotation"
        footer={
          <>
            <Button variant="secondary" onClick={() => setIsAcceptModalOpen(false)} disabled={isProcessing}>
              Cancel
            </Button>
            <Button
              variant="teal"
              onClick={handleConfirmAccept}
              isLoading={isProcessing}
              leftIcon={<CheckCircle2 className="w-4 h-4" />}
            >
              Accept and Save
            </Button>
          </>
        }
      >
        <div className="space-y-3 text-xs text-slate-700">
          <p className="leading-relaxed">
            The current AI segmentation will be saved as the final expert-reviewed annotation.
          </p>
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-emerald-900 space-y-1">
            <p className="font-semibold text-xs flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <span>Status change: Awaiting Review → Finalized</span>
            </p>
            <p className="text-[11px] text-emerald-800">
              The mask overlay color will transition to green and your expert review will be recorded in Annotation History.
            </p>
          </div>
        </div>
      </Modal>

      {/* Reject Suggestion Confirmation Modal */}
      <Modal
        isOpen={isRejectModalOpen}
        onClose={() => setIsRejectModalOpen(false)}
        title="Reject AI suggestion?"
        subtitle="Discard the suggested mask"
        footer={
          <>
            <Button variant="secondary" onClick={() => setIsRejectModalOpen(false)} disabled={isProcessing}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={handleConfirmReject}
              isLoading={isProcessing}
              leftIcon={<XCircle className="w-4 h-4" />}
            >
              Reject and Discard
            </Button>
          </>
        }
      >
        <div className="space-y-3 text-xs text-slate-700">
          <p className="leading-relaxed">
            The suggested mask will be discarded. The original image will remain available for manual annotation.
          </p>

          <div className="space-y-1.5 pt-1">
            <label className="font-medium text-slate-900 block">Reason for Rejection (optional):</label>
            <select
              value={rejectionReason}
              onChange={(e) => setRejectionReason(e.target.value)}
              className="w-full p-2 bg-slate-50 border border-slate-300 rounded-lg text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-teal-500"
            >
              <option value="Normal pleural anatomy / false positive">Normal pleural anatomy / false positive</option>
              <option value="Skin fold or external artifact">Skin fold or external artifact</option>
              <option value="Over-segmentation of parenchyma">Over-segmentation of parenchyma</option>
              <option value="Under-segmented / missed primary apical line">Under-segmented / missed primary apical line</option>
              <option value="Incorrect laterality or lung field">Incorrect laterality or lung field</option>
            </select>
          </div>
        </div>
      </Modal>
    </>
  );
};
