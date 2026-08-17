import React from "react";
import { useNavigate } from "react-router-dom";
import {
  HelpCircle,
  ArrowLeft,
  Sparkles,
  PenTool,
  BarChart2,
  FileCheck,
  Keyboard,
  ShieldAlert,
  Layers,
} from "lucide-react";
import { KEYBOARD_SHORTCUTS, RESEARCH_DISCLAIMER } from "../constants/theme";

export const HelpPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="max-w-4xl mx-auto space-y-6 pb-12">
      {/* Header with Back button */}
      <div>
        <button
          onClick={() => navigate("/dashboard")}
          className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-900 transition-colors mb-1 font-medium"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Dashboard</span>
        </button>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
          <HelpCircle className="w-5 h-5 text-teal-700" />
          <span>Help & Annotation Guidelines</span>
        </h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Workstation documentation, manual annotation controls, AI assistance review workflows, and metrics guide
        </p>
      </div>

      <div className="space-y-6">
        {/* Section 1: Getting Started Workflow */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-4 h-4 text-teal-700" />
            <span>1. Getting Started Workflow</span>
          </h3>

          <ol className="list-decimal list-inside space-y-2 text-xs text-slate-700 leading-relaxed">
            <li>
              <strong>Select New Annotation:</strong> Click <span className="font-semibold text-teal-800">+ New Annotation</span> from the dashboard or sidebar.
            </li>
            <li>
              <strong>Choose a Chest X-ray:</strong> Select a standard radiograph file (<code className="bg-slate-100 px-1 py-0.5 rounded">.png</code>, <code className="bg-slate-100 px-1 py-0.5 rounded">.jpg</code>, or <code className="bg-slate-100 px-1 py-0.5 rounded">.dcm</code>).
            </li>
            <li>
              <strong>Enter/Confirm Filename:</strong> Verify the displayed image name (auto-filled from the selected file).
            </li>
            <li>
              <strong>Select Modality:</strong> Confirm <span className="font-semibold">Chest X-ray (CXR)</span> as the active modality.
            </li>
            <li>
              <strong>Open Workspace:</strong> Click <span className="font-semibold">Open Annotation Workspace</span> to load the image into the high-precision viewport.
            </li>
            <li>
              <strong>Annotate Manually or Request AI Suggestion:</strong> Choose between manual polygon/brush drawing or click <span className="font-semibold">Request AI Suggestion</span> for an automated lesion proposal.
            </li>
            <li>
              <strong>Review the AI Result:</strong> Inspect the cyan proposal mask, finding statement, and model score.
            </li>
            <li>
              <strong>Accept, Edit, or Reject:</strong> Confirm the proposal as final, refine the vertices in edit mode, or discard false positives.
            </li>
            <li>
              <strong>Save Final Annotation:</strong> Confirmed and finalized annotations turn green and are logged in Annotation History.
            </li>
          </ol>
        </div>

        {/* Section 2: Manual Annotation Tools */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <PenTool className="w-4 h-4 text-teal-700" />
            <span>2. Manual Annotation Tools</span>
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-700">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Polygon Tool (P)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Click on anatomical borders to place vertices. Double-click or click near the start point to close the contour.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Brush Tool (B)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Click and drag to paint contiguous mask regions. Adjust brush diameter (4px–60px) in the top toolbar.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Eraser (E)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Erase drawn brush strokes or subtract regions from in-progress manual annotations.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Pan & Zoom (H / Space / + / -)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Scroll wheel or pan tool to navigate high-resolution lung fields without distorting the coordinate grid.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Overlay Opacity (O)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Adjust slider from 10% to 100% or press O to toggle overlay visibility and inspect underlying lung parenchymal texture.
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <strong className="text-slate-900 block">Undo / Redo (Ctrl+Z / Ctrl+Shift+Z)</strong>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                15-step memory buffer to reverse or re-apply stroke and vertex modifications.
              </p>
            </div>
          </div>
        </div>

        {/* Section 3: AI-Assisted Annotation */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-teal-700" />
            <span>3. AI-Assisted Annotation Workflow</span>
          </h3>

          <div className="space-y-2.5 text-xs text-slate-700 leading-relaxed">
            <p>
              The AI Assistance module generates candidate binary masks highlighting potential pneumothorax collections (such as apical air crescents or lateral pleural line displacements).
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div className="p-3 bg-cyan-50/70 border border-cyan-200 rounded-lg space-y-1">
                <strong className="text-cyan-900 block text-xs">Possible Region Detected</strong>
                <p className="text-cyan-800 text-[11px]">
                  The model identified pixels above the operating threshold. Displayed as a cyan overlay. Requires human specialist review.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                <strong className="text-slate-900 block text-xs">No Region Detected</strong>
                <p className="text-slate-600 text-[11px]">
                  No pixel clusters exceeded the operating threshold. The radiologist can still proceed with manual annotation.
                </p>
              </div>
            </div>

            <div className="space-y-1 pt-2">
              <strong className="text-slate-900 block">Review Decisions:</strong>
              <ul className="list-disc list-inside space-y-1 text-slate-600 text-[11px]">
                <li><strong className="text-slate-900">Accept Suggestion:</strong> Approves the AI proposal without modification. Color turns green and status becomes <em>Finalized</em>.</li>
                <li><strong className="text-slate-900">Edit Suggestion:</strong> Loads the AI contour into the manual editor so you can move, add, or erase vertices.</li>
                <li><strong className="text-slate-900">Reject Suggestion:</strong> Completely removes the AI mask and returns the image to <em>Unannotated</em>.</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Section 4: Understanding the Values */}
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-teal-700" />
            <span>4. Understanding the Metrics & Values</span>
          </h3>

          <div className="space-y-2.5 text-xs text-slate-700 leading-relaxed">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                <strong className="text-slate-900 block">Maximum Model Output Score</strong>
                <p className="text-slate-600 text-[11px]">
                  The highest pixel-level output value produced across the image grid. It is an internal model score, not clinical diagnostic confidence.
                </p>
              </div>

              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1">
                <strong className="text-slate-900 block">Suggested Mask Coverage</strong>
                <p className="text-slate-600 text-[11px]">
                  The percentage of total thoracic radiograph area occupied by the proposed segmentation mask.
                </p>
              </div>
            </div>

            <div className="p-3 bg-amber-50/70 border border-amber-200 rounded-lg space-y-1.5">
              <strong className="text-amber-900 block font-semibold">Reference-Based Metrics Notice:</strong>
              <p className="text-amber-800 text-[11px] leading-relaxed">
                Dice coefficient, Intersection-over-Union (IoU), pixel precision, and pixel recall can only be calculated when a verified expert ground-truth reference annotation is available for comparison. When no reference mask exists, these metrics are labeled <em>Not available</em>.
              </p>
            </div>
          </div>
        </div>

        {/* Section 5: Supported Files & Keyboard Shortcuts */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-teal-700" />
              <span>5. Supported File Formats</span>
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              <strong>PNG & JPEG:</strong> Displayed directly in the high-resolution HTML5 canvas viewport with immediate interactivity.
            </p>
            <p className="text-xs text-slate-600 leading-relaxed">
              <strong>DICOM (.dcm):</strong> Supported for file ingestion and header metadata inspection. Native 16-bit array rendering connects through backend adapters.
            </p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Keyboard className="w-4 h-4 text-teal-700" />
              <span>6. Keyboard Shortcuts</span>
            </h3>
            <div className="grid grid-cols-2 gap-1.5 text-[11px]">
              {KEYBOARD_SHORTCUTS.slice(0, 6).map((sc) => (
                <div key={sc.key} className="flex items-center justify-between p-1.5 bg-slate-50 rounded border border-slate-100">
                  <span className="text-slate-600">{sc.description}</span>
                  <kbd className="font-mono bg-white px-1.5 py-0.5 rounded border border-slate-200 text-[10px] font-semibold">{sc.key}</kbd>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Section 6: Research Disclaimer */}
        <div className="bg-navy-900 text-white p-5 rounded-xl border border-navy-800 space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4" />
            <span>Research Prototype Notice</span>
          </h4>
          <p className="text-xs text-slate-300 leading-relaxed">
            {RESEARCH_DISCLAIMER}
          </p>
        </div>
      </div>
    </div>
  );
};
