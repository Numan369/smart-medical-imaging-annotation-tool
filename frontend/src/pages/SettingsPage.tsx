import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useAppData } from "../context/AppDataContext";
import { useToast } from "../context/ToastContext";
import { Button } from "../components/common/Button";
import { Modal } from "../components/common/Modal";
import { KEYBOARD_SHORTCUTS } from "../constants/theme";
import {
  Sliders,
  Keyboard,
  RotateCcw,
  LogOut,
  Palette,
  Eye,
  ArrowLeft,
  Sparkles,
  CheckCircle2,
  Trash2,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { analyzeDuplicateImages, deduplicateImagesAndStorage, DuplicateAnalysisResult } from "../utils/storage";

export const SettingsPage: React.FC = () => {
  const { user, signOut } = useAuth();
  const { images, settings, updateSettings, resetAllDemoData, refreshImages } = useAppData();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [opacity, setOpacity] = useState(settings.defaultOverlayOpacity);
  const [color, setColor] = useState(settings.defaultAnnotationColor);
  const [shortcutsEnabled, setShortcutsEnabled] = useState(settings.enableKeyboardShortcuts);

  // Deduplication state
  const [duplicateReportModal, setDuplicateReportModal] = useState<DuplicateAnalysisResult | null>(null);
  const [isDeduplicating, setIsDeduplicating] = useState(false);

  const handleSavePreferences = (e: React.FormEvent) => {
    e.preventDefault();
    updateSettings({
      defaultOverlayOpacity: opacity,
      defaultAnnotationColor: color,
      enableKeyboardShortcuts: shortcutsEnabled,
    });
    showToast("success", "Settings Saved", "Annotation workstation preferences updated.");
  };

  const handleCheckDuplicates = () => {
    const analysis = analyzeDuplicateImages(images);
    if (analysis.totalDuplicates === 0) {
      showToast("info", "No Duplicates Found", "All images in the workspace have unique hashes and names.");
    } else {
      setDuplicateReportModal(analysis);
    }
  };

  const handleConfirmDeduplicate = async () => {
    setIsDeduplicating(true);
    try {
      const report = deduplicateImagesAndStorage();
      await refreshImages();
      setDuplicateReportModal(null);
      showToast(
        "success",
        "Duplicates Cleaned",
        `Safely removed ${report.removedCount} duplicate records. Annotations and history were preserved.`
      );
    } catch (err) {
      showToast("error", "Error", "Failed to clean duplicates.");
    } finally {
      setIsDeduplicating(false);
    }
  };

  const handleResetData = async () => {
    if (
      window.confirm(
        "Are you sure you want to reset all demo images, annotations, and history logs to the initial clean seed state?"
      )
    ) {
      await resetAllDemoData();
      showToast("info", "Demo Data Reset", "Restored initial clean demo images and annotations.");
    }
  };

  const handleSignOut = async () => {
    await signOut();
    navigate("/login");
  };

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
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Workstation Settings</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Configure annotation defaults, review keyboard shortcuts, and manage demo session
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Left Column: User Profile Card */}
        <div className="space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-navy-900 text-white flex items-center justify-center font-bold text-base">
                {user?.name?.charAt(0) || "U"}
              </div>
              <div className="space-y-0.5">
                <h3 className="font-bold text-sm text-slate-900">{user?.name || "Annotator"}</h3>
                <p className="text-xs text-slate-500">{user?.role || "Annotator"}</p>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 space-y-2 text-xs text-slate-600">
              <div>
                <span className="text-slate-400 block">Email:</span>
                <span className="font-medium text-slate-800">{user?.email || "user@example.com"}</span>
              </div>
              {user?.institution && (
                <div>
                  <span className="text-slate-400 block">Affiliation:</span>
                  <span className="font-medium text-slate-800">{user.institution}</span>
                </div>
              )}
            </div>

            <Button
              variant="secondary"
              size="sm"
              onClick={handleSignOut}
              className="w-full text-red-600 hover:bg-red-50 hover:text-red-700"
              leftIcon={<LogOut className="w-4 h-4" />}
            >
              Sign Out of Session
            </Button>
          </div>

          {/* Clean Duplicate Images Card */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs space-y-2">
            <h4 className="font-semibold text-slate-900 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-teal-700" />
              <span>Workspace Duplicate Cleanup</span>
            </h4>
            <p className="text-slate-500 leading-relaxed text-[11px]">
              Scan and safely merge exact duplicate images (by content hash or normalized name), preserving the highest quality annotations and audit history.
            </p>
            <Button
              variant="secondary"
              size="sm"
              onClick={handleCheckDuplicates}
              className="w-full mt-1 text-xs"
              leftIcon={<Sparkles className="w-3.5 h-3.5 text-teal-600" />}
            >
              Clean Duplicate Images
            </Button>
          </div>

          {/* Reset Demo Data Card */}
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 text-xs space-y-2">
            <h4 className="font-semibold text-slate-900 flex items-center gap-1.5">
              <RotateCcw className="w-3.5 h-3.5 text-teal-700" />
              <span>Demo Session Reset</span>
            </h4>
            <p className="text-slate-500 leading-relaxed text-[11px]">
              Restore all initial test images (`PTX-014-XR`, `PTX-017-XR`, and `PTX-067-XR`) and default annotations.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={handleResetData}
              className="w-full mt-1 text-xs"
            >
              Reset All Demo Data
            </Button>
          </div>
        </div>

        {/* Right Column: Preferences & Keyboard Shortcuts */}
        <div className="md:col-span-2 space-y-6">
          {/* Annotation Preferences */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-4">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-teal-700" />
              <span>Annotation Canvas Defaults</span>
            </h3>

            <form onSubmit={handleSavePreferences} className="space-y-4 text-xs">
              {/* Default Opacity */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <label className="font-medium text-slate-800 flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5 text-slate-500" />
                    <span>Default Mask Overlay Opacity</span>
                  </label>
                  <span className="font-mono font-bold text-slate-900">{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="w-full accent-teal-600 cursor-pointer"
                />
              </div>

              {/* Default Annotation Color */}
              <div className="space-y-1.5">
                <label className="font-medium text-slate-800 flex items-center gap-1.5">
                  <Palette className="w-3.5 h-3.5 text-slate-500" />
                  <span>Default Manual Contour Color</span>
                </label>
                <div className="flex items-center gap-3">
                  <input
                    type="color"
                    value={color}
                    onChange={(e) => setColor(e.target.value)}
                    className="w-8 h-8 rounded border border-slate-300 cursor-pointer p-0.5 bg-white"
                  />
                  <span className="font-mono text-slate-600 text-xs">{color.toUpperCase()}</span>
                  <span className="text-[11px] text-slate-400">
                    (Cyan is reserved for AI proposals; Green is reserved for Finalized annotations)
                  </span>
                </div>
              </div>

              {/* Keyboard Shortcuts Toggle */}
              <div className="pt-2 border-t border-slate-100 flex items-center justify-between">
                <div>
                  <span className="font-medium text-slate-900 block">Enable Hotkey Shortcuts</span>
                  <span className="text-[11px] text-slate-500">
                    Allows single-key tool switches (B, P, E, O, etc.)
                  </span>
                </div>
                <input
                  type="checkbox"
                  checked={shortcutsEnabled}
                  onChange={(e) => setShortcutsEnabled(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500 w-4 h-4 cursor-pointer"
                />
              </div>

              <div className="pt-2 flex justify-end">
                <Button type="submit" variant="teal" size="sm">
                  Save Preferences
                </Button>
              </div>
            </form>
          </div>

          {/* Keyboard Shortcuts Reference */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-card space-y-3">
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Keyboard className="w-4 h-4 text-teal-700" />
              <span>Keyboard Shortcuts Cheat Sheet</span>
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
              {KEYBOARD_SHORTCUTS.map((item) => (
                <div
                  key={item.key}
                  className="flex items-center justify-between p-2 bg-slate-50 rounded-lg border border-slate-200"
                >
                  <span className="text-slate-600 text-[11px]">{item.description}</span>
                  <kbd className="px-2 py-0.5 bg-white border border-slate-300 rounded font-mono text-[10px] text-slate-800 shadow-2xs font-semibold">
                    {item.key}
                  </kbd>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Duplicate Cleanup Confirmation & Preview Modal */}
      {duplicateReportModal && (
        <Modal
          isOpen={true}
          onClose={() => setDuplicateReportModal(null)}
          title="Clean Duplicate Images"
          subtitle={`Detected ${duplicateReportModal.totalDuplicates} exact duplicate records across ${duplicateReportModal.groups.length} image groups`}
          footer={
            <>
              <Button variant="secondary" onClick={() => setDuplicateReportModal(null)} disabled={isDeduplicating}>
                Cancel
              </Button>
              <Button
                variant="danger"
                onClick={handleConfirmDeduplicate}
                isLoading={isDeduplicating}
                leftIcon={<Trash2 className="w-4 h-4" />}
              >
                Clean & Merge {duplicateReportModal.totalDuplicates} Duplicates
              </Button>
            </>
          }
        >
          <div className="space-y-4 text-xs text-slate-700 max-h-96 overflow-y-auto">
            <p className="leading-relaxed">
              For each duplicate set, the system will keep the most valuable record (Priority: Finalized &gt; Awaiting Review &gt; Unannotated) and merge all associated annotation metadata and history.
            </p>

            <div className="space-y-3">
              {duplicateReportModal.groups.map((group, gIdx) => (
                <div key={gIdx} className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-2">
                  <div className="flex items-center justify-between font-semibold text-slate-900">
                    <span className="truncate max-w-xs">{group.best.name}</span>
                    <span className="text-[11px] text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                      {group.duplicates.length} duplicate{group.duplicates.length > 1 ? "s" : ""}
                    </span>
                  </div>

                  <div className="space-y-1 text-[11px] text-slate-600">
                    <div className="flex items-center gap-1 text-emerald-700 font-medium">
                      <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
                      <span>Retaining: {group.best.name} ({group.best.status})</span>
                    </div>
                    {group.duplicates.map((dup, dIdx) => (
                      <div key={dIdx} className="flex items-center gap-1 text-red-600 pl-4">
                        <Trash2 className="w-3 h-3 flex-shrink-0" />
                        <span>Removing duplicate ID: {dup.id} ({dup.status})</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};
