import React, { useEffect, useState, useMemo, useCallback } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import { useAppData } from "../context/AppDataContext";
import { useAuth } from "../context/AuthContext";
import { useToast } from "../context/ToastContext";
import { Annotation, AnnotationShape, MedicalImage, WorkspaceMode } from "../types";
import { AnnotationToolbar, WorkspaceTool } from "../components/workspace/AnnotationToolbar";
import { AnnotationCanvas } from "../components/workspace/AnnotationCanvas";
import { ImageMetadataPanel } from "../components/workspace/ImageMetadataPanel";
import { StatusBadge, ModalityBadge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { aiAnnotationService } from "../services/aiAnnotationService";
import { exportAnnotationAsJson } from "../utils/export";
import { calculateFitToScreen, clampZoom } from "../utils/coordinate";
import {
  ArrowLeft,
  PenTool,
  Sparkles,
  Download,
  PanelRightClose,
  PanelRightOpen,
  Edit2,
} from "lucide-react";

// Fixed internal default parameters (not exposed in GUI)
const DEFAULT_AI_OPTIONS = {
  inferenceMode: "standard" as const,
  threshold: 0.35,
};

export const WorkspacePage: React.FC = () => {
  const { imageId } = useParams<{ imageId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const { images, getAnnotation, saveAnnotation, discardAnnotation, settings } = useAppData();
  const { showToast } = useToast();

  // Active Medical Image
  const image = useMemo(() => images.find((img: MedicalImage) => img.id === imageId), [images, imageId]);

  // Determine initial workspace mode from URL route and image status
  const getInitialMode = useCallback((): WorkspaceMode => {
    if (location.pathname.endsWith("/manual")) return "manual-edit";
    if (location.pathname.endsWith("/ai-review")) return "ai-review";
    if (location.pathname.endsWith("/edit")) return "ai-edit";
    if (image?.status === "finalized") return "finalized";
    if (image?.status === "awaiting-review") return "ai-review";
    return "view";
  }, [location.pathname, image?.status]);

  const [mode, setMode] = useState<WorkspaceMode>(getInitialMode());

  // Annotation Data State
  const [annotation, setAnnotation] = useState<Annotation | null>(null);
  const [shapes, setShapes] = useState<AnnotationShape[]>([]);
  const [originalAiShapes, setOriginalAiShapes] = useState<AnnotationShape[]>([]);
  const [historyPast, setHistoryPast] = useState<AnnotationShape[][]>([]);
  const [historyFuture, setHistoryFuture] = useState<AnnotationShape[][]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [liveCoveragePercent, setLiveCoveragePercent] = useState<number | undefined>(undefined);
  const [isMaskEmpty, setIsMaskEmpty] = useState(false);

  // Viewport and Container Dimensions
  const [containerDimensions, setContainerDimensions] = useState<{ width: number; height: number }>({
    width: 0,
    height: 0,
  });

  // Canvas / Viewport Transform State
  const [activeTool, setActiveTool] = useState<WorkspaceTool>("select");
  const [brushRadius, setBrushRadius] = useState(20);
  const [overlayOpacity, setOverlayOpacity] = useState(settings.defaultOverlayOpacity || 0.55);
  const [isOverlayVisible, setIsOverlayVisible] = useState(true);
  const [activeColor, setActiveColor] = useState(settings.defaultAnnotationColor || "#F59E0B");
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [invert, setInvert] = useState(false);

  // UI Panels State
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [isMetadataPanelOpen, setIsMetadataPanelOpen] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  // Load existing annotation for this image
  useEffect(() => {
    if (!imageId) return;
    let isCancelled = false;

    async function loadData() {
      if (!imageId) return;
      const existing = await getAnnotation(imageId);
      if (isCancelled) return;

      if (existing) {
        setAnnotation(existing);
        setShapes(existing.shapes || []);
        if (existing.originalAiShapes && existing.originalAiShapes.length > 0) {
          setOriginalAiShapes(existing.originalAiShapes);
        } else if (existing.source === "ai" || existing.source === "ai-edited") {
          setOriginalAiShapes(existing.shapes || []);
        }

        if (existing.status === "draft" && (existing.source === "ai" || existing.source === "ai-edited")) {
          setActiveColor("#22D3EE"); // Cyan AI mask
          if (!location.pathname.endsWith("/edit") && !location.pathname.endsWith("/manual")) {
            setMode("ai-review");
          }
        } else if (existing.status === "reviewed") {
          setActiveColor("#16A34A"); // Finalized green
          if (!location.pathname.endsWith("/manual") && !location.pathname.endsWith("/edit")) {
            setMode("finalized");
          }
        }
      } else {
        setAnnotation(null);
        setShapes([]);
        setOriginalAiShapes([]);
        if (location.pathname.endsWith("/manual")) {
          setMode("manual-edit");
        } else {
          setMode("view");
        }
      }
      setHistoryPast([]);
      setHistoryFuture([]);
      setHasUnsavedChanges(false);
    }

    loadData();
    return () => {
      isCancelled = true;
    };
  }, [imageId, getAnnotation, location.pathname]);

  // Sync mode with route changes
  useEffect(() => {
    const currentMode = getInitialMode();
    setMode(currentMode);
    if (currentMode === "manual-edit" || currentMode === "ai-edit") {
      setActiveTool("polygon");
    } else {
      setActiveTool("select");
    }
  }, [getInitialMode]);

  // Fit image to screen whenever container dimensions or image change
  const handleFitToScreen = useCallback(() => {
    if (!image) return;
    const contW = containerDimensions.width > 0 ? containerDimensions.width : 800;
    const contH = containerDimensions.height > 0 ? containerDimensions.height : 600;
    const fit = calculateFitToScreen(image.width, image.height, contW, contH);
    setZoom(fit.zoom);
    setPanX(fit.panX);
    setPanY(fit.panY);
  }, [image, containerDimensions]);

  // Trigger fit-to-screen on image load and container resize
  useEffect(() => {
    if (image && containerDimensions.width > 0 && containerDimensions.height > 0) {
      handleFitToScreen();
    }
  }, [image?.id, containerDimensions.width > 0, containerDimensions.height > 0]);

  // Container resize handler from AnnotationCanvas
  const handleContainerResize = useCallback((width: number, height: number) => {
    setContainerDimensions({ width, height });
  }, []);

  // Coverage callback from AnnotationCanvas
  const handleCoverageCalculated = useCallback((coverage: number, empty: boolean) => {
    setLiveCoveragePercent(coverage);
    setIsMaskEmpty(empty);
  }, []);

  // Shape editing handlers
  const handleShapesChange = (newShapes: AnnotationShape[]) => {
    setHistoryPast((prev) => [...prev.slice(-15), shapes]);
    setHistoryFuture([]);
    setShapes(newShapes);
    setHasUnsavedChanges(true);
  };

  const handleUndo = () => {
    if (historyPast.length === 0) return;
    const prev = historyPast[historyPast.length - 1];
    setHistoryFuture((fut) => [shapes, ...fut.slice(0, 15)]);
    setHistoryPast((past) => past.slice(0, -1));
    setShapes(prev);
    setHasUnsavedChanges(true);
  };

  const handleRedo = () => {
    if (historyFuture.length === 0) return;
    const next = historyFuture[0];
    setHistoryPast((past) => [...past.slice(-15), shapes]);
    setHistoryFuture((fut) => fut.slice(1));
    setShapes(next);
    setHasUnsavedChanges(true);
  };

  const handleClear = () => {
    if (shapes.length === 0) return;
    if (window.confirm("Clear all drawn masks on this image?")) {
      handleShapesChange([]);
    }
  };

  // Direct AI Suggestion Request (No configuration popup!)
  const handleRequestAi = async () => {
    if (!imageId || isAiLoading) return;
    setIsAiLoading(true);
    setMode("ai-processing");

    try {
      const result = await aiAnnotationService.requestAiAnnotation(imageId, {
        inferenceMode: DEFAULT_AI_OPTIONS.inferenceMode,
        threshold: DEFAULT_AI_OPTIONS.threshold,
        hasReferenceMask: image?.hasReferenceMask,
      });

      setAnnotation(result.annotation);
      setShapes(result.annotation.shapes);
      setOriginalAiShapes(result.annotation.shapes);
      setActiveColor("#22D3EE"); // Cyan AI mask

      // Save as AI draft in state
      await saveAnnotation(
        imageId,
        result.annotation.shapes,
        "ai",
        "draft",
        "AI Assistance",
        "AI suggestion generated for review",
        {
          threshold: DEFAULT_AI_OPTIONS.threshold,
          maximumOutputScore: result.maximumOutputScore,
          maskCoveragePercent: result.maskCoveragePercent,
          processingTimeMs: result.inferenceTimeMs,
          finding: result.finding,
          referenceMetrics: result.referenceMetrics,
          originalAiShapes: result.annotation.shapes,
        }
      );

      showToast(
        "info",
        "AI Suggestion Generated",
        result.finding === "possible-region-detected"
          ? "Possible pneumothorax region detected. Human review required."
          : "No pneumothorax region detected by AI. Human review required."
      );

      setMode("ai-review");
      navigate(`/workspace/${imageId}/ai-review`);
    } catch (err) {
      showToast("error", "AI Request Failed", "Could not generate candidate suggestion. Please retry.");
      setMode("view");
    } finally {
      setIsAiLoading(false);
    }
  };

  // 1. Accept Suggestion: saves AI mask, sets green, status finalized
  const handleAcceptAi = async () => {
    if (!imageId) return;
    setIsSaving(true);
    try {
      const acceptedShapes = shapes.map((s) => ({ ...s, color: "#16A34A" }));
      setShapes(acceptedShapes);
      setActiveColor("#16A34A");

      await saveAnnotation(
        imageId,
        acceptedShapes,
        "ai",
        "reviewed",
        user?.name || "Dr. Sarah Jenkins",
        "AI suggestion reviewed and accepted without modification.",
        {
          ...annotation,
          originalAiShapes,
        }
      );

      showToast("success", "AI Suggestion Accepted", "Annotation finalized and saved.");
      setMode("finalized");
      setHasUnsavedChanges(false);
      navigate(`/workspace/${imageId}`);
    } finally {
      setIsSaving(false);
    }
  };

  // 2. Reject Suggestion: discards mask, removes active annotation, returns status to unannotated
  const handleRejectAi = async (reason: string) => {
    if (!imageId) return;
    setIsSaving(true);
    try {
      await discardAnnotation(imageId, reason, user?.name || "Dr. Sarah Jenkins");
      setShapes([]);
      setAnnotation(null);
      setOriginalAiShapes([]);
      setHasUnsavedChanges(false);
      showToast("info", "AI Suggestion Rejected", "Mask discarded. Image returned to unannotated state.");
      setMode("view");
      navigate(`/workspace/${imageId}`);
    } finally {
      setIsSaving(false);
    }
  };

  // 3. Edit Suggestion: enters ai-edit mode, enables drawing & eraser tools with AI mask loaded
  const handleEditAi = () => {
    setMode("ai-edit");
    setActiveTool("polygon");
    setActiveColor("#F59E0B"); // Editing amber
    showToast("info", "Edit Mode Active", "Refine the AI suggestion using Polygon, Brush, or Eraser tools.");
    navigate(`/workspace/${imageId}/edit`);
  };

  // 4. Save Edited Annotation
  const handleSaveEdited = async () => {
    if (!imageId) return;
    if (isMaskEmpty) {
      if (!window.confirm("Save an empty annotation? This indicates no pneumothorax region was retained.")) {
        return;
      }
    }

    setIsSaving(true);
    try {
      const finalizedShapes = shapes.map((s) => ({ ...s, color: "#16A34A" }));
      setShapes(finalizedShapes);
      setActiveColor("#16A34A");

      await saveAnnotation(
        imageId,
        finalizedShapes,
        "ai-edited",
        "reviewed",
        user?.name || "Dr. Sarah Jenkins",
        "AI suggestion edited and finalized by human reviewer.",
        {
          ...annotation,
          originalAiShapes,
          maskCoveragePercent: liveCoveragePercent,
        }
      );

      showToast("success", "Edited Annotation Saved", "Annotation finalized and saved as expert ground truth.");
      setMode("finalized");
      setHasUnsavedChanges(false);
      navigate(`/workspace/${imageId}`);
    } finally {
      setIsSaving(false);
    }
  };

  // 5. Cancel Editing: restores untouched original AI suggestion
  const handleCancelEditing = () => {
    if (hasUnsavedChanges) {
      if (!window.confirm("Discard unsaved mask edits and restore original AI suggestion?")) {
        return;
      }
    }
    setShapes(originalAiShapes);
    setActiveColor("#22D3EE");
    setHasUnsavedChanges(false);
    setMode("ai-review");
    showToast("info", "Editing Cancelled", "Restored original AI suggestion.");
    navigate(`/workspace/${imageId}/ai-review`);
  };

  // 6. Manual Save
  const handleSaveManual = async () => {
    if (!imageId) return;
    if (isMaskEmpty && shapes.length > 0) {
      if (!window.confirm("Save an empty annotation? This indicates no pneumothorax region was retained.")) {
        return;
      }
    }

    setIsSaving(true);
    try {
      const finalizedShapes = shapes.map((s) => ({ ...s, color: "#16A34A" }));
      setShapes(finalizedShapes);
      setActiveColor("#16A34A");

      await saveAnnotation(
        imageId,
        finalizedShapes,
        "manual",
        "reviewed",
        user?.name || "Dr. Sarah Jenkins",
        "Manual annotation saved and finalized."
      );

      showToast("success", "Annotation Saved", "Manual annotation finalized.");
      setMode("finalized");
      setHasUnsavedChanges(false);
      navigate(`/workspace/${imageId}`);
    } finally {
      setIsSaving(false);
    }
  };

  // 7. Cancel Manual Annotation
  const handleCancelManual = () => {
    if (hasUnsavedChanges) {
      if (!window.confirm("Discard unsaved manual annotations?")) {
        return;
      }
    }
    setShapes(annotation?.shapes || []);
    setHasUnsavedChanges(false);
    setMode(image?.status === "finalized" ? "finalized" : "view");
    navigate(`/workspace/${imageId}`);
  };

  // Navigation Back Button Logic
  const handleBackNavigation = () => {
    if (mode === "ai-edit") {
      handleCancelEditing();
      return;
    }
    if (mode === "manual-edit") {
      handleCancelManual();
      return;
    }
    if (mode === "ai-review") {
      navigate(`/workspace/${imageId}`);
      setMode(image?.status === "finalized" ? "finalized" : "view");
      return;
    }
    if (hasUnsavedChanges) {
      if (!window.confirm("You have unsaved changes. Return to dashboard?")) {
        return;
      }
    }
    navigate("/dashboard");
  };

  // Back button label
  const backButtonLabel = useMemo(() => {
    if (mode === "ai-edit") return "Back to AI Review";
    if (mode === "manual-edit") return "Cancel and Return";
    if (mode === "ai-review") return "Back to Workspace";
    return "Back to Dashboard";
  }, [mode]);

  // Keyboard Shortcuts Listener
  useEffect(() => {
    if (!settings.enableKeyboardShortcuts) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (e.ctrlKey && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) handleRedo();
        else handleUndo();
        return;
      }
      if (e.ctrlKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (mode === "ai-edit") handleSaveEdited();
        else if (mode === "manual-edit") handleSaveManual();
        return;
      }

      // Tool switches active in edit modes
      if (mode === "manual-edit" || mode === "ai-edit") {
        const key = e.key.toLowerCase();
        if (key === "b") setActiveTool("brush");
        else if (key === "p") setActiveTool("polygon");
        else if (key === "e") setActiveTool("eraser");
        else if (key === "v") setActiveTool("select");
        else if (key === "h" || e.code === "Space") setActiveTool("pan");
      }

      const key = e.key.toLowerCase();
      if (key === "o") setIsOverlayVisible((v) => !v);
      else if (key === "+") setZoom((z) => clampZoom(z * 1.15));
      else if (key === "-") setZoom((z) => clampZoom(z / 1.15));
      else if (key === "0") handleFitToScreen();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    settings.enableKeyboardShortcuts,
    mode,
    shapes,
    historyPast,
    historyFuture,
    handleFitToScreen,
  ]);

  if (!image) {
    return (
      <div className="p-8 text-center bg-white rounded-xl border border-slate-200 shadow-card max-w-lg mx-auto my-12">
        <h3 className="text-base font-bold text-slate-900 mb-2">Image Not Found</h3>
        <p className="text-xs text-slate-500 mb-6">
          The requested image could not be located in your active workspace session.
        </p>
        <Button variant="primary" onClick={() => navigate("/dashboard")}>
          Return to Dashboard
        </Button>
      </div>
    );
  }

  const isDrawingEnabled = mode === "manual-edit" || mode === "ai-edit";

  return (
    <div className="flex flex-col h-full w-full bg-[#08111f] overflow-hidden select-none">
      {/* Row 1: Workspace Top Header */}
      <div className="bg-white border-b border-slate-200 px-4 py-2.5 flex items-center justify-between gap-4 z-20 shadow-2xs flex-shrink-0">
        {/* Left: Back Button & Image Identifiers */}
        <div className="flex items-center gap-3 min-w-0">
          <Button
            variant="secondary"
            size="sm"
            onClick={handleBackNavigation}
            leftIcon={<ArrowLeft className="w-4 h-4" />}
          >
            {backButtonLabel}
          </Button>

          <div className="flex items-center gap-2 min-w-0 truncate">
            <h2 className="font-bold text-sm text-slate-900 truncate" title={image.name}>
              {image.name}
            </h2>
            <ModalityBadge modality={image.modality} />
            <StatusBadge status={image.status} size="sm" />
            {isAiLoading && (
              <span className="inline-flex items-center gap-1 text-xs text-teal-700 bg-teal-50 px-2 py-0.5 rounded-md border border-teal-200 font-medium">
                <span className="w-2 h-2 rounded-full bg-teal-600 animate-pulse" />
                Generating AI suggestion…
              </span>
            )}
          </div>
        </div>

        {/* Right: Actions according to WorkspaceMode */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* VIEW MODE: Request AI and Manual Annotation (rendered ONLY in top header) */}
          {mode === "view" && (
            <>
              <Button
                variant="teal"
                size="sm"
                onClick={handleRequestAi}
                isLoading={isAiLoading}
                disabled={isAiLoading}
                leftIcon={<Sparkles className="w-4 h-4" />}
              >
                {isAiLoading ? "Generating AI suggestion…" : "Request AI Suggestion"}
              </Button>

              <Button
                variant="primary"
                size="sm"
                onClick={() => {
                  setMode("manual-edit");
                  setActiveTool("polygon");
                  navigate(`/workspace/${imageId}/manual`);
                }}
                disabled={isAiLoading}
                leftIcon={<PenTool className="w-4 h-4" />}
              >
                Manual Annotation
              </Button>
            </>
          )}

          {/* FINALIZED MODE: Edit Annotation and Export */}
          {mode === "finalized" && (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                setMode("manual-edit");
                setActiveTool("polygon");
                navigate(`/workspace/${imageId}/manual`);
              }}
              leftIcon={<Edit2 className="w-4 h-4" />}
            >
              Edit Annotation
            </Button>
          )}

          {/* Export Button (Available across modes when an annotation exists) */}
          <Button
            variant="secondary"
            size="sm"
            onClick={() => annotation && exportAnnotationAsJson(image, annotation)}
            disabled={!annotation || shapes.length === 0}
            title="Export annotation JSON & metadata"
            leftIcon={<Download className="w-4 h-4" />}
          >
            Export
          </Button>

          {/* Toggle Sidebar */}
          <button
            onClick={() => setIsMetadataPanelOpen(!isMetadataPanelOpen)}
            className="p-2 text-slate-500 hover:text-slate-900 hover:bg-slate-100 rounded-md border border-slate-200 transition-colors ml-1"
            title={isMetadataPanelOpen ? "Hide metadata panel" : "Show metadata panel"}
            aria-label="Toggle metadata panel"
          >
            {isMetadataPanelOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Row 2: Annotation Drawing Toolbar */}
      <AnnotationToolbar
        activeTool={activeTool}
        setActiveTool={setActiveTool}
        brushRadius={brushRadius}
        setBrushRadius={setBrushRadius}
        overlayOpacity={overlayOpacity}
        setOverlayOpacity={setOverlayOpacity}
        isOverlayVisible={isOverlayVisible}
        setIsOverlayVisible={setIsOverlayVisible}
        canUndo={historyPast.length > 0 && isDrawingEnabled}
        canRedo={historyFuture.length > 0 && isDrawingEnabled}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onClear={handleClear}
        onZoomIn={() => setZoom((z) => clampZoom(z * 1.15))}
        onZoomOut={() => setZoom((z) => clampZoom(z / 1.15))}
        onFitToScreen={handleFitToScreen}
        onResetView={handleFitToScreen}
        brightness={brightness}
        setBrightness={setBrightness}
        contrast={contrast}
        setContrast={setContrast}
        invert={invert}
        setInvert={setInvert}
        isDrawingDisabled={!isDrawingEnabled}
      />

      {/* Row 3: Main Workspace Viewer & Right Metadata Panel */}
      <div className="flex-1 flex overflow-hidden min-h-0 min-w-0 relative">
        {/* Full Image Viewer Area */}
        <div className="flex-1 relative overflow-hidden min-h-0 min-w-0 bg-[#08111f]">
          <AnnotationCanvas
            imageSrc={image.previewUrl}
            imageWidth={image.width}
            imageHeight={image.height}
            shapes={shapes}
            onShapesChange={handleShapesChange}
            activeTool={isDrawingEnabled ? activeTool : "pan"}
            brushRadius={brushRadius}
            overlayOpacity={overlayOpacity}
            isOverlayVisible={isOverlayVisible}
            activeColor={activeColor}
            zoom={zoom}
            panX={panX}
            panY={panY}
            setZoom={setZoom}
            setPanX={setPanX}
            setPanY={setPanY}
            brightness={brightness}
            contrast={contrast}
            invert={invert}
            onContainerResize={handleContainerResize}
            onCoverageCalculated={handleCoverageCalculated}
          />

          {/* Compact Non-blocking Processing Indicator Overlay */}
          {mode === "ai-processing" && (
            <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur-xs border border-teal-500 text-white px-4 py-2.5 rounded-xl shadow-lg flex items-center gap-3 z-30 pointer-events-none">
              <div className="w-4 h-4 border-2 border-teal-400 border-t-transparent rounded-full animate-spin" />
              <div className="text-xs">
                <span className="font-semibold block">Generating AI suggestion…</span>
                <span className="text-[10px] text-slate-400">Analyzing radiograph for candidate regions</span>
              </div>
            </div>
          )}
        </div>

        {/* Right Metadata & Review Action Panel */}
        {isMetadataPanelOpen && (
          <ImageMetadataPanel
            image={image}
            annotation={annotation}
            mode={mode}
            isOverlayVisible={isOverlayVisible}
            onAccept={handleAcceptAi}
            onReject={handleRejectAi}
            onEdit={handleEditAi}
            onCancelEdit={handleCancelEditing}
            onSaveEdit={handleSaveEdited}
            onCancelManual={handleCancelManual}
            onSaveManual={handleSaveManual}
            isProcessing={isAiLoading}
            isSaving={isSaving}
            liveCoveragePercent={liveCoveragePercent}
            isMaskEmpty={isMaskEmpty}
          />
        )}
      </div>
    </div>
  );
};
