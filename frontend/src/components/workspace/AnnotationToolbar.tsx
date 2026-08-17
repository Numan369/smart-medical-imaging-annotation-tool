import React from "react";
import {
  MousePointer,
  PenTool,
  Paintbrush,
  Eraser,
  Square,
  Hand,
  Undo2,
  Redo2,
  Trash2,
  Eye,
  EyeOff,
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
} from "lucide-react";

export type WorkspaceTool = "select" | "polygon" | "brush" | "eraser" | "bbox" | "pan";

interface AnnotationToolbarProps {
  activeTool: WorkspaceTool;
  setActiveTool: (tool: WorkspaceTool) => void;
  brushRadius: number;
  setBrushRadius: (radius: number) => void;
  overlayOpacity: number;
  setOverlayOpacity: (opacity: number) => void;
  isOverlayVisible: boolean;
  setIsOverlayVisible: React.Dispatch<React.SetStateAction<boolean>>;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onFitToScreen: () => void;
  onResetView: () => void;
  brightness: number;
  setBrightness: (b: number) => void;
  contrast: number;
  setContrast: (c: number) => void;
  invert: boolean;
  setInvert: (inv: boolean) => void;
  isDrawingDisabled?: boolean;
}

export const AnnotationToolbar: React.FC<AnnotationToolbarProps> = ({
  activeTool,
  setActiveTool,
  brushRadius,
  setBrushRadius,
  overlayOpacity,
  setOverlayOpacity,
  isOverlayVisible,
  setIsOverlayVisible,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onClear,
  onZoomIn,
  onZoomOut,
  onFitToScreen,
  onResetView,
  invert,
  setInvert,
  isDrawingDisabled = false,
}) => {
  const tools: { id: WorkspaceTool; label: string; icon: React.ReactNode; shortcut: string; isDrawing?: boolean }[] = [
    { id: "select", label: "Select", icon: <MousePointer className="w-4 h-4" />, shortcut: "V" },
    { id: "polygon", label: "Polygon", icon: <PenTool className="w-4 h-4" />, shortcut: "P", isDrawing: true },
    { id: "brush", label: "Brush", icon: <Paintbrush className="w-4 h-4" />, shortcut: "B", isDrawing: true },
    { id: "eraser", label: "Eraser", icon: <Eraser className="w-4 h-4" />, shortcut: "E", isDrawing: true },
    { id: "bbox", label: "Box", icon: <Square className="w-4 h-4" />, shortcut: "R", isDrawing: true },
    { id: "pan", label: "Pan", icon: <Hand className="w-4 h-4" />, shortcut: "H / Space" },
  ];

  return (
    <div className="bg-white border-b border-slate-200 p-2 flex items-center justify-between gap-3 select-none flex-wrap lg:flex-nowrap shadow-2xs flex-shrink-0">
      {/* Primary Drawing Tools */}
      <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
        {tools.map((t) => {
          const disabled = isDrawingDisabled && t.isDrawing;
          return (
            <button
              key={t.id}
              onClick={() => !disabled && setActiveTool(t.id)}
              disabled={disabled}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors ${
                disabled
                  ? "opacity-35 cursor-not-allowed text-slate-400"
                  : activeTool === t.id
                  ? "bg-navy-900 text-white shadow-xs font-semibold"
                  : "text-slate-700 hover:bg-slate-200 hover:text-slate-900"
              }`}
              title={disabled ? `${t.label} (Select 'Edit Suggestion' or 'Manual Annotation' to enable)` : `${t.label} (${t.shortcut})`}
            >
              {t.icon}
              <span className="hidden sm:inline">{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* Brush Size / Opacity Sliders */}
      <div className="flex items-center gap-3 text-xs text-slate-700">
        {!isDrawingDisabled && (activeTool === "brush" || activeTool === "eraser") && (
          <div className="flex items-center gap-2 bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
            <span className="text-slate-500 font-medium">Brush:</span>
            <input
              type="range"
              min="4"
              max="60"
              value={brushRadius}
              onChange={(e) => setBrushRadius(Number(e.target.value))}
              className="w-16 sm:w-20 accent-teal-600 cursor-pointer"
            />
            <span className="font-mono text-[11px] w-6">{brushRadius}px</span>
          </div>
        )}

        <div className="flex items-center gap-2 bg-slate-50 px-2.5 py-1 rounded-md border border-slate-200">
          <span className="text-slate-500 font-medium">Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(Number(e.target.value))}
            className="w-16 sm:w-20 accent-teal-600 cursor-pointer"
          />
          <span className="font-mono text-[11px] w-8">{Math.round(overlayOpacity * 100)}%</span>
        </div>

        {/* Mask visibility toggle */}
        <button
          onClick={() => setIsOverlayVisible((prev) => !prev)}
          className={`flex items-center gap-1 px-2.5 py-1 rounded-md border text-xs font-medium transition-colors ${
            isOverlayVisible
              ? "bg-slate-100 border-slate-300 text-slate-800"
              : "bg-amber-50 border-amber-300 text-amber-800"
          }`}
          title="Toggle overlay visibility (O)"
        >
          {isOverlayVisible ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          <span className="hidden md:inline">{isOverlayVisible ? "Overlay On" : "Overlay Hidden"}</span>
        </button>
      </div>

      {/* Viewport and History Actions */}
      <div className="flex items-center gap-1 ml-auto">
        {/* Undo / Redo */}
        <button
          onClick={onUndo}
          disabled={!canUndo}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent"
          title="Undo (Ctrl + Z)"
        >
          <Undo2 className="w-4 h-4" />
        </button>
        <button
          onClick={onRedo}
          disabled={!canRedo}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100 disabled:opacity-30 disabled:hover:bg-transparent"
          title="Redo (Ctrl + Shift + Z)"
        >
          <Redo2 className="w-4 h-4" />
        </button>
        <button
          onClick={onClear}
          disabled={isDrawingDisabled}
          className="p-1.5 rounded text-slate-600 hover:text-red-600 hover:bg-red-50 disabled:opacity-30 disabled:hover:bg-transparent"
          title="Clear all manual annotations on this image"
        >
          <Trash2 className="w-4 h-4" />
        </button>

        <div className="h-4 w-px bg-slate-200 mx-1" />

        {/* Viewport Zoom / Fit */}
        <button
          onClick={onZoomIn}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100"
          title="Zoom In (+)"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={onZoomOut}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100"
          title="Zoom Out (-)"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={onFitToScreen}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100"
          title="Fit image to screen (0)"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
        <button
          onClick={onResetView}
          className="p-1.5 rounded text-slate-600 hover:text-navy-900 hover:bg-slate-100"
          title="Reset zoom and center view"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Contrast / Invert Toggle */}
        <button
          onClick={() => setInvert(!invert)}
          className={`text-xs px-2 py-1 rounded font-medium border transition-colors ${
            invert ? "bg-slate-800 text-white border-slate-900" : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100"
          }`}
          title="Invert radiograph polarity (bright bones vs dark bones)"
        >
          Invert
        </button>
      </div>
    </div>
  );
};
