import React, { useRef, useEffect, useState, useCallback } from "react";
import { AnnotationShape, NormalizedPoint } from "../../types";
import { WorkspaceTool } from "./AnnotationToolbar";
import { screenToNormalizedImage, clampZoom } from "../../utils/coordinate";

interface AnnotationCanvasProps {
  imageSrc?: string;
  imageWidth?: number;
  imageHeight?: number;
  shapes: AnnotationShape[];
  onShapesChange: (newShapes: AnnotationShape[]) => void;
  activeTool: WorkspaceTool;
  brushRadius: number;
  overlayOpacity: number;
  isOverlayVisible: boolean;
  activeColor: string;
  zoom: number;
  panX: number;
  panY: number;
  setZoom: React.Dispatch<React.SetStateAction<number>>;
  setPanX: React.Dispatch<React.SetStateAction<number>>;
  setPanY: React.Dispatch<React.SetStateAction<number>>;
  brightness: number;
  contrast: number;
  invert: boolean;
  onContainerResize?: (width: number, height: number) => void;
  onCoverageCalculated?: (coveragePercent: number, isEmpty: boolean) => void;
}

export const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({
  imageSrc,
  imageWidth = 512,
  imageHeight = 512,
  shapes,
  onShapesChange,
  activeTool,
  brushRadius,
  overlayOpacity,
  isOverlayVisible,
  activeColor,
  zoom,
  panX,
  panY,
  setZoom,
  setPanX,
  setPanY,
  brightness,
  contrast,
  invert,
  onContainerResize,
  onCoverageCalculated,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [imageObj, setImageObj] = useState<HTMLImageElement | null>(null);
  const [imageLoadError, setImageLoadError] = useState(false);

  // Interaction State
  const [isPointerDown, setIsPointerDown] = useState(false);
  const [draftPolygonPoints, setDraftPolygonPoints] = useState<NormalizedPoint[]>([]);
  const [currentStrokePoints, setCurrentStrokePoints] = useState<NormalizedPoint[]>([]);
  const [bboxStartPoint, setBboxStartPoint] = useState<NormalizedPoint | null>(null);
  const [lastPanPos, setLastPanPos] = useState<{ x: number; y: number } | null>(null);
  const [mouseClientPos, setMouseClientPos] = useState<{ x: number; y: number } | null>(null);

  // Offscreen mask canvas for rasterized additive and subtractive mask compositing
  const offscreenMaskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenTintCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Initialize offscreen canvases
  useEffect(() => {
    if (!offscreenMaskCanvasRef.current) {
      offscreenMaskCanvasRef.current = document.createElement("canvas");
    }
    if (!offscreenTintCanvasRef.current) {
      offscreenTintCanvasRef.current = document.createElement("canvas");
    }
    offscreenMaskCanvasRef.current.width = imageWidth;
    offscreenMaskCanvasRef.current.height = imageHeight;
    offscreenTintCanvasRef.current.width = imageWidth;
    offscreenTintCanvasRef.current.height = imageHeight;
  }, [imageWidth, imageHeight]);

  // Cancel draft polygon helper
  const cancelDraftPolygon = useCallback(() => {
    setDraftPolygonPoints([]);
    setCurrentStrokePoints([]);
    setBboxStartPoint(null);
  }, []);

  // Clear incomplete draft points whenever activeTool is not "polygon"
  useEffect(() => {
    if (activeTool !== "polygon") {
      setDraftPolygonPoints([]);
    }
  }, [activeTool]);

  // Clear incomplete draft points on image change
  useEffect(() => {
    cancelDraftPolygon();
  }, [imageSrc, cancelDraftPolygon]);

  // Watch container dimensions with ResizeObserver
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          onContainerResize?.(width, height);
        }
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, [onContainerResize]);

  // Load underlying image element
  useEffect(() => {
    if (!imageSrc) {
      setImageObj(null);
      return;
    }
    setImageLoadError(false);
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;
    img.onload = () => {
      setImageObj(img);
      setImageLoadError(false);
    };
    img.onerror = () => {
      console.warn("Could not load image source:", imageSrc);
      setImageLoadError(true);
      setImageObj(null);
    };
  }, [imageSrc]);

  // Complete and commit polygon to saved shapes
  const completePolygon = useCallback(() => {
    if (draftPolygonPoints.length < 3) {
      setDraftPolygonPoints([]);
      return;
    }

    const newShape: AnnotationShape = {
      id: `shp-poly-${Date.now()}`,
      type: "polygon",
      points: draftPolygonPoints,
      color: activeColor,
      label: "Manual Polygon",
      createdAt: new Date().toISOString(),
    };

    onShapesChange([...shapes, newShape]);
    setDraftPolygonPoints([]);
  }, [draftPolygonPoints, activeColor, onShapesChange, shapes]);

  // Keyboard shortcut listener for canvas actions (Escape to cancel, Enter to complete polygon)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        cancelDraftPolygon();
      } else if (e.key === "Enter" && activeTool === "polygon" && draftPolygonPoints.length >= 3) {
        e.preventDefault();
        completePolygon();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [cancelDraftPolygon, activeTool, draftPolygonPoints.length, completePolygon]);

  // Render mask to offscreen canvas and calculate coverage
  const updateOffscreenMask = useCallback(() => {
    const maskCanvas = offscreenMaskCanvasRef.current;
    const tintCanvas = offscreenTintCanvasRef.current;
    if (!maskCanvas || !tintCanvas) return;

    if (maskCanvas.width !== imageWidth || maskCanvas.height !== imageHeight) {
      maskCanvas.width = imageWidth;
      maskCanvas.height = imageHeight;
    }
    if (tintCanvas.width !== imageWidth || tintCanvas.height !== imageHeight) {
      tintCanvas.width = imageWidth;
      tintCanvas.height = imageHeight;
    }

    const maskCtx = maskCanvas.getContext("2d");
    const tintCtx = tintCanvas.getContext("2d");
    if (!maskCtx || !tintCtx) return;

    // 1. Clear mask canvas (transparent background)
    maskCtx.clearRect(0, 0, imageWidth, imageHeight);

    // 2. Render all committed shapes in chronological order
    shapes.forEach((shape) => {
      if (shape.type === "eraser-stroke" && shape.points.length > 0) {
        // Subtractive Erase Operation
        maskCtx.globalCompositeOperation = "destination-out";
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        maskCtx.strokeStyle = "rgba(0,0,0,1)";
        maskCtx.lineWidth = shape.brushRadius || brushRadius;

        maskCtx.beginPath();
        shape.points.forEach((p, idx) => {
          const px = p.x * imageWidth;
          const py = p.y * imageHeight;
          if (idx === 0) maskCtx.moveTo(px, py);
          else maskCtx.lineTo(px, py);
        });
        maskCtx.stroke();
      } else if (shape.type === "brush-stroke" && shape.points.length > 0) {
        // Additive Brush Stroke
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        maskCtx.strokeStyle = "#ffffff";
        maskCtx.lineWidth = shape.brushRadius || brushRadius;

        maskCtx.beginPath();
        shape.points.forEach((p, idx) => {
          const px = p.x * imageWidth;
          const py = p.y * imageHeight;
          if (idx === 0) maskCtx.moveTo(px, py);
          else maskCtx.lineTo(px, py);
        });
        maskCtx.stroke();
      } else if (shape.type === "polygon" && shape.points.length >= 3) {
        // Additive Polygon
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.fillStyle = "#ffffff";
        maskCtx.strokeStyle = "#ffffff";
        maskCtx.lineWidth = 1.5;

        maskCtx.beginPath();
        shape.points.forEach((p, idx) => {
          const px = p.x * imageWidth;
          const py = p.y * imageHeight;
          if (idx === 0) maskCtx.moveTo(px, py);
          else maskCtx.lineTo(px, py);
        });
        maskCtx.closePath();
        maskCtx.fill();
        maskCtx.stroke();
      } else if (shape.type === "bbox" && shape.points.length >= 2) {
        // Additive Bounding Box
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.fillStyle = "#ffffff";
        const p1 = shape.points[0];
        const p2 = shape.points[1];
        const x = Math.min(p1.x, p2.x) * imageWidth;
        const y = Math.min(p1.y, p2.y) * imageHeight;
        const w = Math.abs(p1.x - p2.x) * imageWidth;
        const h = Math.abs(p1.y - p2.y) * imageHeight;
        maskCtx.fillRect(x, y, w, h);
      }
    });

    // 3. Render in-progress active brush or eraser stroke
    if (currentStrokePoints.length > 0) {
      if (activeTool === "eraser") {
        maskCtx.globalCompositeOperation = "destination-out";
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        maskCtx.strokeStyle = "rgba(0,0,0,1)";
        maskCtx.lineWidth = brushRadius;

        maskCtx.beginPath();
        currentStrokePoints.forEach((p, idx) => {
          const px = p.x * imageWidth;
          const py = p.y * imageHeight;
          if (idx === 0) maskCtx.moveTo(px, py);
          else maskCtx.lineTo(px, py);
        });
        maskCtx.stroke();
      } else if (activeTool === "brush") {
        maskCtx.globalCompositeOperation = "source-over";
        maskCtx.lineCap = "round";
        maskCtx.lineJoin = "round";
        maskCtx.strokeStyle = "#ffffff";
        maskCtx.lineWidth = brushRadius;

        maskCtx.beginPath();
        currentStrokePoints.forEach((p, idx) => {
          const px = p.x * imageWidth;
          const py = p.y * imageHeight;
          if (idx === 0) maskCtx.moveTo(px, py);
          else maskCtx.lineTo(px, py);
        });
        maskCtx.stroke();
      }
    }

    // 4. Render in-progress polygon ONLY if activeTool is polygon
    if (activeTool === "polygon" && draftPolygonPoints.length > 0) {
      maskCtx.globalCompositeOperation = "source-over";
      maskCtx.strokeStyle = "#ffffff";
      maskCtx.lineWidth = 1.5;

      maskCtx.beginPath();
      draftPolygonPoints.forEach((p, idx) => {
        const px = p.x * imageWidth;
        const py = p.y * imageHeight;
        if (idx === 0) maskCtx.moveTo(px, py);
        else maskCtx.lineTo(px, py);
      });
      maskCtx.stroke();
    }

    maskCtx.globalCompositeOperation = "source-over";

    // 5. Generate tinted representation
    tintCtx.clearRect(0, 0, imageWidth, imageHeight);
    tintCtx.fillStyle = activeColor;
    tintCtx.fillRect(0, 0, imageWidth, imageHeight);
    tintCtx.globalCompositeOperation = "destination-in";
    tintCtx.drawImage(maskCanvas, 0, 0);
    tintCtx.globalCompositeOperation = "source-over";

    // 6. Recalculate coverage metrics on mask change
    try {
      const imgData = maskCtx.getImageData(0, 0, imageWidth, imageHeight);
      const data = imgData.data;
      let nonZeroCount = 0;
      for (let i = 3; i < data.length; i += 4) {
        if (data[i] > 30) {
          nonZeroCount++;
        }
      }
      const totalPixels = imageWidth * imageHeight;
      const coveragePercent = Number(((nonZeroCount / totalPixels) * 100).toFixed(2));
      onCoverageCalculated?.(coveragePercent, nonZeroCount === 0);
    } catch {
      // Safe fallback
    }
  }, [
    imageWidth,
    imageHeight,
    shapes,
    currentStrokePoints,
    draftPolygonPoints,
    activeTool,
    brushRadius,
    activeColor,
    onCoverageCalculated,
  ]);

  // Redraw the main display canvas on state changes
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width <= 0 || height <= 0) return;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    // Clear background
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#08111f";
    ctx.fillRect(0, 0, width, height);

    const safeZoom = clampZoom(zoom);

    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(safeZoom, safeZoom);

    // 1. Render underlying medical radiograph
    if (imageObj && !imageLoadError) {
      ctx.save();
      const filterString = `${invert ? "invert(100%) " : ""}brightness(${brightness}%) contrast(${contrast}%)`;
      ctx.filter = filterString;
      ctx.drawImage(imageObj, 0, 0, imageWidth, imageHeight);
      ctx.restore();
    } else {
      ctx.fillStyle = "#111c2e";
      ctx.fillRect(0, 0, imageWidth, imageHeight);
      ctx.fillStyle = "#64748b";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(imageLoadError ? "Preview unavailable - re-add image" : "Loading Radiograph...", imageWidth / 2, imageHeight / 2);
    }

    // Border around radiograph frame
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1 / safeZoom;
    ctx.strokeRect(0, 0, imageWidth, imageHeight);

    // 2. Render composite tinted annotation overlay
    if (isOverlayVisible && offscreenTintCanvasRef.current) {
      ctx.save();
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(offscreenTintCanvasRef.current, 0, 0, imageWidth, imageHeight);
      ctx.restore();
    }

    // 3. Render interactive polygon control points ONLY if actively drawing polygon
    if (activeTool === "polygon" && draftPolygonPoints.length > 0) {
      draftPolygonPoints.forEach((p, idx) => {
        const px = p.x * imageWidth;
        const py = p.y * imageHeight;
        ctx.fillStyle = idx === 0 ? "#FDE047" : "#FFFFFF";
        ctx.beginPath();
        ctx.arc(px, py, 4 / safeZoom, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#0F172A";
        ctx.lineWidth = 1 / safeZoom;
        ctx.stroke();
      });
    }

    ctx.restore();

    // 4. Render live circular brush/eraser cursor over pointer
    if (mouseClientPos && (activeTool === "brush" || activeTool === "eraser")) {
      const radiusScreen = (brushRadius / 2) * safeZoom;
      ctx.save();
      ctx.beginPath();
      ctx.arc(mouseClientPos.x, mouseClientPos.y, radiusScreen, 0, Math.PI * 2);
      ctx.strokeStyle = activeTool === "eraser" ? "rgba(239, 68, 68, 0.9)" : "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();
    }
  }, [
    imageObj,
    imageLoadError,
    imageWidth,
    imageHeight,
    panX,
    panY,
    zoom,
    brightness,
    contrast,
    invert,
    isOverlayVisible,
    overlayOpacity,
    draftPolygonPoints,
    mouseClientPos,
    activeTool,
    brushRadius,
  ]);

  // Update offscreen mask first, then redraw display canvas
  useEffect(() => {
    updateOffscreenMask();
    redraw();
  }, [updateOffscreenMask, redraw]);

  // Pointer Down Handlers
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    } catch {
      // Safe ignore
    }

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    // Pan Mode or Middle click
    if (activeTool === "pan" || e.button === 1 || e.buttons === 4) {
      setIsPointerDown(true);
      setLastPanPos({ x: e.clientX, y: e.clientY });
      return;
    }

    const normPt = screenToNormalizedImage(clientX, clientY, panX, panY, zoom, imageWidth, imageHeight);

    if (activeTool === "brush" || activeTool === "eraser") {
      setIsPointerDown(true);
      setCurrentStrokePoints([normPt]);
    } else if (activeTool === "bbox") {
      setIsPointerDown(true);
      setBboxStartPoint(normPt);
    } else if (activeTool === "polygon") {
      if (draftPolygonPoints.length >= 3) {
        const start = draftPolygonPoints[0];
        const dist = Math.hypot(
          (normPt.x - start.x) * imageWidth * clampZoom(zoom),
          (normPt.y - start.y) * imageHeight * clampZoom(zoom)
        );
        if (dist < 18) {
          completePolygon();
          return;
        }
      }
      setDraftPolygonPoints((prev) => [...prev, normPt]);
    }
  };

  // Pointer Move Handlers
  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    setMouseClientPos({ x: clientX, y: clientY });

    if (lastPanPos && (activeTool === "pan" || isPointerDown)) {
      const dx = e.clientX - lastPanPos.x;
      const dy = e.clientY - lastPanPos.y;
      setPanX((prev) => prev + dx);
      setPanY((prev) => prev + dy);
      setLastPanPos({ x: e.clientX, y: e.clientY });
      return;
    }

    if (!isPointerDown) return;

    const normPt = screenToNormalizedImage(clientX, clientY, panX, panY, zoom, imageWidth, imageHeight);

    if (activeTool === "brush" || activeTool === "eraser") {
      setCurrentStrokePoints((prev) => [...prev, normPt]);
    }
  };

  // Safe pointer release & shape committing
  const handleReleasePointer = (e: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      if ((e.target as HTMLElement).hasPointerCapture?.(e.pointerId)) {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      }
    } catch {
      // Safe ignore
    }

    if (lastPanPos) {
      setLastPanPos(null);
      setIsPointerDown(false);
      return;
    }

    if (activeTool === "brush" && currentStrokePoints.length > 0) {
      const newShape: AnnotationShape = {
        id: `shp-brush-${Date.now()}`,
        type: "brush-stroke",
        points: currentStrokePoints,
        brushRadius,
        color: activeColor,
        label: "Manual Brush Stroke",
        createdAt: new Date().toISOString(),
      };
      onShapesChange([...shapes, newShape]);
      setCurrentStrokePoints([]);
    } else if (activeTool === "eraser" && currentStrokePoints.length > 0) {
      // Commit subtractive eraser stroke
      const newShape: AnnotationShape = {
        id: `shp-erase-${Date.now()}`,
        type: "eraser-stroke",
        points: currentStrokePoints,
        brushRadius,
        color: "#000000",
        label: "Subtractive Erase Stroke",
        createdAt: new Date().toISOString(),
      };
      onShapesChange([...shapes, newShape]);
      setCurrentStrokePoints([]);
    } else if (activeTool === "bbox" && bboxStartPoint) {
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const normPt = screenToNormalizedImage(
          e.clientX - rect.left,
          e.clientY - rect.top,
          panX,
          panY,
          zoom,
          imageWidth,
          imageHeight
        );
        const newShape: AnnotationShape = {
          id: `shp-bbox-${Date.now()}`,
          type: "bbox",
          points: [bboxStartPoint, normPt],
          color: activeColor,
          label: "Region Bounding Box",
          createdAt: new Date().toISOString(),
        };
        onShapesChange([...shapes, newShape]);
      }
      setBboxStartPoint(null);
    }

    setIsPointerDown(false);
  };

  const handlePointerLeave = () => {
    setMouseClientPos(null);
  };

  const handleDoubleClick = () => {
    if (activeTool === "polygon" && draftPolygonPoints.length >= 3) {
      completePolygon();
    }
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.88;
    const currentClamped = clampZoom(zoom);
    const newZoom = clampZoom(currentClamped * zoomFactor);

    if (newZoom === currentClamped) return;

    const newPanX = clientX - (clientX - panX) * (newZoom / currentClamped);
    const newPanY = clientY - (clientY - panY) * (newZoom / currentClamped);

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  };

  const displayedZoomPercent = Math.round(clampZoom(zoom) * 100);

  return (
    <div ref={containerRef} className="relative w-full h-full min-h-[300px] bg-[#08111f] overflow-hidden select-none">
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handleReleasePointer}
        onPointerCancel={handleReleasePointer}
        onPointerLeave={handlePointerLeave}
        onLostPointerCapture={handleReleasePointer}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
        className={`w-full h-full touch-none block ${
          activeTool === "pan"
            ? "cursor-grab active:cursor-grabbing"
            : activeTool === "eraser" || activeTool === "brush"
            ? "cursor-none"
            : "cursor-crosshair"
        }`}
      />

      {/* Viewport Floating Info (Only displays draft polygon info when actively drawing a polygon) */}
      <div className="absolute bottom-3 left-3 bg-slate-900/85 backdrop-blur-xs border border-slate-700 text-slate-300 px-3 py-1.5 rounded-lg text-xs font-mono flex items-center gap-3 pointer-events-none z-10">
        <span>Zoom: {displayedZoomPercent}%</span>
        <span>Resolution: {imageWidth}×{imageHeight}</span>
        {activeTool === "polygon" && draftPolygonPoints.length > 0 && (
          <span className="text-amber-400 font-sans">
            Polygon: {draftPolygonPoints.length} pts (Double-click or Enter to close; Esc to cancel)
          </span>
        )}
      </div>
    </div>
  );
};
