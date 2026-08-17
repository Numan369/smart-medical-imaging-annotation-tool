import React from "react";
import clsx from "clsx";
import { AnnotationStatus } from "../../types";
import { CheckCircle2, Clock, AlertTriangle, FileImage } from "lucide-react";

interface StatusBadgeProps {
  status: AnnotationStatus;
  size?: "sm" | "md";
  className?: string;
  showDescription?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = "md",
  className,
}) => {
  const configMap: Record<
    AnnotationStatus,
    { label: string; icon: React.ReactNode; bg: string; text: string; border: string }
  > = {
    "finalized": {
      label: "Finalized",
      icon: <CheckCircle2 className={size === "sm" ? "w-3 h-3" : "w-3.5 h-3.5"} />,
      bg: "bg-emerald-50",
      text: "text-emerald-800",
      border: "border-emerald-200",
    },
    "awaiting-review": {
      label: "Awaiting Review",
      icon: <AlertTriangle className={size === "sm" ? "w-3 h-3" : "w-3.5 h-3.5"} />,
      bg: "bg-amber-50",
      text: "text-amber-800",
      border: "border-amber-200",
    },
    "unannotated": {
      label: "Unannotated",
      icon: <Clock className={size === "sm" ? "w-3 h-3" : "w-3.5 h-3.5"} />,
      bg: "bg-slate-50",
      text: "text-slate-700",
      border: "border-slate-200",
    },
  };

  const item = configMap[status] || configMap["unannotated"];

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 font-medium rounded-full border shadow-2xs select-none",
        item.bg,
        item.text,
        item.border,
        size === "sm" ? "px-2 py-0.5 text-xs" : "px-2.5 py-1 text-xs",
        className
      )}
    >
      <span className="flex-shrink-0">{item.icon}</span>
      <span>{item.label}</span>
    </span>
  );
};

export const ModalityBadge: React.FC<{ modality: string; className?: string }> = ({
  modality,
  className,
}) => {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono font-medium bg-slate-100 text-slate-700 border border-slate-200",
        className
      )}
    >
      <FileImage className="w-3 h-3 text-slate-500" />
      <span>{modality === "xray" ? "CHEST X-RAY" : modality.toUpperCase()}</span>
    </span>
  );
};
