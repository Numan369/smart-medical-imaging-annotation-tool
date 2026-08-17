import React from "react";
import { FolderSearch, PlusCircle } from "lucide-react";
import { Button } from "./Button";

interface EmptyStateProps {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  actionText,
  onAction,
  icon,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center bg-white rounded-xl border border-dashed border-slate-200">
      <div className="w-12 h-12 rounded-full bg-slate-100 text-slate-500 flex items-center justify-center mb-4">
        {icon || <FolderSearch className="w-6 h-6" />}
      </div>
      <h3 className="text-base font-semibold text-slate-900 mb-1">{title}</h3>
      <p className="text-sm text-slate-500 max-w-sm mb-6">{description}</p>
      {actionText && onAction && (
        <Button variant="primary" onClick={onAction} leftIcon={<PlusCircle className="w-4 h-4" />}>
          {actionText}
        </Button>
      )}
    </div>
  );
};
