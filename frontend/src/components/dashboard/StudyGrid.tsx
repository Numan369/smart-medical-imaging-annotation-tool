import React from "react";
import { MedicalImage } from "../../types";
import { StatusBadge, ModalityBadge } from "../common/Badge";
import { Button } from "../common/Button";
import { useNavigate } from "react-router-dom";
import { ExternalLink, Trash2, Calendar, Info } from "lucide-react";

interface StudyGridProps {
  images: MedicalImage[];
  onDeleteImage: (id: string) => void;
  onShowDetails: (image: MedicalImage) => void;
}

export const StudyGrid: React.FC<StudyGridProps> = ({
  images,
  onDeleteImage,
  onShowDetails,
}) => {
  const navigate = useNavigate();

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {images.map((image) => (
        <div
          key={image.id}
          className="bg-white rounded-xl border border-slate-200 shadow-card hover:shadow-md transition-all duration-200 flex flex-col overflow-hidden group"
        >
          {/* Thumbnail Container */}
          <div
            onClick={() => navigate(`/workspace/${image.id}`)}
            className="relative h-44 bg-slate-900 cursor-pointer overflow-hidden flex items-center justify-center"
          >
            {image.previewUrl ? (
              <img
                src={image.previewUrl}
                alt={image.name}
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
                className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
              />
            ) : (
              <span className="text-xs font-mono text-slate-400">CXR PREVIEW</span>
            )}

            <div className="absolute top-2.5 left-2.5">
              <ModalityBadge modality={image.modality} />
            </div>

            <div className="absolute top-2.5 right-2.5">
              <StatusBadge status={image.status} size="sm" />
            </div>
          </div>

          {/* Card Body */}
          <div className="p-4 flex-1 flex flex-col justify-between space-y-3">
            <div>
              <h4 className="font-semibold text-sm text-slate-900 truncate" title={image.name}>
                {image.name}
              </h4>
              <p className="text-xs text-slate-500 mt-0.5">
                {image.status === "finalized"
                  ? `Annotator: ${image.annotator || "Dr. Sarah Jenkins"}`
                  : image.status === "awaiting-review"
                  ? "Awaiting human review"
                  : "Unassigned"}
              </p>
              {image.notes && (
                <p className="text-xs text-slate-600 line-clamp-2 mt-2 leading-relaxed bg-slate-50 p-2 rounded border border-slate-100">
                  {image.notes}
                </p>
              )}
            </div>

            {/* Card Footer: Metadata & Actions */}
            <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
              <div className="flex items-center gap-1">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span>{new Date(image.uploadedAt).toLocaleDateString()}</span>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => onShowDetails(image)}
                  className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded transition-colors"
                  title="View details"
                  aria-label="View image details"
                >
                  <Info className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => onDeleteImage(image.id)}
                  className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors"
                  title="Delete image"
                  aria-label="Delete image"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => navigate(`/workspace/${image.id}`)}
                  className="px-2.5 py-1 text-xs"
                  rightIcon={<ExternalLink className="w-3 h-3" />}
                >
                  Open
                </Button>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
