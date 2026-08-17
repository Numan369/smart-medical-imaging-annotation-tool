import React from "react";
import { MedicalImage } from "../../types";
import { StatusBadge, ModalityBadge } from "../common/Badge";
import { Button } from "../common/Button";
import { useNavigate } from "react-router-dom";
import { ExternalLink, Trash2, Calendar, Info } from "lucide-react";

interface StudyTableProps {
  images: MedicalImage[];
  onDeleteImage: (id: string) => void;
  onShowDetails: (image: MedicalImage) => void;
}

export const StudyTable: React.FC<StudyTableProps> = ({
  images,
  onDeleteImage,
  onShowDetails,
}) => {
  const navigate = useNavigate();

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm text-slate-700">
          <thead className="bg-slate-50/80 border-b border-slate-200 text-xs font-semibold text-slate-600 uppercase tracking-wider">
            <tr>
              <th className="py-3.5 px-4 w-20">Preview</th>
              <th className="py-3.5 px-4">Image Name</th>
              <th className="py-3.5 px-4">Modality</th>
              <th className="py-3.5 px-4">Added</th>
              <th className="py-3.5 px-4">Status</th>
              <th className="py-3.5 px-4">Annotator</th>
              <th className="py-3.5 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-sans">
            {images.map((image) => (
              <tr
                key={image.id}
                onClick={() => navigate(`/workspace/${image.id}`)}
                className="hover:bg-slate-50/70 transition-colors cursor-pointer group"
              >
                {/* Thumbnail Preview */}
                <td className="py-3 px-4" onClick={(e) => e.stopPropagation()}>
                  <div
                    onClick={() => navigate(`/workspace/${image.id}`)}
                    className="w-12 h-12 rounded-lg bg-slate-900 overflow-hidden border border-slate-200 flex items-center justify-center cursor-pointer shadow-2xs group-hover:border-teal-500 transition-colors"
                  >
                    {image.previewUrl ? (
                      <img
                        src={image.previewUrl}
                        alt={image.name}
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = "none";
                        }}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-[10px] font-mono text-slate-400">CXR</span>
                    )}
                  </div>
                </td>

                {/* Name */}
                <td className="py-3 px-4">
                  <span className="font-semibold text-slate-900 group-hover:text-teal-700 transition-colors text-sm">
                    {image.name}
                  </span>
                  {image.notes && (
                    <p className="text-xs text-slate-500 truncate max-w-xs mt-0.5">
                      {image.notes}
                    </p>
                  )}
                </td>

                {/* Modality */}
                <td className="py-3 px-4">
                  <ModalityBadge modality={image.modality} />
                </td>

                {/* Added Date */}
                <td className="py-3 px-4 text-xs text-slate-600">
                  <div className="flex items-center gap-1.5">
                    <Calendar className="w-3.5 h-3.5 text-slate-400" />
                    <span>{new Date(image.uploadedAt).toLocaleDateString()}</span>
                  </div>
                  <span className="text-[11px] text-slate-400 block mt-0.5">
                    {new Date(image.uploadedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </td>

                {/* Annotation Status */}
                <td className="py-3 px-4">
                  <StatusBadge status={image.status} size="sm" />
                </td>

                {/* Annotator */}
                <td className="py-3 px-4 text-xs text-slate-600">
                  <span className="font-medium">
                    {image.status === "finalized"
                      ? (image.annotator || "Dr. Sarah Jenkins")
                      : image.status === "awaiting-review"
                      ? "Awaiting human review"
                      : "Unassigned"}
                  </span>
                </td>

                {/* Actions */}
                <td className="py-3 px-4 text-right" onClick={(e) => e.stopPropagation()}>
                  <div className="flex items-center justify-end gap-1.5">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => onShowDetails(image)}
                      title="View image details"
                      aria-label="View image details"
                      className="px-2"
                    >
                      <Info className="w-3.5 h-3.5 text-slate-600" />
                    </Button>

                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => navigate(`/workspace/${image.id}`)}
                      leftIcon={<ExternalLink className="w-3.5 h-3.5" />}
                    >
                      Workspace
                    </Button>

                    <button
                      onClick={() => onDeleteImage(image.id)}
                      className="p-1.5 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md transition-colors"
                      title="Delete image"
                      aria-label={`Delete ${image.name}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
