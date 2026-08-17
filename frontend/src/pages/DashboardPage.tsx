import React, { useMemo, useState } from "react";
import { useAppData } from "../context/AppDataContext";
import { useToast } from "../context/ToastContext";
import { MedicalImage } from "../types";
import { StatsCard } from "../components/dashboard/StatsCard";
import { FilterToolbar } from "../components/dashboard/FilterToolbar";
import { StudyTable } from "../components/dashboard/StudyTable";
import { StudyGrid } from "../components/dashboard/StudyGrid";
import { TableSkeleton, CardSkeleton } from "../components/common/LoadingSkeleton";
import { EmptyState } from "../components/common/EmptyState";
import { Modal } from "../components/common/Modal";
import { Button } from "../components/common/Button";
import {
  FileText,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Calendar,
  Tag,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
} from "lucide-react";
import { useNavigate } from "react-router-dom";

export const DashboardPage: React.FC = () => {
  const { images, isLoadingImages, filters, setFilters, deleteImage } = useAppData();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [selectedImageDetails, setSelectedImageDetails] = useState<MedicalImage | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const pageSize = 8;

  // Derived filtered & sorted images
  const filteredImages = useMemo(() => {
    let result = [...images];

    // Status filter
    if (filters.status !== "all") {
      result = result.filter((img: MedicalImage) => img.status === filters.status);
    }

    // Modality filter
    if (filters.modality !== "all") {
      result = result.filter((img: MedicalImage) => img.modality === filters.modality);
    }

    // Search query
    if (filters.search.trim()) {
      const q = filters.search.toLowerCase();
      result = result.filter(
        (img: MedicalImage) =>
          img.name.toLowerCase().includes(q) ||
          (img.notes && img.notes.toLowerCase().includes(q)) ||
          (img.annotator && img.annotator.toLowerCase().includes(q))
      );
    }

    // Sorting
    result.sort((a: MedicalImage, b: MedicalImage) => {
      if (filters.sortBy === "name") {
        return filters.sortOrder === "asc"
          ? a.name.localeCompare(b.name)
          : b.name.localeCompare(a.name);
      }
      if (filters.sortBy === "status") {
        return filters.sortOrder === "asc"
          ? a.status.localeCompare(b.status)
          : b.status.localeCompare(a.status);
      }
      // Default: uploadedAt
      const dateA = new Date(a.uploadedAt).getTime();
      const dateB = new Date(b.uploadedAt).getTime();
      return filters.sortOrder === "asc" ? dateA - dateB : dateB - dateA;
    });

    return result;
  }, [images, filters]);

  // Dynamic 4 mutually understandable status metrics
  const stats = useMemo(() => {
    return {
      total: images.length,
      unannotated: images.filter((img: MedicalImage) => img.status === "unannotated").length,
      awaitingReview: images.filter((img: MedicalImage) => img.status === "awaiting-review").length,
      finalized: images.filter((img: MedicalImage) => img.status === "finalized").length,
    };
  }, [images]);

  // Paginated slice
  const totalPages = Math.max(1, Math.ceil(filteredImages.length / pageSize));
  const paginatedImages = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredImages.slice(start, start + pageSize);
  }, [filteredImages, currentPage, pageSize]);

  const handleDeleteImage = async (id: string) => {
    if (window.confirm("Are you sure you want to remove this image from the workspace?")) {
      await deleteImage(id);
      showToast("info", "Image Removed", "The image has been deleted from your workspace.");
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div>
        <h1 className="text-xl font-bold text-slate-900 tracking-tight">Dashboard</h1>
        <p className="text-xs text-slate-500 mt-0.5">
          Review, annotate, and verify chest radiograph pneumothorax images
        </p>
      </div>

      {/* 4 Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatsCard
          title="Total Images"
          value={stats.total}
          description="Images in workspace"
          icon={<FileText className="w-5 h-5 text-navy-900" />}
          colorVariant="navy"
          onClick={() => setFilters((prev) => ({ ...prev, status: "all" }))}
        />
        <StatsCard
          title="Unannotated"
          value={stats.unannotated}
          description="No saved annotation"
          icon={<Clock className="w-5 h-5 text-slate-600" />}
          colorVariant="navy"
          onClick={() => setFilters((prev) => ({ ...prev, status: "unannotated" }))}
        />
        <StatsCard
          title="Awaiting Review"
          value={stats.awaitingReview}
          description="AI suggestion requires human review"
          icon={<AlertTriangle className="w-5 h-5 text-amber-600" />}
          colorVariant="amber"
          onClick={() => setFilters((prev) => ({ ...prev, status: "awaiting-review" }))}
        />
        <StatsCard
          title="Finalized"
          value={stats.finalized}
          description="Annotation reviewed and saved"
          icon={<CheckCircle2 className="w-5 h-5 text-emerald-600" />}
          colorVariant="emerald"
          onClick={() => setFilters((prev) => ({ ...prev, status: "finalized" }))}
        />
      </div>

      {/* Filter and Search Toolbar */}
      <FilterToolbar
        filters={filters}
        setFilters={setFilters}
        totalCount={filteredImages.length}
      />

      {/* Main Images List/Grid or Empty State */}
      {isLoadingImages ? (
        filters.viewMode === "table" ? (
          <TableSkeleton rows={5} />
        ) : (
          <CardSkeleton count={3} />
        )
      ) : filteredImages.length === 0 ? (
        <EmptyState
          title={filters.search ? "No matching images found" : "No images in this category"}
          description={
            filters.search
              ? `No images matched your search query "${filters.search}". Try clearing search filters.`
              : "Select New Annotation to add a chest X-ray and start annotating."
          }
          actionText={filters.search ? "Clear Search" : "+ New Annotation"}
          onAction={() => {
            if (filters.search) {
              setFilters((prev) => ({ ...prev, search: "", status: "all" }));
            } else {
              navigate("/upload");
            }
          }}
        />
      ) : filters.viewMode === "table" ? (
        <StudyTable
          images={paginatedImages}
          onDeleteImage={handleDeleteImage}
          onShowDetails={(img: MedicalImage) => setSelectedImageDetails(img)}
        />
      ) : (
        <StudyGrid
          images={paginatedImages}
          onDeleteImage={handleDeleteImage}
          onShowDetails={(img: MedicalImage) => setSelectedImageDetails(img)}
        />
      )}

      {/* Pagination Controls */}
      {!isLoadingImages && filteredImages.length > pageSize && (
        <div className="flex items-center justify-between bg-white px-4 py-3 border border-slate-200 rounded-xl shadow-card text-xs text-slate-600">
          <span>
            Showing {(currentPage - 1) * pageSize + 1} to{" "}
            {Math.min(currentPage * pageSize, filteredImages.length)} of {filteredImages.length} entries
          </span>
          <div className="flex items-center gap-1.5">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="px-2.5"
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <span className="font-medium px-2">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="px-2.5"
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      )}

      {/* Image Details Modal */}
      <Modal
        isOpen={!!selectedImageDetails}
        onClose={() => setSelectedImageDetails(null)}
        title={selectedImageDetails?.name || "Image Details"}
        subtitle="Medical Image Information"
        maxWidth="lg"
        footer={
          <>
            <Button variant="secondary" onClick={() => setSelectedImageDetails(null)}>
              Close
            </Button>
            {selectedImageDetails && (
              <Button
                variant="primary"
                onClick={() => {
                  navigate(`/workspace/${selectedImageDetails.id}`);
                }}
                leftIcon={<ExternalLink className="w-4 h-4" />}
              >
                Open in Workspace
              </Button>
            )}
          </>
        }
      >
        {selectedImageDetails && (
          <div className="space-y-4 text-xs text-slate-700">
            <div className="flex items-center gap-4 p-3 bg-slate-50 border border-slate-200 rounded-lg">
              <div className="w-16 h-16 bg-slate-900 rounded-md overflow-hidden flex-shrink-0 flex items-center justify-center">
                {selectedImageDetails.previewUrl ? (
                  <img
                    src={selectedImageDetails.previewUrl}
                    alt={selectedImageDetails.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-[10px] text-slate-400 font-mono">DCM</span>
                )}
              </div>
              <div className="space-y-1">
                <h4 className="font-bold text-sm text-slate-900">{selectedImageDetails.name}</h4>
                <div className="flex items-center gap-2 pt-0.5">
                  <span className="font-mono text-slate-700 font-medium">
                    {selectedImageDetails.width} × {selectedImageDetails.height} px
                  </span>
                  <span>•</span>
                  <span>{selectedImageDetails.fileSizeFormatted || "2.4 MB"}</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 bg-white border border-slate-200 rounded-lg space-y-1.5">
                <span className="font-semibold text-slate-900 block flex items-center gap-1.5">
                  <Calendar className="w-3.5 h-3.5 text-teal-700" />
                  <span>Timeline & Status</span>
                </span>
                <p className="text-slate-600">
                  Added: {new Date(selectedImageDetails.uploadedAt).toLocaleString()}
                </p>
                <p className="text-slate-600">
                  Status: <strong className="capitalize">{selectedImageDetails.status.replace("-", " ")}</strong>
                </p>
                <p className="text-slate-600">Annotator: {selectedImageDetails.annotator || "Unassigned"}</p>
              </div>

              <div className="p-3 bg-white border border-slate-200 rounded-lg space-y-1.5">
                <span className="font-semibold text-slate-900 block flex items-center gap-1.5">
                  <Tag className="w-3.5 h-3.5 text-teal-700" />
                  <span>Acquisition Details</span>
                </span>
                <p className="text-slate-600">Modality: Chest Radiograph (CXR)</p>
                <p className="text-slate-600">Format: {selectedImageDetails.fileType.toUpperCase()}</p>
                <p className="text-slate-600">Patient Orientation: PA ERECT</p>
              </div>
            </div>

            {selectedImageDetails.notes && (
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <span className="font-semibold text-slate-900 block mb-1">Notes:</span>
                <p className="text-slate-600 leading-relaxed">{selectedImageDetails.notes}</p>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};
