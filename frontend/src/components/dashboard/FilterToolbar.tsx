import React from "react";
import { FilterState, AnnotationStatus } from "../../types";
import { Search, LayoutGrid, LayoutList, SlidersHorizontal, PlusCircle } from "lucide-react";
import { Button } from "../common/Button";
import { useNavigate } from "react-router-dom";

interface FilterToolbarProps {
  filters: FilterState;
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  totalCount: number;
}

export const FilterToolbar: React.FC<FilterToolbarProps> = ({
  filters,
  setFilters,
  totalCount,
}) => {
  const navigate = useNavigate();

  const statusTabs: { id: "all" | AnnotationStatus; label: string }[] = [
    { id: "all", label: "All Images" },
    { id: "unannotated", label: "Unannotated" },
    { id: "awaiting-review", label: "Awaiting Review" },
    { id: "finalized", label: "Finalized" },
  ];

  return (
    <div className="space-y-3 bg-white p-4 rounded-xl border border-slate-200 shadow-card">
      {/* Top row: Search, Modality, View Toggle, New Annotation Button */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        {/* Search input */}
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by image name, notes, status, or annotator…"
            value={filters.search}
            onChange={(e) => setFilters((prev) => ({ ...prev, search: e.target.value }))}
            className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white transition-all"
          />
        </div>

        {/* Right Actions: Modality, Sort, View, New Annotation */}
        <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
          {/* Modality Filter */}
          <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs text-slate-700">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={filters.modality}
              onChange={(e) =>
                setFilters((prev) => ({ ...prev, modality: e.target.value as FilterState["modality"] }))
              }
              className="bg-transparent border-none text-xs text-slate-800 font-medium focus:outline-none cursor-pointer"
            >
              <option value="all">All Modalities</option>
              <option value="xray">Chest X-ray (CXR)</option>
              <option value="ct" disabled>CT (Future module)</option>
              <option value="mri" disabled>MRI (Future module)</option>
            </select>
          </div>

          {/* Sort By Filter */}
          <select
            value={`${filters.sortBy}-${filters.sortOrder}`}
            onChange={(e) => {
              const [by, order] = e.target.value.split("-") as [FilterState["sortBy"], FilterState["sortOrder"]];
              setFilters((prev) => ({ ...prev, sortBy: by, sortOrder: order }));
            }}
            className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer"
          >
            <option value="uploadedAt-desc">Newest First</option>
            <option value="uploadedAt-asc">Oldest First</option>
            <option value="name-asc">Name (A-Z)</option>
            <option value="status-asc">Status</option>
          </select>

          {/* Table / Grid View Toggle */}
          <div className="flex items-center bg-slate-100 p-1 rounded-lg border border-slate-200">
            <button
              onClick={() => setFilters((prev) => ({ ...prev, viewMode: "table" }))}
              className={`p-1.5 rounded-md transition-colors ${
                filters.viewMode === "table" ? "bg-white text-navy-900 shadow-xs" : "text-slate-500 hover:text-slate-800"
              }`}
              title="Table View"
              aria-label="Table View"
            >
              <LayoutList className="w-4 h-4" />
            </button>
            <button
              onClick={() => setFilters((prev) => ({ ...prev, viewMode: "grid" }))}
              className={`p-1.5 rounded-md transition-colors ${
                filters.viewMode === "grid" ? "bg-white text-navy-900 shadow-xs" : "text-slate-500 hover:text-slate-800"
              }`}
              title="Grid View"
              aria-label="Grid View"
            >
              <LayoutGrid className="w-4 h-4" />
            </button>
          </div>

          {/* New Annotation Button */}
          <Button
            variant="teal"
            size="sm"
            onClick={() => navigate("/upload")}
            leftIcon={<PlusCircle className="w-4 h-4" />}
          >
            + New Annotation
          </Button>
        </div>
      </div>

      {/* Bottom row: Status Filter Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pt-2 border-t border-slate-100 text-xs no-scrollbar">
        {statusTabs.map((tab) => {
          const isActive = filters.status === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setFilters((prev) => ({ ...prev, status: tab.id }))}
              className={`px-3 py-1.5 rounded-lg font-medium whitespace-nowrap transition-colors ${
                isActive
                  ? "bg-navy-900 text-white font-semibold"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
        <span className="text-slate-400 text-xs ml-auto pr-2 hidden md:inline">
          Showing {totalCount} matching {totalCount === 1 ? "image" : "images"}
        </span>
      </div>
    </div>
  );
};
