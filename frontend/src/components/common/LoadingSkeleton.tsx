import React from "react";

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 4 }) => {
  return (
    <div className="w-full animate-pulse divide-y divide-slate-100">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 py-4 px-4">
          <div className="w-12 h-12 bg-slate-200 rounded-md flex-shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-4 bg-slate-200 rounded w-1/3" />
            <div className="h-3 bg-slate-100 rounded w-1/4" />
          </div>
          <div className="h-6 bg-slate-200 rounded-full w-24" />
          <div className="h-8 bg-slate-200 rounded w-28" />
        </div>
      ))}
    </div>
  );
};

export const CardSkeleton: React.FC<{ count?: number }> = ({ count = 4 }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="bg-white rounded-lg border border-slate-200 p-4 animate-pulse space-y-3">
          <div className="h-36 bg-slate-200 rounded-md w-full" />
          <div className="h-4 bg-slate-200 rounded w-3/4" />
          <div className="flex justify-between items-center pt-2">
            <div className="h-5 bg-slate-200 rounded-full w-20" />
            <div className="h-3 bg-slate-100 rounded w-12" />
          </div>
        </div>
      ))}
    </div>
  );
};
