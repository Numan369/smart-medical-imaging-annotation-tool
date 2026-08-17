import React from "react";
import clsx from "clsx";

interface StatsCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  description?: string;
  colorVariant?: "navy" | "amber" | "cyan" | "emerald";
  onClick?: () => void;
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  icon,
  description,
  colorVariant = "navy",
  onClick,
}) => {
  const colorStyles = {
    navy: "border-slate-200 text-navy-900 bg-white hover:border-slate-300",
    amber: "border-amber-200 text-amber-900 bg-amber-50/50 hover:border-amber-300",
    cyan: "border-cyan-200 text-cyan-900 bg-cyan-50/50 hover:border-cyan-300",
    emerald: "border-emerald-200 text-emerald-900 bg-emerald-50/50 hover:border-emerald-300",
  };

  const iconBgStyles = {
    navy: "bg-navy-50 text-navy-900",
    amber: "bg-amber-100 text-amber-800",
    cyan: "bg-cyan-100 text-cyan-800",
    emerald: "bg-emerald-100 text-emerald-800",
  };

  return (
    <div
      onClick={onClick}
      className={clsx(
        "p-4 rounded-xl border shadow-card transition-all duration-150 flex items-center justify-between",
        colorStyles[colorVariant],
        onClick ? "cursor-pointer hover:shadow-md" : ""
      )}
    >
      <div className="space-y-1">
        <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">{title}</p>
        <p className="text-2xl font-bold tracking-tight text-slate-900">{value}</p>
        {description && <p className="text-xs text-slate-500">{description}</p>}
      </div>
      <div className={clsx("w-11 h-11 rounded-lg flex items-center justify-center", iconBgStyles[colorVariant])}>
        {icon}
      </div>
    </div>
  );
};
