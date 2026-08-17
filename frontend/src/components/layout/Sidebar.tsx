import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  PlusCircle,
  History,
  HelpCircle,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
} from "lucide-react";
import clsx from "clsx";
import { BRAND_NAME, RESEARCH_DISCLAIMER } from "../../constants/theme";

interface SidebarProps {
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isCollapsed, setIsCollapsed }) => {
  const navItems = [
    { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { to: "/upload", label: "New Annotation", icon: PlusCircle },
    { to: "/history", label: "Annotation History", icon: History },
    { to: "/help", label: "Help", icon: HelpCircle },
    { to: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <aside
      className={clsx(
        "flex flex-col bg-navy-900 text-slate-200 border-r border-navy-800 transition-all duration-300 z-30 select-none",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      {/* Brand Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-navy-800">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="w-8 h-8 rounded-lg bg-teal-700 text-white flex items-center justify-center font-bold text-base flex-shrink-0 shadow-xs">
            M
          </div>
          {!isCollapsed && (
            <div className="flex flex-col">
              <span className="font-semibold text-white tracking-tight text-sm">
                {BRAND_NAME}
              </span>
              <span className="text-[10px] text-teal-300 font-mono tracking-wider">
                ANNOTATION WORKSPACE
              </span>
            </div>
          )}
        </div>

        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1 rounded text-slate-400 hover:text-white hover:bg-navy-800 transition-colors"
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 py-4 px-2 space-y-1.5 overflow-y-auto">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-colors",
                isActive
                  ? "bg-teal-700 text-white shadow-xs font-semibold"
                  : "text-slate-300 hover:bg-navy-800 hover:text-white"
              )
            }
            title={isCollapsed ? item.label : undefined}
          >
            <item.icon className="w-4 h-4 flex-shrink-0" />
            {!isCollapsed && <span className="truncate">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Research Disclaimer Footer */}
      {!isCollapsed && (
        <div className="p-3 m-3 bg-navy-800/80 rounded-lg border border-navy-700/60 text-[11px] text-slate-300 leading-relaxed">
          <div className="flex items-center gap-1.5 text-amber-400 font-semibold mb-1">
            <ShieldAlert className="w-3.5 h-3.5 flex-shrink-0" />
            <span>Research Notice</span>
          </div>
          <p className="line-clamp-4 text-slate-400 text-[10px]">
            {RESEARCH_DISCLAIMER}
          </p>
        </div>
      )}
    </aside>
  );
};
