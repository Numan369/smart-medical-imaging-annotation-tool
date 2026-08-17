import React, { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

interface AppShellProps {
  children?: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const location = useLocation();
  const isWorkspace = location.pathname.startsWith("/workspace");

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-50">
      {/* Collapsible Left Navigation Sidebar */}
      <Sidebar isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <TopBar />
        <main
          className={`flex-1 min-h-0 overflow-hidden ${
            isWorkspace ? "p-0 bg-slate-950" : "overflow-y-auto p-6 bg-slate-50/70"
          }`}
        >
          {isWorkspace ? (
            children || <Outlet />
          ) : (
            <div className="max-w-7xl mx-auto">{children || <Outlet />}</div>
          )}
        </main>
      </div>
    </div>
  );
};
