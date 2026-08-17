import React from "react";
import { useAuth } from "../../context/AuthContext";
import { LogOut, User, Stethoscope, AlertCircle, Sparkles } from "lucide-react";
import { useNavigate } from "react-router-dom";

export const TopBar: React.FC = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate("/login");
  };

  return (
    <header className="h-16 bg-white border-b border-slate-200 px-6 flex items-center justify-between z-20 shadow-2xs">
      {/* Modality and Active Workstation Info */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 text-slate-800">
          <Stethoscope className="w-5 h-5 text-teal-700" />
          <span className="font-semibold text-sm">Pneumothorax CXR Workstation</span>
        </div>
        <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-teal-50 text-teal-800 border border-teal-200">
          <Sparkles className="w-3 h-3 text-teal-600 animate-pulse" />
          <span>AI Assistance Available</span>
        </span>
      </div>

      {/* Right Side: Prototype Banner & User Profile */}
      <div className="flex items-center gap-4">
        <div className="hidden lg:flex items-center gap-1.5 text-xs text-amber-700 bg-amber-50 px-2.5 py-1 rounded-md border border-amber-200 font-medium">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          <span>Human Review Required</span>
        </div>

        {/* User Badge and Sign Out */}
        <div className="flex items-center gap-3 pl-2 border-l border-slate-200">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-navy-900 text-white flex items-center justify-center text-xs font-semibold">
              <User className="w-4 h-4" />
            </div>
            <div className="hidden md:flex flex-col text-left">
              <span className="text-xs font-semibold text-slate-900 leading-tight">
                {user?.name || "Dr. Sarah Jenkins"}
              </span>
              <span className="text-[10px] text-slate-500">{user?.role || "Annotator"}</span>
            </div>
          </div>

          <button
            onClick={handleSignOut}
            className="p-1.5 rounded-md text-slate-500 hover:text-red-600 hover:bg-red-50 transition-colors"
            title="Sign out of annotation workstation"
            aria-label="Sign out"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </header>
  );
};
