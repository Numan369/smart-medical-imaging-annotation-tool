import React from "react";
import { Link } from "react-router-dom";
import { Button } from "../components/common/Button";
import { FileQuestion, ArrowLeft } from "lucide-react";

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-navy-900 text-white flex items-center justify-center mb-4 shadow-md">
        <FileQuestion className="w-8 h-8 text-teal-400" />
      </div>
      <h1 className="text-3xl font-bold text-slate-900 mb-2">404 — Page Not Found</h1>
      <p className="text-sm text-slate-600 max-w-md mb-8">
        The requested medical image or workstation view could not be found. Check the URL or return to the dashboard.
      </p>
      <Link to="/dashboard">
        <Button variant="primary" leftIcon={<ArrowLeft className="w-4 h-4" />}>
          Back to Dashboard
        </Button>
      </Link>
    </div>
  );
};
