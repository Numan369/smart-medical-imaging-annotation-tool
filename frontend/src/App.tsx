import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import { AppShell } from "./components/layout/AppShell";

// Auth Pages
import { SignInPage } from "./pages/auth/SignInPage";
import { SignUpPage } from "./pages/auth/SignUpPage";
import { ForgotPasswordPage } from "./pages/auth/ForgotPasswordPage";

// Workstation Pages
import { DashboardPage } from "./pages/DashboardPage";
import { UploadPage } from "./pages/UploadPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { HistoryPage } from "./pages/HistoryPage";
import { SettingsPage } from "./pages/SettingsPage";
import { HelpPage } from "./pages/HelpPage";
import { NotFoundPage } from "./pages/NotFoundPage";

// Protected Route Guard
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center text-white select-none">
        <div className="flex flex-col items-center gap-4 text-center p-6">
          <div className="w-14 h-14 rounded-2xl bg-teal-500/10 border border-teal-500/30 text-teal-400 flex items-center justify-center shadow-xl">
            <div className="w-7 h-7 border-3 border-teal-400 border-t-transparent rounded-full animate-spin" />
          </div>
          <div>
            <h3 className="text-base font-bold tracking-tight text-white">MediMask AI</h3>
            <p className="text-xs text-slate-400 mt-1 font-mono">Loading workspace session…</p>
          </div>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <AppShell>{children}</AppShell>;
};

export const App: React.FC = () => {
  return (
    <Routes>
      {/* Public Auth Routes */}
      <Route path="/login" element={<SignInPage />} />
      <Route path="/sign-in" element={<Navigate to="/login" replace />} />
      <Route path="/signup" element={<SignUpPage />} />
      <Route path="/register" element={<Navigate to="/signup" replace />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />

      {/* Protected Medical Workstation Routes */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Navigate to="/dashboard" replace />
          </ProtectedRoute>
        }
      />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
            <UploadPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/workspace/:imageId"
        element={
          <ProtectedRoute>
            <WorkspacePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/workspace/:imageId/manual"
        element={
          <ProtectedRoute>
            <WorkspacePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/workspace/:imageId/ai-review"
        element={
          <ProtectedRoute>
            <WorkspacePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/workspace/:imageId/edit"
        element={
          <ProtectedRoute>
            <WorkspacePage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <HistoryPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/help"
        element={
          <ProtectedRoute>
            <HelpPage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <SettingsPage />
          </ProtectedRoute>
        }
      />

      {/* 404 Fallback */}
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
};

export default App;
