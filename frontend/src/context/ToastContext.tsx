import React, { createContext, useContext, useState, useCallback } from "react";

export type ToastType = "success" | "error" | "warning" | "info";

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  durationMs?: number;
}

interface ToastContextType {
  toasts: ToastItem[];
  showToast: (type: ToastType, title: string, message?: string, durationMs?: number) => void;
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    (type: ToastType, title: string, message?: string, durationMs = 4000) => {
      const id = `toast-${Date.now()}-${Math.random()}`;
      const newToast: ToastItem = { id, type, title, message, durationMs };

      setToasts((prev) => [...prev, newToast]);

      if (durationMs > 0) {
        setTimeout(() => {
          removeToast(id);
        }, durationMs);
      }
    },
    [removeToast]
  );

  return (
    <ToastContext.Provider value={{ toasts, showToast, removeToast }}>
      {children}
      {/* Toast viewport */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-start gap-3 p-4 rounded-lg shadow-lg border text-sm transition-all duration-300 transform translate-y-0 ${
              toast.type === "success"
                ? "bg-white border-green-200 text-slate-800"
                : toast.type === "error"
                ? "bg-white border-red-200 text-slate-800"
                : toast.type === "warning"
                ? "bg-white border-amber-200 text-slate-800"
                : "bg-white border-teal-200 text-slate-800"
            }`}
          >
            <div className="flex-shrink-0 mt-0.5">
              {toast.type === "success" && (
                <div className="w-5 h-5 rounded-full bg-green-100 text-green-600 flex items-center justify-center font-bold text-xs">
                  ✓
                </div>
              )}
              {toast.type === "error" && (
                <div className="w-5 h-5 rounded-full bg-red-100 text-red-600 flex items-center justify-center font-bold text-xs">
                  ✕
                </div>
              )}
              {toast.type === "warning" && (
                <div className="w-5 h-5 rounded-full bg-amber-100 text-amber-600 flex items-center justify-center font-bold text-xs">
                  !
                </div>
              )}
              {toast.type === "info" && (
                <div className="w-5 h-5 rounded-full bg-teal-100 text-teal-600 flex items-center justify-center font-bold text-xs">
                  i
                </div>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-slate-900">{toast.title}</p>
              {toast.message && <p className="text-slate-600 mt-0.5 text-xs">{toast.message}</p>}
            </div>
            <button
              onClick={() => removeToast(toast.id)}
              className="text-slate-400 hover:text-slate-600 text-xs p-1"
              aria-label="Close notification"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a ToastProvider");
  }
  return context;
}
