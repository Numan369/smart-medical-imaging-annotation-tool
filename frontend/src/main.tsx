import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { ToastProvider } from "./context/ToastContext";
import { AuthProvider } from "./context/AuthContext";
import { AppDataProvider } from "./context/AppDataContext";
import App from "./App";
import "./index.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element '#root' not found in DOM.");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <BrowserRouter>
        <ToastProvider>
          <AuthProvider>
            <AppDataProvider>
              <App />
            </AppDataProvider>
          </AuthProvider>
        </ToastProvider>
      </BrowserRouter>
    </ErrorBoundary>
  </React.StrictMode>
);
