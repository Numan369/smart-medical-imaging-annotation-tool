import { Component, ErrorInfo, ReactNode } from "react";
import { resetDemoStorage } from "../../utils/storage";
import { AlertOctagon, RotateCcw, RefreshCw, LogIn } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught rendering exception caught by ErrorBoundary:", error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReload = () => {
    window.location.reload();
  };

  private handleResetData = () => {
    try {
      resetDemoStorage();
    } catch (e) {
      console.error("Failed to reset storage", e);
    }
    window.location.href = "/dashboard";
  };

  private handleReturnToSignIn = () => {
    localStorage.removeItem("smart_med_session_token");
    window.location.href = "/login";
  };

  public render() {
    if (this.state.hasError) {
      const isDev = process.env.NODE_ENV !== "production";

      return (
        <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-6 text-white select-none">
          <div className="max-w-lg w-full bg-slate-800 border border-slate-700 rounded-2xl p-6 sm:p-8 shadow-2xl space-y-6 text-center">
            <div className="w-14 h-14 bg-red-500/20 border border-red-500/40 text-red-400 rounded-2xl mx-auto flex items-center justify-center shadow-lg">
              <AlertOctagon className="w-8 h-8" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold tracking-tight text-white">
                The application could not be displayed.
              </h2>
              <p className="text-xs text-slate-400 leading-relaxed">
                An unexpected interface error occurred during rendering. You can reload the workstation, restore default demo data, or return to sign in.
              </p>
            </div>

            {/* Error Message Box */}
            <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-700/80 text-left">
              <span className="text-[11px] font-semibold uppercase text-red-400 font-mono block mb-1">
                Runtime Error:
              </span>
              <p className="font-mono text-xs text-slate-300 break-words">
                {this.state.error?.message || "Unknown rendering exception"}
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-2.5 pt-2">
              <button
                onClick={this.handleReload}
                className="w-full sm:w-auto px-4 py-2 bg-teal-600 hover:bg-teal-500 text-white font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-1.5 shadow-xs"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reload Application</span>
              </button>

              <button
                onClick={this.handleResetData}
                className="w-full sm:w-auto px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-1.5 border border-slate-600 shadow-xs"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset Demo Data</span>
              </button>

              <button
                onClick={this.handleReturnToSignIn}
                className="w-full sm:w-auto px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 font-semibold text-xs rounded-lg transition-colors flex items-center justify-center gap-1.5 border border-slate-600 shadow-xs"
              >
                <LogIn className="w-3.5 h-3.5" />
                <span>Return to Sign In</span>
              </button>
            </div>

            {/* Technical Stack Trace in Dev Mode */}
            {isDev && this.state.errorInfo && (
              <details className="text-left pt-2 border-t border-slate-700">
                <summary className="text-[11px] font-mono text-slate-400 cursor-pointer hover:text-slate-300">
                  Technical Diagnostics & Component Stack
                </summary>
                <pre className="mt-2 p-2.5 bg-slate-950 rounded text-[10px] font-mono text-slate-400 overflow-x-auto max-h-44 border border-slate-800">
                  {this.state.error?.stack}
                  {"\n\nComponent Stack:\n"}
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
