import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { useToast } from "../../context/ToastContext";
import { Button } from "../../components/common/Button";
import { BRAND_NAME, PRODUCT_NAME, RESEARCH_DISCLAIMER } from "../../constants/theme";
import { Eye, EyeOff, Lock, Mail, Stethoscope, ArrowRight } from "lucide-react";

export const SignInPage: React.FC = () => {
  const { signIn } = useAuth();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [email, setEmail] = useState("s.jenkins@radiology.hospital.org");
  const [password, setPassword] = useState("demo123456");
  const [rememberMe, setRememberMe] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!email || !email.includes("@")) {
      setError("Please enter a valid institutional email address.");
      return;
    }
    if (!password || password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    setIsLoading(true);
    try {
      await signIn({ email, password, rememberMe });
      showToast("success", "Welcome to MediMask AI", "Signed in to medical annotation workstation.");
      navigate("/dashboard");
    } catch (err) {
      setError("Authentication failed. Please check your credentials.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickDemoLogin = async () => {
    setIsLoading(true);
    try {
      await signIn({ email: "s.jenkins@radiology.hospital.org", password: "demo123456", rememberMe: true });
      showToast("success", "Demo Mode Active", "Signed in as Dr. Sarah Jenkins (Consultant Radiologist)");
      navigate("/dashboard");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <div className="w-12 h-12 bg-navy-900 text-white rounded-xl mx-auto flex items-center justify-center shadow-md mb-3">
          <Stethoscope className="w-6 h-6 text-teal-400" />
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">{BRAND_NAME}</h2>
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mt-1">{PRODUCT_NAME}</p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md px-4">
        <div className="bg-white py-8 px-6 shadow-card rounded-2xl border border-slate-200 sm:px-10">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                Institutional Email
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="name@hospital.org"
                  className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                Password
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-9 pr-10 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {error && (
              <div className="text-xs text-red-600 bg-red-50 p-2.5 rounded-lg border border-red-200">
                {error}
              </div>
            )}

            <div className="flex items-center justify-between text-xs">
              <label className="flex items-center gap-2 text-slate-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="rounded border-slate-300 text-teal-600 focus:ring-teal-500"
                />
                <span>Remember this workstation</span>
              </label>
              <Link to="/forgot-password" className="text-teal-700 hover:underline font-medium">
                Forgot password?
              </Link>
            </div>

            <Button type="submit" variant="primary" className="w-full" isLoading={isLoading}>
              Sign In to Workstation
            </Button>
          </form>

          {/* Quick Demo Login Option */}
          <div className="mt-5 pt-4 border-t border-slate-100">
            <Button
              type="button"
              variant="secondary"
              className="w-full text-xs"
              onClick={handleQuickDemoLogin}
              disabled={isLoading}
              rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
            >
              Quick Demo Access (Dr. Sarah Jenkins)
            </Button>
          </div>

          {/* Sign Up Link */}
          <div className="mt-4 text-center text-xs text-slate-500">
            Need an annotator account?{" "}
            <Link to="/signup" className="text-teal-700 font-semibold hover:underline">
              Create account
            </Link>
          </div>
        </div>

        {/* Research Notice */}
        <p className="mt-4 text-center text-[11px] text-slate-400 leading-relaxed px-4">
          {RESEARCH_DISCLAIMER}
        </p>
      </div>
    </div>
  );
};
