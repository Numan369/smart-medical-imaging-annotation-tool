import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { useToast } from "../../context/ToastContext";
import { Button } from "../../components/common/Button";
import { RESEARCH_DISCLAIMER } from "../../constants/theme";
import { Eye, EyeOff, Lock, Mail, User, Stethoscope } from "lucide-react";

export const SignUpPage: React.FC = () => {
  const { signUp } = useAuth();
  const { showToast } = useToast();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedName = name.trim();
    const trimmedEmail = email.trim();

    if (!trimmedName) {
      setError("Please enter your full name.");
      return;
    }

    if (!trimmedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (!password || password.length < 8 || !/[A-Za-z]/.test(password) || !/\d/.test(password)) {
      setError("Password must be at least 8 characters and contain at least one letter and one number.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setIsLoading(true);
    try {
      await signUp({ name: trimmedName, email: trimmedEmail, password });
      showToast("success", "Account Created", "Your annotator account has been created successfully.");
      navigate("/dashboard");
    } catch (err) {
      setError("Failed to create account. Please retry.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center py-10 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <div className="w-12 h-12 bg-navy-900 text-white rounded-xl mx-auto flex items-center justify-center shadow-md mb-3">
          <Stethoscope className="w-6 h-6 text-teal-400" />
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">Create your account</h2>
        <p className="text-xs text-slate-500 mt-1">Start reviewing and annotating medical images</p>
      </div>

      <div className="mt-6 sm:mx-auto sm:w-full sm:max-w-md px-4">
        <div className="bg-white py-8 px-6 shadow-card rounded-2xl border border-slate-200 sm:px-10">
          <form className="space-y-4" onSubmit={handleSubmit}>
            {/* 1. Full Name */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                Full Name
              </label>
              <div className="relative">
                <User className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Sarah Jenkins"
                  className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                  required
                />
              </div>
            </div>

            {/* 2. Email Address */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                Email Address
              </label>
              <div className="relative">
                <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="e.g. sarah.jenkins@hospital.org"
                  className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                  required
                />
              </div>
            </div>

            {/* 3. Password */}
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
                  placeholder="At least 8 characters (with letters and numbers)"
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
              <p className="text-[11px] text-slate-400 mt-1">Minimum 8 characters, at least one letter and one number</p>
            </div>

            {/* 4. Confirm Password */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Re-enter password"
                  className="w-full pl-9 pr-3 py-2 bg-slate-50 border border-slate-300 rounded-lg text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                  required
                />
              </div>
            </div>

            {error && (
              <div className="text-xs text-red-600 bg-red-50 p-2.5 rounded-lg border border-red-200">
                {error}
              </div>
            )}

            <Button type="submit" variant="teal" className="w-full mt-2" isLoading={isLoading}>
              Create Account
            </Button>
          </form>

          <div className="mt-4 text-center text-xs text-slate-500">
            Already registered?{" "}
            <Link to="/login" className="text-teal-700 font-semibold hover:underline">
              Sign in to workspace
            </Link>
          </div>
        </div>

        <p className="mt-4 text-center text-[11px] text-slate-400 leading-relaxed px-4">
          {RESEARCH_DISCLAIMER}
        </p>
      </div>
    </div>
  );
};
