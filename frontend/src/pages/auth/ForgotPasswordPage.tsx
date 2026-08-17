import React, { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "../../components/common/Button";
import { BRAND_NAME, PRODUCT_NAME } from "../../constants/theme";
import { Mail, ArrowLeft, CheckCircle2, Stethoscope } from "lucide-react";

export const ForgotPasswordPage: React.FC = () => {
  const [email, setEmail] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email && email.includes("@")) {
      setIsSubmitted(true);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md text-center">
        <div className="w-12 h-12 bg-navy-900 text-white rounded-xl mx-auto flex items-center justify-center shadow-md mb-3">
          <Stethoscope className="w-6 h-6 text-teal-400" />
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">Reset password</h2>
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mt-1">{BRAND_NAME} — {PRODUCT_NAME}</p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md px-4">
        <div className="bg-white py-8 px-6 shadow-card rounded-2xl border border-slate-200 sm:px-10">
          {isSubmitted ? (
            <div className="text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center mx-auto">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <h3 className="font-semibold text-slate-900 text-base">Check your institutional email</h3>
              <p className="text-xs text-slate-600 leading-relaxed">
                In a production system, a password reset link would be dispatched to <strong className="text-slate-900">{email}</strong>.
              </p>
              <div className="pt-2">
                <Link to="/login">
                  <Button variant="primary" className="w-full">
                    Return to Sign In
                  </Button>
                </Link>
              </div>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <p className="text-xs text-slate-600 leading-relaxed">
                Enter your institutional email address and we'll send instructions to reset your password.
              </p>

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

              <Button type="submit" variant="primary" className="w-full">
                Send Reset Link
              </Button>

              <div className="text-center pt-2">
                <Link to="/login" className="inline-flex items-center gap-1.5 text-xs text-teal-700 font-medium hover:underline">
                  <ArrowLeft className="w-3.5 h-3.5" />
                  <span>Back to Sign In</span>
                </Link>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
