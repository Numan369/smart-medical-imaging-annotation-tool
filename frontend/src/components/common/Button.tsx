import React from "react";
import clsx from "clsx";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost" | "outline" | "teal";
  size?: "sm" | "md" | "lg";
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = "primary",
  size = "md",
  isLoading = false,
  leftIcon,
  rightIcon,
  className,
  disabled,
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center justify-center font-medium rounded-md transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed select-none";

  const sizeStyles = {
    sm: "px-2.5 py-1.5 text-xs gap-1.5",
    md: "px-3.5 py-2 text-sm gap-2",
    lg: "px-5 py-2.5 text-base gap-2.5",
  };

  const variantStyles = {
    primary:
      "bg-navy-900 text-white hover:bg-navy-800 focus:ring-navy-700 active:bg-navy-900 border border-transparent shadow-sm",
    secondary:
      "bg-white text-slate-700 hover:bg-slate-50 hover:text-slate-900 border border-slate-300 focus:ring-teal-500 shadow-sm",
    teal:
      "bg-teal-700 text-white hover:bg-teal-600 focus:ring-teal-500 active:bg-teal-800 border border-transparent shadow-sm",
    danger:
      "bg-red-600 text-white hover:bg-red-700 focus:ring-red-500 active:bg-red-800 border border-transparent shadow-sm",
    outline:
      "bg-transparent text-navy-900 border border-navy-900 hover:bg-navy-50 focus:ring-navy-700",
    ghost:
      "bg-transparent text-slate-600 hover:text-slate-900 hover:bg-slate-100 focus:ring-slate-400 border border-transparent",
  };

  return (
    <button
      className={clsx(
        baseStyles,
        sizeStyles[size],
        variantStyles[variant],
        className
      )}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <span className="animate-spin inline-block w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
      ) : (
        leftIcon
      )}
      <span>{children}</span>
      {!isLoading && rightIcon}
    </button>
  );
};
