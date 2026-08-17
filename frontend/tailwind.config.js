/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          900: '#0F2744',
          800: '#163357',
          700: '#1E426D',
          600: '#2A568C',
          50: '#F0F5FA',
        },
        teal: {
          700: '#0E7490',
          600: '#0891B2',
          500: '#06B6D4',
          100: '#CFFAFE',
          50: '#ECFEFF',
        },
        aimask: {
          cyan: '#22D3EE',
          border: '#06B6D4',
        },
        status: {
          accepted: '#16A34A',
          acceptedBg: '#F0FDF4',
          acceptedBorder: '#BBF7D0',
          review: '#D97706',
          reviewBg: '#FFFBEB',
          reviewBorder: '#FDE68A',
          rejected: '#DC2626',
          rejectedBg: '#FEF2F2',
          rejectedBorder: '#FECACA',
          uploaded: '#64748B',
          uploadedBg: '#F8FAFC',
          uploadedBorder: '#E2E8F0',
          ai: '#0891B2',
          aiBg: '#ECFEFF',
          aiBorder: '#A5F3FC',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(15, 39, 68, 0.05), 0 1px 2px 0 rgba(15, 39, 68, 0.03)',
        'card': '0 2px 6px -1px rgba(15, 39, 68, 0.08), 0 2px 4px -2px rgba(15, 39, 68, 0.04)',
        'modal': '0 12px 24px -6px rgba(15, 39, 68, 0.18), 0 4px 8px -4px rgba(15, 39, 68, 0.1)',
      }
    },
  },
  plugins: [],
}
