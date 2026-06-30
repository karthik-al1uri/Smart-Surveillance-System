/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        severity: {
          low: '#3B82F6',
          medium: '#F59E0B',
          high: '#EF4444',
          critical: '#DC2626',
        },
      },
    },
  },
  plugins: [require('@tailwindcss/forms')],
};

