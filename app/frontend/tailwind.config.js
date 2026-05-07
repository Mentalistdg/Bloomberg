/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Paleta sobria institucional para finanzas
        ink: '#0b0f17',
        panel: '#111827',
        muted: '#9ca3af',
        line: '#1f2937',
        accent: '#2563eb',     // azul institucional
        positive: '#10b981',
        negative: '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
