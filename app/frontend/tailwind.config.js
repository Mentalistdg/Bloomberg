/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#000000',
        panel: '#111111',
        'panel-hover': '#1a1a1a',
        muted: '#737373',
        line: '#222222',
        'line-light': '#333333',
        accent: '#CF2141',
        'accent-light': '#e8274a',
        'accent-dark': '#a81a34',
        positive: '#00c853',
        negative: '#CF2141',
        text: '#f5f5f5',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
