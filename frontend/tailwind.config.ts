import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        primary: {
          400: '#4ade80',
          500: '#22c55e',
          600: '#16a34a',
        },
        surface: {
          DEFAULT: '#0d0d1a',
          elevated: '#12122a',
          border: '#1e1e3a',
        },
        detection: {
          high: '#22c55e',
          medium: '#f59e0b',
          low: '#ef4444',
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
