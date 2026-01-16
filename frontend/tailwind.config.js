/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        night: "#0f172a",
        slate: "#334155",
        mist: "#e2e8f0",
        sun: "#f97316",
        surf: "#0ea5e9",
        leaf: "#10b981",
      },
      boxShadow: {
        glow: "0 24px 60px rgba(14, 165, 233, 0.2)",
        warm: "0 18px 40px rgba(249, 115, 22, 0.25)",
      },
      animation: {
        rise: "rise 700ms ease-out both",
        floatSlow: "floatSlow 10s ease-in-out infinite",
      },
      keyframes: {
        rise: {
          "0%": { opacity: 0, transform: "translateY(16px)" },
          "100%": { opacity: 1, transform: "translateY(0px)" },
        },
        floatSlow: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-18px)" },
        },
      },
    },
  },
  plugins: [],
};
