/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        // Distinctive choices — explicitly NOT Inter / Roboto / Space Grotesk.
        display: ['"Fraunces"', "ui-serif", "Georgia", "serif"],
        body: ['"Newsreader"', "ui-serif", "Georgia", "serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        // Dark literary theme — see theme.css for the canonical CSS variables.
        ink: "#0E1116",
        cream: "#F0E6D6",
        brass: "#C28840",
        oxblood: "#5C1A1B",
        ash: "#2A2C32",
      },
      keyframes: {
        "ink-pulse": {
          "0%, 100%": { opacity: "0.45", transform: "scale(0.85)" },
          "50%": { opacity: "1", transform: "scale(1.05)" },
        },
      },
      animation: {
        "ink-pulse": "ink-pulse 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
