import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on http://localhost:8000 by default.
const BACKEND_URL = process.env.VITE_BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // The frontend calls "/api/..." and Vite strips the prefix so the
      // backend routes (which live at the root, e.g. /chat) are reached.
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
