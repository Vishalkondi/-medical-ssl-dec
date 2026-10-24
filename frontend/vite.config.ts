import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the frontend calls /api/... and Vite forwards it to FastAPI,
// so no CORS setup is needed. Override the target with VITE_API_TARGET.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "node",
  },
});
