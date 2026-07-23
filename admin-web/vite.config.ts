import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, /api is proxied to the caregiver-server so the SPA and API share an origin (no CORS).
// Override the target with VITE_DEV_API_TARGET. In production the SPA calls VITE_API_BASE_URL directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_TARGET || 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
})
