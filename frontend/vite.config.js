import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // In dev, proxy API calls to the gateway so the browser never needs CORS and the
    // same relative paths work unchanged once nginx takes over this job in prod.
    proxy: {
      '/v1': 'http://localhost:8000',
      '/stats': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/requests': 'http://localhost:8000',
      '/admin': 'http://localhost:8000',
    },
  },
})
