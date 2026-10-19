import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Inference engine routes forwarded by the dev server and `vite preview`, so the
// frontend can use relative URLs when VITE_API_BASE_URL is not set.
const API_TARGET = 'http://127.0.0.1:8001'
const apiProxy = Object.fromEntries(
  ['/scan', '/predict', '/extract', '/health', '/stats', '/api'].map((path) => [
    path,
    // xfwd adds X-Forwarded-For so the backend can rate-limit per browser client.
    { target: API_TARGET, changeOrigin: true, xfwd: true },
  ]),
)

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Read VITE_* variables from the repository-root .env shared with the backend.
  envDir: '..',
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: apiProxy,
    watch: {
      ignored: [
        '**/Dataset/**',
        '**/data/**',
        '**/models/**',
        '**/outputs/**',
        '**/.git/**',
      ],
    },
  },
  preview: {
    proxy: apiProxy,
  },
  test: {
    // Keep tests independent of the developer's local .env.
    env: { VITE_API_BASE_URL: '' },
  },
})
