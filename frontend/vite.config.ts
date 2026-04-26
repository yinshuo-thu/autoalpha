import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, __dirname, '')
  const backendUrl = env.AUTOALPHA_BACKEND_URL || 'http://127.0.0.1:8080'
  const backendWsUrl = env.AUTOALPHA_BACKEND_WS_URL || backendUrl.replace(/^http/i, 'ws')
  const appBase = env.AUTOALPHA_APP_BASE || '/v2/'

  return {
    base: appBase,
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: 3000,
      host: true,
      allowedHosts: ['autoalpha', 'autoalpha.cn'],
      proxy: {
        '/api': {
          target: backendUrl,
          changeOrigin: true,
        },
        '/ws': {
          target: backendWsUrl,
          ws: true,
        },
      },
    },
    build: {
      sourcemap: false,
      chunkSizeWarningLimit: 900,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom'],
            charts: ['recharts'],
            ui: ['lucide-react', 'framer-motion'],
          },
        },
      },
    },
  }
})
