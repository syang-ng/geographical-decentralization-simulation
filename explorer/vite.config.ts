import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3200,
    proxy: {
      '/api': {
        target: 'http://localhost:3201',
        changeOrigin: true,
      },
    },
  },
})
