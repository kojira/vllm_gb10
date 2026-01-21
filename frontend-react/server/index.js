import express from 'express'
import { createProxyMiddleware } from 'http-proxy-middleware'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const app = express()
const PORT = process.env.PORT || 3000
const API_TARGET = process.env.API_TARGET || 'http://localhost:8080'

// Proxy API requests
app.use('/v1', createProxyMiddleware({
  target: API_TARGET,
  changeOrigin: true,
}))

app.use('/health', createProxyMiddleware({
  target: API_TARGET,
  changeOrigin: true,
}))

// Serve static files from dist
app.use(express.static(path.join(__dirname, '../dist')))

// SPA fallback
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../dist/index.html'))
})

app.listen(PORT, () => {
  console.log(`Frontend server running on port ${PORT}`)
  console.log(`API proxy target: ${API_TARGET}`)
})

