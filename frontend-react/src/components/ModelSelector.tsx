import { useState, useEffect } from 'react'
import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  LinearProgress,
  Typography,
  Alert,
  SelectChangeEvent,
} from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import { Model, Status } from '../types'
import { fetchModels, loadModel, unloadModel } from '../services/api'

interface ModelSelectorProps {
  status: Status | null
  onStatusChange: () => void
}

export function ModelSelector({ status, onStatusChange }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([])
  const [engine, setEngine] = useState<'vllm' | 'llamacpp'>('vllm')
  const [selectedModel, setSelectedModel] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
  }, [])

  const filteredModels = models.filter((m) => {
    if (engine === 'llamacpp') {
      return m.type === 'gguf'
    }
    return m.type === 'transformers'
  })

  const currentEngineStatus = engine === 'vllm' ? status?.vllm : status?.llamacpp
  const isLoading = currentEngineStatus?.status === 'loading'
  const isLoaded = currentEngineStatus?.status === 'loaded'

  const handleEngineChange = (e: SelectChangeEvent) => {
    setEngine(e.target.value as 'vllm' | 'llamacpp')
    setSelectedModel('')
  }

  const handleLoad = async () => {
    if (!selectedModel) return
    setLoading(true)
    setError(null)
    try {
      await loadModel(engine, selectedModel)
      onStatusChange()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Load failed')
    } finally {
      setLoading(false)
    }
  }

  const handleUnload = async () => {
    setLoading(true)
    setError(null)
    try {
      await unloadModel(engine)
      onStatusChange()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unload failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', gap: 2 }}>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>エンジン</InputLabel>
          <Select value={engine} label="エンジン" onChange={handleEngineChange}>
            <MenuItem value="vllm">vLLM</MenuItem>
            <MenuItem value="llamacpp">llama.cpp</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ flex: 1, minWidth: 200 }}>
          <InputLabel>モデル</InputLabel>
          <Select
            value={selectedModel}
            label="モデル"
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {filteredModels.map((m) => (
              <MenuItem key={m.path} value={m.path}>
                {m.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button
          variant="contained"
          startIcon={<PlayArrowIcon />}
          onClick={handleLoad}
          disabled={!selectedModel || loading || isLoading}
        >
          ロード
        </Button>

        <Button
          variant="outlined"
          color="error"
          startIcon={<StopIcon />}
          onClick={handleUnload}
          disabled={!isLoaded || loading}
        >
          アンロード
        </Button>
      </Box>

      {(isLoading || loading) && (
        <Box>
          <LinearProgress
            variant="determinate"
            value={(currentEngineStatus?.progress || 0) * 100}
          />
          <Typography variant="caption" color="text.secondary">
            {currentEngineStatus?.progress_message || 'ロード中...'}
          </Typography>
        </Box>
      )}

      {isLoaded && (
        <Alert severity="success" sx={{ py: 0.5 }}>
          {currentEngineStatus?.model?.split('/').pop()} がロードされています
        </Alert>
      )}

      {error && <Alert severity="error">{error}</Alert>}
    </Box>
  )
}

