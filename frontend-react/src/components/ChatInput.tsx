import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Box, TextField, IconButton, Slider, Typography, Tooltip } from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import SettingsIcon from '@mui/icons-material/Settings'

interface ChatInputProps {
  onSend: (message: string, maxTokens: number, temperature: number) => void
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0.7)
  const [isComposing, setIsComposing] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus()
    }
  }, [disabled])

  const handleSend = () => {
    const trimmed = message.trim()
    if (trimmed && !disabled) {
      onSend(trimmed, maxTokens, temperature)
      setMessage('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
      {showSettings && (
        <Box sx={{ mb: 2, px: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
            <Typography variant="body2" sx={{ minWidth: 100 }}>
              最大トークン: {maxTokens}
            </Typography>
            <Slider
              value={maxTokens}
              onChange={(_, v) => setMaxTokens(v as number)}
              min={64}
              max={4096}
              step={64}
              size="small"
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" sx={{ minWidth: 100 }}>
              Temperature: {temperature.toFixed(1)}
            </Typography>
            <Slider
              value={temperature}
              onChange={(_, v) => setTemperature(v as number)}
              min={0}
              max={2}
              step={0.1}
              size="small"
            />
          </Box>
        </Box>
      )}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
        <Tooltip title="設定">
          <IconButton
            onClick={() => setShowSettings(!showSettings)}
            color={showSettings ? 'primary' : 'default'}
          >
            <SettingsIcon />
          </IconButton>
        </Tooltip>
        <TextField
          inputRef={inputRef}
          fullWidth
          multiline
          maxRows={6}
          placeholder="メッセージを入力..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          disabled={disabled}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
            },
          }}
        />
        <IconButton
          color="primary"
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            '&:hover': { bgcolor: 'primary.dark' },
            '&:disabled': { bgcolor: 'action.disabledBackground' },
          }}
        >
          <SendIcon />
        </IconButton>
      </Box>
    </Box>
  )
}

