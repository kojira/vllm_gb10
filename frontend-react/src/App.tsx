import { useState, useRef, useEffect } from 'react'
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Paper,
  Chip,
  CircularProgress,
} from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import { ChatMessage } from './components/ChatMessage'
import { ChatInput } from './components/ChatInput'
import { ModelSelector } from './components/ModelSelector'
import { SessionList } from './components/SessionList'
import { useStatus } from './hooks/useStatus'
import { Message, Session } from './types'
import { createSession, fetchSession, streamCompletion } from './services/api'

function App() {
  const { status, refresh } = useStatus(5000)
  const [messages, setMessages] = useState<Message[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [sessionDrawerOpen, setSessionDrawerOpen] = useState(false)
  const chatContainerRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight
    }
  }, [messages])

  const currentEngine = status?.vllm.status === 'loaded' ? 'vllm' : 
                        status?.llamacpp.status === 'loaded' ? 'llamacpp' : null
  const currentModel = currentEngine === 'vllm' ? status?.vllm.model : status?.llamacpp.model

  const handleSend = async (message: string, maxTokens: number, temperature: number) => {
    if (!currentEngine || !currentModel) {
      alert('モデルをロードしてください')
      return
    }

    // Create session if not exists
    let currentSessionId = sessionId
    if (!currentSessionId) {
      try {
        const title = message.substring(0, 30) + (message.length > 30 ? '...' : '')
        currentSessionId = await createSession(title, currentEngine, currentModel)
        setSessionId(currentSessionId)
      } catch (e) {
        console.error('Failed to create session:', e)
      }
    }

    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: message,
    }
    setMessages((prev) => [...prev, userMessage])

    // Add placeholder for assistant message
    const assistantId = Date.now() + 1
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: 'assistant', content: '' },
    ])

    setIsGenerating(true)

    try {
      let fullText = ''
      for await (const chunk of streamCompletion({
        prompt: message,
        engine: currentEngine,
        model: currentModel,
        maxTokens,
        temperature,
        stream: true,
        sessionId: currentSessionId || undefined,
      })) {
        fullText += chunk
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: fullText } : m
          )
        )
      }
    } catch (e) {
      console.error('Generation error:', e)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: `エラー: ${e instanceof Error ? e.message : '不明なエラー'}` }
            : m
        )
      )
    } finally {
      setIsGenerating(false)
    }
  }

  const handleSelectSession = async (session: Session) => {
    try {
      const fullSession = await fetchSession(session.id)
      setSessionId(session.id)
      setMessages(fullSession.messages || [])
    } catch (e) {
      console.error('Failed to load session:', e)
    }
  }

  const handleNewSession = () => {
    setSessionId(null)
    setMessages([])
  }

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <AppBar position="static" color="transparent" elevation={0}>
        <Toolbar>
          <IconButton
            edge="start"
            color="inherit"
            onClick={() => setSessionDrawerOpen(true)}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" sx={{ flexGrow: 1, ml: 2 }}>
            LLM Chat
          </Typography>
          {currentEngine && (
            <Chip
              label={`${currentEngine.toUpperCase()} - ${currentModel?.split('/').pop()}`}
              color="success"
              size="small"
            />
          )}
        </Toolbar>
      </AppBar>

      <Paper
        elevation={0}
        sx={{
          mx: 2,
          mb: 2,
          p: 2,
          bgcolor: 'background.paper',
        }}
      >
        <ModelSelector status={status} onStatusChange={refresh} />
      </Paper>

      <Box
        ref={chatContainerRef}
        sx={{
          flex: 1,
          overflow: 'auto',
          px: 2,
          py: 2,
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Typography color="text.secondary">
              メッセージを入力して会話を開始してください
            </Typography>
          </Box>
        ) : (
          messages.map((msg) => (
            <ChatMessage key={msg.id} role={msg.role} content={msg.content} />
          ))
        )}
        {isGenerating && messages[messages.length - 1]?.content === '' && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        )}
      </Box>

      <ChatInput onSend={handleSend} disabled={isGenerating || !currentEngine} />

      <SessionList
        open={sessionDrawerOpen}
        onClose={() => setSessionDrawerOpen(false)}
        currentSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
      />
    </Box>
  )
}

export default App

