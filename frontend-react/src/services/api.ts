import { Model, Session, Status } from '../types'

const API_BASE = ''

export async function fetchStatus(): Promise<Status> {
  const response = await fetch(`${API_BASE}/v1/status`)
  if (!response.ok) throw new Error('Failed to fetch status')
  return response.json()
}

export async function fetchModels(): Promise<Model[]> {
  const response = await fetch(`${API_BASE}/v1/models/list`)
  if (!response.ok) throw new Error('Failed to fetch models')
  const data = await response.json()
  return data.models
}

export async function loadModel(engine: string, modelPath: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v1/models/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engine, model_path: modelPath }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to load model')
  }
}

export async function unloadModel(engine: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v1/models/unload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ engine }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to unload model')
  }
}

export async function downloadModel(modelId: string, filename?: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v1/models/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId, filename }),
  })
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Failed to download model')
  }
}

export async function fetchSessions(): Promise<Session[]> {
  const response = await fetch(`${API_BASE}/v1/sessions`)
  if (!response.ok) throw new Error('Failed to fetch sessions')
  const data = await response.json()
  return data.sessions
}

export async function createSession(title: string, engine: string, model: string): Promise<string> {
  const response = await fetch(`${API_BASE}/v1/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title, engine, model }),
  })
  if (!response.ok) throw new Error('Failed to create session')
  const data = await response.json()
  return data.id
}

export async function fetchSession(sessionId: string): Promise<Session> {
  const response = await fetch(`${API_BASE}/v1/sessions/${sessionId}`)
  if (!response.ok) throw new Error('Failed to fetch session')
  return response.json()
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/v1/sessions/${sessionId}`, {
    method: 'DELETE',
  })
  if (!response.ok) throw new Error('Failed to delete session')
}

export interface CompletionOptions {
  prompt: string
  engine: string
  model?: string
  maxTokens: number
  temperature: number
  stream: boolean
  sessionId?: string
}

export async function* streamCompletion(options: CompletionOptions): AsyncGenerator<string> {
  const response = await fetch(`${API_BASE}/v1/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: options.prompt,
      engine: options.engine,
      model: options.model,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
      stream: options.stream,
      session_id: options.sessionId,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || 'Completion failed')
  }

  if (!options.stream) {
    const data = await response.json()
    yield data.choices[0].text
    return
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim()
        if (data === '[DONE]') return
        try {
          const parsed = JSON.parse(data)
          if (parsed.choices?.[0]?.text) {
            yield parsed.choices[0].text
          }
        } catch {
          // Ignore parse errors
        }
      }
    }
  }
}

