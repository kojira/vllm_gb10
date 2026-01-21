export interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at?: string
}

export interface Session {
  id: string
  title: string
  engine: string | null
  model: string | null
  created_at: string
  updated_at: string
  messages?: Message[]
}

export interface Model {
  name: string
  path: string
  type: 'transformers' | 'gguf'
  gguf_files: string[] | null
}

export interface EngineStatus {
  status: 'idle' | 'loading' | 'loaded' | 'error'
  model: string | null
  progress: number
  progress_message: string
  process_alive?: boolean
}

export interface Status {
  vllm: EngineStatus
  llamacpp: EngineStatus
  download: {
    status: string
    model_id: string | null
    progress: number
    message: string
  }
}

