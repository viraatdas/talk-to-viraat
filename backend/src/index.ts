import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import { fetch } from 'undici'

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3001

// Middleware
app.use(cors())
app.use(express.json())

// Types
interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatRequest {
  messages: ChatMessage[]
}

interface ChatResponse {
  content: string
  thinking?: string
}

// Mock responses that simulate Viraat's style
const mockResponses = [
  "hey what's going on\n\npretty cool you're testing this",
  "lol yeah\n\nthis is just a placeholder\n\nonce the real model is hooked up it'll be way better",
  "basically simulating my text style here\n\nshort messages\n\nwith line breaks",
  "the actual fine-tuned model will be much more accurate\n\nbut this gives you the idea",
  "yo\n\ntesting out the interface?\n\nseems to be working pretty well",
  "nice\n\nthe UI looks clean\n\nonce we get the real model connected it'll be fire",
  "pretty solid setup\n\njust need to swap in the actual model API\n\nshould be straightforward"
]

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { messages }: ChatRequest = req.body

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages format' })
    }

    const hfToken = process.env.HF_API_TOKEN
    const hfModelId = process.env.HF_MODEL_ID

    // If HF config is missing, fall back to mock for local dev
    if (!hfToken || !hfModelId) {
      const response: ChatResponse = await generateMockResponse(messages)
      return res.json(response)
    }

    // Build a simple chat-style prompt
    const prompt = buildPrompt(messages)

    const responseText = await callHuggingFaceInference(hfModelId, hfToken, prompt)

    const cleaned = postProcessModelOutput(responseText)

    const response: ChatResponse = {
      content: cleaned,
    }

    res.json(response)
  } catch (error) {
    console.error('Chat error:', error)
    // Fallback to mock on failure to keep UX smooth
    try {
      const { messages }: ChatRequest = req.body
      const response: ChatResponse = await generateMockResponse(messages)
      return res.json(response)
    } catch (e) {
      return res.status(500).json({ error: 'Internal server error' })
    }
  }
})

function buildPrompt(messages: ChatMessage[]): string {
  const systemPreface = `You are Viraat. Respond concisely in short, casual bursts separated by blank lines. If the user greets you, greet back. Keep it friendly.`
  const lines: string[] = []
  lines.push(systemPreface)
  lines.push('')
  for (const message of messages) {
    if (message.role === 'user') {
      lines.push(`User: ${message.content}`)
    } else {
      lines.push(`Assistant: ${message.content}`)
    }
  }
  lines.push('Assistant:')
  return lines.join('\n')
}

async function callHuggingFaceInference(modelId: string, token: string, prompt: string): Promise<string> {
  const url = `https://api-inference.huggingface.co/models/${encodeURIComponent(modelId)}`
  const body = {
    inputs: prompt,
    parameters: {
      max_new_tokens: 256,
      temperature: 0.7,
      top_p: 0.95,
      return_full_text: false,
    },
    options: {
      use_cache: true,
      wait_for_model: true,
    },
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`HF Inference error ${response.status}: ${text}`)
  }

  const data = await response.json()
  // Try common shapes
  if (Array.isArray(data) && data.length > 0) {
    const first = data[0] as any
    if (typeof first?.generated_text === 'string') return first.generated_text as string
    if (typeof first === 'string') return first
  }
  if (typeof (data as any)?.generated_text === 'string') return (data as any).generated_text as string
  if (Array.isArray((data as any)?.choices) && (data as any).choices[0]?.text) return (data as any).choices[0].text
  return typeof data === 'string' ? data : JSON.stringify(data)
}

function postProcessModelOutput(output: string): string {
  // Remove a leading "Assistant:" if the model echoes the role
  let text = output.trim()
  if (text.toLowerCase().startsWith('assistant:')) {
    text = text.slice('assistant:'.length).trim()
  }
  return text
}

async function generateMockResponse(messages: ChatMessage[]): Promise<ChatResponse> {
  // Simulate some processing time
  await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1500))

  const lastMessage = messages[messages.length - 1]?.content || ''
  
  // Simple keyword-based responses for demo
  let response = mockResponses[Math.floor(Math.random() * mockResponses.length)]
  
  if (lastMessage.toLowerCase().includes('hello') || lastMessage.toLowerCase().includes('hi')) {
    response = "hey what's up\n\nhow's it going?"
  } else if (lastMessage.toLowerCase().includes('how are you')) {
    response = "pretty good\n\njust chilling\n\nhow about you?"
  } else if (lastMessage.toLowerCase().includes('model') || lastMessage.toLowerCase().includes('ai')) {
    response = "yeah this is the fine-tuned model\n\ntrained on my iMessage data\n\npretty cool right?"
  }

  return {
    content: response,
    thinking: `Analyzing user message: "${lastMessage}". Generating response in Viraat's style with short, casual text bursts.`
  }
}

// Health check
app.get('/health', (req, res) => {
  const hfModelId = process.env.HF_MODEL_ID || null
  const usingMock = !process.env.HF_API_TOKEN || !process.env.HF_MODEL_ID
  res.json({ status: 'ok', timestamp: new Date().toISOString(), model: hfModelId, usingMock })
})

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
  console.log(`📱 Chat API available at http://localhost:${PORT}/api/chat`)
})

export default app
