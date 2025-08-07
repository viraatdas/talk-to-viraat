import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'

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

    // TODO: Replace this with actual model inference
    // For now, return a mock response
    const response: ChatResponse = await generateMockResponse(messages)

    res.json(response)
  } catch (error) {
    console.error('Chat error:', error)
    res.status(500).json({ error: 'Internal server error' })
  }
})

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
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`)
  console.log(`📱 Chat API available at http://localhost:${PORT}/api/chat`)
})

export default app
