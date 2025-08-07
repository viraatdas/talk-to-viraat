const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3001/api'

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatResponse {
  content: string
  thinking?: string
}

export class ApiService {
  static async sendMessage(messages: ChatMessage[]): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ messages }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      return await response.json()
    } catch (error) {
      console.error('API Error:', error)
      // Fallback to mock response if API is not available
      return this.getMockResponse(messages[messages.length - 1].content)
    }
  }

  private static getMockResponse(userMessage: string): ChatResponse {
    const mockResponses = [
      "hey what's up\n\npretty cool you're testing this out",
      "lol yeah this is just a mock response\n\nthe real model will be way better",
      "basically this is simulating my text style\n\nshort bursts\n\nwith newlines",
      "once we get the actual model hooked up\n\nit'll be way more accurate",
      "this is just placeholder text\n\nbut you get the idea right?",
    ]
    
    const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)]
    
    return {
      content: randomResponse,
      thinking: "This is just a mock response since the actual model isn't connected yet. The real model will analyze the user's message and respond in Viraat's style."
    }
  }
}


