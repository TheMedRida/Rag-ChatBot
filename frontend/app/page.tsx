"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendQuestion = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      if (!response.ok) throw new Error("Failed to get answer");

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: "assistant",
        content: "Error: Could not connect to backend. Make sure the API is running at http://localhost:8000",
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      
      <header className="bg-white border-b border-gray-200 p-4">
        <h1 className="text-2xl font-bold text-gray-800">RAG Q&A Chatbot</h1>
        <p className="text-sm text-gray-600">Ask questions about your documents</p>
      </header>

      
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg">👋 Welcome! Ask me anything about the documents.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-2xl rounded-lg p-4 ${
                msg.role === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-300">
                  <p className="text-xs font-semibold mb-1">Sources:</p>
                  <ul className="text-xs space-y-1">
                    {msg.sources.map((source, i) => (
                      <li key={i}>📄 {source}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <p className="text-gray-600">Thinking...</p>
            </div>
          </div>
        )}

        
        <div ref={messagesEndRef} />
      </div>

      
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && !loading && sendQuestion()}
            placeholder="Type your question here..."
            disabled={loading}
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
          />
          <button
            onClick={sendQuestion}
            disabled={loading || !input.trim()}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed font-medium"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}