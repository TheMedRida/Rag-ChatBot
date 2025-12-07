# RAG Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your documents. Built with FastAPI, Next.js, and ChromaDB.

## Quick Start

### Prerequisites
- Docker
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/TheMedRida/Rag-ChatBot.git
cd Rag-ChatBot

# Make docker.sh executable and run
chmod +x docker.sh
./docker.sh build
./docker.sh up

# Process documents in backend/documents/
# Only run this if you haven't ingested documents yet or if you want to re-ingest with new documents
./docker.sh ingest
```

### Access the Application
- Frontend Chat UI: http://localhost:3000
- Backend API: http://localhost:8000

## 📊 **How It Works**

1. **Documents → Chunks**: PDF/TXT/MD files are split into optimized chunks
2. **Chunks → Vectors**: Converted to embeddings using nomic-embed-text (via Ollama)
3. **Store in ChromaDB**: Vectors saved for fast similarity search
4. **Question → Answer**: Find relevant chunks, feed to llama3.1:8b for answer
5. **Show Sources**: Displays which documents were used to generate the answer

## Docker Management

The `docker.sh` script provides the following commands:

| Command | Description |
|---------|-------------|
| `./docker.sh up` | Start all services |
| `./docker.sh down` | Stop all services |
| `./docker.sh build` | Rebuild Docker images |
| `./docker.sh ingest` | Process and index documents |
| `./docker.sh logs` | View container logs |


## Local Development (Without Docker)

### Prerequisites

1. **Install Ollama**: https://ollama.com/download
2. **Pull required models**:
   ```bash
   ollama pull nomic-embed-text  # Embedding model
   ollama pull llama3.1:8b       # LLM model
   ```
3. Start Ollama

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run ingestion (first time or to update documents)
python ingest.py

# Start the FastAPI server
python app.py
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Testing

```bash
# Test the API
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Test question"}'
```

## Tech Stack

### Backend
- FastAPI
- ChromaDB
- LangChain
- PyMuPDF

### Frontend
- Next.js 16
- TypeScript
- Tailwind CSS

### Models (for Local Development)
- Ollama (llama3.1:8b & nomic-embed-text)

### Infrastructure
- Docker


## Author

**Mohamed Rida Lajghal**
- GitHub: [@TheMedRida](https://github.com/TheMedRida)
- Email: mohamedridalajghal@gmail.com