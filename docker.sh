#!/bin/bash

set -e

case "$1" in
    build)
        echo "Building Docker images..."
        docker-compose build
        echo "Build complete!"
        ;;
    
    up)
        echo "Starting services..."
        docker-compose up -d
        
        echo "Waiting for Ollama to start..."
        sleep 5
        
        # Pull models if not present
        if ! docker exec rag-ollama ollama list | grep -q "nomic-embed-text"; then
            echo "Pulling nomic-embed-text model..."
            docker exec rag-ollama ollama pull nomic-embed-text
        fi
        
        if ! docker exec rag-ollama ollama list | grep -q "llama3.1:8b"; then
            echo "Pulling llama3.1:8b model..."
            docker exec rag-ollama ollama pull llama3.1:8b
        fi
        
        echo "Services started!"
        echo "Frontend: http://localhost:3000"
        echo "Backend: http://localhost:8000"
        ;;
    
    down)
        echo "Stopping services..."
        docker-compose down
        echo "Services stopped!"
        ;;
    
    ingest)
        echo "📚 Starting document ingestion..."
        echo "⚠️  This will delete the existing chroma_db"
        echo ""
        
        # Restart backend to ensure clean state
        echo "🔄 Restarting backend..."
        docker restart rag-backend
        
        # Delete chroma_db inside container
        echo "🗑️  Cleaning up old vector store..."
        docker exec rag-backend rm -rf /app/chroma_db 2>/dev/null || true
        
        # Run ingestion
        echo "🔄 Starting document processing..."
        docker exec rag-backend python ingest.py
        
        # Restart again to load the new vector store
        echo "🔄 Reloading backend with new documents..."
        docker restart rag-backend
        
        echo "✅ Ingestion complete! Backend reloaded with new documents."
        ;;
    
    logs)
        if [ -z "$2" ]; then
            docker-compose logs -f
        else
            docker-compose logs -f "$2"
        fi
        ;;
    
    *)
        echo "Usage: ./docker.sh {build|up|down|ingest|logs}"
        exit 1
        ;;
esac