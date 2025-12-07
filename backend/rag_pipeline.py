"""
RAG Pipeline Module - Optimized for Containers
Handles document ingestion, embedding generation, and question answering.
"""

import ollama
import os
import gc
import shutil
import time
from typing import Dict, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for document-based question answering.
    Uses Ollama models for embeddings (nomic-embed-text) and generation (llama3.1:8b).
    """
    
    def __init__(self, persist_directory: str = "./chroma_db", skip_load: bool = False):
        """
        Initialize the RAG pipeline.
        Args:
            persist_directory: Path to store ChromaDB vector database
            skip_load: Skip loading existing vector store (for ingestion)
        """
        self.persist_directory = persist_directory
        
        # Get Ollama host from environment
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        print("=" * 60)
        print("RAG Pipeline initialized")
        print(f"Ollama host: {self.ollama_host}")
        print(f"Embedding model: nomic-embed-text")
        print(f"Generation model: llama3.1:8b")
        print(f"Persist directory: {persist_directory}")
        print("=" * 60)
        
        self.vectorstore = None
        self.retriever = None
        
        if not skip_load:
            self._load_existing_vectorstore()
    
    def _create_embeddings_class(self):
        """Create a memory-safe embeddings class for containers."""
        from langchain.embeddings.base import Embeddings
        
        class ContainerSafeEmbeddings(Embeddings):
            """Embeddings that process one document at a time for memory safety."""
            
            def __init__(self, host, model="nomic-embed-text"):
                self.host = host
                self.model = model
                self.client = ollama.Client(host=host)
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Process one document at a time - memory safe for containers."""
                embeddings = []
                
                for i, text in enumerate(texts):
                    try:
                        # Process ONE document at a time
                        response = self.client.embeddings(
                            model=self.model,
                            prompt=text[:8000]
                        )
                        embeddings.append(response["embedding"])
                        
                        # Show progress
                        if (i + 1) % 10 == 0:
                            print(f"  Embedded {i+1}/{len(texts)} documents")
                            
                        # Small delay to prevent overwhelming Ollama
                        time.sleep(0.05)
                        
                    except Exception as e:
                        print(f"  Warning: Error embedding document {i+1}: {str(e)[:100]}")
                        # Use a small random vector as fallback (better than zero vector)
                        import random
                        fallback = [random.uniform(-0.01, 0.01) for _ in range(768)]
                        embeddings.append(fallback)
                
                return embeddings
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                try:
                    response = self.client.embeddings(
                        model=self.model,
                        prompt=text[:8000]
                    )
                    return response["embedding"]
                except Exception as e:
                    print(f"Query embedding error: {e}")
                    # Return a small random vector
                    import random
                    return [random.uniform(-0.01, 0.01) for _ in range(768)]
        
        return ContainerSafeEmbeddings(host=self.ollama_host)
    
    def _load_existing_vectorstore(self) -> bool:
        """
        Load existing vector store from disk if available.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        if os.path.exists(self.persist_directory):
            try:
                print(f"Loading existing vector store from: {self.persist_directory}")
                
                # Use our memory-safe embeddings
                custom_embeddings = self._create_embeddings_class()
                
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=custom_embeddings
                )
                
                # Configure retriever with good defaults
                self.retriever = self.vectorstore.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={
                        "k": 6,           # Return 5 most relevant docs
                        "fetch_k": 10,    # Fetch 10 docs before MMR
                        "lambda_mult": 0.1  # Balance relevance/diversity
                    }
                )
                
                print("✓ Vector store loaded successfully")
                return True
                
            except Exception as e:
                print(f"Warning: Could not load existing vector store: {e}")
                print("Will create new one during ingestion")
        
        return False
    
    def is_ingested(self) -> bool:
        """Check if documents have been ingested."""
        return self.retriever is not None
    
    def _chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into optimized chunks for retrieval."""
        print("Splitting documents into chunks...")
        
        # Optimal chunking parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1700,      
            chunk_overlap=300,   
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True  # Useful for source tracking
        )
        
        split_docs = text_splitter.split_documents(docs)
        
        # Add metadata for better tracking
        for i, doc in enumerate(split_docs):
            doc.metadata['chunk_id'] = i
            if 'page' in doc.metadata:
                doc.metadata['context'] = f"Page {doc.metadata['page']}"
        
        print(f"Created {len(split_docs)} chunks")
        return split_docs
    
    def ingest_documents(self, documents_dir: str = "./documents") -> bool:
        """
        Ingest documents from a directory.
        Memory-safe for container environments.
        """
        print(f"Looking for documents in: {documents_dir}")
        
        if not os.path.exists(documents_dir):
            print(f"Error: Directory not found: {documents_dir}")
            return False
        
        # Find supported files
        files = []
        for f in os.listdir(documents_dir):
            if f.endswith(('.pdf', '.txt', '.md')):
                files.append(f)
        
        print(f"Found {len(files)} supported files")
        
        if not files:
            print("No PDF, TXT, or MD files found")
            return False
        
        all_docs = []
        
        # Load documents
        for filename in files:
            file_path = os.path.join(documents_dir, filename)
            print(f"Loading: {filename}")
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['source'] = filename
                    all_docs.extend(docs)
                    print(f"  → Loaded {len(docs)} pages")
                    
                elif filename.endswith(('.txt', '.md')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    all_docs.append(Document(
                        page_content=text,
                        metadata={"source": filename}
                    ))
                    print(f"  → Loaded text file")
                    
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        if not all_docs:
            print("Error: No documents could be loaded")
            return False
        
        print(f"Total documents loaded: {len(all_docs)}")
        
        # Split into chunks
        split_docs = self._chunk_documents(all_docs)
        
        # Clean up old vector store if it exists
        if os.path.exists(self.persist_directory):
            print(f"Removing old vector store...")
            self._close_vectorstore()
            time.sleep(1)
            try:
                shutil.rmtree(self.persist_directory)
                print("Old vector store removed")
            except Exception as e:
                print(f"Warning: Could not fully remove old vector store: {e}")
        
        print("Generating embeddings (memory-safe, one at a time)...")
        print("This may take a few minutes...")
        
        # Create our memory-safe embeddings
        custom_embeddings = self._create_embeddings_class()
        
        # Create vector store with first batch, then add rest
        # Using small batch size for container safety
        batch_size = 10
        total_docs = len(split_docs)
        
        for i in range(0, total_docs, batch_size):
            batch = split_docs[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} "
                  f"({len(batch)} documents)")
            
            if i == 0:
                # Create new vector store
                self.vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=custom_embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                # Add to existing vector store
                self.vectorstore.add_documents(batch)
            
            # Memory cleanup after each batch
            gc.collect()
            
            # Save progress
            self.vectorstore.persist()
        
        # Create retriever with optimized settings
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 10,
                "lambda_mult": 0.1
            }
        )
        
        print("=" * 60)
        print("✓ Documents ingested successfully!")
        print(f"Vector store saved to: {self.persist_directory}")
        print("=" * 60)
        
        gc.collect()
        return True
    
    def _close_vectorstore(self):
        """Close vector store to release resources."""
        if self.vectorstore is not None:
            try:
                del self.vectorstore
                self.vectorstore = None
                self.retriever = None
                gc.collect()
            except Exception as e:
                print(f"Warning: Error closing vector store: {e}")
    
    def _rerank_documents(self, docs: List[Document], question: str) -> List[Document]:
        """Simple keyword-based reranking for better relevance."""
        question_lower = question.lower()
        question_words = set(word for word in question_lower.split() if len(word) > 3)
        
        if not question_words:
            return docs[:4]  # Return first 4 if no meaningful words
        
        scored_docs = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            matches = sum(1 for word in question_words if word in content_lower)
            scored_docs.append((matches, doc))
        
        # Sort by score, keep top 5
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:5]]
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build optimized prompt for the language model."""
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {question}

IMPORTANT RULES:
1. Answer using ONLY information from the context
2. If the answer is not in the context, say "I don't have enough information"
3. Be specific and concise

ANSWER:"""
        return prompt
    
    def get_answer(self, question: str) -> Dict[str, any]:
        """Get answer to a question using RAG."""
        if not self.retriever:
            return {
                "answer": "No documents ingested yet. Please run ingestion first.",
                "sources": []
            }
        
        print(f"Processing question: {question}")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retriever.invoke(question)
            print(f"Retrieved {len(retrieved_docs)} relevant chunks")
            
            if not retrieved_docs:
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "sources": []
                }
            
            # Rerank for better relevance
            reranked_docs = self._rerank_documents(retrieved_docs, question)
            
            # Build context from top documents
            context_parts = []
            for i, doc in enumerate(reranked_docs[:5], 1):  # Use top 
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                content = doc.page_content[:1700]  # Limit context length
                context_parts.append(
                    f"[Source: {source}, Page: {page}]\n{content}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Build and send prompt
            prompt = self._build_prompt(question, context)
            
            print("Generating answer...")
            
            client = ollama.Client(host=self.ollama_host)
            response = client.chat(
                model="llama3.1:8b",
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.2,      # Low for factual accuracy
                    'num_predict': 500,      # Allow detailed answers
                    'num_ctx': 8192,         # Use full context window
                }
            )
            
            final_answer = response['message']['content'].strip()
            
            # Extract unique sources
            sources = []
            seen = set()
            for doc in reranked_docs[:5]:  # Return up to 5 sources
                source = doc.metadata.get('source', 'Unknown')
                if source not in seen:
                    sources.append(source)
                    seen.add(source)
            
            print("✓ Answer generated successfully")
            
            return {
                "answer": final_answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": []
            }