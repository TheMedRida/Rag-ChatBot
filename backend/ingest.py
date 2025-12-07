from rag_pipeline import RAGPipeline
import sys

def main():
    """
    Main function to run document ingestion.
    
    Processes all documents in the ./documents directory,
    generates embeddings, and stores them in ChromaDB.
    """
    print("=" * 60)
    print("Starting Document Ingestion")
    print("=" * 60)
    
    pipeline = RAGPipeline(skip_load=True)
    
    success = pipeline.ingest_documents()
    
    print("=" * 60)
    if success:
        print("All documents ingested successfully")
        print("You can now start the server with: python app.py")
        print("=" * 60)
    else:
        print("Ingestion failed")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()