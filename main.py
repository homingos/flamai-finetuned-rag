from src.pipeline import RAGQuestionAnswering

def main():
    """Main function to run the RAG system."""
    try:
        # Create and run the RAG system
        rag_system = RAGQuestionAnswering(config_path="./src/config/config.json")
        rag_system.run_complete_pipeline()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()