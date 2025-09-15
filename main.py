#!/usr/bin/env python3
"""
Main entry point for the RAG Question Answering system.
This script loads the model once and then waits for processing requests.
"""

import sys
import os
import argparse
import json
import yaml
import csv
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src directory to path to import pipeline module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import RAGQuestionAnswering


def validate_csv_structure(csv_path: str) -> bool:
    """
    Validate that the CSV file has the required 'question' column.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        bool: True if CSV has required structure, False otherwise
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'question' not in reader.fieldnames:
                print(f"❌ CSV file '{csv_path}' must have a 'question' column.")
                print(f"   Found columns: {', '.join(reader.fieldnames or [])}")
                return False
            
            # Count valid questions
            question_count = 0
            for row in reader:
                if row['question'].strip():
                    question_count += 1
            
            if question_count == 0:
                print(f"❌ No valid questions found in '{csv_path}'")
                return False
            
            print(f"✅ Found {question_count} valid questions in CSV")
            return True
            
    except Exception as e:
        print(f"❌ Error validating CSV structure: {e}")
        return False


def validate_config_files(config_path: str) -> bool:
    """
    Validate that the configuration file exists and has required structure.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        bool: True if config file is valid, False otherwise
    """
    try:
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_config_keys = ['model_path', 'output_path', 'retrieval_params', 'generation_params']
        missing_keys = []
        
        for key in required_config_keys:
            if key not in config:
                missing_keys.append(key)
        
        if missing_keys:
            print("❌ Missing required configuration keys:")
            for key in missing_keys:
                print(f"   - {key}")
            return False
        
        # Check if model file exists
        if not os.path.exists(config['model_path']):
            print(f"❌ Model file not found: {config['model_path']}")
            return False
        
        print("✅ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return False


def validate_request_file(request_path: str) -> bool:
    """
    Validate that the request file exists and has required structure.
    
    Args:
        request_path (str): Path to the request file
        
    Returns:
        bool: True if request file is valid, False otherwise
    """
    try:
        if not os.path.exists(request_path):
            print(f"❌ Request file not found: {request_path}")
            return False

        with open(request_path, 'r') as f:
            request = json.load(f)
        
        required_request_keys = ['pdf_path', 'questions_path']
        missing_keys = []
        
        for key in required_request_keys:
            if key not in request:
                missing_keys.append(key)
        
        if missing_keys:
            print("❌ Missing required request keys:")
            for key in missing_keys:
                print(f"   - {key}")
            return False
        
        # Check if files exist
        if not os.path.exists(request['pdf_path']):
            print(f"❌ PDF file not found: {request['pdf_path']}")
            return False
        
        if not os.path.exists(request['questions_path']):
            print(f"❌ Questions file not found: {request['questions_path']}")
            return False
        
        # Validate CSV structure
        if not validate_csv_structure(request['questions_path']):
            return False
        
        print("✅ Request file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Error validating request: {e}")
        return False


def preview_questions_from_csv(csv_path: str, num_preview: int = 5) -> List[str]:
    """
    Preview questions from CSV file using the same logic as pipeline.py.
    
    Args:
        csv_path (str): Path to the CSV file
        num_preview (int): Number of questions to preview
        
    Returns:
        List[str]: List of preview questions
    """
    try:
        questions = []
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'question' not in reader.fieldnames:
                print(f"❌ CSV file '{csv_path}' must have a 'question' column.")
                return []
            
            for row in reader:
                q = row['question'].strip()
                if q:
                    questions.append(q)
                    if len(questions) >= num_preview:
                        break
        
        return questions
    except Exception as e:
        print(f"❌ Error reading CSV preview: {e}")
        return []


def run_interactive_mode(rag_system: RAGQuestionAnswering) -> None:
    """
    Run the system in interactive mode for single questions.
    
    Args:
        rag_system (RAGQuestionAnswering): Initialized RAG system
    """
    print("\n" + "="*70)
    print("🤖 RAG Question Answering - Interactive Mode")
    print("="*70)
    print("Commands:")
    print("  - Type a question to get answers")
    print("  - 'preview' - See sample questions from current request")
    print("  - 'load <request.json>' - Load a new request file")
    print("  - 'status' - Show current system status")
    print("  - 'quit' or 'exit' - Stop the system")
    print("-"*70)
    
    while True:
        try:
            user_input = input("\n💭 Your input: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'status':
                print(f"\n📊 System Status:")
                print(f"   Model initialized: {'✅' if rag_system.is_initialized else '❌'}")
                print(f"   Request loaded: {'✅' if rag_system.current_request else '❌'}")       
                print(f"   Vectorstore ready: {'✅' if rag_system.vectorstore else '❌'}")
                if rag_system.current_request:
                    print(f"   Current PDF: {rag_system.current_request['pdf_path']}")
                    print(f"   Current Questions: {rag_system.current_request['questions_path']}")
                continue
            
            if user_input.lower().startswith('load '):
                request_path = user_input[5:].strip()
                if not request_path:
                    print("❌ Please specify a request file path: load <request.json>")
                    continue
                
                if validate_request_file(request_path):
                    try:
                        rag_system.initialize_request_system(request_path)
                        print(f"✅ Request loaded and system initialized!")
                    except Exception as e:
                        print(f"❌ Error loading request: {e}")
                continue
            
            if user_input.lower() == 'preview':
                if rag_system.current_request:
                    preview_questions = preview_questions_from_csv(rag_system.current_request['questions_path'])
                    if preview_questions:
                        print("\n📋 Sample questions from current request:")
                        print("-" * 50)
                        for i, q in enumerate(preview_questions, 1):
                            print(f"{i}. {q}")
                        print("-" * 50)
                else:
                    print("❌ No request loaded. Use 'load <request.json>' first.")
                continue
            
            if not user_input:
                print("Please enter a valid command or question.")
                continue
            
            # Check if system is ready for questions
            if not rag_system.is_initialized:
                print("❌ Model not initialized. Please restart the system.")
                continue
            
            if rag_system.vectorstore is None or not rag_system.chains:
                print("❌ No request loaded. Use 'load <request.json>' first.")
                continue
            
            # Process the question
            print(f"\n🔍 Processing: {user_input}")
            print("-" * 50)
            
            answers = rag_system.answer_question(user_input)
            
            print(f"\n📋 Generated {len(answers)} response styles:")
            print("=" * 50)
            
            style_names = ["Professional", "Friendly", "Reassuring", "Simple"]
            for i, (answer_key, answer_text) in enumerate(answers.items(), 1):
                style_name = style_names[i-1] if i <= len(style_names) else f"Style {i}"
                print(f"\n{i}. {style_name} Response:")
                print(f"   {answer_text}")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing input: {e}")


def run_batch_mode(rag_system: RAGQuestionAnswering, request_path: str, save_results: bool = True) -> List[Dict[str, Any]]:
    """
    Run the system in batch mode to process all questions from request file.
    
    Args:
        rag_system (RAGQuestionAnswering): Initialized RAG system
        request_path (str): Path to request file
        save_results (bool): Whether to save results to file
        
    Returns:
        List[Dict[str, Any]]: Results from processing all questions
    """
    print("\n" + "="*70)
    print("📁 RAG Question Answering - Batch Mode")
    print("="*70)
    
    try:
        # Initialize request system
        rag_system.initialize_request_system(request_path)
        
        # Process questions
        results = rag_system.process_questions_from_file()
        
        if save_results:
            rag_system.save_results(results)
        else:
            print(f"\n✅ Processing complete! Generated {len(results)} question sets")
            print("💡 Results not saved (use --save to save to file)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        raise


def display_results_summary(results: List[Dict[str, Any]]) -> None:
    """
    Display a summary of the results.
    
    Args:
        results (List[Dict[str, Any]]): Results from processing questions
    """
    if not results:
        print("No results to display.")
        return
    
    print(f"\n📊 Results Summary:")
    print("=" * 50)
    print(f"Total questions processed: {len(results)}")
    print(f"Total answers generated: {len(results) * 4}")
    
    print(f"\n📝 Sample Results:")
    print("-" * 30)
    
    # Show first question's results as example
    if results:
        sample = results[0]
        print(f"Question: {sample['question'][:100]}...")
        
        style_names = ["Professional", "Friendly", "Reassuring", "Simple"]
        for i, (answer_key, answer_text) in enumerate(sample['answers'].items(), 1):
            style_name = style_names[i-1] if i <= len(style_names) else f"Style {i}"
            print(f"{style_name}: {answer_text[:80]}...")


def main():
    """Main function to run the RAG Question Answering system."""
    parser = argparse.ArgumentParser(
        description="RAG Question Answering System with Persistent Model Loading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive                     # Interactive mode (load requests on demand)
  python main.py --request request.json           # Batch mode with specific request
  python main.py --request request.json --no-save # Batch mode without saving
  python main.py --validate-only                  # Only validate configuration

Architecture:
  - config.yaml: Model configuration (loads once)
  - request.json: PDF and questions paths (loads per request)
  - Model persists in memory between requests

Note: CSV files must have a 'question' column header and contain valid questions.
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='src/config/config.yaml',
        help='Path to YAML configuration file (default: src/config/config.yaml)'
    )
    
    parser.add_argument(
        '--request', '-r',
        help='Path to request file for batch processing'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode (model loaded once, process requests on demand)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Process questions but do not save results to file'
    )
    
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Display results summary after processing'
    )
    
    parser.add_argument(
        '--validate-only', '-v',
        action='store_true',
        help='Only validate configuration, then exit'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print("🚀 RAG Question Answering System")
    print("=" * 50)
    print("Powered by LLaMA with persistent model loading")
    print(f"Configuration: {args.config}")
    print("-" * 50)
    
    # Validate configuration
    if not validate_config_files(args.config):
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    if args.validate_only:
        print("✅ Configuration validation successful")
        sys.exit(0)
    
    # Validate request file if provided
    if args.request and not validate_request_file(args.request):
        print("❌ Request validation failed")
        sys.exit(1)
    
    # Determine mode
    if not args.interactive and not args.request:
        print("❌ Either --interactive or --request must be specified")
        print("Use --help for usage information")
        sys.exit(1)
    
    try:
        # Initialize the RAG system with model loading
        print("\n🔧 Initializing RAG system...")
        print("⏳ Loading model (this may take a few moments)...")
        start_time = time.time()
        
        rag_system = RAGQuestionAnswering(config_path=args.config)
        rag_system.initialize_model_system()
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        print("🔄 Model ready for processing requests!")
        
        # Run in appropriate mode
        if args.interactive:
            run_interactive_mode(rag_system)
        else:
            results = run_batch_mode(rag_system, args.request, save_results=not args.no_save)
            
            if args.summary and results:
                display_results_summary(results)
    
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()