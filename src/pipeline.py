import os
import json
import yaml
import re
import unicodedata
import csv
from operator import itemgetter
from typing import List, Dict, Tuple, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


class RAGQuestionAnswering:
    """
    A class for RAG-based question answering with multiple response styles.
    """
    
    # Style-specific prompts
    PROMPTS = {
        "professional": """You are a professional business consultant. Answer the question using only the provided information.

Information: {context}

Question: {question}

Give a direct, authoritative business answer. Be concise and factual. Under 35 words.

Answer:""",
        
        "friendly": """You are a helpful colleague. Answer the question using only the provided information.

Information: {context}

Question: {question}

Answer in a warm, friendly way like you're helping a coworker. Be encouraging. Under 35 words.

Answer:""",
        
        "reassuring": """You are a supportive advisor. Answer the question using only the provided information.

Information: {context}

Question: {question}

Answer with confidence to put the person at ease. Be warm and reassuring. Under 35 words.

Answer:""",
        
        "simple": """You are explaining to someone who wants a clear, simple answer. Use only the provided information.

Information: {context}

Question: {question}

Give the simplest, most straightforward answer possible. No fancy words. Under 35 words.

Answer:"""
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the RAG Question Answering system.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config = self.load_config(config_path)
        self.vectorstore = None
        self.retriever = None
        self.llm_instances = {}
        self.chains = {}
        self.is_initialized = False
        self.current_request = None
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("--- Configuration Loaded ---")
        print(f"Model Path: {config['model_path']}")
        print(f"Output Path: {config['output_path']}")
        print("--------------------------\n")
        
        return config
    
    def load_request(self, request_path: str) -> Dict[str, Any]:
        """Load the request file with PDF and questions paths."""
        if not os.path.exists(request_path):
            raise FileNotFoundError(f"Request file not found at '{request_path}'")
        
        with open(request_path, 'r') as f:
            request = json.load(f)
        
        print("--- Request Loaded ---")
        print(f"PDF Path: {request['pdf_path']}")
        print(f"Questions Path: {request['questions_path']}")
        print("---------------------\n")
        
        self.current_request = request
        return request

    def read_questions_from_file(self, filepath: str) -> List[str]:
        """
        Read questions from a CSV file.
        The CSV file should have a header row, and one of the columns should be named 'question'.
        Returns a list of questions (strings).
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Questions file not found at '{filepath}'")
        
        questions = []
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'question' not in reader.fieldnames:
                raise ValueError(f"CSV file '{filepath}' must have a 'question' column.")
            for row in reader:
                q = row['question'].strip()
                if q:
                    questions.append(q)
        print(f"Found {len(questions)} questions in '{filepath}'.\n")
        return questions
    
    def clean_for_tts(self, text: str) -> str:
        """Enhanced cleaning function for TTS output."""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove quotes
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        # Remove unwanted robotic phrases
        unwanted_phrases = [
            "Answer:", "Response:", "Based on the information", "According to the document",
            "The information shows", "The context mentions", "The document states",
            "From what I can see", "According to what", "The provided information",
            "Based on what", "The information provided", "Looking at the information"
        ]
        
        for phrase in unwanted_phrases:
            pattern = re.escape(phrase) + r'[,:\s]*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove trailing incomplete thoughts
        text = re.sub(r'\.\s+However[,.]?.*$', '.', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\s+Therefore[,.]?.*$', '.', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\s+It\'s important.*$', '.', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\s+Please note.*$', '.', text, flags=re.IGNORECASE)
        
        # Remove markdown and special formatting
        text = re.sub(r'[*#_`\[\]()]', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading unnecessary words
        text = re.sub(r'^(that|it appears|it seems)\s+', '', text, flags=re.IGNORECASE)
        
        # Clean up sentence structure
        text = text.strip()
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def initialize_llm(self, style_name: str) -> LlamaCpp:
        """Initialize a single LLM instance for a specific style."""
        return LlamaCpp(
            model_path=self.config["model_path"],
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=self.config["generation_params"]["n_ctx"],
            temperature=0.05,
            max_tokens=50,
            verbose=False,
            stop=["\n\n", "Question:", "Information:", "Context:"],
            use_mlock=True,
            use_mmap=True,
            n_threads=4
        )
    
    def setup_vectorstore(self, pdf_path: str = None):
        """Set up the vector store and retriever."""
        if pdf_path is None:
            if self.current_request is None:
                raise ValueError("No request loaded. Call load_request() first or provide pdf_path.")
            pdf_path = self.current_request["pdf_path"]
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at '{pdf_path}'")
        
        print(f"Loading PDF: '{os.path.basename(pdf_path)}'...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        print("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["retrieval_params"]["chunk_size"],
            chunk_overlap=self.config["retrieval_params"]["chunk_overlap"]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Creating local embeddings and vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def setup_llm_instances(self):
        """Initialize all LLM instances for different styles."""
        print("Initializing 4 LLM instances for different styles...")
        
        # Check for CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
                print(f"✓ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("⚠ CUDA not available in PyTorch")
        except ImportError:
            print("⚠ PyTorch not available for CUDA check")
        
        # Initialize LLM instances for each style
        for style_name in self.PROMPTS.keys():
            print(f"Loading {style_name.title()} model...")
            self.llm_instances[style_name] = self.initialize_llm(style_name)
            print(f"✓ {style_name.title()} model loaded")
    
    def setup_chains(self):
        """Set up RAG chains for each style."""
        base_retrieval = {
            "context": itemgetter("question") | self.retriever,
            "question": itemgetter("question")
        }
        
        for style_name, prompt_template in self.PROMPTS.items():
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.chains[style_name] = (
                base_retrieval | 
                prompt | 
                self.llm_instances[style_name] | 
                StrOutputParser()
            )
    
    def initialize_model_system(self):
        """Initialize only the model components (LLM instances and chains)."""
        if self.is_initialized:
            print("Model system already initialized.")
            return
        
        self.setup_llm_instances()
        self.is_initialized = True
        print("✓ Model system initialized!")
    
    def initialize_request_system(self, request_path: str = None):
        """Initialize the request-specific components (vectorstore and chains)."""
        if request_path:
            self.load_request(request_path)
        
        if self.current_request is None:
            raise ValueError("No request loaded. Call load_request() first.")
        
        self.setup_vectorstore()
        self.setup_chains()
        print("✓ Request system initialized!")
    
    def validate_answer(self, answer: str) -> str:
        """Validate and adjust answer quality for appropriate word count."""
        if not answer or len(answer.strip()) < 5:
            return "I don't have that information in the document."
        
        word_count = len(answer.split())
        
        # If too short, add disclaimer
        if word_count < 60:
            answer += " Please refer to your policy document for complete details."
        
        # If too long, truncate intelligently
        elif word_count > 120:
            words = answer.split()
            truncate_point = 100
            for i in range(95, 105):
                if i < len(words) and words[i].endswith(('.', '!', '?')):
                    truncate_point = i + 1
                    break
            
            answer = " ".join(words[:truncate_point])
            if not answer.endswith(('.', '!', '?')):
                answer += '.'
        
        return answer
    
    def answer_question(self, question: str) -> Dict[str, str]:
        """Answer a single question using all available styles."""
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize_system() first.")
        
        print(f"\nProcessing question: {question}")
        answers = {}
        
        for i, (style_name, chain) in enumerate(self.chains.items(), 1):
            print(f"--- {style_name.title()} Model ({i}/4) ---")
            
            try:
                input_data = {"question": question}
                
                # Check context size
                retrieved_docs = self.retriever.invoke(question)

                context_text = "\n".join([doc.page_content for doc in retrieved_docs])
                estimated_tokens = len(context_text.split()) + len(question.split()) + 100
                
                if estimated_tokens > 3500:
                    print(f"Warning: Estimated tokens ({estimated_tokens}) may be too large")
                    context_text = context_text[:2000]
                
                raw_answer = chain.invoke(input_data)
                clean_answer = self.clean_for_tts(raw_answer)
                validated_answer = self.validate_answer(clean_answer)
                
                answers[f"answer_{i}"] = validated_answer
                
                word_count = len(validated_answer.split())
                print(f"OUTPUT: {validated_answer}")
                print(f"Word count: {word_count} words")
                print(f"Estimated tokens used: {estimated_tokens}\n")
                
            except Exception as e:
                print(f"Error with {style_name} model: {e}")
                answers[f"answer_{i}"] = "I encountered an error processing this question."
                print(f"OUTPUT: Error occurred\n")
        
        return answers
    
    def process_questions_from_file(self, questions_path: str = None) -> List[Dict[str, Any]]:
        """Process all questions from the specified questions file."""
        if not self.is_initialized:
            raise RuntimeError("Model system not initialized. Call initialize_model_system() first.")
        
        if self.vectorstore is None or self.chains == {}:
            raise RuntimeError("Request system not initialized. Call initialize_request_system() first.")
        
        if questions_path is None:
            if self.current_request is None:
                raise ValueError("No request loaded. Call load_request() first or provide questions_path.")
            questions_path = self.current_request["questions_path"]
        
        questions = self.read_questions_from_file(questions_path)
        all_results = []
        
        print("\n--- Starting Answer Generation with 4 Specialized Models ---\n")
        
        for i, question in enumerate(questions):
            print(f"\n{'='*70}")
            print(f"Processing Question {i+1}/{len(questions)}")
            print(f"QUESTION: {question}")
            print(f"{'='*70}\n")
            
            answers = self.answer_question(question)
            question_result = {"question": question, "answers": answers}
            all_results.append(question_result)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """Save results to a JSON file."""
        if output_path is None:
            output_path = self.config["output_path"]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✓ All questions processed successfully!")
        print(f"✓ Results saved to '{output_path}'")
        print(f"✓ Generated {len(results)} questions × 4 styles = {len(results) * 4} total answers")
        print(f"{'='*70}")
    
    def run_complete_pipeline(self, request_path: str = None):
        """Run the complete question answering pipeline."""
        try:
            if not self.is_initialized:
                self.initialize_model_system()
            
            self.initialize_request_system(request_path)
            results = self.process_questions_from_file()
            # self.save_results(results)
            return results
        except Exception as e:
            print(f"Error in pipeline: {e}")
            raise

