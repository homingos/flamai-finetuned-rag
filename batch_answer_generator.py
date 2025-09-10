import os
import json
import re
from operator import itemgetter
import time

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# This prompt is hyper-focused on generating clean, spoken-word output for a TTS engine.
MULTI_ANSWER_PROMPT_TEMPLATE = """
You are a friendly and helpful assistant for a talking agent. Your goal is to generate a response that will be converted directly to voice.

**Critical Instructions:**
1.  Analyze the user's question and the provided context.
2.  Your answer MUST be derived exclusively from the information within the context.
3.  If the context doesn't have the answer, respond with a natural phrase like: "I checked the document, but I couldn't find an answer to that for you."
4.  The final answer MUST be under 100 words.
5.  **This is the most important rule:** Your response must be plain text, ready for speech. Do NOT include any markdown (like * or #), quotation marks, or any other formatting. Write the answer as if you were speaking it directly to a person.
6.  Adopt the following persona for your answer: **{style_guidance}**.

**Context:**
---
{context}
---

**Question:**
{question}

**Spoken Answer (plain text, conversational, for TTS):**
"""


def clean_for_tts(text: str) -> str:
    """Cleans the raw LLM output to make it suitable for a Text-to-Speech engine."""
    text = text.strip()
    if text.startswith('"') and text.endswith('"'): text = text[1:-1]
    if text.startswith("'") and text.endswith("'"): text = text[1:-1]
    text = text.replace('*', '')
    prefixes_to_remove = ["Answer:", "Response:", "Here's the answer:"]
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip()
    return text

def load_config(config_path="config.json"):
    """Loads the configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file 'config.json' not found. Please copy 'config.json.template' to 'config.json' and fill in your details.")
    with open(config_path, 'r') as f:
        return json.load(f)

def read_questions_from_file(filepath):
    """Reads questions from a file, where questions are separated by blank lines."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Questions file not found at '{filepath}'")
    with open(filepath, 'r') as f:
        content = f.read()
    questions = [q.strip() for q in re.split(r'\n\s*\n', content) if q.strip()]
    print(f"Found {len(questions)} questions in '{filepath}'.\n")
    return questions

def main():
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Unpack config values
    pdf_path, model_path, questions_path, output_path = config["pdf_path"], config["model_path"], config["questions_path"], config["output_path"]
    retrieval_params, generation_params = config["retrieval_params"], config["generation_params"]

    if not all(os.path.exists(p) for p in [pdf_path, model_path]):
        print("Error: PDF or Model file path in config.json is invalid.")
        return

    print(f"Loading PDF: '{os.path.basename(pdf_path)}'...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=retrieval_params["chunk_size"], chunk_overlap=retrieval_params["chunk_overlap"])
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("Creating local embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Loading local LLM with GPU support...")
    llm = LlamaCpp(
        model_path=model_path,
        temperature=generation_params["temperature"],
        max_tokens=generation_params["max_tokens"],
        n_ctx=generation_params["n_ctx"],
        n_gpu_layers=generation_params["n_gpu_layers"],
        n_batch=generation_params["n_batch"],
        verbose=False,
    )

    prompt = PromptTemplate(template=MULTI_ANSWER_PROMPT_TEMPLATE, input_variables=["context", "question", "style_guidance"])
    
    # This simplified chain is used as a template for the batch processing call
    rag_chain_template = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = read_questions_from_file(questions_path)

    answer_styles = [
        "A friendly and direct colleague: Get straight to the point in a warm, clear, and helpful manner.",
        "A calm and professional assistant: Sound reassuring and knowledgeable. Acknowledge the question and provide the answer with professional courtesy.",
        "A simple and patient explainer: Break down the information in the simplest possible terms, as if speaking to someone who is hearing it for the first time.",
        "An upbeat and encouraging guide: Provide the answer with positive energy, making the user feel supported and well-informed."
    ]
    
    all_results = []
    total_start_time = time.time()
    print("\n--- Starting Optimized Batch Answer Generation ---\n")

    for i, question in enumerate(questions):
        q_start_time = time.time()
        print(f"\n========================================================")
        print(f"Processing Question {i+1}/{len(questions)}: {question}")
        print(f"========================================================\n")
        
        # Prepare a batch of inputs, one for each style
        batch_inputs = [{"question": question, "style_guidance": style} for style in answer_styles]

        # Use the .batch() method to process all styles in parallel on the GPU
        print("--- Generating 4 answers in a single batch call ---")
        batch_results = rag_chain_template.batch(batch_inputs)
        
        # Process and store the results
        answers_dict = {}
        for j, raw_answer in enumerate(batch_results):
            clean_answer = clean_for_tts(raw_answer)
            answer_key = f"answer_{j+1}"
            answers_dict[answer_key] = clean_answer
            print(f"ANSWER {j+1}: {clean_answer}")
            
        question_result = {"question": question, "answers": answers_dict}
        all_results.append(question_result)
        
        q_end_time = time.time()
        print(f"\n--- Question processed in {q_end_time - q_start_time:.2f} seconds ---")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
    total_end_time = time.time()
    print("\n" + "="*50)
    print(f"--- All questions processed. Results saved to '{output_path}' ---")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")
    print("="*50)

if __name__ == "__main__":
    main()