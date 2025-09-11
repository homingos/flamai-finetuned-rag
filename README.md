# Local RAG Batch Answer Generator

This project provides a powerful, offline solution for asking questions about a PDF document and generating multiple, contextually-correct, and stylistically-diverse answers for each question. It is highly optimized for GPU acceleration on both NVIDIA (CUDA) and Apple Silicon (Metal) hardware, leveraging `llama-cpp-python` for fast, local inference.


The output is a structured JSON file containing clean, conversational text ready for use in Text-to-Speech (TTS) applications, making it ideal for building talking agents or voice-based assistants.

## Features

-   **100% Offline & Private:** Your documents, questions, and generated answers never leave your machine.
-   **GPU Accelerated:** Optimized for high-speed batch processing on NVIDIA (CUDA) and Apple Silicon (Metal) GPUs.
-   **Multiple Diverse Answers:** Generates four distinct answers for each question, each with a different conversational persona (e.g., friendly, professional, simple).
-   **Factually Grounded:** All answers are strictly derived from the content of the provided PDF document using a Retrieval-Augmented Generation (RAG) pipeline.
-   **TTS-Ready Output:** Answers are automatically cleaned of markdown and artifacts, producing plain text perfect for any Text-to-Speech engine.
-   **Batch Processing:** Efficiently processes a list of questions from a plain text file.
-   **Structured JSON Output:** Saves all questions and their corresponding answers in a clean, easy-to-parse JSON file.

## How It Works

1.  **Load:** The script loads a PDF document and a text file of questions.
2.  **Chunk & Embed:** It splits the PDF into manageable text chunks and creates vector embeddings for each chunk using a local sentence transformer.
3.  **Index:** The embeddings are stored in a local FAISS vector store for fast retrieval.
4.  **Retrieve:** For each question, the system retrieves the most relevant text chunks from the PDF.
5.  **Generate in Batch:** It prepares four unique prompts (one for each persona) and sends them to a local LLM (like Mistral 7B) in a single, parallel batch call.
6.  **Clean & Save:** The generated answers are cleaned for TTS and saved to a structured JSON file.

## Project Structure

```
.
├── .gitignore
├── README.md
├── batch_answer_generator.py   # The main Python script
├── config.json.template        # Template for your configuration
└── requirements.txt            # Python dependencies
```

## Setup and Installation

Follow these steps to set up the project on your machine.

### 1. Prerequisites

-   Python 3.10+
-   **For NVIDIA GPUs:** Latest NVIDIA drivers and the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
-   **For Apple Silicon:** Xcode Command Line Tools (`xcode-select --install`).

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Set Up the Python Virtual Environment

```bash
# Create the environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate   # On Windows
```

### 4. Install Dependencies

First, install the core packages from `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Next, install `llama-cpp-python` with the correct hardware acceleration flags. **Choose only one** of the following commands based on your system.

**For NVIDIA GPUs (CUDA):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

**For Apple Silicon (Metal):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

### 5. Download a Language Model

This project is optimized for GGUF-format models. The **Mistral 7B Instruct** model is a great starting point.

1.  Download the model: [**`mistral-7b-instruct-v0.2.Q4_K_M.gguf`**](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf) (~4.37 GB).
2.  Place the downloaded `.gguf` file in a convenient location on your machine.

## Configuration

1.  **Create your `config.json`:**
    Copy the template file to create your local configuration.
    ```bash
    cp config.json.template config.json
    ```

2.  **Edit `config.json`:**
    Open `config.json` and update the paths and parameters.

    -   `pdf_path`: Absolute path to the PDF document you want to query.
    -   `questions_path`: Path to your questions file (e.g., `./questions.txt`).
    -   `model_path`: Absolute path to your downloaded `.gguf` model file.
    -   `output_path`: Where to save the final JSON results.
    -   `n_gpu_layers`: Set to `-1` to offload all possible layers to the GPU for maximum performance.
    -   `n_batch`: The token processing batch size. `512` is a good default. Increase it for high-VRAM GPUs, or decrease it if you encounter memory errors.

## Usage

1.  **Add your PDF:** Place your PDF document on your machine and update its path in `config.json`.
2.  **Create `questions.txt`:** Create a file named `questions.txt` in the project root. Add your questions, separated by blank lines.
    ```
    What is the policy for sick leave notification?

    How many days of bereavement leave are employees entitled to?
    ```
3.  **Run the script:**
    Make sure your virtual environment is active, then run the generator.
    ```bash
    python batch_answer_generator.py
    ```

The script will process each question and save the results to the file specified in `output_path` (e.g., `results.json`).

## Example Output (`results.json`)

The output is a JSON array, with each object containing a question and a dictionary of its four generated answers.

```json
[
    {
        "question": "What is the policy for sick leave notification?",
        "answers": {
            "answer_1": "Sure thing. You just need to let your supervisor know at least two hours before your shift starts if you're taking a sick day.",
            "answer_2": "I can certainly clarify that for you. The policy states that employees should inform their supervisor of sick leave no less than two hours before their scheduled start time.",
            "answer_3": "It's quite simple. If you're sick, the rule is to just contact your supervisor. The main thing to remember is to do it at least two hours before you were supposed to start work.",
            "answer_4": "Of course! To make sure everything is covered, just be sure to notify your supervisor two hours before your shift begins. Hope that helps!"
        }
    }
]
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.