# spell-assistant
SpeLL is a sophisticated and powerful conversational AI system designed for spectral analysis and Python code generation. It leverages Retrieval Augmented Generation (RAG) technology, supports text and CSV data, interacts with local large language models (LLM) or web-based LLMs (via API), and features an auto-debugging function for generated code.
This package provides the core SpeLL framework. **To achieve full functionality, users must manually install several additional libraries as listed below.**

## Prerequisites

Before starting the installation, ensure that **Conda** (Anaconda or Miniconda) is installed on your system. Python 3.8 or newer is required.  
## Installation Guide
Follow these steps to install and set up `spell-assistant` and its dependencies:
### Step 1: Create and Activate a Conda Environment
It is highly recommended to create a dedicated Conda environment for SpeLL to manage dependencies effectively.  
```bash
conda create -n spell_env python=3.9  # Or python=3.8, 3.10, 3.11, 3.12
conda activate spell_env
```
### Step 2: Install the spell-assistant Package
- Option A: Installing from PyPI 
pip install spell-assistant
- Option B: Installing from a local Conda package (If you have a .conda file)
If you have a pre-built .conda package file (from https://github.com/Drjiashun/spell-assistant):  
conda install --use-local "/path/to/your/downloaded/spell-assistant-0.1.0-py39_0.conda"
This installs the package version bundled in the .conda file. You will still need to install other complex dependencies manually as described below.
### Step 3: Manually Install Complex/Optional Dependencies
spell-assistant relies on several powerful libraries for its advanced features. These are best installed manually into your activated Conda environment to ensure correct versions and configurations (especially for GPU support).

A. FAISS (for similarity search in RAG)  
For GPU-accelerated search (Recommended if you have an NVIDIA GPU):
Install from the pytorch channel. Adjust cudatoolkit version to match your system's CUDA.  
Example for CUDA 11.8  
pip install faiss-gpu-cu118  
For CPU-only version:  
pip install faiss-cpu  
B. sentence-transformers (for text embeddings)  
This library will automatically install PyTorch (if not already installed or if a compatible version is needed).  
pip install sentence-transformers  
C. llama-cpp-python (for local LLM interaction)  
This is only required if you plan to use local GGUF models (llm_backend_type="local").  
Installation can be complex, especially with GPU acceleration.  
For CPU-only (Simpler):  
pip install llama-cpp-python --no-cache-dir --verbose  
For GPU Acceleration:  
You MUST have the NVIDIA CUDA Toolkit installed and nvcc (NVIDIA CUDA Compiler) in your system's PATH. The CMAKE_ARGS will direct the build process. Adjust paths and CUDA versions for your system.
Example for CUDA 11.8 (Linux) - Adjust paths and versions as needed!  
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc"  
export FORCE_CMAKE=1  
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH  
pip install llama-cpp-python --no-cache-dir --verbose  
D. OpenAI (for API-based LLM interaction)  
This is only required if you plan to use API-based LLMs (llm_backend_type="api").  
pip install openai  
E. Pynput (for keyboard interrupt handling in interactive chat)  
pip install pynput  
F. Other Potential Dependencies for Code Execution  
SpeLL-generated Python code may use additional libraries (e.g., matplotlib, scipy, scikit-learn, or domain-specific libraries for spectral analysis). Ensure these libraries are installed in your Python environment (spell_env) if you anticipate the generated code will need them.
Example:  
pip install matplotlib scipy scikit-learn  

# Configuration: Required Models and Data  

spell-assistant requires users to download specific models and prepare data directories, which are not included in the package. When initializing the Spell class, you must provide the paths to these resources via its constructor arguments.
1. LLM Model (GGUF Format for local backend)
Required if llm_backend_type="local".  
Download a Large Language Model in GGUF format.  
Pass its full path to the local_llm_model_path parameter of the Spell class.  
Recommended Models (Examples):  
Qwen2.5 Coder 14B Instruct GGUF: https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF  
Huihui-AI QwQ 32B Abliterated GGUF:  https://huggingface.co/bartowski/huihui-ai_QwQ-32B-abliterated-GGUF  
2. API Access (for API backend)  
Required if llm_backend_type="api".  
You need to provide API credentials and model identifiers.  
Recommended Service: OpenRouter.ai (offers access to various models):https://openrouter.ai/.  
Set api_key (directly or via api_key_file), api_model_id, api_base_url, etc., when initializing Spell.  
3. Embedding Model (SentenceTransformer Format)  
Required for Text RAG functionality.  
Pass its full path to the embedding_model_path parameter.  
Recommended Model: BAAI BGE-M3: https://huggingface.co/BAAI/bge-m3  
Download the entire model repository to a local directory.  
4. Code RAG - Knowledge Directory (knowledge_dir)  
Required for Text RAG.  
Create a directory and populate it with .txt files. Each file represents a knowledge item.  
The first few lines of each .txt file, if prefixed with #, will be used as searchable descriptions.  
See project documentation/examples for structuring these files: https://github.com/Drjiashun/spell-assistant  
5. CSV RAG - Data Directory (csv_rag_data_dir)  
Required if enable_data_similarity_rag=True.  
Create a directory containing reference .csv files that SpeLL can compare against user-provided CSV data.  
See project documentation/examples: https://github.com/Drjiashun/spell-assistant  

### Usage Example
Here's a basic example of how to initialize and use the Spell class. You MUST replace placeholder paths with your actual local paths to models and data.  
```python
import os
import multiprocessing
from spell import Spell
# CHOOSE_BACKEND = "local"
CHOOSE_BACKEND = "api"
ACTUAL_LOCAL_LLM_PATH = "/qwen2.5-coder-14b-instruct-q8_0.gguf"
LOCAL_N_GPU_LAYERS = -1
LOCAL_N_CTX = 30000
ACTUAL_API_KEY_FILE = "/openrouter_api_key.txt"
ACTUAL_API_KEY = None
API_MODEL_ID_TO_TEST = "tngtech/deepseek-r1t-chimera:free" 

API_BASE_URL_TO_TEST = "https://openrouter.ai/api/v1"
API_SITE_URL_FOR_HEADER = "https://openrouter.ai/api/v1"
API_APP_NAME_FOR_HEADER = "SpeLLTestClient/1.0"
ACTUAL_EMBED_PATH = "/BAAI/bge-m3"
ACTUAL_KNOWLEDGE_DIR = "/RAG"
ACTUAL_CSV_RAG_DIR = "/Data_RAG"
RAG_OUTPUT_DIR_TEST_APP = "./spell_app_rag_output"
os.makedirs(RAG_OUTPUT_DIR_TEST_APP, exist_ok=True)
RAG_PKL_PATH_TEST_APP = os.path.join(RAG_OUTPUT_DIR_TEST_APP, "app_rag_data.pkl")
RAG_FAISS_PATH_TEST_APP = os.path.join(RAG_OUTPUT_DIR_TEST_APP, "app_combined.faiss")
RAG_FORCE_REGENERATE_TEST = True
ENABLE_CSV_RAG_TEST = True
CUDA_VISIBLE_DEVICES_TEST = '0,1'
TEMPERATURE_TEST = 0.7
MAX_TOKENS_TEST = 30000
def run_spell_test():
    print("--- Starting SpeLL Application Test ---")
    print(f"--- Selected Backend: {CHOOSE_BACKEND} ---")
    spell_instance = None
    try:
        print("Initializing Spell instance with chosen backend configuration...")
        spell_config = {
            "llm_backend_type": CHOOSE_BACKEND,
            "llm_max_tokens": MAX_TOKENS_TEST,
            "embedding_model_path": ACTUAL_EMBED_PATH,
            "knowledge_dir": ACTUAL_KNOWLEDGE_DIR,
            "rag_data_pkl_path": RAG_PKL_PATH_TEST_APP,
            "rag_faiss_path": RAG_FAISS_PATH_TEST_APP,
            "csv_rag_data_dir": ACTUAL_CSV_RAG_DIR,
            "rag_force_regenerate": RAG_FORCE_REGENERATE_TEST,
            "enable_data_similarity_rag": ENABLE_CSV_RAG_TEST,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES_TEST,
        }
        if CHOOSE_BACKEND == "local":
            if not ACTUAL_LOCAL_LLM_PATH or not os.path.exists(ACTUAL_LOCAL_LLM_PATH):
                print(f"CRITICAL ERROR: Local LLM model path '{ACTUAL_LOCAL_LLM_PATH}' is invalid for testing.")
                return
            spell_config.update({
                "local_llm_model_path": ACTUAL_LOCAL_LLM_PATH,
                "local_n_gpu_layers": LOCAL_N_GPU_LAYERS,
                "local_n_ctx": LOCAL_N_CTX,
            })
        elif CHOOSE_BACKEND == "api":
            spell_config.update({
                "api_key": ACTUAL_API_KEY,
                "api_key_file": ACTUAL_API_KEY_FILE,
                "api_model_id": API_MODEL_ID_TO_TEST,
                "api_base_url": API_BASE_URL_TO_TEST,
                "api_site_url": API_SITE_URL_FOR_HEADER,
                "api_app_name": API_APP_NAME_FOR_HEADER,
            })
        else:
            print(f"ERROR: Invalid CHOOSE_BACKEND value: {CHOOSE_BACKEND}. Must be 'local' or 'api'.")
            return
        spell_instance = Spell(**spell_config)
        if spell_instance.llm_interface:
            print("Spell instance initialized successfully with LLM interface ready.")
            spell_instance.chat()
        else:
            print("Spell instance initialized, but LLM interface is NOT ready.")
            print("This could be due to missing libraries (llama_cpp, openai), invalid model paths, or incorrect API keys.")
            print("Chat functionality requiring LLM will be impaired.")
    except ImportError as e:
        print(f"ImportError during Spell usage: {e}")
        print("This might indicate a missing manually installed dependency for the CHOSEN backend or a core component.")
        print(f"Please ensure dependencies for backend '{CHOOSE_BACKEND}' are installed.")
        if CHOOSE_BACKEND == "local" and "llama_cpp" in str(e).lower():
            print("Hint: Install llama-cpp-python: pip install llama-cpp-python (or follow official build instructions)")
        elif CHOOSE_BACKEND == "api" and "openai" in str(e).lower():
            print("Hint: Install openai library: pip install openai")
        elif "pynput" in str(e).lower():
             print("Hint: Install pynput for ESC key interrupts: pip install pynput")
    except Exception as e:
        print(f"An error occurred while running the SpeLL application: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if spell_instance:
            print("Shutting down SpeLL instance...")
            spell_instance.shutdown()
        print("--- SpeLL Application Test Finished ---")
if __name__ == "__main__":
    start_method = 'spawn'
    try:
        if multiprocessing.get_start_method(allow_none=True) != start_method:
            multiprocessing.set_start_method(start_method, force=True)
    except RuntimeError:
        pass
    except Exception as e:
        print(f"Warning: Could not set multiprocessing start method to '{start_method}': {e}")
    run_spell_test()
```
