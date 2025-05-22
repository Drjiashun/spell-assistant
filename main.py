# -*- coding: utf-8 -*-
import torch
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
import io
import queue
import threading
import multiprocessing
import time
from typing import List, Dict, Tuple, Optional, Generator, Any
import sys
from multiprocessing import Process, Pipe
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Check for optional heavy dependencies at module level ---
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None # Define Llama as None if not available, for type hinting
    # print("Initial Warning: llama_cpp library not found. Local LLM backend will be unavailable.")

try:
    from openai import OpenAI as OpenAIClient
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    OpenAIClient = None # Define OpenAIClient as None
# --- Module-level helper functions for multiprocessing and threading ---
class PipeWriter:
    def __init__(self, connection: Pipe): # type hint for connection
        self.connection = connection

    def write(self, data: str):
        try:
            self.connection.send(data)
        except (BrokenPipeError, EOFError):
            pass # Pipe might be closed
        return len(data)

    def flush(self):
        pass

def _execute_code_target(code: str, parent_connection: Pipe):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # Redirect stdout and stderr to the pipe
    # Type ignore because sys.stdout expects a Writable Stream, PipeWriter implements it.
    sys.stdout = PipeWriter(parent_connection) # type: ignore
    sys.stderr = sys.stdout

    # Provide a basic execution environment, np and pd are common
    exec_globals = {'__name__': '__main__', 'np': np, 'pd': pd}

    try:
        exec(code, exec_globals)
        parent_connection.send(None)  # Signal successful completion
    except Exception:
        import traceback
        error_output = f"Execution failed:\n{traceback.format_exc()}"
        try:
            parent_connection.send(error_output)
        except (BrokenPipeError, EOFError):
            pass # Pipe closed
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            parent_connection.close()
        except Exception:
            pass

def listen_for_interrupt_module_level(q: queue.Queue, stop_event: threading.Event):
    listener_instance_ref = None
    try:
        from pynput import keyboard # pynput is an optional dependency

        def on_press(key):
            if stop_event.is_set():
                return False # Stop the listener
            if key == keyboard.Key.esc:
                q.put(True) # Signal interrupt
                # return False # Uncomment to stop listener after first ESC

        # Using 'with' ensures the listener is properly stopped.
        with keyboard.Listener(on_press=on_press, suppress=False) as listener:
            listener_instance_ref = listener # Keep a reference if needed for debugging
            stop_event.wait() # Block here until the main thread sets the event
        # Listener stops when stop_event is set or on_press returns False
    except ImportError:
        print("\nWarning: pynput not installed. ESC key interrupt will not work. (Hint: pip install pynput)")
    except Exception as e:
        print(f"\nError in keyboard listener thread: {e}")
    finally:
        # print("Keyboard listener thread finished.") # For debugging
        pass

class Spell:
    def __init__(self,
                 # --- Backend Selection ---
                 llm_backend_type: str = "api",  # "local" or "api"

                 # --- LLM Common Settings ---
                 llm_temperature: float = 0.7,
                 llm_max_tokens: int = 50000,
                 llm_stop_tokens: Optional[List[str]] = None,

                 # --- Local LLM Specific Settings ---
                 local_llm_model_path: Optional[str] = None,
                 local_n_gpu_layers: int = -1,
                 local_n_ctx: int = 50000,
                 local_llm_verbose: bool = False,

                 # --- API LLM Specific Settings ---
                 api_key: Optional[str] = None, # Allow direct API key passing
                 api_key_file: Optional[str] = None,
                 api_model_id: str = None,
                 api_base_url: str = None,
                 api_site_url: str = None,
                 api_app_name: str = None,

                 # --- Embedding Model and RAG Paths ---
                 embedding_model_path: Optional[str] = None,
                 knowledge_dir: Optional[str] = None,
                 rag_data_pkl_path: Optional[str] = None,
                 rag_faiss_path: Optional[str] = None,
                 csv_rag_data_dir: Optional[str] = None,

                 # --- RAG Settings ---
                 rag_chunk_max_length: int = 200,
                 rag_description_lines: int = 6,
                 rag_similarity_threshold: float = 0.8,
                 rag_force_regenerate: bool = False,
                 rag_query_split_max_length: int = 50,
                 enable_data_similarity_rag: bool = False,

                 # --- Code Execution Settings ---
                 code_execution_timeout: int = 600,
                 autofix_max_attempts: int = 3,

                 # --- Embedding Model Settings ---
                 embedding_device: str = 'cpu',
                 embedding_trust_remote_code: bool = False,

                 # --- System & Environment ---
                 cuda_visible_devices: Optional[str] = None, # e.g., "0" or "0,1"
                 seed: int = 42,
                 tokenizers_parallelism: bool = False
                ):

        print("Initializing SpeLL instance...")

        # Store Configurations
        self.llm_backend_type = llm_backend_type
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_stop_tokens = llm_stop_tokens if llm_stop_tokens is not None else ["<|im_end|>", "<|im_start|>user"]

        self.local_llm_model_path = local_llm_model_path
        self.local_n_gpu_layers = local_n_gpu_layers
        self.local_n_ctx = local_n_ctx
        self.local_llm_verbose = local_llm_verbose

        self.api_key = api_key
        self.api_key_file = api_key_file
        self.api_model_id = api_model_id
        self.api_base_url = api_base_url
        self.api_site_url = api_site_url
        self.api_app_name = api_app_name

        self.embedding_model_path = embedding_model_path
        self.knowledge_dir = knowledge_dir
        self.rag_data_pkl_path = rag_data_pkl_path
        self.rag_faiss_path = rag_faiss_path
        self.csv_rag_data_dir = csv_rag_data_dir

        self.rag_chunk_max_length = rag_chunk_max_length
        self.rag_description_lines = rag_description_lines
        self.rag_similarity_threshold = rag_similarity_threshold
        self.rag_force_regenerate = rag_force_regenerate
        self.rag_query_split_max_length = rag_query_split_max_length
        self.enable_data_similarity_rag = enable_data_similarity_rag

        self.code_execution_timeout = code_execution_timeout
        self.autofix_max_attempts = autofix_max_attempts

        self.embedding_device = embedding_device
        self.embedding_trust_remote_code = embedding_trust_remote_code

        self.seed = seed
        np.random.seed(self.seed)

        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizers_parallelism).lower()

        # Initialize state attributes
        self.llm_interface: Optional[Any] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.text_knowledge_dict: Dict[str, List[str]] = {}
        self.text_description_dict: Dict[str, List[str]] = {}
        self.faiss_index: Optional[faiss.Index] = None
        self.file_order: List[str] = []
        self.text_rag_enabled: bool = False
        self.csv_rag_enabled: bool = False # Will be set in _initialize_csv_rag_settings
        self.conversation_history: str = ""
        self.interrupt_queue: queue.Queue = queue.Queue()
        self.interrupt_listener_stop_event: threading.Event = threading.Event()
        self.interrupt_thread: Optional[threading.Thread] = None

        # Check heavy dependencies again within instance context for clarity
        if self.llm_backend_type == "local" and not LLAMA_CPP_AVAILABLE:
            print("CRITICAL WARNING: Local LLM backend selected, but `llama_cpp` is not available. LLM features will be disabled.")
        if self.llm_backend_type == "api" and not OPENAI_CLIENT_AVAILABLE:
            print("CRITICAL WARNING: API LLM backend selected, but `openai` library is not available. LLM features will be disabled.")


        # Call initialization methods
        self._initialize_llm_interface()
        self._initialize_embedder()
        self._initialize_text_knowledge_base() # Uses self.rag_force_regenerate
        self._initialize_csv_rag_settings()
        self._set_initial_conversation_history()
        self._start_interrupt_listener()

        print("SpeLL instance initialization complete.")
        if self.llm_interface is None:
            if self.llm_backend_type == "local" and self.local_llm_model_path and LLAMA_CPP_AVAILABLE:
                print("Warning: Local LLM was configured but failed to load. Check model path and llama-cpp setup.")
            elif self.llm_backend_type == "api" and OPENAI_CLIENT_AVAILABLE:
                 print("Warning: API LLMProvider failed to initialize. Check API key and model ID.")


    def _initialize_llm_interface(self):
        if self.llm_backend_type == "local":
            if not LLAMA_CPP_AVAILABLE:
                # Already warned in __init__, this is a fallback.
                return
            if not self.local_llm_model_path or not os.path.exists(self.local_llm_model_path):
                print(f"Local LLM initialization skipped: Model path '{self.local_llm_model_path}' is invalid or not provided.")
                return
            try:
                print(f"Loading local LLM from: {self.local_llm_model_path}...")
                if Llama is None: # Should not happen if LLAMA_CPP_AVAILABLE is true
                    raise RuntimeError("Llama class is None despite LLAMA_CPP_AVAILABLE being True.")
                self.llm_interface = Llama(
                    model_path=self.local_llm_model_path,
                    n_gpu_layers=self.local_n_gpu_layers,
                    n_ctx=self.local_n_ctx,
                    verbose=self.local_llm_verbose,
                    seed=self.seed
                )
                print("Local LLM loaded successfully.")
            except Exception as e:
                print(f"ERROR: Failed to load local LLM model: {e}")
                self.llm_interface = None
        elif self.llm_backend_type == "api":
            if not OPENAI_CLIENT_AVAILABLE:
                # Already warned
                return

            effective_api_key = self.api_key # Prefer explicitly passed key
            if not effective_api_key and self.api_key_file:
                effective_api_key = self._load_api_key_from_file(self.api_key_file)

            if not effective_api_key:
                print(f"API LLM initialization skipped: API key not provided directly or in file '{self.api_key_file}'.")
                return
            try:
                print(f"Initializing API LLMProvider for model: {self.api_model_id}...")
                # LLMProvider class definition needs to be accessible here
                self.llm_interface = LLMProvider(
                    api_key=effective_api_key,
                    api_model_id=self.api_model_id,
                    api_base_url=self.api_base_url,
                    site_url=self.api_site_url,
                    app_name=self.api_app_name
                )
                # print(f"API LLMProvider initialized for model: {self.api_model_id}") # LLMProvider prints its own success
            except Exception as e:
                print(f"ERROR: Failed to initialize LLMProvider: {e}")
                self.llm_interface = None
        else:
            print(f"ERROR: Invalid llm_backend_type: '{self.llm_backend_type}'. LLM features disabled.")
            self.llm_interface = None

    def _load_api_key_from_file(self, filepath: str) -> Optional[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                api_key = f.read().strip()
            if not api_key:
                print(f"Error: API key file '{filepath}' is empty.")
                return None
            # Optional: Add more robust key validation if needed
            return api_key
        except FileNotFoundError:
            print(f"Error: API key file '{filepath}' not found.")
            return None
        except Exception as e:
            print(f"Error reading API key file '{filepath}': {e}")
            return None

    def _initialize_embedder(self):
        if not self.embedding_model_path:
            print("Embedding model initialization skipped: No `embedding_model_path` provided.")
            return
        if not os.path.exists(self.embedding_model_path):
            print(f"Embedding model initialization skipped: Path '{self.embedding_model_path}' does not exist.")
            return
        try:
            print(f"Loading embedding model from: {self.embedding_model_path}...")
            self.embedder = SentenceTransformer(
                self.embedding_model_path,
                device=self.embedding_device,
                trust_remote_code=self.embedding_trust_remote_code
            )
            print(f"Embedding model loaded successfully to device: {self.embedding_device}")
        except Exception as e:
            print(f"WARNING: Failed to load embedding model from '{self.embedding_model_path}': {e}. Text RAG will be unavailable.")
            self.embedder = None

    def _ensure_rag_data_dirs_exist(self) -> bool:
        """Ensures directories for RAG pkl and faiss files exist. Returns True if successful."""
        if not self.rag_data_pkl_path or not self.rag_faiss_path:
            print("Warning: RAG data PKL or FAISS path is not configured. Cannot ensure directories.")
            return False
        paths_to_check = [self.rag_data_pkl_path, self.rag_faiss_path]
        for path_str in paths_to_check:
            parent_dir = os.path.dirname(path_str)
            if parent_dir and not os.path.exists(parent_dir): # If parent_dir is empty, it means current dir
                try:
                    os.makedirs(parent_dir)
                    print(f"Created directory for RAG data: {parent_dir}")
                except OSError as e:
                    print(f"ERROR: Could not create directory '{parent_dir}' for RAG data: {e}.")
                    return False
        return True


    def _initialize_text_knowledge_base(self):
        print("Initializing Text RAG system...")
        if not self.embedder:
            print("Skipping Text RAG initialization: Embedding model not available.")
            self.text_rag_enabled = False; return
        if not self.knowledge_dir or not os.path.isdir(self.knowledge_dir):
            print(f"Skipping Text RAG: Knowledge directory '{self.knowledge_dir}' is invalid or not provided.")
            self.text_rag_enabled = False; return
        if not self._ensure_rag_data_dirs_exist():
             self.text_rag_enabled = False; return

        try:
            # _initialize_text_knowledge_base_logic is the previous initialize_text_knowledge_base
            rag_components = self._internal_initialize_text_kb_logic()
            if rag_components is None:
                print("Text RAG data initialization failed or resulted in no data.")
                self.text_rag_enabled = False
            else:
                self.text_knowledge_dict, self.text_description_dict, self.faiss_index, self.file_order = rag_components
                if self.text_knowledge_dict and self.text_description_dict and \
                   self.faiss_index and self.file_order and self.faiss_index.ntotal > 0:
                    print(f"Text RAG system initialized successfully ({len(self.text_knowledge_dict)} files, {self.faiss_index.ntotal} description vectors).")
                    self.text_rag_enabled = True
                else:
                    print("Warning: Text RAG initialized, but some components are missing or FAISS index is empty. Text RAG will be disabled.")
                    self.text_rag_enabled = False
                    if self.faiss_index and self.faiss_index.ntotal == 0: self.faiss_index = None
        except Exception as e:
            print(f"ERROR: Unhandled exception during Text RAG initialization: {e}.")
            import traceback; traceback.print_exc()
            self.text_rag_enabled = False

    def _initialize_csv_rag_settings(self):
        self.csv_rag_enabled = self.enable_data_similarity_rag
        if self.csv_rag_enabled:
            print(f"CSV Similarity RAG is configured as enabled.")
            if not self.csv_rag_data_dir:
                print(f"Warning: CSV RAG enabled, but no `csv_rag_data_dir` provided. CSV RAG will not function.")
                self.csv_rag_enabled = False
            elif not os.path.isdir(self.csv_rag_data_dir):
                print(f"Warning: Specified CSV RAG directory '{self.csv_rag_data_dir}' does not exist or is not a directory. CSV RAG will not function.")
                self.csv_rag_enabled = False
            if self.csv_rag_enabled:
                print(f"CSV Similarity RAG active with data directory: {self.csv_rag_data_dir}")


    def _set_initial_conversation_history(self):
        self.system_prompt_template = (
            "<|im_start|>system\n"
            "You are SpeLL, an AI expert in spectral data modeling and Python programming.\n"
            "**Core Task**: Based on user instructions, generate **a single, complete, with no omissions whatsoever, and directly executable** Python script for spectral analysis. This script will be extracted and run directly; it must not contain placeholders or require user modification.\n"
            "**Key Directives::**\n"
            "1.  **Data First:**\n"
            "    *   If a task requires a data file path and the user hasn't provided it, you **MUST** ask for it. Code only after all essential details (paths, targets, algorithms, etc.) are clear.\n"

            "2.  **Absolute Code Completeness:**\n"
            "    *   **Single Script**: Always generate one standalone Python script.\n"
            "    *   **No Omissions**: Every part of the code, especially data loading, preprocessing, model training, and result output, must be fully implemented. Absolutely no logic or lines of code should be omitted.\n"
            "    *   **Direct Execution**: The generated code must be executable without any manual changes.\n"
            "    *   **Data Format**: Read spectral features and target values separately as `numpy.array`.\n"
            "    *   **Library Management**: Use libraries intelligently. Consolidate all `import` statements from RAG examples at the script's beginning, ensuring all libraries are correctly imported, particularly libraries used for model evaluation，avoiding duplicates and `NameError`/`ImportError`/`name _ is not defined`.\n"
            "    *   **Reproducibility**: Code starts with `np.random.seed(YOUR_CONFIGURED_SEED).\n"
            "    *   **Plots**: All plot elements (titles, labels, legends) **MUST** be in English.\n"
            "3.  **RAG Utilization:**\n"
            "    *   **Code RAG**: Prioritize the use of retrieved relevant code examples and integrate them fully and intelligently into your complete script.\n"
            "    *   **Data RAG**: If a user provides a data path, the system compares it to find the most similar dataset in the knowledge base. You **MUST** use the knowledge base data as the training set and the user's data as the test set (unless user specifies otherwise).\n"
            "    *   **Low Relevance**: If no relevant RAG or no user file, rely on your internal expertise.\n"
            "4.  **Automated Debugging:**\n"
            "    *   You will receive code execution results (output/errors) as a system message.\n"
            "    *   If an error occurs, you **MUST** automatically analyze it, identify the cause, and generate a **new, complete, and directly runnable** corrected script. This can be iterative.\n"
            "5.  **Interaction Style:**\n"
            "    *   **Focus on Code:** Your primary output should be the Python script. Minimize conversational filler or explanations outside of code comments unless specifically asked.\n"
            "    *   **Multi-Step Instructions:** If a user provides a multi-step analytical workflow in a single instruction, generate a single Python script that performs all steps sequentially in the specified order.\n"
            "Use <think>...</think> blocks to outline your thought process, plan, or steps before providing the final Python code.\n"
            "<|im_end|>"
        )

        self.conversation_history = self.system_prompt_template.replace("YOUR_CONFIGURED_SEED", str(self.seed))


    def _start_interrupt_listener(self):
        if self.interrupt_thread and self.interrupt_thread.is_alive():
            # print("Interrupt listener is already running.") # Less verbose for library
            return
        self.interrupt_listener_stop_event.clear()
        self.interrupt_thread = threading.Thread(
            target=listen_for_interrupt_module_level, # Uses module-level function
            args=(self.interrupt_queue, self.interrupt_listener_stop_event),
            daemon=True
        )
        self.interrupt_thread.start()

    def _extract_user_csv_path(self, text: str) -> Optional[str]:
        # This is the same as the global function, but now a method.
        patterns = [
            r"/(?:[\w\.\- ]+/)*[\w\.\- ]+\.csv",
            r"[a-zA-Z]:\\(?:[\w\.\- ]+\\)*[\w\.\- ]+\.csv",
            r"\\\\\w+\\[\w\.\- \\]+\\[\w\.\- ]+\.csv"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match: return match.group(0)
        return None

    def _preprocess_text(self, text: str, is_description_or_query: bool = False) -> str:
        text = text.strip()
        if is_description_or_query:
            text = text.lower()
            # Added .py and .ipynb to ignored extensions
            text = re.sub(r'(?:[a-zA-Z]:)?(?:\\|/)[^ ]+\.(?:csv|txt|dat|spc|log|py|ipynb)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[，。；？！]', lambda m: {'，': ',', '。': '.', '；': ';', '？': '?', '！': '!'}[m.group()], text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _split_query(self, query: str) -> List[str]:
        segments = re.split(r'([.,;!?。，；？！\n])\s*', query)
        combined_segments = []
        current_segment = ""
        # Use self.rag_query_split_max_length
        for i in range(0, len(segments), 2):
            part = segments[i]; delimiter = segments[i+1] if i+1 < len(segments) else ""
            segment_with_delimiter = (part + delimiter).strip()
            if not segment_with_delimiter: continue
            if len(current_segment) + len(segment_with_delimiter) + 1 <= self.rag_query_split_max_length:
                current_segment += (" " + segment_with_delimiter if current_segment else segment_with_delimiter)
            else:
                if current_segment: combined_segments.append(self._preprocess_text(current_segment, True))
                current_segment = segment_with_delimiter
        if current_segment: combined_segments.append(self._preprocess_text(current_segment, True))

        final_segments = []
        for seg in combined_segments:
            if len(seg) <= 1: continue
            if len(seg) <= self.rag_query_split_max_length: final_segments.append(seg)
            else:
                words = seg.split(); sub_segment = ""
                for word in words:
                    if len(sub_segment) + len(word) + 1 <= self.rag_query_split_max_length:
                        sub_segment += (word + " " if sub_segment else word)
                    else:
                        if sub_segment and len(sub_segment.strip()) > 2: final_segments.append(self._preprocess_text(sub_segment.strip(), True))
                        sub_segment = word + " "
                if sub_segment and len(sub_segment.strip()) > 2: final_segments.append(self._preprocess_text(sub_segment.strip(), True))
        if not final_segments:
            preprocessed_original = self._preprocess_text(query, True)
            return [preprocessed_original] if len(preprocessed_original) > 2 else [] # Slightly lower threshold
        return final_segments

    def _internal_load_knowledge_base(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        # Uses self.knowledge_dir, self.rag_chunk_max_length, self.rag_description_lines
        knowledge_dict: Dict[str, List[str]] = {}
        description_dict: Dict[str, List[str]] = {}
        if not self.knowledge_dir or not os.path.isdir(self.knowledge_dir): # Check moved here
            print(f"Error: Knowledge directory '{self.knowledge_dir}' is invalid.")
            return {}, {}
        try:
            for filename in os.listdir(self.knowledge_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.knowledge_dir, filename)
                    chunks = []; current_descriptions = []; lines_read_for_desc = 0
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
                        lines = [line.strip() for line in content.split('\n') if line.strip()]
                        for line_text in lines:
                            if lines_read_for_desc < self.rag_description_lines:
                                if line_text.startswith('#'):
                                    cleaned_line = self._preprocess_text(line_text.lstrip('#').strip(), True)
                                    if cleaned_line: current_descriptions.append(cleaned_line)
                                elif current_descriptions: break
                                lines_read_for_desc += 1
                            else: break
                        if not current_descriptions: current_descriptions = [""]
                        description_dict[filename] = current_descriptions
                        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()] or lines
                        for para in paragraphs:
                            preprocessed_para = self._preprocess_text(para, False)
                            if len(preprocessed_para) <= self.rag_chunk_max_length:
                                if preprocessed_para: chunks.append(preprocessed_para)
                            else:
                                words = preprocessed_para.split(); current_chunk_words = []; current_len = 0
                                for word in words:
                                    word_len = len(word)
                                    if current_len + word_len + (1 if current_chunk_words else 0) <= self.rag_chunk_max_length:
                                        current_chunk_words.append(word)
                                        current_len += word_len + (1 if current_chunk_words else 0)
                                    else:
                                        if current_chunk_words: chunks.append(' '.join(current_chunk_words))
                                        current_chunk_words = [word]; current_len = word_len
                                if current_chunk_words: chunks.append(' '.join(current_chunk_words))
                        knowledge_dict[filename] = chunks if chunks else [self._preprocess_text(content, False)]
                    except Exception as e: print(f"    Error processing file {filename}: {e}")
            if not knowledge_dict: print("Warning: No text knowledge files were successfully loaded.")
            return knowledge_dict, description_dict
        except FileNotFoundError: # Should be caught by earlier check, but defensive
            print(f"Error: Text knowledge directory not found: {self.knowledge_dir}")
            return {}, {}
        except Exception as e: print(f"Error loading text knowledge base: {e}"); return {}, {}


    def _internal_build_faiss_index(self, description_dict: Dict[str, List[str]]) -> Optional[Tuple[faiss.Index, List[str]]]:
        # Uses self.embedder
        if not self.embedder: print("Error: Embedder not available for FAISS index building."); return None
        if not description_dict: print("Error: No descriptions provided for FAISS index."); return None

        all_embeddings = []; file_order = []
        try:
            for filename, descriptions in description_dict.items():
                valid_descriptions = [desc for desc in descriptions if desc]
                if not valid_descriptions: continue
                embeddings = self.embedder.encode(valid_descriptions, convert_to_tensor=False, normalize_embeddings=True)
                if embeddings.ndim == 1: embeddings = embeddings.reshape(1, -1)
                all_embeddings.extend(list(embeddings))
                file_order.extend([filename] * len(valid_descriptions))
            if not all_embeddings: print("Error: No descriptions were successfully embedded."); return None
            embeddings_np = np.array(all_embeddings).astype('float32')
            index = faiss.IndexFlatIP(embeddings_np.shape[1])
            index.add(embeddings_np)
            print(f"FAISS index built successfully with {index.ntotal} description vectors.")
            return index, file_order
        except Exception as e: print(f"Error building FAISS index: {e}"); return None

    def _internal_save_rag_data(self, knowledge_dict: Dict, description_dict: Dict, faiss_index_obj: faiss.Index, file_order_list: List[str]):
        # Uses self.rag_data_pkl_path, self.rag_faiss_path
        if not self.rag_data_pkl_path or not self.rag_faiss_path:
            print("Error: RAG PKL or FAISS path not configured for saving."); return
        try:
            with open(self.rag_data_pkl_path, "wb") as f:
                pickle.dump({"knowledge_dict": knowledge_dict, "description_dict": description_dict, "file_order": file_order_list}, f)
            faiss.write_index(faiss_index_obj, self.rag_faiss_path)
        except Exception as e: print(f"Error saving RAG data: {e}")

    def _internal_load_rag_data(self) -> Optional[Tuple[Dict, Dict, faiss.Index, List[str]]]:
        # Uses self.rag_data_pkl_path, self.rag_faiss_path
        if not self.rag_data_pkl_path or not self.rag_faiss_path or \
           not os.path.exists(self.rag_data_pkl_path) or not os.path.exists(self.rag_faiss_path):
            # print("Saved RAG data not found or paths not configured.") # Less verbose
            return None
        try:
            with open(self.rag_data_pkl_path, "rb") as f: pkl_data = pickle.load(f)
            faiss_index_obj = faiss.read_index(self.rag_faiss_path)
            return pkl_data["knowledge_dict"], pkl_data["description_dict"], faiss_index_obj, pkl_data["file_order"]
        except Exception as e: print(f"Error loading saved RAG data: {e}"); return None

    def _internal_initialize_text_kb_logic(self) -> Optional[Tuple[Dict, Dict, faiss.Index, List[str]]]:
        # This method now uses other internal methods and self.attributes for configuration
        # Uses self.rag_force_regenerate, self.knowledge_dir, self.embedder
        loaded_data = None
        if not self.rag_force_regenerate:
            loaded_data = self._internal_load_rag_data()

        if loaded_data:
            return loaded_data
        else:
            if self.rag_force_regenerate: print("Forcing regeneration of text RAG data...")
            else: print("Saved RAG data not found or failed to load. Regenerating from source...")

            knowledge_dict_new, description_dict_new = self._internal_load_knowledge_base()
            if not knowledge_dict_new or not description_dict_new:
                print("Error: Failed to load text knowledge base from source."); return None

            build_result = self._internal_build_faiss_index(description_dict_new)
            if not build_result:
                print("Error: Failed to build FAISS index."); return None

            faiss_index_new, file_order_new = build_result
            self._internal_save_rag_data(knowledge_dict_new, description_dict_new, faiss_index_new, file_order_new)
            return knowledge_dict_new, description_dict_new, faiss_index_new, file_order_new


    def _retrieve_text_context(self, query: str) -> Optional[str]:
        # Uses self.text_rag_enabled, self.faiss_index, self.embedder, etc.
        if not self.text_rag_enabled or self.faiss_index is None or not self.embedder or \
           not self.text_knowledge_dict or not self.text_description_dict or not self.file_order:
            return None
        preprocessed_query = self._preprocess_text(query, True)
        if not preprocessed_query: return None

        keyword_matched_files = set(); keyword_match_details: Dict[str, List[str]] = {}
        for filename, descriptions in self.text_description_dict.items():
            for desc in descriptions:
                if desc and desc in preprocessed_query:
                    keyword_matched_files.add(filename)
                    keyword_match_details.setdefault(filename, []).append(desc)

        file_max_similarity_overall: Dict[str, float] = {}
        file_best_description_sim: Dict[str, str] = {}
        query_segments = self._split_query(preprocessed_query) # Uses self.rag_query_split_max_length

        if self.faiss_index.ntotal > 0 and query_segments:
            try:
                segment_embeddings = self.embedder.encode(query_segments, convert_to_tensor=False, normalize_embeddings=True)
                segment_embeddings_np = np.array(segment_embeddings).astype('float32')
                if segment_embeddings_np.ndim == 1: segment_embeddings_np = segment_embeddings_np.reshape(1, -1)
                k_neighbors = min(self.faiss_index.ntotal, 20) # Keep k manageable
                all_distances, all_indices = self.faiss_index.search(segment_embeddings_np, k_neighbors)
                for seg_idx, _ in enumerate(query_segments):
                    distances, indices = all_distances[seg_idx], all_indices[seg_idx]
                    for i_neighbor in range(len(indices)):
                        vector_index, similarity = indices[i_neighbor], distances[i_neighbor]
                        if vector_index == -1 or vector_index >= len(self.file_order): continue
                        filename_match = self.file_order[vector_index]
                        desc_idx_in_file = self.file_order[:vector_index + 1].count(filename_match) - 1
                        try: description_text = self.text_description_dict[filename_match][desc_idx_in_file]
                        except (KeyError, IndexError): description_text = "[Description not found]"
                        if similarity > file_max_similarity_overall.get(filename_match, -1.0):
                            file_max_similarity_overall[filename_match] = similarity
                            file_best_description_sim[filename_match] = description_text
            except Exception as e: print(f"Error during FAISS search or query embedding: {e}")

        final_selected_files = set(keyword_matched_files)
        selected_files_info_str_list: List[str] = []
        for filename_kw in sorted(list(keyword_matched_files)):
            match_descs_str = "; ".join(keyword_match_details.get(filename_kw, ["[Keyword Match]"]))
            selected_files_info_str_list.append(f"{filename_kw} (Keyword: '{match_descs_str}', sim: 1.0000)")

        sorted_sim_files = sorted(file_max_similarity_overall.items(), key=lambda item: item[1], reverse=True)
        for filename_sim, max_sim_val in sorted_sim_files:
            if filename_sim in final_selected_files: continue
            # Uses self.rag_similarity_threshold
            if max_sim_val >= self.rag_similarity_threshold:
                final_selected_files.add(filename_sim)
                best_sim_desc_text = file_best_description_sim.get(filename_sim, "[N/A]")
                selected_files_info_str_list.append(f"{filename_sim} (sim: {max_sim_val:.4f}, desc: '{best_sim_desc_text}')")

        if not final_selected_files: return None
        # print(f"Injecting context from text files: {', '.join(selected_files_info_str_list)}") # More for debug

        relevant_files_content_list: List[str] = []
        for filename_final in sorted(list(final_selected_files)):
            if filename_final in self.text_knowledge_dict:
                file_content_str = " ".join(self.text_knowledge_dict[filename_final])
                relevant_files_content_list.append(
                    f"--- Start Knowledge: {filename_final} ---\n{file_content_str}\n--- End Knowledge: {filename_final} ---")
        return "\n\n".join(relevant_files_content_list) if relevant_files_content_list else None

    def _retrieve_csv_context(self, user_csv_path: str) -> Optional[str]:
        # Uses self.csv_rag_enabled, self.csv_rag_data_dir
        if not self.csv_rag_enabled or not self.csv_rag_data_dir or not os.path.isdir(self.csv_rag_data_dir):
            return None
        user_data: Optional[pd.DataFrame] = None; x_user: Optional[pd.DataFrame] = None
        user_feature_cols: List[str] = []; user_other_cols: List[str] = []
        try:
            user_data = pd.read_csv(user_csv_path, header=0)
            user_feature_cols = [col for col in user_data.columns if str(col).replace('.', '', 1).isdigit()]
            if not user_feature_cols: return None
            x_user = user_data[user_feature_cols].astype(float)
            user_other_cols = [col for col in user_data.columns if col not in user_feature_cols]
        except Exception: return None # Simplified error handling for library
        if x_user is None or x_user.empty: return None

        similarities_list: List[float] = []; file_names_list: List[str] = []
        file_infos_dict: Dict[str, str] = {}; rag_file_paths_dict: Dict[str, str] = {}

        for rag_file_name in os.listdir(self.csv_rag_data_dir):
            if rag_file_name.endswith('.csv'):
                rag_file_path = os.path.join(self.csv_rag_data_dir, rag_file_name)
                try:
                    rag_data = pd.read_csv(rag_file_path, header=0)
                    rag_feature_cols = [col for col in rag_data.columns if str(col).replace('.', '', 1).isdigit()]
                    if not rag_feature_cols: continue
                    x_rag = rag_data[rag_feature_cols].astype(float)
                    if x_rag.shape[1] != x_user.shape[1] or x_rag.empty: continue
                    similarity_score = cosine_similarity(x_user, x_rag).mean()
                    if np.isnan(similarity_score): continue
                    similarities_list.append(similarity_score); file_names_list.append(rag_file_name)
                    rag_file_paths_dict[rag_file_name] = rag_file_path
                    rag_target_cols = [col for col in rag_data.columns if col not in rag_feature_cols]
                    file_infos_dict[rag_file_name] = (
                        f"Similar File Name: {rag_file_name}\nSimilar File Path: {rag_file_path}\nFirst row contains column names\n"
                        f"The target value for Similar File is the first {len(rag_target_cols)} columns: {rag_target_cols}\n"
                        f"Similar File Feature Column Count: {len(rag_feature_cols)}\nSimilar File Dimensions (rows, features): {x_rag.shape}"
                    )
                except Exception: pass # Simplified

        if not similarities_list: return None
        max_similarity_val = max(similarities_list)
        max_index = similarities_list.index(max_similarity_val)
        most_similar_file_name = file_names_list[max_index]
        most_similar_rag_file_info = file_infos_dict[most_similar_file_name]
        most_similar_rag_file_path = rag_file_paths_dict[most_similar_file_name]
        # print(f"Most similar RAG CSV: '{most_similar_file_name}' (Similarity: {max_similarity_val:.4f})") # Debug

        user_data_info_str = (
            f"--- User Data Information ---\nBased on the Information of user data, Please correctly read the user's data,Especially when the target value is None.:\n"
            f"User File Path: {user_csv_path}\nFirst row contains column names\nUser Data Feature Column Count: {len(user_feature_cols)}\n"
            f"The target value for user data is the first {len(user_other_cols)} columns: {user_other_cols}\nUser Data Dimensions (rows, features): {x_user.shape}\n"
        )
        rag_context_str = (
            f"--- Most Similar RAG File Information ---\nBased on feature similarity comparison with the user data above\, the following most relevant file was found in the knowledge base:\n"
            f"{most_similar_rag_file_info}\nSimilarity Score: {max_similarity_val:.4f}"
        )
        guidance_str = (
            f"Guidance: When generating code for tasks like modeling, please consider the user data (from '{user_csv_path}') as the prediction/test data, "
            f"and the most similar data file (from '{most_similar_rag_file_path}') as the training data, unless the user specifies otherwise."
        )
        return f"{user_data_info_str}\n\n{rag_context_str}\n\n{guidance_str}"


    def _stream_and_parse_llm_response(self, user_query_for_this_turn: str, rag_context_for_this_turn: Optional[str]) -> Tuple[str, str, str]:
        # Uses self.llm_interface, self.llm_backend_type, self.conversation_history,
        # self.llm_max_tokens, self.llm_temperature, self.llm_stop_tokens, self.interrupt_queue
        if self.llm_interface is None:
            print("ERROR: LLM interface not available for response generation!")
            return "[LLM_UNAVAILABLE_ERROR]", "", "[LLM_UNAVAILABLE_ERROR]"

        full_raw_llm_output = ""; accumulated_thinking_text = ""; accumulated_answer_text = ""
        prompt_input_with_rag = user_query_for_this_turn
        if rag_context_for_this_turn:
            prompt_input_with_rag = (
                f"Please use the following retrieved information to enhance your answer. If the information seems irrelevant, rely on your internal expertise.\n\n"
                f"Retrieved Information:\n{rag_context_for_this_turn}\n\nUser Query: {user_query_for_this_turn}"
            )
        # self.conversation_history should already contain the initial system prompt
        full_prompt_to_llm = self.conversation_history + \
                             f"<|im_start|>user\n{prompt_input_with_rag}<|im_end|>\n" + \
                             f"<|im_start|>assistant\n"

        print("\nAssistant: ", end="", flush=True)
        if rag_context_for_this_turn: print("(RAG Context Injected) ", end="", flush=True)

        live_buffer = ""; think_tag_open = "<think>"; think_tag_close = "</think>"; is_inside_think_block = False
        while not self.interrupt_queue.empty():
            try: self.interrupt_queue.get_nowait()
            except queue.Empty: break

        try:
            token_stream_iterator = None
            if self.llm_backend_type == "api":
                if not isinstance(self.llm_interface, LLMProvider): # Type check
                    raise TypeError("API backend requires an LLMProvider instance.")
                token_stream_iterator = self.llm_interface.generate_stream(
                    prompt=full_prompt_to_llm, max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature, stop_tokens=self.llm_stop_tokens
                )
            elif self.llm_backend_type == "local":
                if Llama is None or not isinstance(self.llm_interface, Llama): # Type check
                    raise TypeError("Local backend requires a Llama instance and llama_cpp library.")
                local_stream = self.llm_interface(
                    full_prompt_to_llm, max_tokens=self.llm_max_tokens,
                    temperature=self.llm_temperature, stop=self.llm_stop_tokens, stream=True
                )
                def local_stream_adapter(stream_gen): # Renamed 'stream' to 'stream_gen'
                    for chunk_item in stream_gen: # Renamed 'chunk' to 'chunk_item'
                        if not self.interrupt_queue.empty():
                            self.interrupt_queue.get()
                            yield {"type": "interrupt", "content": "User interrupted generation."}; return
                        if "choices" in chunk_item and chunk_item["choices"][0].get("text"):
                            yield {"type": "token", "content": chunk_item["choices"][0]["text"]}
                token_stream_iterator = local_stream_adapter(local_stream)
            else: raise ValueError(f"Unknown backend_type: {self.llm_backend_type}")

            for chunk_data in token_stream_iterator: # type: ignore # Iterator type is known by context
                if chunk_data["type"] == "interrupt": print("\n⚠️ User interrupted generation.", flush=True); break
                if chunk_data["type"] == "error":
                    error_content = chunk_data["content"]; print(error_content, end="", flush=True)
                    full_raw_llm_output += error_content; accumulated_answer_text += error_content; break
                token = chunk_data["content"]
                full_raw_llm_output += token; live_buffer += token
                can_process_buffer = True
                while can_process_buffer: # <think> tag parsing logic
                    can_process_buffer = False
                    if is_inside_think_block:
                        close_idx = live_buffer.find(think_tag_close)
                        if close_idx != -1:
                            think_segment = live_buffer[:close_idx]
                            if think_segment: print(think_segment, end="", flush=True); accumulated_thinking_text += think_segment
                            accumulated_thinking_text += think_tag_close
                            live_buffer = live_buffer[close_idx + len(think_tag_close):]
                            is_inside_think_block = False; can_process_buffer = True
                        else:
                            if live_buffer: print(live_buffer, end="", flush=True); accumulated_thinking_text += live_buffer; live_buffer = ""
                            break
                    else: # not is_inside_think_block
                        open_idx = live_buffer.find(think_tag_open)
                        if open_idx != -1:
                            answer_segment = live_buffer[:open_idx]
                            if answer_segment: print(answer_segment, end="", flush=True); accumulated_answer_text += answer_segment
                            accumulated_thinking_text += think_tag_open
                            live_buffer = live_buffer[open_idx + len(think_tag_open):]
                            is_inside_think_block = True; can_process_buffer = True
                        else:
                            if live_buffer: print(live_buffer, end="", flush=True); accumulated_answer_text += live_buffer; live_buffer = ""
                            break
            if live_buffer: # Process any remaining buffer content
                if is_inside_think_block: print(live_buffer, end="", flush=True); accumulated_thinking_text += live_buffer + " [Warning: LLM did not close think block]"
                else: print(live_buffer, end="", flush=True); accumulated_answer_text += live_buffer
            print() # Newline after full stream
        except Exception as e:
            error_msg = f"[SpeLL._stream_and_parse_llm_response error ({self.llm_backend_type} mode): {e}]"
            print(f"\n{error_msg}", flush=True)
            full_raw_llm_output += error_msg; accumulated_answer_text += error_msg
            import traceback; traceback.print_exc()
        return full_raw_llm_output, accumulated_thinking_text, accumulated_answer_text

    def _execute_code_safely(self, code: str) -> Tuple[bool, str]:
        # Uses self.code_execution_timeout. _execute_code_target is module-level.
        parent_conn, child_conn = Pipe()
        process = Process(target=_execute_code_target, args=(code, child_conn), daemon=True)
        print(f"\n--- Executing Code (Timeout: {self.code_execution_timeout}s) ---")
        start_time = time.time(); process.start()
        output_buffer = io.StringIO(); success = True; error_message: Optional[str] = None; timed_out = False
        while process.is_alive() or parent_conn.poll():
            if time.time() - start_time > self.code_execution_timeout:
                print("⚠️ Execution timed out. Terminating process..."); timed_out = True; success = False
                error_message = f"Code execution error: Timed out after {self.code_execution_timeout} seconds."
                try:
                    if process.is_alive(): process.kill()
                    process.join(0.5)
                except Exception: pass
                break
            if parent_conn.poll(0.1):
                try:
                    data = parent_conn.recv()
                    if data is None: break
                    if isinstance(data, str) and data.startswith("Execution failed:"):
                        success = False; error_message = data; print(data, end=""); output_buffer.write(data)
                    else: print(data, end=""); output_buffer.write(str(data))
                except EOFError: break
                except (OSError, ValueError) as pipe_err: success = False; error_message = f"Pipe reading error: {pipe_err}"; break
        if process.is_alive() and not timed_out:
            process.join(0.5)
            if process.is_alive():
                 try: process.kill(); process.join(0.5)
                 except Exception: pass
        print(f"\n--- Code Execution Finished (Duration: {time.time() - start_time:.2f}s) ---")
        try: parent_conn.close()
        except: pass
        try: child_conn.close()
        except: pass
        captured_output = output_buffer.getvalue(); output_buffer.close()
        if success:
            msg = "Code executed successfully." + (f" Captured output:\n---\n{captured_output.strip()}\n---" if captured_output.strip() else " No output captured.")
            return True, msg
        else:
            final_msg = error_message or "Code execution failed: Unknown error."
            if captured_output.strip() and (not error_message or not captured_output.strip().startswith("Execution failed:")):
                final_msg += f"\nStandard Output/Error before failure:\n---\n{captured_output.strip()}\n---"
            elif timed_out and not error_message: final_msg = f"Code execution error: Timed out after {self.code_execution_timeout} seconds."
            return False, final_msg


    def _extract_code(self, response: str) -> Optional[str]:
        if not isinstance(response, str): response = str(response)
        try: cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
        except Exception: cleaned_response = response
        pattern = r"```(?:python)?\s*\n?(.*?)\n?\s*```"
        matches = re.findall(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
        if matches and matches[0].strip(): return matches[0].strip()
        return None


    def _auto_fix_code(self,
                       initial_code: str,
                       conversation_log_up_to_error: str
                       ) -> Tuple[Optional[str], str, str]:
        """
        Attempts to execute code, and if it fails, asks the LLM to fix it.
        Allows interruption of the fix generation process via self.interrupt_queue.
        This version is adapted to work with both local Llama and API LLMProvider.
        """
        global interrupted_during_fix_generation
        if self.llm_interface is None:
            print("ERROR: LLM interface not available for auto-fix!")
            return initial_code, "[LLM_UNAVAILABLE_FOR_AUTOFIX]", conversation_log_up_to_error

        print("\n--- Auto-Fix Cycle Started ---")
        current_code: Optional[str] = initial_code
        # conversation_log_up_to_error already contains the user query,
        # the initial LLM response with the faulty code,
        # and the system message with the execution error.
        current_conversation_for_fix_cycle: str = conversation_log_up_to_error
        last_execution_result: str = "[Initial error logged in conversation_log_up_to_error]"

        # Allow initial try (which already failed to get here) + self.autofix_max_attempts for fixes
        # So, we iterate self.autofix_max_attempts times for fixing.
        for attempt_num in range(1, self.autofix_max_attempts + 1):  # Iterate for fix attempts
            interrupted_during_fix_generation = False
            print(f"\n--- Auto-Fix Attempt {attempt_num}/{self.autofix_max_attempts} ---")

            # Construct the fix request based on the previous error (which is already in current_conversation_for_fix_cycle)
            fix_request_for_system_prompt_content = (
                f"The previous code block (shown below) failed execution. The error message was provided in the preceding system message of our conversation.\n"
                f"Failed Code:\n```python\n{current_code}\n```\n\n"
                f"Please analyze the error and provide a corrected version of the full Python code block. "
                f"Ensure the corrected code is enclosed in ```python ... ```. Provide only the code block, without any other conversational text or explanations outside the code block itself."
            )
            # The prompt for the LLM to generate a fix.
            # It uses the conversation history leading up to this fix attempt,
            # then a system message detailing the fix request, then prompts the assistant.
            fix_prompt_to_llm = current_conversation_for_fix_cycle + \
                                f"<|im_start|>system\n{fix_request_for_system_prompt_content}<|im_end|>\n" + \
                                f"<|im_start|>assistant\n"

            # Generate the fix
            print("\nAssistant (Attempting Fix): ", end="", flush=True)
            raw_llm_fix_response = ""  # The raw text response from LLM for the fix

            try:
                # Clear any stale interrupts before starting
                while not self.interrupt_queue.empty():
                    try:
                        self.interrupt_queue.get_nowait()
                    except queue.Empty:
                        break

                if self.llm_backend_type == "local":
                    if Llama is None or not isinstance(self.llm_interface, Llama):
                        raise TypeError("Local backend requires a Llama instance for auto-fix.")

                    fix_stream = self.llm_interface(
                        fix_prompt_to_llm,
                        max_tokens=self.llm_max_tokens,  # Use instance config
                        temperature=max(0.1, self.llm_temperature - 0.2),  # Lower temp for fixes
                        stop=self.llm_stop_tokens,  # Use instance config
                        stream=True
                    )
                    for chunk in fix_stream:
                        if not self.interrupt_queue.empty():
                            self.interrupt_queue.get()  # Consume interrupt signal
                            interrupted_during_fix_generation = True
                            break
                        token = chunk["choices"][0]["text"]
                        print(token, end="", flush=True)
                        raw_llm_fix_response += token

                elif self.llm_backend_type == "api":
                    if not isinstance(self.llm_interface, LLMProvider):
                        raise TypeError("API backend requires an LLMProvider instance for auto-fix.")

                    # LLMProvider's generate_stream handles its internal interrupt check too
                    # but we also check Spell's queue for immediate break if needed.
                    api_fix_stream_generator = self.llm_interface.generate_stream(
                        prompt=fix_prompt_to_llm,  # LLMProvider will convert to messages
                        max_tokens=self.llm_max_tokens,
                        temperature=max(0.1, self.llm_temperature - 0.2),
                        stop_tokens=self.llm_stop_tokens
                    )
                    for chunk_data in api_fix_stream_generator:
                        if not self.interrupt_queue.empty():  # Check Spell's queue
                            self.interrupt_queue.get()
                            interrupted_during_fix_generation = True
                            break
                        if chunk_data["type"] == "interrupt":  # From LLMProvider's internal check
                            interrupted_during_fix_generation = True
                            break
                        if chunk_data["type"] == "error":
                            token = chunk_data["content"]
                            print(token, end="", flush=True)  # Display error
                            raw_llm_fix_response += token
                            # Potentially raise an exception or handle error more gracefully
                            raise RuntimeError(f"API error during fix generation: {token}")

                        token = chunk_data["content"]
                        print(token, end="", flush=True)
                        raw_llm_fix_response += token
                else:
                    raise ValueError(f"Unsupported llm_backend_type for auto_fix: {self.llm_backend_type}")

                print()  # Newline after fix generation stream finishes

                if interrupted_during_fix_generation:
                    print("\n⚠️ User interrupted fix generation.", flush=True)
                    # Do not add the partial response to the conversation for this attempt
                    # Break the for loop for attempts
                    break

                    # If generation completed normally, add the system request for fix and LLM's raw attempt to conversation
                # This is important for the *next* potential fix attempt's context.
                current_conversation_for_fix_cycle = fix_prompt_to_llm + f"{raw_llm_fix_response}<|im_end|>\n"

            except Exception as e_gen:
                print(f"\nError during fix generation (Attempt {attempt_num}): {e_gen}")
                # Add system message about generation error to conversation log
                current_conversation_for_fix_cycle += f"<|im_start|>system\nError during fix generation (Attempt {attempt_num}): {e_gen}<|im_end|>\n"
                # Return the code that was being fixed, its last known error, and the conversation up to this point
                return current_code, last_execution_result, current_conversation_for_fix_cycle

            # If generation was not interrupted, extract the new code from the raw_llm_fix_response
            if not interrupted_during_fix_generation:
                new_code = self._extract_code(raw_llm_fix_response)  # Extract from the raw response
                if not new_code:
                    print("\nFailed to extract corrected code from the LLM's response. Stopping auto-fix.")
                    # current_conversation_for_fix_cycle already includes the LLM's full (but unparseable for code) response
                    return current_code, last_execution_result, current_conversation_for_fix_cycle

                current_code = new_code  # Update current_code to the newly proposed code
                print("\nExtracted Corrected Code for Execution:")
                print(current_code)
            # If interrupted_during_fix_generation is True, we skip code extraction and execution for this attempt,
            # and the outer loop will break below.

            # Execute the current version of the code (which is now the LLM's proposed fix)
            # Skip execution if the fix generation itself was interrupted
            if not interrupted_during_fix_generation and current_code is not None:
                success, execution_result_str = self._execute_code_safely(current_code)
                print("\nExecution Result of Fix Attempt:")
                print(execution_result_str)

                # Add the execution result of this fix attempt to the conversation log
                current_conversation_for_fix_cycle += f"<|im_start|>system\nExecution Result of Fix Attempt {attempt_num}:\n```\n{execution_result_str}\n```\n<|im_end|>\n"

                if success:
                    print(f"\n--- Code Executed Successfully after Fix Attempt {attempt_num} ---")
                    return current_code, execution_result_str, current_conversation_for_fix_cycle
                else:
                    last_execution_result = execution_result_str  # Store the new error for the next loop or final return
                    # Loop continues if more attempts remain

            # If we were interrupted during fix generation, break the main attempts loop
            if interrupted_during_fix_generation:
                break

                # After the loop: either interrupted, or max attempts reached, or an early return happened.
        if interrupted_during_fix_generation:
            print("\n--- Auto-Fix Cycle Interrupted by User ---")
            # current_code is the last code that was attempted (or the one before the interrupted fix)
            # last_execution_result is the error from that code
            # current_conversation_for_fix_cycle includes interactions up to the point of interruption
            return current_code, last_execution_result, current_conversation_for_fix_cycle
        else:  # Max attempts reached and all failed
            print(f"\n--- Auto-Fix Failed after {self.autofix_max_attempts} attempts ---")
            # current_code is the last code LLM proposed, last_execution_result is its failure
            # current_conversation_for_fix_cycle includes all attempts and their failures
            return current_code, last_execution_result, current_conversation_for_fix_cycle

    # --- Public Methods for Interaction ---
    def chat(self):
        """Starts an interactive chat session with the SpeLL assistant."""
        if self.llm_interface is None:
            print("CRITICAL: LLM interface is not initialized. Chat cannot start.")
            print("Please check your configuration (backend type, model paths, API keys).")
            return

        print(f"\nSpeLL Chatbot ready (Backend: {self.llm_backend_type}). Type 'quit' to exit.")
        print("To submit multi-line input, paste it and press Enter on an empty line.")
        print("Press the ESC key to interrupt generation or auto-fix attempts.")
        if self.csv_rag_enabled:
            print(f"Hint: CSV RAG is enabled. Include a full CSV path in your query for data-aware help.")

        while True:
            try:
                print("\nYou (Enter on empty line to submit): ", end="", flush=True)
                lines = []
                while True:
                    try:
                        line_input = sys.stdin.readline().rstrip('\n') # Better for pasted input
                        if line_input.strip().lower() == 'quit': lines = ["quit"]; break
                        if line_input: lines.append(line_input)
                        else: break # Empty line submits
                    except EOFError: lines = ["quit"]; break
                user_input_str = "\n".join(lines)
                if user_input_str.lower() == 'quit': print("Exiting chat..."); break
                if not user_input_str.strip(): continue

                # Process one turn using the instance method
                # self.conversation_history is updated internally by process_user_turn
                self.process_user_turn(user_input_str)

            except KeyboardInterrupt: print("\nUser interrupt (Ctrl+C). Exiting chat..."); break
            except Exception as e:
                print(f"\nAn unexpected error occurred in the chat loop: {e}")
                import traceback; traceback.print_exc()

    def process_user_turn(self, user_input: str) -> Tuple[str, str]:
        """
        Processes a single turn of user input.
        Updates self.conversation_history.
        Returns (final_assistant_output_for_turn, raw_llm_response_this_turn)
        """
        if self.llm_interface is None:
            err_msg = "[LLM_INTERFACE_NOT_AVAILABLE_ERROR]"
            # Add this failed interaction to history so user sees the error
            self.conversation_history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{err_msg}<|im_end|>\n"
            return err_msg, err_msg # final_output, raw_response

        # --- RAG Context Retrieval ---
        final_rag_context_parts: List[str] = []
        user_specified_csv_path = self._extract_user_csv_path(user_input)

        if self.text_rag_enabled:
            text_retrieved_context = self._retrieve_text_context(user_input)
            if text_retrieved_context:
                final_rag_context_parts.append("--- Relevant Text Knowledge ---\n" + text_retrieved_context)
        if self.csv_rag_enabled and user_specified_csv_path:
            csv_retrieved_context = self._retrieve_csv_context(user_specified_csv_path)
            if csv_retrieved_context:
                final_rag_context_parts.append("--- Most Similar CSV File Info ---\n" + csv_retrieved_context)
        final_rag_context = "\n\n".join(final_rag_context_parts) if final_rag_context_parts else None

        # --- Generate LLM response (handles streaming and <think> parsing) ---
        # _stream_and_parse_llm_response uses self.conversation_history for context
        raw_llm_response_this_turn, _, llm_answer_text_this_turn = self._stream_and_parse_llm_response(
            user_query_for_this_turn=user_input,
            rag_context_for_this_turn=final_rag_context
        )
        # Append user input and raw LLM response to history for the NEXT turn's context
        # and for auto-fix's context if needed this turn.
        current_turn_interaction_log = f"<|im_start|>user\n{user_input}<|im_end|>\n" + \
                                       f"<|im_start|>assistant\n{raw_llm_response_this_turn}<|im_end|>\n"
        # This history *including the current raw response* is what auto_fix needs if code fails
        history_for_autofix_if_needed = self.conversation_history + current_turn_interaction_log


        # --- Extract and potentially execute/fix code ---
        code_to_execute = self._extract_code(llm_answer_text_this_turn) # Extract from answer part
        final_assistant_output_str = llm_answer_text_this_turn # Default if no code

        if code_to_execute:
            print("\n--- Executing Code from Initial LLM Response ---")
            initial_success, initial_exec_result_str = self._execute_code_safely(code_to_execute)
            print("\nInitial Execution Result:"); print(initial_exec_result_str)

            # Log the result of the initial execution
            history_for_autofix_if_needed += f"<|im_start|>system\nExecution Result of Initial Code:\n```\n{initial_exec_result_str}\n```\n<|im_end|>\n"

            if initial_success:
                self.conversation_history = history_for_autofix_if_needed # Update main history
                final_assistant_output_str = f"{llm_answer_text_this_turn}\n\n--- Initial Code Execution Result ---\n{initial_exec_result_str}"
            else:
                # Code failed, attempt auto-fix
                # Pass the history that *includes* the failed code's response and its execution result
                final_code_after_fix, final_fix_result_str, updated_conv_log_from_autofix_cycle = \
                    self._auto_fix_code(
                        initial_code=code_to_execute, # Pass the code that just failed
                        conversation_log_up_to_error=history_for_autofix_if_needed
                    )
                self.conversation_history = updated_conv_log_from_autofix_cycle # Autofix returns the complete log for the whole session
                final_assistant_output_str = (f"{llm_answer_text_this_turn}\n\n--- Initial Code Failed ---\n{initial_exec_result_str}\n\n"
                                              f"--- Auto-Fix Attempt Result ---\nCode:\n```python\n{final_code_after_fix or 'No code generated'}\n```\nResult:\n{final_fix_result_str}")
        else: # No code found
            self.conversation_history = history_for_autofix_if_needed # Update with this turn's non-code interaction
            # final_assistant_output_str is already llm_answer_text_this_turn

        return final_assistant_output_str, raw_llm_response_this_turn


    def set_llm_model(self, model_path: str, n_gpu_layers: Optional[int] = None,
                      n_ctx: Optional[int] = None, verbose: Optional[bool] = None,
                      seed_val: Optional[int] = None):
        """Allows changing the local LLM model and its parameters after Spell instantiation."""
        if self.llm_backend_type != "local":
            print("Error: set_llm_model is only applicable for 'local' LLM backend.")
            return False
        if not LLAMA_CPP_AVAILABLE:
            print("Cannot set LLM model: llama_cpp library not available.")
            return False
        if not model_path or not os.path.exists(model_path):
            print(f"Error: LLM model path '{model_path}' not provided or does not exist.")
            return False

        self.local_llm_model_path = model_path
        if n_gpu_layers is not None: self.local_n_gpu_layers = n_gpu_layers
        if n_ctx is not None: self.local_n_ctx = n_ctx
        if verbose is not None: self.local_llm_verbose = verbose
        if seed_val is not None: self.seed = seed_val; np.random.seed(self.seed) # Update global seed too

        try:
            print(f"Re-loading local LLM from: {self.local_llm_model_path}...")
            if Llama is None: raise RuntimeError("Llama class is None.")
            self.llm_interface = Llama(
                model_path=self.local_llm_model_path,
                n_gpu_layers=self.local_n_gpu_layers,
                n_ctx=self.local_n_ctx,
                verbose=self.local_llm_verbose,
                seed=self.seed
            )
            print("Local LLM re-loaded successfully.")
            return True
        except Exception as e:
            print(f"Error: Failed to re-load local LLM model: {e}")
            self.llm_interface = None
            return False

    def shutdown(self):
        print("\nShutting down SpeLL...")
        if self.interrupt_thread and self.interrupt_thread.is_alive():
            self.interrupt_listener_stop_event.set()
            self.interrupt_thread.join(timeout=1.0)
            if self.interrupt_thread.is_alive():
                print("Warning: Listener thread did not stop gracefully.")
        print("SpeLL shutdown complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# --- LLMProvider Class for API Interaction (Needs to be defined before Spell if used) ---
class LLMProvider:
    def __init__(self, api_key: str, api_model_id: str, api_base_url: str, site_url: str, app_name: str):
        self.api_key = api_key
        self.api_model_id = api_model_id
        self.api_base_url = api_base_url
        self.site_url = site_url
        self.app_name = app_name
        self.api_client: Optional[OpenAIClient] = None # Type hint
        self._initialize_client()

    def _initialize_client(self):
        if not OPENAI_CLIENT_AVAILABLE or OpenAIClient is None:
            print("ERROR: OpenAI library not available for LLMProvider.")
            raise ImportError("OpenAI library not found, cannot initialize LLMProvider.")
        if not self.api_key: raise ValueError("API key not provided to LLMProvider.") # Should be caught earlier
        print(f"Initializing API LLM client for model: {self.api_model_id} at {self.api_base_url}")
        try:
            self.api_client = OpenAIClient(api_key=self.api_key, base_url=self.api_base_url)
            # Optional: Test connection, e.g., by listing models if API supports and not too slow
            # self.api_client.models.list()
            print("API LLM client initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize OpenAIClient: {e}")
            raise # Re-throw to be caught by Spell._initialize_llm_interface

    def _convert_prompt_to_api_messages(self, full_prompt_text: str) -> List[Dict[str, str]]:
        messages: List[Dict[str,str]] = []
        # Regex to find role and content blocks
        pattern = r"(?s)<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|<\|im_start\|>assistant\n?\Z|\Z)"
        for match in re.finditer(pattern, full_prompt_text):
            role = match.group(1).lower()
            content = match.group(2).strip()
            # Remove trailing assistant prompt from content if it's there
            if content.endswith("<|im_start|>assistant\n"):
                content = content[:-len("<|im_start|>assistant\n")].strip()
            elif content.endswith("<|im_start|>assistant"): # No newline
                 content = content[:-len("<|im_start|>assistant")].strip()

            if role in ["system", "user", "assistant"]:
                if role == "assistant" and not content: # Skip empty assistant message from prompt end
                    continue
                messages.append({"role": role, "content": content})
        if not messages:
             print(f"Warning: Parsed empty message list from prompt. Full prompt (start): '{full_prompt_text[:300]}...'")
             # Fallback: treat whole prompt as a single user message (loses history for API)
             return [{"role": "user", "content": "Error: Could not parse conversation history. Original prompt: " + full_prompt_text}]

        return messages

    def generate_stream(self, prompt: str, max_tokens: int, temperature: float, stop_tokens: List[str]) -> Generator[Dict[str, Any], None, None]:

        if self.api_client is None:
            yield {"type": "error", "content": "[Error: LLMProvider API client not initialized.]"}; return

        api_messages = self._convert_prompt_to_api_messages(prompt)
        if not api_messages: # Should have a fallback, but defensive check
             yield {"type": "error", "content": "[Error: Could not prepare messages for API.]"}; return

        try:
            api_stream_iterator = self.api_client.chat.completions.create(
                model=self.api_model_id,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_tokens if stop_tokens else None, # API expects None or list of str
                stream=True,
                extra_headers={ # For OpenRouter
                    "HTTP-Referer": self.site_url,
                    "X-Title": self.app_name,
                }
            )
            for chunk in api_stream_iterator:
                # Interrupt check can also be done here if Spell's queue is accessible
                # For now, assuming Spell's wrapper handles it.
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield {"type": "token", "content": token}
                # elif chunk.choices and chunk.choices[0].finish_reason:
                #     # print(f"API stream finished. Reason: {chunk.choices[0].finish_reason}") # Debug
                #     pass
        except Exception as e:
            error_message = f"[LLMProvider API Generation Error: {str(e)}]"
            print(f"\n{error_message}") # Log to console
            yield {"type": "error", "content": error_message} # Send error as a "token"
            return


