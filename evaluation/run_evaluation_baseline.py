# rag_script.py

# Catatan: Perintah pip berikut berasal dari file asli.
# Saat menjalankan sebagai skrip .py, Anda biasanya akan menginstal dependensi ini
# sekali dari baris perintah atau melalui file requirements.txt.
#
# %pip install -U bitsandbytes
# %pip install -U transformers accelerate
# %pip install -U sentence-transformers
# %pip install -U cross-encoder
# %pip install -U faiss-gpu-cu12
# %pip install -U openai

import os
import json
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import faiss
import pickle
from kaggle_secrets import UserSecretsClient
from tqdm import tqdm

# --- Mengatur direktori cache untuk pustaka Hugging Face ---
cache_directory = "/kaggle/working/huggingface_cache"
os.environ['HF_HOME'] = cache_directory
os.environ['TRANSFORMERS_CACHE'] = cache_directory
os.makedirs(cache_directory, exist_ok=True)
print(f"Direktori cache Hugging Face telah diatur ke: {cache_directory}")

# --- Memastikan direktori hasil Kaggle ada ---
os.makedirs("/kaggle/working/results", exist_ok=True)

# --- Konfigurasi sistem (DIUBAH UNTUK KAGGLE) ---
KAGGLE_INPUT_DIR = "/kaggle/input/rag-data" # <--- Ganti nama dataset Anda di sini

CONFIG = {
    # --- Konfigurasi model API (untuk generator dan penulis ulang) ---
    "llm_api_model_name": "qwen2.5-7b-instruct",
    "llm_api_key_env": "DASHSCOPE_API_KEY",
    "llm_api_base_url_env": "DASHSCOPE_BASE_URL",

    # --- Jalur model lokal (DIUBAH: Planner menggunakan model lokal) ---
    "planner_model_path": "/kaggle/input/qwen2-5-7b-instruct/qwen2.5-7b-instruct",
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-large",

    # --- File data & hasil (DIUBAH: Nama file output telah diubah) ---
    "hotpotqa_dev_path": f"{KAGGLE_INPUT_DIR}/hotpot_train_v1.1.json",
    "pkl_database_path": f"{KAGGLE_INPUT_DIR}/hotpot_contexts_deduplicated.pkl",
    "faiss_index_path": f"{KAGGLE_INPUT_DIR}/hotpotqa_faiss_deduplicated.index",
    "results_output_path": "/kaggle/working/results/rag_evaluation_results.json", # <-- Diubah: File output untuk evaluasi

    # --- Hiperparameter alur RAG ---
    "retriever_top_k": 5,
    "reranker_top_n": 2,
    
    # --- Konfigurasi perangkat ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # --- Konfigurasi cache ---
    "cache_dir": cache_directory,
}

# ==============================================================================
# 2. Komponen RAG
# ==============================================================================

# Kelas layanan LLM (untuk panggilan API, Generator dan Rewriter masih menggunakan)
class LLMService:
    def __init__(self, config):
        self.client = None
        self.model_name = config["llm_api_model_name"]
        api_key_env = config["llm_api_key_env"]
        base_url_env = config["llm_api_base_url_env"]
        
        print(f"--- Menginisialisasi layanan API model bahasa ({self.model_name}) ---")
        try:
            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret(api_key_env)
            base_url = user_secrets.get_secret(base_url_env)
        except Exception as e:
            print(f"Kesalahan fatal: Tidak dapat mengambil konfigurasi API dari Kaggle Secrets: {e}")
            api_key, base_url = None, None

        if api_key and base_url:
            try:
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                print("--- Klien API OpenAI telah berhasil dibuat. ---")
            except Exception as e:
                print(f"Kesalahan fatal: Gagal menginisialisasi klien OpenAI: {e}")
                self.client = None
        else:
            print(f"Peringatan: Gagal mendapatkan '{api_key_env}' atau '{base_url_env}' dari Kaggle Secrets. Layanan API tidak akan tersedia.")

    def call_api(self, messages: list, max_tokens: int, temperature: float, top_p: float) -> str:
        if not self.client: return "Kesalahan: Klien API belum diinisialisasi."
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Kesalahan panggilan API: {e}")
            return f"Terjadi kesalahan saat panggilan API: {e}"

# Planner (PERUBAHAN BESAR: Beralih ke Model Lokal)
class Planner:
    def __init__(self, config):
        print(f"--- Menginisialisasi perencana kueri (Planner - model lokal: {config['planner_model_path']}) ---")
        self.device = config["device"]
        self.model_path = config["planner_model_path"]
        self.cache_dir = config["cache_dir"]

        # Menggunakan BitsAndBytes untuk konfigurasi kuantisasi 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                cache_dir=self.cache_dir,
                use_fast=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16, # Pastikan tipe data cocok
                device_map="auto",
                cache_dir=self.cache_dir
            )
            print(f"--- Model perencana lokal '{self.model_path}' berhasil dimuat ---")
        except Exception as e:
            print(f"Kesalahan fatal: Gagal memuat model Planner lokal: {e}")
            self.tokenizer = None
            self.model = None

    def _parse_plan(self, response_text: str):
        try:
            json_start, json_end = response_text.find('['), response_text.rfind(']')
            if json_start != -1 and json_end != -1:
                json_part = response_text[json_start : json_end + 1]
                sub_questions = json.loads(json_part)
                if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions) and sub_questions:
                    return sub_questions
            raise ValueError("Array JSON yang valid tidak ditemukan dalam respons model.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gagal mengurai rencana: {e}. Teks respons lengkap: '{response_text}'")
            return None

    def generate_plan(self, question: str, max_new_tokens: int = 256) -> list[str]:
        if not self.model or not self.tokenizer:
            print("Kesalahan: Model Planner tidak dimuat dengan benar, mengembalikan pertanyaan asli.")
            return [question]

        prompt_template = f"""You are a world-class planning expert. Your job is to break down a complex question into a series of simple, sequential sub-questions. The answer to a previous sub-question can help answer the next. The plan you formulate must be answerable based on the provided encyclopedic text.Do not make any assumptions about the form of the content (e.g., do not assume the existence of lyrics, comments, or conversations).

Respond with ONLY a valid JSON array of strings, where each string is a sub-question.

**Example 1: Simple Comparison**
Complex Question: "Were Scott Derrickson and Ed Wood of the same nationality?"
Output:
[
  "What is the nationality of Scott Derrickson?",
  "What is the nationality of Ed Wood?"
]

**Example 2: Intersection/Constraint Query**
Complex Question: "Gunmen from Laredo starred which narrator of 'Frontier'?"
Output:
[
  "Who was the narrator of the series 'Frontier'?",
  "Who were the main actors in the movie 'Gunmen from Laredo'?",
  "Did the narrator of 'Frontier', Walter Darwin Coy, also star in 'Gunmen from Laredo'?"
]

Now, generate a plan for the following question.
Complex Question: "{question}"
Output:
"""
        messages = [{"role": "user", "content": prompt_template}]
        
        # Menggunakan apply_chat_template untuk memformat input
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Menggunakan generasi deterministik untuk mendapatkan rencana yang konsisten
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        parsed_plan = self._parse_plan(response_text)
        return parsed_plan if parsed_plan else [question]

# QueryRewriter (Menggunakan API LLMService)
class QueryRewriter:
    def __init__(self, llm_service: LLMService):
        print("--- Menginisialisasi penulis ulang kueri (QueryRewriter) ---")
        self.llm_service = llm_service
        if self.llm_service.client: print("--- Penulis ulang kueri telah terhubung ke layanan API ---")
        else: print("--- Peringatan: Penulis ulang kueri gagal terhubung ke layanan API ---")

    def rewrite(self, history: str, sub_q: str) -> str:
        if not self.llm_service.client or not history.strip(): return sub_q
        
        system_prompt = "You are an expert query rewriter. Your task is to reformulate a follow-up question into a self-contained query based on a conversation history. Output MUST be ONLY the rewritten query."
        user_prompt = f"[Conversation History]\n{history}\n[Follow-up Question]\n{sub_q}\n[Rewritten Query]"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        try:
            rewritten_query = self.llm_service.call_api(messages=messages, max_tokens=100, temperature=0.0, top_p=1.0)
            return rewritten_query.strip('"') or sub_q
        except Exception as e:
            print(f"    - Peringatan: Penulisan ulang kueri API gagal: {e}. Menggunakan kueri asli.")
            return sub_q

# ContextRetriever (Lokal)
class ContextRetriever:
    def __init__(self, config, corpus: list[str]):
        print("--- Menginisialisasi pengambil konteks (ContextRetriever) ---")
        self.device = config["device"]
        self.corpus = corpus
        self.index_path = config["faiss_index_path"]
        self.model = SentenceTransformer(config["embedding_model"], device=self.device, cache_folder=config["cache_dir"])
        if os.path.exists(self.index_path):
            print(f"--- Memuat indeks FAISS: '{self.index_path}' ---")
            self.index = faiss.read_index(self.index_path)
            print(f"--- Indeks berhasil dimuat, berisi {self.index.ntotal} dokumen. ---")
        else:
            self.index = None
            print(f"--- Peringatan: File indeks FAISS tidak ditemukan di '{self.index_path}'. ---")

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.index: return []
        query_embedding = self.model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        _, I = self.index.search(query_embedding.cpu().numpy(), k)
        return [self.corpus[i] for i in I[0]]

# Reranker (Lokal)
class Reranker:
    def __init__(self, config):
        print("--- Menginisialisasi pengurut ulang (Reranker) ---")
        self.model = CrossEncoder(config["reranker_model"], device=config["device"], max_length=512, cache_folder=config["cache_dir"])
        print("--- Inisialisasi pengurut ulang berhasil ---")

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        if not docs: return []
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]

# Generator (Menggunakan API LLMService)
class Generator:
    def __init__(self, llm_service: LLMService):
        print(f"--- Menginisialisasi generator (Generator via API: {llm_service.model_name}) ---")
        self.llm_service = llm_service
        if self.llm_service.client: print("--- Generator telah terhubung ke layanan API ---")
        else: print("--- Peringatan: Generator gagal terhubung ke layanan API ---")

    def _generate_with_api(self, messages: list) -> str:
        if not self.llm_service.client: return "Kesalahan: Layanan API LLM belum diinisialisasi."
        return self.llm_service.call_api(messages=messages, max_tokens=512, temperature=0.1, top_p=0.9)

    def generate(self, question: str, final_context: list[str]) -> str:
        if not final_context: return "Informasi tidak cukup, tidak dapat menjawab."
        context_str = "\n\n".join([f"Potongan relevan {i+1}:\n{doc}" for i, doc in enumerate(final_context)])
        system_prompt = "You are a meticulous AI assistant. Your task is to answer a question based *only* on the provided context. Respond only with the answer itself, in English. YOU MUST NOT USE YOUR OWN KNOWLEDGE."
        user_prompt = f"--- Context Start ---\n{context_str}\n--- Context End ---\n\nQuestion: {question}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self._generate_with_api(messages)

    def generate_final_synthesis(self, original_question: str, intermediate_steps: list) -> str:
        synthesis_prompt = "You are a professional answer synthesizer. Your task is to combine the given sub-question answers to form a final, coherent answer for the original, complex question. Base your final answer strictly on the provided sub-answers. If info was not available, reflect this limitation. Synthesize the facts into a direct answer.\n\n--- Sub-Questions and Answers Review ---\n"
        for i, step in enumerate(intermediate_steps):
            synthesis_prompt += f"{i+1}. Sub-Question: {step['sub_question']}\n   Answer: {step['sub_answer']}\n\n"
        synthesis_prompt += f"--- Final Task ---\nNow, please answer this initial question in English based on the above information: \"{original_question}\""
        messages = [{"role": "system", "content": "You are a professional answer integration expert."}, {"role": "user", "content": synthesis_prompt}]
        return self._generate_with_api(messages)


# ==============================================================================
# 2. Alur Evaluasi RAG
# ==============================================================================

def execute_full_rag_pipeline(question, plan, retriever, reranker, generator, rewriter, config):
    """
    Menjalankan alur RAG lengkap dan menangkap semua langkah perantara untuk evaluasi.
    """
    q_and_a_history = ""
    intermediate_steps_data = []
    
    for sub_q in plan:
        # 1. Tulis ulang kueri
        query_for_retriever = rewriter.rewrite(q_and_a_history, sub_q)
        
        # 2. Ambil
        retrieved_docs = retriever.retrieve(query_for_retriever, k=config["retriever_top_k"])
        
        # 3. Urutkan ulang
        reranked_docs = reranker.rerank(query_for_retriever, retrieved_docs)
        
        # 4. Siapkan konteks pembuatan
        context_for_sub_q = reranked_docs[:config["reranker_top_n"]]
        
        final_context_for_gen = []
        if q_and_a_history:
            history_context = f"Confirmed Background Knowledge (from previous steps):\n{q_and_a_history.strip()}"
            final_context_for_gen.append(history_context)
        final_context_for_gen.extend(context_for_sub_q)
        
        # 5. Hasilkan sub-jawaban
        sub_answer = generator.generate(sub_q, final_context_for_gen)
        
        # 6. Catat detail langkah perantara
        intermediate_steps_data.append({
            "sub_question": sub_q,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "sub_answer": sub_answer
        })
        
        # 7. Perbarui riwayat
        if "tidak dapat menjawab" not in sub_answer and "informasi tidak cukup" not in sub_answer and "error" not in sub_answer.lower():
            q_and_a_history += f"Q: {sub_q}\nA: {sub_answer}\n"
            
    # 8. Sintesiskan jawaban akhir
    if len(plan) > 1:
        final_answer = generator.generate_final_synthesis(question, intermediate_steps_data)
    elif intermediate_steps_data:
        final_answer = intermediate_steps_data[0]['sub_answer']
    else:
        final_answer = "Gagal menghasilkan jawaban."
        
    return {
        "final_generated_answer": final_answer,
        "intermediate_steps": intermediate_steps_data
    }


# ==============================================================================
# 3. Alur Eksekusi Utama
# ==============================================================================
def main():
    # --- 1. Memuat data dan layanan inti ---
    print("\n--- Memuat data dan menginisialisasi semua komponen sistem RAG ---")

    print(f"Memuat data dari jalur lokal: {CONFIG['pkl_database_path']}")
    with open(CONFIG["pkl_database_path"], 'rb') as f:
        corpus_data = pickle.load(f)
    corpus = [item['context'] for item in corpus_data]

    with open(CONFIG["hotpotqa_dev_path"], 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Inisialisasi layanan API LLM (untuk Generator dan Rewriter)
    print("\n--- Menginisialisasi layanan API model bahasa... ---")
    llm_service = LLMService(CONFIG)
    if not llm_service.client:
        print("\n\nPeringatan: Inisialisasi layanan API model bahasa gagal. Fungsi Generator dan Rewriter akan terbatas.\n")
    
    # Inisialisasi semua komponen (Planner sekarang menggunakan model lokal)
    planner = Planner(CONFIG)
    rewriter = QueryRewriter(llm_service)
    generator = Generator(llm_service)
    retriever = ContextRetriever(CONFIG, corpus=corpus)
    reranker = Reranker(CONFIG)

    # Periksa apakah model Planner berhasil dimuat
    if not planner.model:
        print("\n\nKesalahan: Model lokal Planner gagal dimuat. Program akan keluar.\n")
        return

    print("\n--- Semua komponen berhasil diinisialisasi ---\n")

    # --- 2. Perulangan untuk menjalankan alur RAG dan menghasilkan hasil evaluasi ---
    print(f"\n{'='*25} Mulai menghasilkan hasil evaluasi RAG {'='*25}\n")

    evaluation_results = []
    
    # Catatan: Untuk demonstrasi, hanya sejumlah kecil data yang diproses di sini. Anda dapat menyesuaikan rentang irisan.
    for item in tqdm(dataset[100:250], desc="Menghasilkan hasil evaluasi RAG"):
        question = item["question"]
        ground_truth_answer = item["answer"]
        item_id = item["_id"]

        # 1. Hasilkan rencana kueri menggunakan model lokal
        plan = planner.generate_plan(question)
        
        # 2. Jalankan alur RAG lengkap
        rag_output = execute_full_rag_pipeline(
            question, plan, retriever, reranker, generator, rewriter, CONFIG
        )

        # 3. Format hasil agar sesuai dengan struktur JSON yang ditentukan
        result_entry = {
            "id": item_id,
            "original_question": question,
            "ground_truth_answer": ground_truth_answer,
            "planner_plan": plan,
            "intermediate_steps": rag_output["intermediate_steps"],
            "final_generated_answer": rag_output["final_generated_answer"]
        }
        evaluation_results.append(result_entry)

    # --- 3. Menyimpan hasil evaluasi akhir ---
    output_path = CONFIG["results_output_path"]
    print(f"\n--- Menyimpan {len(evaluation_results)} hasil evaluasi ke file: {output_path} ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=4)
        print(f"--- Hasil evaluasi RAG telah berhasil disimpan ke {output_path} ---")
    except Exception as e:
        print(f"--- Terjadi kesalahan saat menyimpan hasil: {e} ---")

if __name__ == '__main__':
    main()