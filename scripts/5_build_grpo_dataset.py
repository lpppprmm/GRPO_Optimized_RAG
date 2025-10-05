# build_grpo_dataset.py

# Catatan: Perintah pip berikut berasal dari sel pertama notebook.
# Saat menjalankan sebagai skrip .py, Anda biasanya akan menginstal dependensi ini
# sekali dari baris perintah atau melalui file requirements.txt.
#
# pip install -U bitsandbytes
# pip install -U transformers accelerate
# pip install -U faiss-gpu-cu12

import os
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm

# --- 1. Konfigurasi Lingkungan & Jalur ---

# Mengatur direktori cache untuk pustaka Hugging Face di lingkungan Kaggle
cache_directory = "/kaggle/working/huggingface_cache"
os.environ['HF_HOME'] = cache_directory
os.environ['TRANSFORMERS_CACHE'] = cache_directory
os.makedirs(cache_directory, exist_ok=True)
print(f"Direktori cache Hugging Face telah diatur ke: {cache_directory}")

# Memastikan direktori output yang dapat ditulisi di Kaggle ada
os.makedirs("/kaggle/working/results", exist_ok=True)

# Mendefinisikan jalur untuk direktori input Kaggle
KAGGLE_HF_MODELS_DIR = "/kaggle/input/huggingface-models"
KAGGLE_DATA_DIR = "/kaggle/input/rag-data"
BGE_M3_HASH = "5617a9f61b028005a4858fdac845db406aefb181"  # <--- !!! Silakan isi nilai hash di sini
RERANKER_HASH = "55611d7bca2a7133960a6d3b71e083071bbfc312" # <--- !!! Silakan isi nilai hash di sini

# Kamus konfigurasi global
CONFIG = {
    # --- Konfigurasi Jalur Model Terpadu ---
    # Jalur model dasar (model sebelum fine-tuning)
    "base_model_path": f"/kaggle/input/qwen2-5-7b-instruct/qwen2.5-7b-instruct",
    # Jalur adaptor untuk Planner (output fine-tuning Llama Factory)
    "adapter_path": "/kaggle/input/qwen2-dpo-output-full-archive/qwen2_dpo_output", # <--- !!! Silakan ganti dengan jalur adaptor Anda sendiri

    # --- Konfigurasi Lainnya Tetap Sama ---
    "embedding_model": f"/kaggle/input/bge-model/models--BAAI--bge-m3/snapshots/{BGE_M3_HASH}",
    "reranker_model": f"/kaggle/input/bge-model/models--BAAI--bge-reranker-large/snapshots/{RERANKER_HASH}",

    "hotpotqa_dev_path": f"{KAGGLE_DATA_DIR}/hotpot_train_v1.1.json",
    "pkl_database_path": f"{KAGGLE_DATA_DIR}/hotpot_contexts_deduplicated.pkl",
    "faiss_index_path": f"{KAGGLE_DATA_DIR}/hotpotqa_faiss_deduplicated.index",
    "results_output_path": "/kaggle/working/results/GRPO_11800_12000.json",
    "retriever_top_k": 5,
    "reranker_top_n": 2,
    "min_preference_gap": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cache_dir": cache_directory,
}


# ==============================================================================
# 2. Pengelola Model Lokal Terpadu (LocalModelHandler)
# ==============================================================================
class LocalModelHandler:
    """
    Kelas terpadu untuk memuat model dasar besar dan adaptor PEFT, serta menyediakan semua fungsi inferensi yang diperlukan:
    1.  Pembuatan teks (perilaku dapat dibedakan dengan mengaktifkan/menonaktifkan adaptor)
    2.  Perhitungan skor reward (biasanya menggunakan model dasar)
    """
    def __init__(self, base_model_path: str, adapter_path: str, device: str, cache_dir: str): # <<< DIUBAH: Parameter diubah
        print(f"--- Menginisialisasi pengelola model lokal terpadu ---")
        self.device = device
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # <<< MULAI DIUBAH: Muat model dasar terlebih dahulu, lalu muat adaptor
        print(f"--- Memuat LLM dasar dari '{base_model_path}' (kuantisasi 4-bit)... ---")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, cache_dir=cache_dir)
        print(f"--- LLM dasar '{base_model_path}' dan tokenizer berhasil dimuat ---")

        print(f"--- Memuat dan menerapkan adaptor PEFT dari '{adapter_path}'... ---")
        # Muat adaptor ke model dasar
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"--- Adaptor PEFT '{adapter_path}' berhasil dimuat ---")
        # <<< AKHIR DIUBAH

    # <<< METODE BARU: Menambahkan metode untuk mengontrol status adaptor
    def set_adapter_state(self, enabled: bool):
        """
        Mengaktifkan atau menonaktifkan adaptor PEFT yang telah dimuat.
        """
        if enabled:
            self.model.enable_adapter_layers()
        else:
            self.model.disable_adapter_layers()

    def apply_chat_template(self, messages: list) -> str:
        """Menerapkan templat obrolan model"""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt_str: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
        """Fungsi pembuatan teks umum"""
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        generation_output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response_text = self.tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response_text.strip()

    def calculate_reward_score(self, context_str: str, question: str, ground_truth_answer: str) -> float:
        """
        Menghitung fungsi reward. Fungsi ini secara default harus menggunakan model dasar (adaptor dinonaktifkan).
        """
        # <<< DIUBAH: Nonaktifkan adaptor sebelum perhitungan, pulihkan setelahnya untuk memastikan keamanan status
        try:
            self.set_adapter_state(enabled=False)
            # 1. Bangun prompt lengkap untuk menghasilkan jawaban
            system_prompt = "You are a meticulous AI assistant. Your task is to answer a question based *only* on the provided context."
            user_prompt = f"--- Context Start ---\\n{context_str}\\n--- Context End ---\\n\\nQuestion: {question}"
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            prompt_str = self.apply_chat_template(messages)
            
            # 2. Langkah-langkah tokenisasi...
            prompt_tokens = self.tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False).to(self.device)
            answer_tokens = self.tokenizer.encode(ground_truth_answer, return_tensors="pt", add_special_tokens=False).to(self.device)
            if answer_tokens.shape[1] == 0: return -1e9
            input_ids = torch.cat([prompt_tokens, answer_tokens], dim=-1)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits
            
            answer_logits = logits[:, prompt_tokens.shape[1]-1:-1, :]
            log_probs = F.log_softmax(answer_logits, dim=-1)
            target_log_probs = log_probs.gather(2, answer_tokens.unsqueeze(-1)).squeeze(-1)
            total_reward = target_log_probs.sum().item()
            
            is_invalid = torch.isinf(torch.tensor(total_reward)) or torch.isnan(torch.tensor(total_reward))
            return total_reward if not is_invalid else -1e9
        finally:
            # Aktifkan kembali adaptor baik berhasil maupun gagal, untuk menghindari memengaruhi Planner
            self.set_adapter_state(enabled=True)

# ==============================================================================
# 3. Definisi Komponen RAG (Semua menggunakan model lokal)
# ==============================================================================

class Planner:
    def __init__(self, local_model_handler: LocalModelHandler):
        print("--- Menginisialisasi perencana kueri (Planner - menggunakan model lokal) ---")
        self.local_model = local_model_handler

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

    def generate_plan_with_sampling(self, question: str, top_p: float = 0.9, temperature: float = 0.8, max_new_tokens: int = 256) -> list[str]:
        # --- PERBAIKAN: Memperbaiki sintaks f-string dari f\"\"\" menjadi f""" ---
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
        prompt_str = self.local_model.apply_chat_template(messages)
        # <<< DIUBAH: Aktifkan adaptor sebelum memanggil, pulihkan ke default (nonaktif) setelahnya
        try:
            self.local_model.set_adapter_state(enabled=True) # Aktifkan adaptor fine-tuned untuk Planner
            print("Planner: Adaptor diaktifkan.")
            response_text = self.local_model.generate(prompt_str, max_new_tokens, temperature, top_p, do_sample=True)
        finally:
            self.local_model.set_adapter_state(enabled=False) # Kembalikan ke status model dasar
            print("Planner: Adaptor dinonaktifkan.")
            
        parsed_plan = self._parse_plan(response_text)
        return parsed_plan if parsed_plan else [question]


class QueryRewriter:
    def __init__(self, local_model_handler: LocalModelHandler):
        print("--- Menginisialisasi penulis ulang kueri (QueryRewriter - menggunakan model lokal) ---")
        self.local_model = local_model_handler

    def rewrite(self, history: str, sub_q: str) -> str:
        if not history.strip():
            return sub_q
        
        system_prompt = """You are an expert query rewriter. Your task is to reformulate a follow-up question into a self-contained query based on a conversation history.

Rules:
1. If the "Follow-up Question" is already a standalone, complete question that doesn't rely on the history, output it as is.
2. If the "Follow-up Question" contains pronouns (like "he", "she", "it", "they") or is otherwise dependent on the "Conversation History", rewrite it by replacing the dependency with the correct entity from the history.
3. Your output MUST be ONLY the rewritten query, with no introductions or explanations like "Here is the rewritten query:".
"""
        user_prompt = f"[Conversation History]\\n{history}\\n[Follow-up Question]\\n{sub_q}\\n[Rewritten Query]"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt_str = self.local_model.apply_chat_template(messages)

        rewritten_query = ""
        try:
            self.local_model.set_adapter_state(enabled=False) # Pastikan menggunakan model dasar
            rewritten_query = self.local_model.generate(prompt_str, max_new_tokens=100, temperature=0.01, top_p=1.0, do_sample=False)
        finally:
            pass # Status default adalah nonaktif, tidak perlu dipulihkan

        # Membersihkan output
        if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
            rewritten_query = rewritten_query[1:-1]
        return rewritten_query or sub_q


class ContextRetriever:
    def __init__(self, config, corpus: list[str]):
        print("--- Menginisialisasi pengambil konteks (ContextRetriever) ---")
        self.device = config["device"]
        self.corpus = corpus
        self.index_path = config["faiss_index_path"]
        print(f"--- Memuat model embedding: '{config['embedding_model']}' ---")
        self.model = SentenceTransformer(config["embedding_model"], device=self.device, cache_folder=config["cache_dir"])
        
        if os.path.exists(self.index_path):
            print(f"--- Memuat indeks FAISS: '{self.index_path}' ---")
            self.index = faiss.read_index(self.index_path)
            print(f"--- Indeks FAISS berhasil dimuat, berisi {self.index.ntotal} dokumen. ---")
        else:
            self.index = None
            print(f"--- Peringatan: File indeks FAISS tidak ditemukan di '{self.index_path}'. Pengambil tidak akan tersedia. ---")

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.index: return []
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        _, indices = self.index.search(query_embedding_np, k)
        return [self.corpus[i] for i in indices[0]]


class Reranker:
    def __init__(self, config):
        print("--- Menginisialisasi pengurut ulang (Reranker) ---")
        print(f"--- Memuat model pengurut ulang: '{config['reranker_model']}' ---")
        self.model = CrossEncoder(config["reranker_model"], device=config["device"], cache_folder=config["cache_dir"])
        print("--- Inisialisasi pengurut ulang berhasil ---")

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        if not docs: return []
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]


class Generator:
    def __init__(self, local_model_handler: LocalModelHandler):
        print(f"--- Menginisialisasi generator (Generator - menggunakan model lokal) ---")
        self.local_model = local_model_handler

    def _generate(self, messages: list) -> str:
        prompt_str = self.local_model.apply_chat_template(messages)
        try:
            self.local_model.set_adapter_state(enabled=False) # Pastikan menggunakan model dasar
            return self.local_model.generate(prompt_str, max_new_tokens=512, temperature=0.1, top_p=0.9, do_sample=True)
        finally:
            pass # Status default adalah nonaktif, tidak perlu dipulihkan
        

    def generate(self, question: str, final_context: list[str]) -> str:
        if not final_context: return "Informasi tidak cukup, tidak dapat menjawab."
        context_str = "\\n\\n".join([f"Potongan relevan {i+1}:\\n{doc}" for i, doc in enumerate(final_context)])
        system_prompt = "You are a meticulous AI assistant. Your task is to answer a question based *only* on the provided context.\\n\\n**CRITICAL RULES:**\\n1.  **Format**: Respond only with the answer itself, in English.\\n\\n**WARNING: YOU MUST NOT USE YOUR OWN KNOWLEDGE.**"
        user_prompt = f"--- Context Start ---\\n{context_str}\\n--- Context End ---\\n\\nQuestion: {question}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self._generate(messages)

    def generate_final_synthesis(self, original_question: str, intermediate_steps: list) -> str:
        synthesis_prompt = "You are a professional answer synthesizer. Your task is to combine the given sub-question answers to form a final, coherent answer for the original, complex question.\\n\\n**CRITICAL RULES:**\\n1.  **Foundation of Truth**: You MUST base your final answer strictly on the provided sub-answers.\\n2.  **Handle Insufficiency**: If any sub-answer indicates that information was not available, reflect this limitation in your final answer.\\n3.  **Be Direct**: Synthesize the facts into a direct answer to the original question.\\n\\n--- Sub-Questions and Answers Review ---\\n"
        for i, step in enumerate(intermediate_steps):
            synthesis_prompt += f"{i+1}. Sub-Question: {step['sub_question']}\\n   Answer: {step['sub_answer']}\\n\\n"
        synthesis_prompt += f"--- Final Task ---\\nNow, please answer this initial question in English based on the above information: \\\"{original_question}\\\""
        messages = [{"role": "system", "content": "You are a professional answer integration expert."}, {"role": "user", "content": synthesis_prompt}]
        return self._generate(messages)

# ==============================================================================
# 4. Evaluasi & Pembuatan Pasangan Preferensi (berdasarkan fungsi reward baru)
# ==============================================================================

class RewardCalculator:
    """
    Evaluator baru yang menggunakan fungsi reward dari LocalModelHandler untuk menghitung skor.
    """
    def __init__(self, local_model_handler: LocalModelHandler):
        print("--- Menginisialisasi kalkulator reward (RewardCalculator - berdasarkan probabilitas model generatif) ---")
        self.local_model = local_model_handler

    def score(self, aggregated_context: str, question: str, ground_truth_answer: str) -> dict:
        """
        Memanggil pengelola model lokal untuk menghitung skor reward.
        """
        if not aggregated_context.strip() or not ground_truth_answer.strip():
            return {"reward": -1e9} # Kembalikan skor yang sangat rendah jika konteks atau jawaban kosong
        
        reward_val = self.local_model.calculate_reward_score(aggregated_context, question, ground_truth_answer)
        return {"reward": reward_val}


def execute_full_rag_pipeline(question, plan, retriever, reranker, generator, rewriter, config):
    """
    Menjalankan alur RAG lengkap.
    DIUBAH: Fungsi ini sekarang mengembalikan jawaban akhir dan konteks agregat dari semua langkah untuk perhitungan reward.
    """
    q_and_a_history = ""
    intermediate_steps = []
    aggregated_contexts = [] # Baru: Untuk mengumpulkan konteks yang diambil dari semua langkah
    
    for sub_q in plan:
        # 1. Tulis ulang kueri
        query_for_retriever = rewriter.rewrite(q_and_a_history, sub_q)
        # 2. Ambil
        retrieved_docs = retriever.retrieve(query_for_retriever, k=config["retriever_top_k"])
        # 3. Urutkan ulang
        reranked_docs = reranker.rerank(query_for_retriever, retrieved_docs)
        context_for_sub_q = reranked_docs[:config["reranker_top_n"]]
        
        # Kumpulkan konteks untuk perhitungan reward akhir
        aggregated_contexts.extend(context_for_sub_q)
        
        # Siapkan konteks untuk pembuatan sub-jawaban
        final_context_for_gen = []
        if q_and_a_history:
            history_context = f"Confirmed Background Knowledge (from previous steps):\\n{q_and_a_history.strip()}"
            final_context_for_gen.append(history_context)
        final_context_for_gen.extend(context_for_sub_q)
        
        # 4. Hasilkan sub-jawaban
        sub_answer = generator.generate(sub_q, final_context_for_gen)
        intermediate_steps.append({"sub_question": sub_q, "sub_answer": sub_answer})
        
        # 5. Perbarui riwayat (hanya jika jawabannya valid)
        if "tidak dapat menjawab" not in sub_answer and "informasi tidak cukup" not in sub_answer and "error" not in sub_answer.lower():
            q_and_a_history += f"Q: {sub_q}\\nA: {sub_answer}\\n"
            
    # 6. Sintesiskan jawaban akhir
    if len(plan) > 1:
        final_answer = generator.generate_final_synthesis(question, intermediate_steps)
    else:
        final_answer = intermediate_steps[0]['sub_answer'] if intermediate_steps else "Gagal menghasilkan jawaban."
        
    # Gabungkan semua konteks unik menjadi satu string untuk perhitungan reward
    unique_contexts = list(dict.fromkeys(aggregated_contexts))
    aggregated_context_str = "\\n\\n".join(unique_contexts)

    return {"final_answer": final_answer, "aggregated_context": aggregated_context_str}


def create_preference_pair(evaluated_plans: list, min_gap: float):
    """
    Membuat pasangan preferensi (chosen, rejected) berdasarkan skor reward.
    """
    if len(evaluated_plans) < 2:
        return None

    # Urutkan berdasarkan skor 'reward' baru dari tertinggi ke terendah
    sorted_by_reward = sorted(evaluated_plans, key=lambda x: x['scores']['reward'], reverse=True)

    chosen_plan_data = sorted_by_reward[0]
    rejected_plan_data = sorted_by_reward[-1]

    chosen_score = chosen_plan_data['scores']['reward']
    rejected_score = rejected_plan_data['scores']['reward']

    # Jika selisih antara skor tertinggi dan terendah terlalu kecil, abaikan sampel untuk memastikan preferensi yang jelas
    if (chosen_score - rejected_score) < min_gap:
        return None

    return {
        "chosen": chosen_plan_data['plan'],
        "rejected": rejected_plan_data['plan'],
        "chosen_reward": chosen_score,
        "rejected_reward": rejected_score
    }


# ==============================================================================
# 5. Alur Eksekusi Utama
# ==============================================================================
def main():
    # --- 1. Memuat data ---
    print("\\n--- Memuat data ---")
    print(f"Memuat korpus dari jalur lokal: {CONFIG['pkl_database_path']}")
    with open(CONFIG["pkl_database_path"], 'rb') as f:
        corpus_data = pickle.load(f)
    corpus = [item['context'] for item in corpus_data]

    print(f"Memuat dataset dari jalur lokal: {CONFIG['hotpotqa_dev_path']}")
    with open(CONFIG["hotpotqa_dev_path"], 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # --- 2. Menginisialisasi model lokal terpadu ---
    print("\\n--- Menginisialisasi pengelola model bahasa lokal terpadu... ---")
    
    # <<< DIUBAH: Hanya buat satu pengelola
    unified_model_handler = LocalModelHandler(
        base_model_path=CONFIG["base_model_path"],
        adapter_path=CONFIG["adapter_path"],
        device=CONFIG["device"],
        cache_dir=CONFIG["cache_dir"]
    )
    # Secara default, atur adaptor ke status nonaktif, karena sebagian besar operasi menggunakan model dasar
    unified_model_handler.set_adapter_state(enabled=False)

    # --- 3. Menginisialisasi semua komponen sistem RAG ---
    print("\\n--- Menginisialisasi komponen RAG lainnya... ---")
    # <<< DIUBAH: Semua komponen menggunakan pengelola yang sama
    planner = Planner(unified_model_handler)
    rewriter = QueryRewriter(unified_model_handler)
    generator = Generator(unified_model_handler)
    # RewardCalculator juga menggunakan pengelola yang sama, ia akan secara otomatis menonaktifkan adaptor di dalamnya
    reward_calculator = RewardCalculator(unified_model_handler)
    
    # Inisialisasi komponen non-LLM
    retriever = ContextRetriever(CONFIG, corpus=corpus)
    reranker = Reranker(CONFIG)

    print("\\n--- Semua komponen berhasil diinisialisasi ---\\n")
    

    # --- 4. Perulangan untuk menghasilkan dataset DPO ---
    print(f"\\n{'='*25} Memulai pembuatan pasangan preferensi untuk dataset DPO {'='*25}\\n")

    dpo_dataset = []
    num_samples_per_question = 3 # Hasilkan 3 rencana kandidat untuk setiap pertanyaan
    min_gap_threshold = CONFIG.get("min_preference_gap", 0.1)

    # Untuk demonstrasi cepat, kami hanya memproses sebagian kecil dari dataset
    for item in tqdm(dataset[11800:12000], desc="Menghasilkan pasangan preferensi DPO"):
        question = item["question"]
        ground_truth_answer = item["answer"]
        
        # Langkah A: Hasilkan beberapa rencana kandidat yang berbeda untuk satu pertanyaan
        candidate_plans = []
        unique_plans = set()
        for _ in range(num_samples_per_question):
            plan = planner.generate_plan_with_sampling(question)
            plan_str = json.dumps(plan, sort_keys=True) # Serialisasi untuk deduplikasi
            if plan_str not in unique_plans:
                unique_plans.add(plan_str)
                candidate_plans.append({"plan": plan})

        # Jika keragaman yang dihasilkan tidak cukup (kurang dari 2 rencana unik), lewati pertanyaan ini
        if len(candidate_plans) < 2:
            continue
            
        # Langkah B: Jalankan alur RAG untuk setiap rencana dan hitung reward
        evaluated_plans = []
        for plan_data in candidate_plans:
            plan = plan_data['plan']
            # Jalankan alur RAG lengkap
            pipeline_output = execute_full_rag_pipeline(question, plan, retriever, reranker, generator, rewriter, CONFIG)
            
            # Beri skor menggunakan kalkulator reward baru
            scores = reward_calculator.score(
                aggregated_context=pipeline_output["aggregated_context"],
                question=question,
                ground_truth_answer=ground_truth_answer
            )
            
            evaluated_plans.append({
                "plan": plan, 
                "final_answer": pipeline_output["final_answer"], 
                "scores": scores
            })
        
        # Langkah C: Buat pasangan preferensi berdasarkan skor reward
        preference_pair = create_preference_pair(evaluated_plans, min_gap=min_gap_threshold)

        if preference_pair:
            dpo_dataset.append({
                "prompt": question,
                "chosen": json.dumps(preference_pair['chosen'], ensure_ascii=False),
                "rejected": json.dumps(preference_pair['rejected'], ensure_ascii=False),
                "chosen_reward": preference_pair['chosen_reward'],
                "rejected_reward": preference_pair['rejected_reward']
            })

    # --- 5. Menyimpan dataset akhir ---
    output_path = CONFIG["results_output_path"]
    print(f"\\n--- Menyimpan {len(dpo_dataset)} data preferensi DPO ke file: {output_path} ---")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_dataset, f, ensure_ascii=False, indent=4)
    print(f"--- Dataset DPO telah berhasil disimpan ke {output_path} ---")


if __name__ == '__main__':
    main()