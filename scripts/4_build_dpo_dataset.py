# build_dpo_dataset.py

# Catatan: Perintah pip berikut berasal dari sel pertama notebook.
# Saat menjalankan sebagai skrip .py, Anda biasanya akan menginstal dependensi ini
# sekali dari baris perintah atau melalui file requirements.txt,
# daripada menjalankannya di dalam skrip itu sendiri.
#
# pip install -U bitsandbytes
# pip install -U transformers accelerate
# pip install -U rouge-score
# pip install -U faiss-gpu-cu12

import os
import json
import torch
from openai import OpenAI # <--- diubah: Impor pustaka OpenAI untuk panggilan API
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
from kaggle_secrets import UserSecretsClient
from tqdm import tqdm
from rouge_score import rouge_scorer

# --- Mengatur direktori cache untuk pustaka Hugging Face ---
# Direktori yang dapat ditulisi di Kaggle adalah /kaggle/working/
cache_directory = "/kaggle/working/huggingface_cache"
os.environ['HF_HOME'] = cache_directory
os.environ['TRANSFORMERS_CACHE'] = cache_directory
os.makedirs(cache_directory, exist_ok=True)
print(f"Direktori cache Hugging Face telah diatur ke: {cache_directory}")

# --- Pastikan direktori hasil Kaggle ada ---
os.makedirs("/kaggle/working/results", exist_ok=True)

# --- Konfigurasi sistem (DIUBAH UNTUK KAGGLE) ---
# Ganti <your-dataset-name> dengan nama dataset yang Anda unggah!
KAGGLE_INPUT_DIR = "/kaggle/input/rag-data" # <--- Ganti nama dataset Anda di sini

CONFIG = {
    # --- Konfigurasi Model & API (bagian ini tidak perlu diubah karena akan membaca dari Kaggle Secrets) ---
    "llm_api_model_name": "qwen2.5-7b-instruct",
    "llm_api_key_env": "DASHSCOPE_API_KEY",
    "llm_api_base_url_env": "DASHSCOPE_BASE_URL",

    # --- Jalur model lokal (bagian ini tidak perlu diubah) ---
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-large",

    # --- File Data & Hasil (jalur telah disesuaikan untuk Kaggle) ---
    "hotpotqa_dev_path": f"{KAGGLE_INPUT_DIR}/hotpot_train_v1.1.json",
    "pkl_database_path": f"{KAGGLE_INPUT_DIR}/hotpot_contexts_deduplicated.pkl",
    "faiss_index_path": f"{KAGGLE_INPUT_DIR}/hotpotqa_faiss_deduplicated.index",
    "results_output_path": "/kaggle/working/results/rag_results_from_db.json", # Output ke direktori yang dapat ditulisi

    # --- Hiperparameter alur RAG (bagian ini tidak perlu diubah) ---
    "retriever_top_k": 5,
    "reranker_top_n": 2,
    "min_preference_gap": 0,
    
    # --- Konfigurasi perangkat (bagian ini tidak perlu diubah, akan mendeteksi GPU secara otomatis) ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # --- Konfigurasi cache (jalur telah disesuaikan untuk Kaggle) ---
    "cache_dir": cache_directory,
}

# ==============================================================================
# 2. Komponen RAG
# ==============================================================================

# Kelas Layanan LLM (PERUBAHAN BESAR: Beralih ke API)
class LLMService:
    def __init__(self, config):
        """
        Menginisialisasi layanan API LLM.
        Tidak lagi memuat model lokal, tetapi mengonfigurasi klien API.
        """
        # --- Inisialisasi atribut instance ---
        self.client = None # <--- Baru: Pertama, inisialisasi klien ke None
        self.model_name = config["llm_api_model_name"] # <--- Diubah: Simpan model_name sebagai atribut instance

        api_key_env = config["llm_api_key_env"]      # "DASHSCOPE_API_KEY"
        base_url_env = config["llm_api_base_url_env"]# "DASHSCOPE_BASE_URL"
        
        print(f"--- Menginisialisasi layanan API model bahasa ({self.model_name}) ---")
        
        # --- Menggunakan UserSecretsClient untuk mendapatkan Secret secara langsung ---
        try:
            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret(api_key_env)
            base_url = user_secrets.get_secret(base_url_env)
        except Exception as e:
            print(f"Kesalahan fatal: Tidak dapat mengambil konfigurasi API dari Kaggle Secrets: {e}")
            api_key = None
            base_url = None

        # --- Memeriksa dan membuat klien API ---
        if api_key and base_url:
            try:
                # vvvvvvvvvvvv Ini adalah kode baru yang paling penting vvvvvvvvvvvv
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                print("--- Klien API OpenAI telah berhasil dibuat. ---")
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            except Exception as e:
                print(f"Kesalahan fatal: Gagal menginisialisasi klien OpenAI: {e}")
                self.client = None # Pastikan klien adalah None jika inisialisasi gagal
        else:
            print(f"Peringatan: Gagal mendapatkan '{api_key_env}' atau '{base_url_env}' dari Kaggle Secrets. Layanan API tidak akan tersedia.")


    def call_api(self, messages: list, max_tokens: int, temperature: float, top_p: float, do_sample: bool) -> str:
        """
        Memanggil model bahasa melalui API.
        """
        if not self.client:
            return "Kesalahan: Klien API belum diinisialisasi."

        # Jika do_sample tidak didukung, dapat disimulasikan dengan mengatur suhu
        if not do_sample:
            temperature = 0.01 # Atur suhu yang sangat rendah tetapi tidak nol untuk output deterministik
        
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
            return f"Terjadi kesalahan selama panggilan API: {e}"


# Planner (DIUBAH: Menggunakan API LLMService)
class Planner:
    def __init__(self, llm_service: LLMService):
        print("--- Menginisialisasi perencana kueri (Planner) ---")
        self.llm_service = llm_service
        if self.llm_service.client:
            print("--- Perencana kueri telah terhubung ke layanan API ---")
        else:
            print("--- Peringatan: Perencana kueri gagal terhubung ke layanan API ---")

    def _parse_plan(self, response_text: str):
        try:
            # Respons API mungkin lebih bersih, tetapi kami mempertahankan logika penguraian yang kuat
            json_start, json_end = response_text.find('['), response_text.rfind(']')
            if json_start != -1 and json_end != -1:
                json_part = response_text[json_start : json_end + 1]
                sub_questions = json.loads(json_part)
                if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions) and sub_questions:
                    return sub_questions
                else: raise ValueError("Konten JSON bukan daftar string atau kosong.")
            else: raise ValueError("Array JSON yang valid tidak ditemukan dalam respons API.")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Gagal mengurai rencana: {e}. Teks respons lengkap: '{response_text}'")
            return None

    def generate_plan_with_sampling(
        self, question: str, do_sample: bool = True, top_p: float = 0.9, temperature: float = 0.8, max_new_tokens: int = 256
    ) -> list[str]:
        # --- PERBAIKAN: Mengembalikan prompt ke Bahasa Inggris untuk konsistensi dan keandalan ---
        # --- dan memperbaiki sintaks f-string f"""...""" ---
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
        
        response_text = self.llm_service.call_api(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        parsed_plan = self._parse_plan(response_text)
        return parsed_plan if parsed_plan else [question]


# QueryRewriter (DIUBAH: Menggunakan API LLMService dan prompt yang diperbaiki)
class QueryRewriter:
    def __init__(self, llm_service: LLMService):
        print("--- Menginisialisasi penulis ulang kueri (QueryRewriter) ---")
        self.llm_service = llm_service
        if self.llm_service.client:
            print("--- Penulis ulang kueri telah terhubung ke layanan API ---")
        else:
            print("--- Peringatan: Penulis ulang kueri gagal terhubung ke layanan API ---")

    def rewrite(self, history: str, sub_q: str) -> str:
        if not self.llm_service.client: return sub_q
        if not history.strip(): return sub_q
        
        # --- PERBAIKAN: Mengembalikan prompt ke Bahasa Inggris untuk konsistensi ---
        system_prompt = """You are an expert query rewriter. Your task is to reformulate a follow-up question into a self-contained query based on a conversation history.

Rules:
1. If the "Follow-up Question" is already a standalone, complete question that doesn't rely on the history, output it as is.
2. If the "Follow-up Question" contains pronouns (like "he", "she", "it", "they") or is otherwise dependent on the "Conversation History", rewrite it by replacing the dependency with the correct entity from the history.
3. Your output MUST be ONLY the rewritten query, with no introductions or explanations like "Here is the rewritten query:".
"""
        user_prompt = f"[Conversation History]\\n{history}\\n[Follow-up Question]\\n{sub_q}\\n[Rewritten Query]"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        try:
            rewritten_query = self.llm_service.call_api(
                messages=messages,
                max_tokens=100,
                temperature=0.1,
                top_p=1.0,
                do_sample=False
            )
            if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
                rewritten_query = rewritten_query[1:-1]
            return rewritten_query or sub_q
        except Exception as e:
            print(f"    - Peringatan: Penulisan ulang kueri API gagal: {e}. Menggunakan kueri asli.")
            return sub_q

# ContextRetriever (Tidak perlu diubah, menggunakan model lokal melalui CONFIG)
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
            print(f"--- Peringatan: File indeks FAISS tidak ditemukan di '{self.index_path}'. ---")
            self.index = None

    def retrieve(self, query: str, k: int) -> list[str]:
        if not self.index: return []
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        _, I = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), k)
        return [self.corpus[i] for i in I[0]]


# Reranker (Tidak perlu diubah, menggunakan model lokal melalui CONFIG)
class Reranker:
    def __init__(self, config):
        print("--- Menginisialisasi pengurut ulang (Reranker) ---")
        self.model = CrossEncoder(config["reranker_model"], device=config["device"], cache_folder=config["cache_dir"])
        print("--- Inisialisasi pengurut ulang berhasil ---")

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        if not docs: return []
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]


# Generator (DIUBAH: Menggunakan API LLMService)
class Generator:
    def __init__(self, llm_service: LLMService):
        print(f"--- Menginisialisasi generator (Generator melalui API: {llm_service.model_name}) ---")
        self.llm_service = llm_service
        if self.llm_service.client:
            print("--- Generator telah terhubung ke layanan API ---")
        else:
            print("--- Peringatan: Generator gagal terhubung ke layanan API ---")

    def _generate_with_api(self, messages: list) -> str:
        """
        Fungsi pembantu untuk memanggil API LLM.
        """
        if not self.llm_service.client: return "Kesalahan: Layanan API LLM belum diinisialisasi."
        
        try:
            final_answer = self.llm_service.call_api(
                messages=messages,
                max_tokens=512,
                temperature=0.1,
                top_p=0.9,
                do_sample=True
            )
            return final_answer or "Model API mengembalikan respons kosong."
        except Exception as e:
            return f"Terjadi kesalahan saat menghasilkan dengan model API: {e}"

    def generate(self, question: str, final_context: list[str]) -> str:
        if not final_context: return "Informasi tidak cukup untuk menjawab."
        context_str = "\\n\\n".join([f"Potongan relevan {i+1}:\\n{doc}" for i, doc in enumerate(final_context)])
        system_prompt = "Anda adalah asisten AI yang teliti. Tugas Anda adalah menjawab pertanyaan *hanya* berdasarkan konteks yang disediakan.\\n\\n**ATURAN PENTING:**\\n1. **Format**: Tanggapi hanya dengan jawabannya saja, dalam bahasa Inggris.\\n\\n**PERINGATAN: ANDA TIDAK BOLEH MENGGUNAKAN PENGETAHUAN ANDA SENDIRI.**"
        user_prompt = f"--- Mulai Konteks ---\\n{context_str}\\n--- Akhir Konteks ---\\n\\nPertanyaan: {question}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        return self._generate_with_api(messages)

    def generate_final_synthesis(self, original_question: str, intermediate_steps: list) -> str:
        synthesis_prompt = "Anda adalah penyintesis jawaban profesional. Tugas Anda adalah menggabungkan jawaban sub-pertanyaan yang diberikan untuk membentuk jawaban akhir yang koheren untuk pertanyaan asli yang kompleks.\\n\\n**ATURAN PENTING:**\\n1. **Dasar Kebenaran**: Anda HARUS mendasarkan jawaban akhir Anda secara ketat pada sub-jawaban yang disediakan.\\n2. **Tangani Ketidakcukupan**: Jika ada sub-jawaban yang menunjukkan bahwa informasi tidak tersedia, cerminkan batasan ini dalam jawaban akhir Anda.\\n3. **Langsung**: Sintesiskan fakta-fakta menjadi jawaban langsung untuk pertanyaan asli.\\n\\n--- Tinjauan Sub-Pertanyaan dan Jawaban ---\\n"
        for i, step in enumerate(intermediate_steps):
            synthesis_prompt += f"{i+1}. Sub-Pertanyaan: {step['sub_question']}\\n   Jawaban: {step['sub_answer']}\\n\\n"
        synthesis_prompt += f"--- Tugas Akhir ---\\nSekarang, harap jawab pertanyaan awal ini dalam bahasa Inggris berdasarkan informasi di atas: \\\"{original_question}\\\""
        messages = [{"role": "system", "content": "Anda adalah ahli integrasi jawaban profesional."}, {"role": "user", "content": synthesis_prompt}]
        return self._generate_with_api(messages)

# ==============================================================================
# 2. Evaluasi & Pembuatan Pasangan Preferensi (Tidak perlu diubah)
# ==============================================================================

class Evaluator:
    def __init__(self):
        print("--- Menginisialisasi evaluator (Evaluator - hanya ROUGE) ---")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        print("--- Inisialisasi evaluator selesai ---")

    def score(self, prediction: str, reference: str) -> dict:
        if not prediction.strip() or not reference.strip():
            return {"rouge_l": 0.0}
        rouge_results = self.rouge_scorer.score(reference, prediction)
        return {"rouge_l": rouge_results['rougeL'].fmeasure}


def execute_full_rag_pipeline(question, plan, retriever, reranker, generator, rewriter, config):
    q_and_a_history = ""
    intermediate_steps = []
    for sub_q in plan:
        query_for_retriever = rewriter.rewrite(q_and_a_history, sub_q)
        retrieved_docs = retriever.retrieve(query_for_retriever, k=config["retriever_top_k"])
        reranked_docs = reranker.rerank(query_for_retriever, retrieved_docs)
        context_for_sub_q = reranked_docs[:config["reranker_top_n"]]
        
        final_context_for_gen = []
        if q_and_a_history:
            history_context = f"Pengetahuan Latar Belakang yang Dikonfirmasi (dari langkah sebelumnya):\\n{q_and_a_history.strip()}"
            final_context_for_gen.append(history_context)
        final_context_for_gen.extend(context_for_sub_q)
        
        sub_answer = generator.generate(sub_q, final_context_for_gen)
        intermediate_steps.append({"sub_question": sub_q, "sub_answer": sub_answer})
        
        # Hanya tambahkan ke riwayat jika jawabannya valid
        if "tidak dapat menjawab" not in sub_answer and "informasi tidak cukup" not in sub_answer and "error" not in sub_answer.lower():
            q_and_a_history += f"T: {sub_q}\\nJ: {sub_answer}\\n"
            
    if len(plan) > 1:
        final_answer = generator.generate_final_synthesis(question, intermediate_steps)
    else:
        final_answer = intermediate_steps[0]['sub_answer'] if intermediate_steps else "Gagal menghasilkan jawaban."
        
    return final_answer


def create_preference_pair(evaluated_plans: list, min_gap: float):
    if len(evaluated_plans) < 2:
        return None

    sorted_by_rouge = sorted(evaluated_plans, key=lambda x: x['scores']['rouge_l'], reverse=True)

    chosen_plan_data = sorted_by_rouge[0]
    rejected_plan_data = sorted_by_rouge[-1]

    chosen_score = chosen_plan_data['scores']['rouge_l']
    rejected_score = rejected_plan_data['scores']['rouge_l']

    if (chosen_score - rejected_score) < min_gap:
        return None

    return {
        "chosen": chosen_plan_data['plan'],
        "rejected": rejected_plan_data['plan'],
        "chosen_reward": chosen_score,
        "rejected_reward": rejected_score
    }


# ==============================================================================
# 3. Alur Eksekusi Utama
# ==============================================================================
def main():
    # --- 1. Memuat data dan layanan inti ---
    # Pastikan variabel lingkungan QWEN_API_KEY dan QWEN_BASE_URL diatur sebelum berjalan
    print("\\n--- Memuat data dan menginisialisasi semua komponen sistem RAG ---")

    print(f"Memuat data dari jalur lokal: {CONFIG['pkl_database_path']}")
    with open(CONFIG["pkl_database_path"], 'rb') as f:
        corpus_data = pickle.load(f)
    corpus = [item['context'] for item in corpus_data]

    with open(CONFIG["hotpotqa_dev_path"], 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Inisialisasi layanan API LLM, itu akan membaca kunci API dan URL dari variabel lingkungan
    print("\\n--- Menginisialisasi layanan API model bahasa, pastikan variabel lingkungan telah diatur... ---")
    llm_service = LLMService(CONFIG)

    if not llm_service.client:
        print("\\n\\nKesalahan: Inisialisasi layanan API model bahasa gagal. Program akan keluar.\\n")
        return

    # Inisialisasi semua komponen lainnya
    planner = Planner(llm_service)
    rewriter = QueryRewriter(llm_service)
    generator = Generator(llm_service)
    retriever = ContextRetriever(CONFIG, corpus=corpus)
    reranker = Reranker(CONFIG)
    evaluator = Evaluator()

    print("\\n--- Semua komponen berhasil diinisialisasi ---\\n")

    # --- 2. Perulangan untuk menghasilkan dataset DPO ---
    print(f"\\n{'='*25} Memulai pembuatan pasangan preferensi untuk dataset DPO {'='*25}\\n")

    dpo_dataset = []
    num_samples_per_question = 3
    dataset_sample_size = 10  # Untuk kemudahan demonstrasi, gunakan sampel dataset yang lebih kecil
    min_gap_threshold = CONFIG.get("min_preference_gap", 0.1)

    for item in tqdm(dataset[11000:12000], desc="Menghasilkan pasangan preferensi DPO"):
        question = item["question"]
        ground_truth_answer = item["answer"]
        candidate_plans = []
        unique_plans = set()
        for _ in range(num_samples_per_question):
            plan = planner.generate_plan_with_sampling(question)
            plan_str = json.dumps(plan, sort_keys=True)
            if plan_str not in unique_plans:
                unique_plans.add(plan_str)
                candidate_plans.append({"plan": plan})

        if len(candidate_plans) < 2:
            continue

        evaluated_plans = []
        for plan_data in candidate_plans:
            plan = plan_data['plan']
            final_answer = execute_full_rag_pipeline(question, plan, retriever, reranker, generator, rewriter, CONFIG)
            scores = evaluator.score(final_answer, ground_truth_answer)
            evaluated_plans.append({
                "plan": plan, "final_answer": final_answer, "scores": scores
            })

        preference_pair = create_preference_pair(evaluated_plans, min_gap=min_gap_threshold)

        if preference_pair:
            dpo_dataset.append({
                "prompt": question,
                "chosen": json.dumps(preference_pair['chosen'], ensure_ascii=False),
                "rejected": json.dumps(preference_pair['rejected'], ensure_ascii=False),
                "chosen_reward": preference_pair['chosen_reward'],
                "rejected_reward": preference_pair['rejected_reward']
            })

    # --- 3. Menyimpan dataset akhir ---
    # Jalur output telah diubah ke direktori results lokal
    output_path = os.path.join("results", "dpo_planner_dataset_with_rewards.json")
    print(f"\\n--- Menyimpan {len(dpo_dataset)} data preferensi DPO ke file: {output_path} ---")
    try:
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_dataset, f, ensure_ascii=False, indent=4)
        print(f"--- Dataset DPO telah berhasil disimpan ke {output_path} ---")
    except Exception as e:
        print(f"--- Terjadi kesalahan saat menyimpan hasil: {e} ---")

if __name__ == '__main__':
    main()