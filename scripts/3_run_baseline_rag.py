import os
import json
import torch
import time
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import faiss
import pickle

# ==============================================================================
# 1. 配置与准备
# ==============================================================================
# --- 设置 Hugging Face 库的缓存目录 ---
cache_directory = "D:/huggingface_models_cache"
os.environ['HF_HOME'] = cache_directory
os.environ['TRANSFORMERS_CACHE'] = cache_directory
os.makedirs(cache_directory, exist_ok=True)
print(f"Hugging Face 模型缓存目录已设置为: {os.environ.get('HF_HOME')}")

# --- 系统配置 ---
CONFIG = {
    # --- 模型与API配置 ---
    "cache_dir": cache_directory,
    "planner_model": "Qwen/Qwen2.5-7B-Instruct",
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-large",
    "generator_api_model_name": "qwen-plus",
    "generator_api_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": os.getenv("DASHSCOPE_API_KEY"),

    # --- 数据与结果文件 ---
    "hotpotqa_dev_path": "hotpot_train_v1.1.json",
    "pkl_database_path": "hotpot_contexts_deduplicated.pkl", 
    "faiss_index_path": "hotpotqa_faiss_deduplicated.index", # <-- 新增：FAISS索引的保存路径
    "results_output_path": "rag_results_from_db.json", 

    # --- RAG 流程超参数 ---
    "retriever_top_k": 5, 
    "reranker_top_n": 2,

    # --- 设备配置 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==============================================================================
# 2. RAG 组件
# ==============================================================================

class Planner:
    def __init__(self, config):
        print("--- 初始化查询规划器 (Planner v2 - 支持交集查询) ---")
        self.device = config["device"]
        self.model_id = config["planner_model"]
        self.cache_dir = config["cache_dir"]
        print(f"正在从 {self.model_id} 加载模型...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map=self.device,
                quantization_config=quantization_config, cache_dir=self.cache_dir,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            print("--- 查询规划器初始化完成 ---")
        except Exception as e:
            print(f"错误: 初始化Planner时加载模型失败: {e}")
            self.model, self.tokenizer = None, None
    
    def generate_plan(self, question: str) -> list[str]:
        if not self.model or not self.tokenizer:
            print("Planner 模型未成功加载，跳过规划步骤。")
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
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        try:
            generated_ids = self.model.generate(
                model_inputs.input_ids, max_new_tokens=256, do_sample=False
            )
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            assistant_response_start = decoded.rfind("assistant")
            if assistant_response_start == -1:
                 assistant_response_start = decoded.rfind('Output:')
            json_part_raw = decoded[assistant_response_start:]
            json_start, json_end = json_part_raw.find('['), json_part_raw.rfind(']')
            if json_start != -1 and json_end != -1:
                json_part = json_part_raw[json_start : json_end + 1]
                sub_questions = json.loads(json_part)
                if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
                    return sub_questions
                else: raise ValueError("JSON content is not a list of strings.")
            else: raise ValueError("Could not find a valid JSON array in the assistant's response.")
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"警告: Planner未能生成有效的JSON格式计划。错误: {e}\n原始输出: {decoded}")
            return [question]

# --- MODIFIED: ContextRetriever now saves and loads the pre-built FAISS index ---
class ContextRetriever:
    def __init__(self, config, corpus: list[str]):
        print("--- 初始化上下文检索器 (ContextRetriever - FAISS Database) ---")
        self.device = config["device"]
        self.corpus = corpus  # 存储原始文本文档
        self.index_path = config["faiss_index_path"]
        
        # 加载嵌入模型
        self.model = SentenceTransformer(
            config["embedding_model"], device=self.device, cache_folder=config["cache_dir"]
        )
        
        # 检查索引文件是否已存在
        if os.path.exists(self.index_path):
            print(f"--- 正在从 '{self.index_path}' 加载已存在的FAISS索引... ---")
            self.index = faiss.read_index(self.index_path)
            print(f"--- FAISS索引加载完成，共包含 {self.index.ntotal} 个文档。 ---")
        else:
            print(f"--- 未找到FAISS索引文件。正在为整个知识库 ({len(self.corpus)}个文档) 构建新的索引... ---")
            # 对整个语料库进行编码
            corpus_embeddings = self.model.encode(
                self.corpus, 
                convert_to_tensor=True, 
                normalize_embeddings=True, 
                show_progress_bar=True
            )
            
            # 构建FAISS索引
            d = corpus_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(corpus_embeddings.cpu().numpy())
            
            print(f"--- FAISS索引构建完成，共索引了 {self.index.ntotal} 个文档。---")
            print(f"--- 正在将新索引保存到 '{self.index_path}'... ---")
            faiss.write_index(self.index, self.index_path)
            print("--- 索引已成功保存。 ---")

        print("--- 上下文检索器初始化完成 ---")

    def retrieve(self, query: str, k: int) -> list[str]:
        """
        从预先构建的FAISS索引中检索最相关的k个文档。
        """
        if not hasattr(self, 'index'):
            print("错误: FAISS 索引未初始化。")
            return []
            
        # 对查询进行编码
        query_embedding = self.model.encode(
            query, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        # 搜索索引
        D, I = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), k)
        
        # 从原始语料库返回相应的文档
        return [self.corpus[i] for i in I[0]]

class Reranker:
    def __init__(self, config):
        print("--- 初始化精排器 (Reranker) ---")
        self.device = config["device"]
        self.model = CrossEncoder(
            config["reranker_model"], device=self.device, cache_folder=config["cache_dir"]
        )
        print("--- 精排器初始化完成 ---")

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        if not docs: return []
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [doc for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]

class Generator:
    def __init__(self, config):
        print(f"--- 初始化生成器 (API 模型: {config['generator_api_model_name']}) ---")
        self.model_name = config["generator_api_model_name"]
        self.api_key = config["api_key"]
        self.base_url = config.get("generator_api_base_url")
        if not self.api_key or "DASHSCOPE_API_KEY" in self.api_key:
            print("警告: 生成器API Key尚未配置。")
            self.client = None
            return
        if not self.base_url:
            print("警告: 生成器API base_url 尚未配置。")
            self.client = None
            return
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print("--- 生成器初始化完成 ---")

    def generate(self, question: str, final_context: list[str]) -> str:
        if not self.client: return "错误: API客户端未初始化。"
        if not final_context: return "信息不足，无法回答。"
        context_str = "\n\n".join([f"相关片段 {i+1}:\n{doc}" for i, doc in enumerate(final_context)])
        system_prompt = """You are a meticulous AI assistant. Your task is to answer a question based *only* on the provided context.

**CRITICAL RULES:**
1.  **Format**: Respond only with the answer itself, in English.

**WARNING: YOU MUST NOT USE YOUR OWN KNOWLEDGE.**
"""
        user_prompt = f"--- 上下文开始 ---\n{context_str}\n--- 上下文结束 ---\n\n问题: {question}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            completion = self.client.chat.completions.create(model=self.model_name, messages=messages, temperature=0.0)
            return completion.choices[0].message.content or "API返回为空。"
        except Exception as e:
            return f"生成答案时发生API错误: {e}"

    def generate_final_synthesis(self, original_question: str, intermediate_steps: list) -> str:
        if not self.client: return "错误: API客户端未初始化。"
        synthesis_prompt = f"""You are a professional answer synthesizer. Your task is to combine the given sub-question answers to form a final, coherent answer for the original, complex question.

**CRITICAL RULES:**
1.  **Foundation of Truth**: You MUST base your final answer strictly on the provided sub-answers. Do not introduce any external knowledge or make new inferences.
2.  **Handle Insufficiency**: If any sub-answer indicates that information was not available (e.g., "The answer is not available..."), reflect this limitation in your final answer. Do not try to fill in the gaps.
3.  **Be Direct**: Synthesize the facts into a direct answer to the original question.

--- Sub-Questions and Answers Review ---
"""
        for i, step in enumerate(intermediate_steps):
            synthesis_prompt += f"{i+1}. Sub - problems: {step['sub_question']}\n   answers: {step['sub_answer']}\n\n"
        synthesis_prompt += f"""--- Final task ---
Now, please answer this initial question in English based on the above information: "{original_question}"
"""
        messages = [{"role": "system", "content": "You are a professional answer integration expert."}, {"role": "user", "content": synthesis_prompt}]
        try:
            completion = self.client.chat.completions.create(model=self.model_name, messages=messages)
            return completion.choices[0].message.content or "API返回为空。"
        except Exception as e:
            return f"生成最终答案时发生API错误: {e}"

def rewrite_query(history: str, sub_q: str, client: OpenAI, model_name: str) -> str:  
    if not history.strip():
        print("    - 信息: 未找到历史记录，使用原始查询。")
        return sub_q

    system_prompt = """You are an expert query rewriter. Your task is to reformulate a follow-up question into a self-contained query based on a conversation history.

Rules:
1. If the "Follow-up Question" is already a standalone, complete question that doesn't rely on the history, output it as is.
2. If the "Follow-up Question" contains pronouns (like "he", "she", "it", "they") or is otherwise dependent on the "Conversation History", rewrite it by replacing the dependency with the correct entity from the history.
3. Your output MUST be ONLY the rewritten query, with no introductions or explanations like "Here is the rewritten query:".
"""
    user_prompt = f"""[Conversation History]{history}[Follow-up Question]{sub_q}[Rewritten Query]"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        rewritten_query = completion.choices[0].message.content.strip()
        if rewritten_query.startswith('"') and rewritten_query.endswith('"'):
            rewritten_query = rewritten_query[1:-1]
        print(f"    - 重写后的查询: \"{rewritten_query}\"")
        return rewritten_query
    except Exception as e:
        print(f"    - 警告: 查询重写失败，错误: {e}. 回退到原始查询。")
        return sub_q
        
# ==============================================================================
# 3. 主执行流程
# ==============================================================================

def main():
    if not CONFIG["api_key"]:
        print("\n\n错误: 环境变量 DASHSCOPE_API_KEY 未设置。请设置API密钥后重试。\n")
        return

    # 从PKL数据库加载知识库
    pkl_db_path = CONFIG["pkl_database_path"]
    if not os.path.exists(pkl_db_path):
        print(f"错误: PKL 数据库文件未在 '{pkl_db_path}' 找到")
        return
    try:
        with open(pkl_db_path, 'rb') as f:
            corpus_data = pickle.load(f)
        corpus = [item['context'] for item in corpus_data]
        print(f"成功从PKL数据库加载知识库，共包含 {len(corpus)} 个文档。")
    except Exception as e:
        print(f"错误: 加载PKL文件失败: {e}")
        return

    # 加载测试问题数据集
    dev_data_path = CONFIG["hotpotqa_dev_path"]
    if not os.path.exists(dev_data_path):
        print(f"错误: HotpotQA 文件未在 '{dev_data_path}' 找到")
        return
    with open(dev_data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"成功加载测试数据集，包含 {len(dataset)} 个问题。")


    print("\n--- 正在初始化所有RAG系统组件 ---")
    planner = Planner(CONFIG)
    # 将加载的知识库传递给检索器进行索引（或加载已有的索引）
    retriever = ContextRetriever(CONFIG, corpus=corpus)
    reranker = Reranker(CONFIG)
    generator = Generator(CONFIG)
    print("--- 所有组件初始化完成 ---\n")

    print(f"\n{'='*25} 开始处理数据集中的问题 {'='*25}\n")
    
    all_results = []
    
    sample_count = 5
    for i, item in enumerate(tqdm(dataset[:sample_count], desc="处理问题中")):
        question = item["question"]
        ground_truth_answer = item["answer"]
        
        print(f"\n--- [问题 ID {i+1}] ---")
        print(f"原始问题 Q: {question}")
        
        sub_questions = planner.generate_plan(question)
        if len(sub_questions) > 1:
            print(f"Planner 生成的子问题: {sub_questions}")
        else:
            print("Planner 认为无需分解，直接回答原始问题。")
        
        q_and_a_history = ""
        intermediate_steps = []

        for sub_q_index, sub_q in enumerate(sub_questions):
            print(f"  -> 正在回答子问题 {sub_q_index+1}/{len(sub_questions)}: {sub_q}")
            
            query_for_retriever = rewrite_query(q_and_a_history, sub_q, generator.client, CONFIG["generator_api_model_name"])
            
            # 从全局数据库中检索
            retrieved_docs = retriever.retrieve(query_for_retriever, k=CONFIG["retriever_top_k"])
            
            reranked_docs = reranker.rerank(query_for_retriever, retrieved_docs)
            
            context_for_sub_q = reranked_docs[:CONFIG["reranker_top_n"]]
            
            final_context_for_gen = []
            if q_and_a_history:
                history_context = f"Confirmed Background Knowledge (from previous steps):\n{q_and_a_history.strip()}"
                final_context_for_gen.append(history_context)
            final_context_for_gen.extend(context_for_sub_q)
            sub_answer = generator.generate(sub_q, final_context_for_gen)
            print(f"  -> 子问题答案: {sub_answer}")
            
            intermediate_steps.append({
                "sub_question": query_for_retriever,
                "retrieved_docs": retrieved_docs, # 可选：记录初筛结果
                "reranked_docs": context_for_sub_q, # 记录精排后的上下文
                "sub_answer": sub_answer
            })
            
            if "无法回答" not in sub_answer and "信息不足" not in sub_answer:
                q_and_a_history += f"Q: {sub_q}\nA: {sub_answer}\n"
        
        if len(sub_questions) > 1:
            print("--- 正在综合所有子答案以生成最终答案 ---")
            generated_answer = generator.generate_final_synthesis(question, intermediate_steps)
        else:
            generated_answer = intermediate_steps[0]['sub_answer'] if intermediate_steps else "未能生成答案。"

        print(f"A (模型最终生成): {generated_answer}")
        print(f"真实答案: {ground_truth_answer}")
        print("-" * (20 + len(str(i+1))))

        all_results.append({
            "id": item.get('_id', i + 1), # 使用原始ID
            "original_question": question,
            "ground_truth_answer": ground_truth_answer,
            "planner_plan": sub_questions,
            "intermediate_steps": intermediate_steps,
            "final_generated_answer": generated_answer,
        })

    output_path = CONFIG["results_output_path"]
    print(f"\n--- 正在将 {len(all_results)} 条结果保存到文件: {output_path} ---")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        print("--- 结果已成功保存！ ---")
    except Exception as e:
        print(f"--- 保存结果时发生错误: {e} ---")

if __name__ == '__main__':
    main()