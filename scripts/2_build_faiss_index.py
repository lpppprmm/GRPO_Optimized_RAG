import os
import pickle
import faiss
import numpy as np
import torch

# --- 关键修改：设置Hugging Face的缓存目录 ---
# 将此代码块放在所有其他导入（尤其是transformers或sentence_transformers）之前！
# 这样做可以将模型下载和缓存的位置从默认的C盘重定向到您指定的路径。
#
# 【请根据您的实际情况修改这里的路径】
cache_dir = "D:\\huggingface_models_cache"  # 使用双反斜杠'\\'来表示Windows路径
# 
# 确保目标文件夹存在，或者您的程序有权限创建它
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
print(f"Hugging Face 模型缓存目录已成功设置为: {cache_dir}")
# --- 缓存目录设置结束 ---


from sentence_transformers import SentenceTransformer

# --- 0. 系统配置 ---
# 在这里可以方便地修改所有模型和文件路径
CONFIG = {
    # 【修改】请确保这里的知识库文件路径是正确的
    "corpus_path": "hotpot_contexts_deduplicated.pkl", 
    "faiss_index_path": "hotpotqa_faiss_deduplicated.index",
    "embedding_model": "BAAI/bge-m3",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def load_corpus(path):
    """
    加载知识库文件。
    """
    print(f"加载知识库文件: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"知识库文件未找到: {path}。请确保数据路径正确。")
    with open(path, 'rb') as f:
        corpus = pickle.load(f)
    print(f"知识库加载成功，包含 {len(corpus)} 篇文档。")
    return corpus

def build_faiss_index(config):
    """
    构建或加载FAISS索引。
    如果索引文件已存在，则直接加载；否则，创建新索引。
    """
    index_path = config["faiss_index_path"]
    
    if os.path.exists(index_path):
        print(f"检测到已存在的FAISS索引 '{index_path}'，无需重新构建。")
        return

    print("--- 未找到FAISS索引，开始构建新的索引 ---")
    
    # 1. 加载知识库
    corpus_docs = load_corpus(config["corpus_path"])
    
    # 2. 初始化嵌入模型 (它现在会使用您在上面设置的缓存路径)
    print(f"正在加载嵌入模型: {config['embedding_model']}...")
    embedding_model = SentenceTransformer(config["embedding_model"], device=config["device"])
    print("嵌入模型加载完成。")

    # 3. 分块编码文档并构建索引
    # 定义分块大小，以优化内存使用并提供进度反馈
    chunk_size = 10000  # 每块处理10000个文档
    all_embeddings = []
    num_docs = len(corpus_docs)
    num_chunks = (num_docs + chunk_size - 1) // chunk_size

    print(f"知识库共 {num_docs} 篇文档，将分 {num_chunks} 块进行处理。")

    for i in range(0, num_docs, chunk_size):
        chunk_num = (i // chunk_size) + 1
        start_index = i
        end_index = min(i + chunk_size, num_docs)
        
        print(f"\n--- 正在处理第 {chunk_num}/{num_chunks} 块文档 (文档索引 {start_index} 到 {end_index})... ---")
        
        # 原代码:
        # chunk_docs = corpus_docs[start_index:end_index]
        
        # 【修正】: 从字典列表中提取出纯文本内容
        chunk_texts = [doc['context'] for doc in corpus_docs[start_index:end_index]]

        # 对当前块进行编码，tqdm进度条会在此处显示
        chunk_embeddings = embedding_model.encode(
            chunk_texts,  # 【修正】: 传递纯文本列表
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        all_embeddings.append(chunk_embeddings.cpu().numpy())
        # ==================== 在这里添加以下代码 ====================

    print("\n--- 所有文档块处理完毕，开始整合向量并构建FAISS索引 ---")

    # 1. 将所有块的 embeddings 合并成一个大的 numpy 数组
    # np.vstack 会垂直地堆叠数组，正好是我们需要的
    final_embeddings = np.vstack(all_embeddings)

    # 2. 获取向量维度
    embedding_dim = final_embeddings.shape[1]

    # 3. 初始化FAISS索引
    # IndexFlatL2 是最基础的索引，进行精确的暴力L2距离搜索。
    # 对于更高性能的需求，可以考虑使用如 'IVF_SQ8' 等更复杂的索引类型。
    print(f"正在创建FAISS索引 (IndexFlatL2)，向量维度为: {embedding_dim}")
    faiss_index = faiss.IndexFlatL2(embedding_dim)

    # 4. 将向量添加到索引中
    print(f"正在将 {final_embeddings.shape[0]} 个向量添加到索引中...")
    faiss_index.add(final_embeddings)

    # 5. 保存索引到文件
    index_path = config["faiss_index_path"]
    print(f"正在将索引保存到: {index_path}")
    faiss.write_index(faiss_index, index_path)

    print(f"--- FAISS索引构建并保存成功！'{index_path}' 已生成。---")
    # ============================================================

# --- 主执行入口 ---
if __name__ == '__main__':
    # 确保运行此脚本前，CONFIG中的 "corpus_path" 指向您正确的知识库文件
    # 脚本执行后，将在同一目录下生成 'hotpotqa_faiss.index' 文件
    build_faiss_index(CONFIG)