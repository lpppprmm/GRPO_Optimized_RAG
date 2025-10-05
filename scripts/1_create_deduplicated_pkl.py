import json
import pickle

# 定义要读取的JSON文件名
json_filename = 'hotpot_train_v1.1.json'
# 定义要保存的PKL文件名
pkl_filename = 'hotpot_contexts_deduplicated.pkl'

try:
    # 打开并读取JSON文件
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备一个列表来存储处理后的上下文数据
    processed_contexts = []
    # 创建一个集合来存储已经见过的上下文，用于去重
    seen_contexts = set()

    # 遍历JSON数据中的每一个字典
    for entry in data:
        # 检查'context'键是否存在于字典中
        if 'context' in entry and isinstance(entry['context'], list):
            # 获取当前字典的唯一标识符，方便后续与supporting_facts对应
            entry_id = entry.get('_id', 'N/A')
            
            # 遍历context中的每一个条目（由标题和句子列表组成）
            for title, sentences in entry['context']:
                # 将标题和所有句子拼接成一个完整的段落
                # 标题和句子之间用换行符分隔，句子之间用空格连接
                full_context = title + "\n" + " ".join(sentences)
                
                # ------ 去重逻辑 ------
                # 只有当这个上下文没有被添加过时，才进行处理
                if full_context not in seen_contexts:
                    # 将新的上下文添加到集合中，标记为已见过
                    seen_contexts.add(full_context)
                    
                    # 将处理后的上下文以及其来源信息存储起来
                    # 这样的结构方便您在RAG系统中检索，并能追溯到原文
                    processed_contexts.append({
                        'id': entry_id,
                        'title': title,
                        'context': full_context
                    })

    # 将处理好的上下文列表保存到PKL文件中
    with open(pkl_filename, 'wb') as f_out:
        pickle.dump(processed_contexts, f_out)

    print(f"成功处理了 {len(data)} 个原始条目。")
    print(f"去重后，提取并处理了 {len(processed_contexts)} 条唯一的上下文内容。")
    print(f"所有处理后的唯一上下文已保存到 '{pkl_filename}' 文件中。")
    print()

except FileNotFoundError:
    print(f"错误：找不到文件 '{json_filename}'。请确保文件名正确并且文件与脚本在同一个目录下。")
except json.JSONDecodeError:
    print(f"错误：文件 '{json_filename}' 的内容不是有效的JSON格式。")
except Exception as e:
    print(f"发生了未知错误: {e}")