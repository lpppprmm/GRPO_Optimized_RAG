import json
import re
from collections import Counter

def tokenize(text):
    """
    一个智能分词器，能正确处理单词和各种格式的数字。
    它将文本分解成一个由小写单词和数字组成的集合。
    """
    if not isinstance(text, str):
        return set()
    
    # 1. 提取所有数字（整数和浮点数）
    numbers = set(re.findall(r'\d+\.\d+|\d+', text))
    
    # 2. 从文本中移除数字，为提取单词做准备
    text_without_numbers = re.sub(r'\d+\.\d+|\d+', ' ', text)
    
    # 3. 提取所有剩余的单词，并转为小写
    words = set(re.findall(r'\b\w+\b', text_without_numbers.lower()))
    
    # 4. 合并数字和单词集合
    return numbers.union(words)

def evaluate_single_answer(record):
    """
    使用一套经过优化的通用规则来评估单个答案。
    这个版本逻辑清晰，只有一个返回路径。
    """
    gt = record.get('ground_truth_answer', '').strip()
    gen = record.get('final_generated_answer', '').strip()

    if not gt:
        return "Incorrect", "标准答案为空。"

    # --- 规则1: 完全匹配 (最高优先级) ---
    if gt.lower() == gen.lower():
        return "Correct", "生成答案与标准答案完全匹配。"

    # --- 规则2: 核心信息单元匹配 (核心规则) ---
    # 使用分词器获取两个答案的核心信息单元
    gt_tokens = tokenize(gt)
    gen_tokens = tokenize(gen)
    
    # 如果标准答案的所有核心信息单元都出现在生成答案中，则判定为正确。
    # 这能处理生成答案包含更多上下文，或者标准答案包含非必要词的情况。
    #issubset(): 判断标准答案的核心词集合是否是生成答案核心词集合的子集。
    if gt_tokens and gt_tokens.issubset(gen_tokens):
        return "Correct", "标准答案中的所有核心关键词都出现在生成答案中。"

    # --- 规则3: 部分匹配 (备用规则) ---
    # 仅在核心信息不完全匹配时，检查是否有部分重叠。
    # 例如，只匹配了数字，但文字部分不匹配。
    gt_nums = {token for token in gt_tokens if re.match(r'^\d+\.?\d*$', token)}
    gen_nums = {token for token in gen_tokens if re.match(r'^\d+\.?\d*$', token)}
    
    # 如果标准答案有数字，并且这些数字都出现在生成答案中，但其他词不完全匹配，则算部分正确。
    if gt_nums and gt_nums.issubset(gen_nums):
        return "Partially Correct", f"仅匹配了核心数字 {gt_nums}，但其他关键词不完全匹配。"
        
    # 如果标准答案中没有任何数字，但至少有一个词匹配，也可以算部分正确。
    if not gt_nums and (gt_tokens & gen_tokens): # & 是集合的交集操作
        return "Partially Correct", f"部分关键词匹配: {gt_tokens & gen_tokens}"

    # --- 规则4: 错误 (最低优先级) ---
    # 如果以上所有规则都不满足，则判定为错误。
    return "Incorrect", "生成答案未能匹配标准答案的核心信息。"


def evaluate_rag_and_save_report(input_file_path, output_file_path):
    """
    主函数，用于读取、评估和保存报告。
    (此函数无需修改，保持原样即可)
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {input_file_path} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析文件 {input_file_path} 中的JSON。")
        return

    evaluations = []
    counts = Counter()

    for i, record in enumerate(data):
        record['id'] = record.get('id', i + 1)
        category, reason = evaluate_single_answer(record)
        counts[category] += 1
        evaluations.append({
            "id": record["id"],
            "question": record.get("original_question", "N/A"),
            "ground_truth": record.get("ground_truth_answer", "N/A"),
            "generated_answer": record.get("final_generated_answer", "N/A"),
            "evaluation": category,
            "reason": reason
        })

    # (报告构建和保存部分的代码省略，与之前版本相同)
    report_lines = []
    report_lines.append(f"# 评估报告\n")
    report_lines.append(f"**输入文件**: `{input_file_path}`\n")
    report_lines.append("## 详细评估结果\n")
    for eval_item in evaluations:
        report_lines.append(f"### ID: {eval_item['id']}")
        report_lines.append(f"- **问题**: {eval_item['question']}")
        report_lines.append(f"- **标准答案**: `{eval_item['ground_truth']}`")
        report_lines.append(f"- **生成答案**: {eval_item['generated_answer']}")
        report_lines.append(f"- **评估结果**: **{eval_item['evaluation']}** ({eval_item['reason']})\n")
    report_lines.append("\n---\n\n## 评估摘要\n")
    total = len(data)
    accuracy = (counts["Correct"] / total) * 100 if total > 0 else 0
    partial_accuracy = ((counts["Correct"] + counts["Partially Correct"]) / total) * 100 if total > 0 else 0
    report_lines.append("| 评估类别 | 数量 | 百分比 |")
    report_lines.append("|---|---|---|")
    report_lines.append(f"| 正确 (Correct) | {counts['Correct']}/{total} | {accuracy:.1f}% |")
    report_lines.append(f"| 部分正确 (Partially Correct) | {counts['Partially Correct']}/{total} | {(counts['Partially Correct']/total*100):.1f}% |")
    report_lines.append(f"| 错误 (Incorrect) | {counts['Incorrect']}/{total} | {(counts['Incorrect']/total*100):.1f}% |")
    report_lines.append("\n---")
    report_lines.append(f"\n**严格准确率 (仅含“正确”):** {accuracy:.1f}%")
    report_lines.append(f"**宽松准确率 (包含“正确”和“部分正确”):** {partial_accuracy:.1f}%")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"评估报告已成功保存至文件: '{output_file_path}'")
    except IOError as e:
        print(f"错误：无法写入文件 {output_file_path}。原因: {e}")

# --- 主程序入口 ---
if __name__ == "__main__":
    input_file = 'rag_results_2.json'
    output_file = 'evaluation_report.md'
    
    evaluate_rag_and_save_report(input_file, output_file)