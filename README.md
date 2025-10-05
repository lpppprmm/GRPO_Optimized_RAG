# GRPO-RAG: 基于生成式反馈的RAG端到端优化研究

本项目探索并实现了一个基于生成式反馈的端到端优化框架（灵感源于 **GRPO** 算法），旨在显著提升检索增强生成（RAG）系统在处理需要多步推理的复杂问题时的性能。

研究的核心是解决高级 RAG 系统中的一个关键瓶颈：**如何让作为“规划器”（Planner）的大语言模型（LLM）能深刻理解并完美适应一个高度定制化的私有知识库？**

本项目完整复现了从构建基线 RAG 系统、通过直接偏好优化（DPO）进行初步对齐，到最终利用生成反馈信号实现“全局最优”的完整研究流程。

## 核心贡献

1.  **方法创新**: 首次将前沿的对齐技术 **DPO** 应用于 RAG 系统的规划器（Planner）组件，有效解决了通用 LLM Agent 在特定环境中“水土不服”的问题。
2.  **框架创新**: 设计并实现了一个基于**生成反馈**的 RAG 端到端优化框架。该框架将最终的生成质量（以模型对数概率为奖励信号）直接用于优化前端的规划策略，突破了传统依赖“代理信号”（如文本相似度）的局部优化局限，实现了真正的“全局优化”。

## 项目结构

```
GRPO-Optimized-RAG/
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   └── .gitkeep
│
├── evaluation/
│   ├── run_evaluation_baseline.py
│   ├── run_evaluation_dpo.py
│   ├── run_evaluation_grpo.py
│   └── generate_evaluation_report.py
│
├── evaluation_results/
│   ├── 1_baseline_results.md
│   ├── 2_dpo_finetune_results.md
│   └── 3_grpo_finetune_results.md
│
├── samples/
│   ├── dpo_planner_dataset_sample.json
│   └── grpo_planner_dataset_sample.json
│
└── scripts/
    ├── 1_create_deduplicated_pkl.py
    ├── 2_build_faiss_index.py
    ├── 3_run_baseline_rag.py
    ├── 4_build_dpo_dataset.py
    └── 5_build_grpo_dataset.py
```

## 复现指南

### 步骤 0：项目设置

首先，克隆本仓库并安装所需的 Python 依赖包。

```bash
git clone https://github.com/<Your_Username>/GRPO-Optimized-RAG.git
cd GRPO-Optimized-RAG
pip install -r requirements.txt
```

### 步骤 1：配置环境变量

项目中部分脚本需要调用 API。请将您的密钥配置在环境变量中。

1.  将 `.env.example` 文件复制一份并重命名为 `.env`。
2.  打开 `.env` 文件，填入您自己的 API Key 和 Base URL。

### 步骤 2：数据准备

1.  **下载原始数据**: 请从 [HotpotQA 数据集官网](https://hotpotqa.github.io/) 下载 `hotpot_train_v1.1.json` 文件，并将其放置在 `data/` 目录中。

2.  **创建去重知识库 (PKL)**:
    ```bash
    python scripts/1_create_deduplicated_pkl.py
    ```

3.  **构建 FAISS 索引**:
    ```bash
    python scripts/2_build_faiss_index.py
    ```

### 步骤 3：阶段一 - DPO 初步对齐

1.  **生成 DPO 偏好数据集**: 此阶段使用 ROUGE 分数作为“代理信号”，为规划器生成偏好数据集。
    ```bash
    python scripts/4_build_dpo_dataset.py
    ```
    > 运行此脚本将生成完整的偏好数据集。您可以在 `samples/dpo_planner_dataset_sample.json` 中查看该数据集的格式样本。

2.  **使用 Llama-Factory 微调 Planner**: 本项目使用 [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架进行 DPO 微调。请使用上一步生成的数据集，选择 `Qwen/Qwen2-7B-Instruct` 作为基础模型进行 DPO 训练。

### 步骤 4：阶段二 - GRPO 端到端优化

1.  **生成 GRPO 偏好数据集**: 这是本研究的核心。我们使用上一阶段微调过的 Planner，并基于生成模型给出的**对数概率**作为最终奖励信号，来构建更高质量的偏好数据集。
    ```bash
    # 运行前，请确保脚本 5 中正确配置了您的基础模型和 LoRA 适配器路径
    python scripts/5_build_grpo_dataset.py
    ```
    > 您可以在 `samples/grpo_planner_dataset_sample.json` 中查看此阶段数据集的格式样本。

2.  **第二轮微调**: 再次使用 Llama-Factory，在 DPO 模型的基础上，用 GRPO 数据集进行第二轮 DPO 微调。

### 步骤 5：运行自动化评估

本项目提供了一套完整的脚本，用于自动化评估基线、DPO 和 GRPO 三个阶段的模型性能。`evaluation_results/` 目录中的所有 `.md` 报告都是由这些脚本生成的，确保了结果的完全可复现性。

#### 5.1 生成各阶段模型的原始JSON结果

*   **评估基线模型**:
    ```bash
    python evaluation/run_evaluation_baseline.py
    ```

*   **评估DPO微调模型**:
    ```bash
    python evaluation/run_evaluation_dpo.py
    ```

*   **评估GRPO微调模型**:
    ```bash
    python evaluation/run_evaluation_grpo.py
    ```
> **注意**: 在运行这些脚本前，请确保脚本内部的 `planner_model_path` 指向了正确的模型路径（原始模型、DPO适配器合并后的模型、GRPO适配器合并后的模型）。

#### 5.2 生成最终的 Markdown 评估报告

当您获得了三个 `.json` 格式的原始结果后，运行以下脚本，它会自动处理这些数据，并生成格式化好的 `.md` 评估报告。

```bash
python evaluation/generate_evaluation_report.py
```

## 实验结果与分析

我们在 HotpotQA 数据集的 100 个样本上，对基线、DPO 微调和 GRPO 微调三个阶段的模型进行了评估。完整的评估日志可以在 `evaluation_results/` 目录下查看。

### 核心指标对比

| 模型阶段              | 规划器 (Planner)           | 严格准确率 (Correct) | 宽松准确率 (Correct + Partial) |
| :-------------------- | :------------------------- | :------------------- | :----------------------------- |
| **阶段一: 基线**      | `Qwen2-7B-Instruct` (原始) | 65.0%                | 81.0%                          |
| **阶段二: DPO 对齐**  | `Planner-DPO-v1`           | **73.0%**            | **86.0%**                      |
| **阶段三: GRPO 优化** | `Planner-GRPO-v2`          | 69.0%                | **86.0%**                      |

### 结果解读与局限性分析

1.  **DPO 效果显著**: 从基线到 DPO 阶段，模型的**严格准确率从 65% 大幅提升至 73%**。这有力地证明，使用 ROUGE 分数作为奖励信号进行初步对齐，能够有效让规划器适应知识库的结构，生成更精准的查询路径。

2.  **GRPO 的挑战与未来方向**:
    有趣的是，GRPO 阶段的严格准确率（69%）相比 DPO 略有下降，但宽松准确率保持在了 86% 的高位。我们深入分析后认为，这主要暴露了在**构建高质量 GRPO 偏好数据集时的核心挑战**。
    
    在本次实验中，我们仅使用了 `hotpot_train_v1.1.json` 的一小部分数据，通过 `5_build_grpo_dataset.py` 脚本生成了约6000个候选偏好对，并从中挑选了2000个用于第二轮微调。然而，我们发现这2000个样本中，**仍有大量偏好对的奖励分数（log-probability）差距过小**。
    
    这说明，对于许多问题，Planner 生成的不同查询规划路径，最终导向的上下文对生成正确答案的“贡献度”是相似的。这些低区分度的样本对于模型来说是“噪音”，它们未能提供清晰的优化信号，从而限制了第二轮微调的效果。
    
    **那么，未来可以生成更多偏好对吗？答案是肯定的，这正是最关键的优化方向。** GRPO 框架的理论优势是明确的，但其效果高度依赖于能否生成一个**大规模、高信噪比**的偏好数据集。通过在更大规模的 `hotpot_train_v1.1.json` 数据上运行 `5_build_grpo_dataset.py` 脚本，我们可以将候选池从当前的6000个扩展到数万甚至数十万个。一个更大的候选池，意味着我们可以采用**更严格的筛选策略**（例如，在脚本中提高 `min_preference_gap` 的阈值），大胆地丢弃所有奖励分数相近的“噪音”样本，从而筛选出一个规模可观且信噪比极高的最终数据集。这才是真正释放 GRPO 框架全部潜力的钥匙。

### 定性案例分析

尽管存在数据挑战，GRPO 模型在处理某些复杂问题上依然展现了其独特的优势。以问题 `ID: 5a7da9145542990b8f503a12` 为例：

**问题**: "Imperial Ballroom was a 1982 album released by the singer born whom?" (发布1982年专辑《Imperial Ballroom》的歌手出生时叫什么名字？)

*   **基线模型答案**:
    > `...cannot answer the question using only the given information.` (无法回答)
    > **评估**: **Incorrect**

*   **DPO 微调后答案**:
    > `根据提供的信息，没有直接提到任何艺术家...` (根据提供的信息，无法确定)
    > **评估**: **Incorrect**

*   **GRPO 微调后答案**:
    > `Imperial Ballroom was a 1982 album released by Elvis Costello, who was born Declan Patrick MacManus.`
    > **评估**: **Correct**

**分析**: 我们可以看到，对于这个需要多步推理和信息整合的复杂问题，基线和 DPO 模型都因无法有效规划检索路径而失败。然而，经过 GRPO 端到端优化的模型，成功地规划了正确的检索策略（先找到专辑 -> 再找到歌手 -> 最后找到歌手的原名），并给出了完全正确的答案。这有力地证明了 GRPO 框架在提升高级 RAG 系统解决复杂问题能力上的巨大潜力。

## 技术选型

*   **核心框架**: PyTorch, Hugging Face Transformers
*   **向量检索**: FAISS
*   **规划/生成模型**: Qwen2-7B-Instruct
*   **嵌入模型**: BAAI/bge-m3
*   **重排模型**: BAAI/bge-reranker-large
*   **微调框架**: Llama-Factory

## 许可证

本项目采用 [GNU General Public License v3.0](LICENSE) 开源许可证。