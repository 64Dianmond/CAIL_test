# gufe test

## 项目概述

本项目基于大语言模型和本地规则引擎相结合的方式，对刑事案件进行量刑预测。项目通过分析案件事实，自动提取量刑相关情节，并根据《量刑指导意见（试行）》2024版计算刑期区间。

## 目录结构

```
.
├── 1102/                   # 实验版本1（基于不同模型的Task1实现）
├── 1104/                   # 实验版本2（基于相似案例检索的Task2实现）
├── 1106/                   # 实验版本3（基于提取的被告人信息和案情描述的不同模型的完整流程实现，主要缩减了模型的输入长度，qwen3max成绩最佳为：Final_score: 0.420578, Task1_avg_F1: 0.4768, Task2_avg_Winkler: 0.3643，尚未优化提示词 ）
├── data/                   # 数据文件目录
├── rules/                  # 量刑规则引擎
├── evaluation/             # 评估脚本（已弃用）
├── result/                 # 预测结果
├── output/                 # 输出文件
├── prompts/                # 提示词模板（已弃用）
├── run_infer.py            # 主推理脚本（混合模式）
├── run_infer_llm.py        # 纯LLM推理脚本
├── run_infer_without_rule.py # 不使用规则的推理脚本
├── runinfer*.py            # 不同模型的推理脚本
├── requirements.txt        # 依赖包列表
└── README.md              # 项目说明文档
```

## 各版本说明

### 1102版本

基于不同大语言模型实现的Task1量刑情节提取功能：
- `runinfer_qwen3max_task1.py`: 使用通义千问模型的实现
- `runinfer_deepseek_task1.py`: 使用DeepSeek模型的实现
- 增加了对"拒不供述"等特殊量刑情节的识别规则

### 1104版本

基于相似案例检索的Task2刑期预测实现：
- 使用FAISS向量检索技术
- 通过案例相似度匹配进行刑期预测
- `predict_interval_task2.py`: 主要预测脚本
- `build_index_for_task2.py`: 案例索引构建脚本

### 1106版本

完整的端到端量刑预测流程实现：
- 集成不同大语言模型的推理脚本
- 包含完整的数据处理和预测流程
- `predict_with_llm.py`: 使用大语言模型的完整预测流程
- `process.py`: 数据处理脚本
- 包含多个实验结果文件

## 功能介绍

### Task1: 量刑情节提取
使用大语言模型从案件事实中提取影响量刑的关键情节，包括：
- 犯罪金额
- 犯罪次数
- 特殊情节（如自首、坦白、累犯等）
- 犯罪形态（未遂、中止等）
- 特殊规则情节（如拒不供述等）

### Task2: 刑期区间预测
基于提取的情节和本地规则引擎计算刑期预测区间：
- 支持30种常见犯罪类型
- 根据《量刑指导意见》实现规则计算
- 采用"最窄区间策略"优化预测结果
- 支持基于相似案例检索的预测方法

## 安装依赖

```bash
pip install -r requirements.txt
```

## 依赖说明

项目依赖的主要第三方库包括：
- openai: 用于调用各类大语言模型API
- python-dotenv: 用于管理环境变量
- tqdm: 用于显示进度条
- faiss-cpu/gpu: 用于相似案例检索
- numpy: 科学计算基础库
- pandas: 数据处理和分析库
- requests: HTTP请求库
- orjson: 高性能JSON处理库

## 使用方法

### 1. 配置环境变量

创建 `.env` 文件并配置 API 密钥：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_api_base_url
OPENAI_MODEL=your_preferred_model
```

### 2. 运行推理

```bash
# 混合模式推理（LLM+规则引擎）
python run_infer.py

# 纯LLM模式推理
python run_infer_llm.py

#模型计算器工具调用 
python runinfer_qwen3max_function_Call.py

# 不使用规则的推理
python run_infer_without_rule.py

# 1102版本 - 不同模型的Task1实现
python 1102/runinfer_qwen3max_task1.py
python 1102/runinfer_deepseek_task1.py

# 1104版本 - 基于案例检索的Task2实现
python 1104/predict_interval_task2.py

# 1106版本 - 完整流程实现
python 1106/predict_with_llm.py
```


## 核心组件

### 量刑规则引擎 (rules/sentencing_rules.py)

实现了基于《量刑指导意见》的本地规则引擎，支持：
- 30种常见犯罪类型的刑期计算
- 法定量刑情节的识别和处理
- 基准刑计算和调节
- 置信度评估
- 支持数额类、暴力类等多种犯罪类型

### 推理器 (run_infer.py)

混合推理器结合了大语言模型和规则引擎：
- Task1 使用 LLM 提取量刑情节
- Task2 使用本地规则引擎计算刑期区间

## 评估指标

采用 Winkler Score 评估刑期预测准确性：
- 区间覆盖真实值时，得分为区间宽度
- 区间未覆盖真实值时，额外惩罚偏离部分
- 最终得分通过转换函数归一化到 0-1 区间

## 文件说明

- `submission_final.jsonl`: 最终提交结果
- `task6.jsonl`: 测试数据集
- `*.py`: 各类推理和处理脚本
- `evaluation/`: 评估脚本和结果

## 开发团队

本项目由参赛团队开发，用于 CAIL2025 量刑预测任务。

## 许可证

本项目仅供学习和研究使用。
