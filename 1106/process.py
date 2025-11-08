import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# --- 全局配置 ---
DATA_PATH = "data/task6.jsonl"
EXTRACTED_INFO_PATH = "extracted_info1.json"

DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen3-max")
TEMPERATURE = 1.0
MAX_TOKENS = 8192

# --- 初始化客户端 ---
client = OpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL
)


def extract_info_with_llm(fact_text: str) -> dict:
    """
    使用大模型提取被告人信息和案情描述
    """
    prompt = f"""请从以下法律文书中提取两部分信息：

1. **被告人信息**：提取以"被告人"开头的段落，包含被告人的姓名、出生日期、民族、文化程度、户籍地、前科、羁押情况等基本信息。可能有多个被告人，请全部提取。

2. **案情描述**：提取以"经审理查明"开头或者“公诉机关指控”以及“XX检察院指控”的完整段落及其后续内容，这部分描述了案件的具体事实经过。

3. 如果"经审理查明"后续内容没有具体的案情描述，请重新将“公诉机关指控”或者以及“XX检察院指控”的完整段落内容设置为案情描述。

请严格按照以下JSON格式返回，不要添加任何其他内容：
{{
    "defendant_info": "这里填写被告人信息的完整文本",
    "case_description": "这里填写案情描述的完整文本，如果没有则为空字符串"
}}

原文如下：
{fact_text}
"""

    try:
        response = client.chat.completions.create(
            model=DASHSCOPE_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的法律文书信息提取助手，擅长从判决书中准确提取关键信息。"},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        result_text = response.choices[0].message.content.strip()

        # 尝试解析JSON
        # 移除可能的markdown代码块标记
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        result = json.loads(result_text)
        return result

    except Exception as e:
        print(f"提取失败: {e}")
        return {
            "defendant_info": "",
            "case_description": ""
        }


def process_data():
    """
    处理数据集，提取被告人信息和案情描述
    """
    results = []

    # 读取数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"开始处理 {len(lines)} 条数据...")

    for line in tqdm(lines):
        data = json.loads(line)
        case_id = data['id']
        fact = data['fact']

        # 使用大模型提取信息
        extracted = extract_info_with_llm(fact)

        result = {
            "id": case_id,
            "defendant_info": extracted["defendant_info"],
            "case_description": extracted["case_description"]
        }

        results.append(result)

    # 保存结果
    with open(EXTRACTED_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"提取完成！结果已保存至 {EXTRACTED_INFO_PATH}")
    return results


def verify_extraction(results: list):
    """
    验证提取结果的质量
    """
    total = len(results)
    has_defendant = sum(1 for r in results if r['defendant_info'].strip())
    has_case_desc = sum(1 for r in results if r['case_description'].strip())

    print("\n=== 提取结果统计 ===")
    print(f"总数据量: {total}")
    print(f"成功提取被告人信息: {has_defendant} ({has_defendant / total * 100:.2f}%)")
    print(f"成功提取案情描述: {has_case_desc} ({has_case_desc / total * 100:.2f}%)")
    print(f"案情描述缺失: {total - has_case_desc} ({(total - has_case_desc) / total * 100:.2f}%)")


if __name__ == "__main__":
    # 执行提取
    results = process_data()

    # 验证结果
    verify_extraction(results)

    # 显示几个样例
    print("\n=== 提取样例 ===")
    for i in range(min(2, len(results))):
        print(f"\n--- 案例 {results[i]['id']} ---")
        print(f"被告人信息: {results[i]['defendant_info'][:200]}...")
        print(f"案情描述: {results[i]['case_description'][:200]}...")
