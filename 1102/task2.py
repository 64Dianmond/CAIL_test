import os
import re
import json
import time

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict, Any, Tuple

load_dotenv()
# --- 1. API 配置 ---
# 请确保您已设置 DASHSCOPE_API_KEY 环境变量
DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置 OPENAI_API_KEY 环境变量")

DASHSCOPE_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL_NAME = os.getenv("OPENAI_MODEL", "qwen3-max")  # 使用 qwen3-max
TEMPERATURE = 0.01
MAX_TOKENS = 8192

# 初始化 API 客户端
try:
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL
    )
except Exception as e:
    print(f"初始化OpenAI客户端失败: {e}")
    exit()


# --- 2. 文件路径配置 ---
class FILE_CONFIG:
    # Task 1 的输出文件 (包含 id 和 answer1)，这是唯一的输入
    INPUT_FILE = './result/submission_qwen3max_with_refusal.jsonl'

    # 最终提交的结果文件
    OUTPUT_FILE = 'submission_final_with_task2_qwen3max.jsonl'


# --- 3. Task 2 提示词模板 (已移除 fact) ---

TASK2_THEFT_PROMPT = """
你是一个专业的刑事法官助手，你的任务是严格遵循《最高人民法院、最高人民检察院关于常见犯罪的量刑指导意见（试行）》的规定，对一个盗窃罪案件进行刑期预测。

你必须按照以下量刑步骤进行：
1.  **确定量刑起点**：根据基本犯罪事实（如数额、情节）在法定刑幅度内确定。
2.  **确定基准刑**：根据其他犯罪事实（如具体数额、次数）调节量刑起点。
3.  **确定宣告刑**：根据所有量刑情节（如自首、累犯、退赔等）调节基准刑，得出最终刑期。

**【重要约束】**
* 你必须输出一个以**月**为单位的刑期**区间** `[l, u]`。
* 为了优化Winkler Score，这个区间的**宽度**（即 u - l）**不得超过12个月**。
* 你的分析过程必须清晰地展示上述三个步骤。
* **你的最终回答必须且只能以 `最终刑期: [l, u]` 结尾，不要有任何多余的文字。**

**【量刑指导意见 - 盗窃罪】**
1.  **量刑起点**：
    (1) 达到数额较大起点或扒窃的，在**一年以下（0-12月）**有期徒刑、拘役幅度内确定量刑起点。
    (2) 达到数额巨大起点或有其他严重情节的，在**三年至四年（36-48月）**有期徒刑幅度内确定量刑起点。
    (3) 达到数额特别巨大起点或有其他特别严重情节的，在**十年至十二年（120-144月）**有期徒刑幅度内确定量刑起点。
2.  **宣告刑调节（常见情节）**：
    * 累犯：增加基准刑的10%-40%。
    * 前科（非累犯）：增加基准刑的10%以下。
    * 自首：可以减少基准刑的40%以下。
    * 坦白：可以减少基准刑的20%以下。
    * 积极赔偿并取得谅解：可以减少基准刑的40%以下。
    * 认罪认罚：可以减少基准刑的30%以下。

---

**【待处理案件】**

**已提取的量刑情节 (answer1):**
{已提取的量刑情节}

---

请开始你的分析和计算，并严格按照格式要求在最后给出最终刑期：

**量刑步骤分析:**
1.  **确定量刑起点**: 
2.  **确定基准刑**: 
3.  **确定宣告刑**: 

**最终刑期: [l, u]**
"""

TASK2_ASSAULT_PROMPT = """
你是一个专业的刑事法官助手，你的任务是严格遵循《最高人民法院、最高人民检察院关于常见犯罪的量刑指导意见（试行）》的规定，对一个故意伤害罪案件进行刑期预测。

你必须按照以下量刑步骤进行：
1.  **确定量刑起点**：根据基本犯罪事实（如伤害等级）在法定刑幅度内确定。
2.  **确定基准刑**：根据其他犯罪事实（如伤残等级、手段）调节量刑起点。
3.  **确定宣告刑**：根据所有量刑情节（如起因、自首、赔偿等）调节基准刑，得出最终刑期。

**【重要约束】**
* 你必须输出一个以**月**为单位的刑期**区间** `[l, u]`。
* 为了优化Winkler Score，这个区间的**宽度**（即 u - l）**不得超过12个月**。
* 你的分析过程必须清晰地展示上述三个步骤。
* **你的最终回答必须且只能以 `最终刑期: [l, u]` 结尾，不要有任何多余的文字。**

**【量刑指导意见 - 故意伤害罪】**
1.  **量刑起点**：
    (1) 故意伤害致一人轻伤的，在**二年以下（0-24月）**有期徒刑、拘役幅度内确定量刑起点。
    (2) 故意伤害致一人重伤的，在**三年至五年（36-60月）**有期徒刑幅度内确定量刑起点。
    (3) 以特别残忍手段故意伤害致一人重伤，造成六级严重残疾的，在**十年至十三年（120-156月）**有期徒刑幅度内确定量刑起点。
2.  **宣告刑调节（常见情节）**：
    * 累犯：增加基准刑的10%-40%。
    * 前科（非累犯）：增加基准刑的10%以下。
    * 自首：可以减少基准刑的40%以下。
    * 坦白：可以减少基准刑的20%以下。
    * 积极赔偿并取得谅解：可以减少基准刑的40%以下。
    * 认罪认罚：可以减少基准刑的30%以下。

---

**【待处理案件】**

**已提取的量刑情节 (answer1):**
{已提取的量刑情节}

---

请开始你的分析和计算，并严格按照格式要求在最后给出最终刑期：

**量刑步骤分析:**
1.  **确定量刑起点**: 
2.  **确定基准刑**: 
3.  **确定宣告刑**: 

**最终刑期: [l, u]**
"""

TASK2_FRAUD_PROMPT = """
你是一个专业的刑事法官助手，你的任务是严格遵循《最高人民法院、最高人民检察院关于常见犯罪的量刑指导意见（试行）》的规定，对一个诈骗罪案件进行刑期预测。

你必须按照以下量刑步骤进行：
1.  **确定量刑起点**：根据基本犯罪事实（如数额、情节）在法定刑幅度内确定。
2.  **确定基准刑**：根据其他犯罪事实（如具体数额）调节量刑起点。
3.  **确定宣告刑**：根据所有量刑情节（如自首、累犯、退赔等）调节基准刑，得出最终刑期。

**【重要约束】**
* 你必须输出一个以**月**为单位的刑期**区间** `[l, u]`。
* 为了优化Winkler Score，这个区间的**宽度**（即 u - l）**不得超过12个月**。
* 你的分析过程必须清晰地展示上述三个步骤。
* **你的最终回答必须且只能以 `最终刑期: [l, u]` 结尾，不要有任何多余的文字。**

**【量刑指导意见 - 诈骗罪】**
1.  **量刑起点**：
    (1) 达到数额较大起点的，在**一年以下（0-12月）**有期徒刑、拘役幅度内确定量刑起点。
    (2) 达到数额巨大起点或有其他严重情节的，在**三年至四年（36-48月）**有期徒刑幅度内确定量刑起点。
    (3) 达到数额特别巨大起点或有其他特别严重情节的，在**十年至十二年（120-144月）**有期徒刑幅度内确定量刑起点。
2.  **宣告刑调节（常见情节）**：
    * 累犯：增加基准刑的10%-40%。
    * 前科（非累犯）：增加基准刑的10%以下。
    * 自首：可以减少基准刑的40%以下。
    * 坦白：可以减少基准刑的20%以下。
    * 积极赔偿并取得谅解：可以减少基准刑的40%以下。
    * 认罪认罚：可以减少基准刑的30%以下。

---

**【待处理案件】**

**已提取的量刑情节 (answer1):**
{已提取的量刑情节}

---

请开始你的分析和计算，并严格按照格式要求在最后给出最终刑期：

**量刑步骤分析:**
1.  **确定量刑起点**: 
2.  **确定基准刑**: 
3.  **确定宣告刑**: 

**最终刑期: [l, u]**
"""


# --- 4. 辅助函数 ---

def parse_model_output(response_text: str, case_id: str) -> List[int]:
    """
    从模型的完整回复中解析出 [l, u] 区间。
    (此函数与上一版相同)
    """
    # 默认的备用区间，以防解析失败
    fallback_range = [6, 18]

    # 1. 尝试用正则表达式精确匹配 "最终刑期: [l, u]"
    match = re.search(r'最终刑期:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)

    if match:
        try:
            l = int(match.group(1))
            u = int(match.group(2))

            # 确保 l <= u
            if l > u:
                l, u = u, l

            # 强制约束：宽度不超过12个月
            if u - l > 12:
                print(f"⚠️ [ID: {case_id}] 警告：模型输出宽度 {u - l} > 12。自动修正为 [{l}, {l + 12}]")
                u = l + 12

            return [l, u]

        except Exception as e:
            print(f"❌ [ID: {case_id}] 错误：解析匹配项失败 {e}。原始文本: {response_text}")
            return fallback_range

    # 2. 如果精确匹配失败，尝试从文本末尾查找最后一个 [...]
    last_bracket_match = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', response_text)
    if last_bracket_match:
        try:
            l, u = map(int, last_bracket_match[-1])

            if l > u:
                l, u = u, l

            if u - l > 12:
                print(f"⚠️ [ID: {case_id}] 警告：(备用解析) 模型输出宽度 {u - l} > 12。自动修正为 [{l}, {l + 12}]")
                u = l + 12

            print(f"✅ [ID: {case_id}] 备用解析成功。")
            return [l, u]
        except Exception:
            pass  # 忽略解析错误，继续使用 fallback

    print(
        f"❌ [ID: {case_id}] 错误：无法从模型输出中解析刑期。使用备用区间 {fallback_range}。原始文本: {response_text[-200:]}...")
    return fallback_range


def predict_sentencing_range(factors: List[str], case_id: str) -> List[int]:
    """
    (已修改) 调用大模型 API 预测刑期区间 (仅使用 factors)。
    """
    if not factors:
        print(f"❌ [ID: {case_id}] 错误：量刑情节列表为空。")
        return [6, 18]  # 备用

    crime_type = factors[0]

    if crime_type == "盗窃罪":
        template = TASK2_THEFT_PROMPT
    elif crime_type == "故意伤害罪":
        template = TASK2_ASSAULT_PROMPT
    elif crime_type == "诈骗罪":
        template = TASK2_FRAUD_PROMPT
    else:
        print(f"⚠️ [ID: {case_id}] 警告：未知的罪名 '{crime_type}'。将使用 '盗窃罪' 模板作为备用。")
        template = TASK2_THEFT_PROMPT  # 默认使用盗窃罪

    # 将列表转换为格式化的字符串
    factors_str = json.dumps(factors, ensure_ascii=False, indent=2)

    # (已修改) 仅格式化 "已提取的量刑情节"
    user_prompt = template.format(已提取的量刑情节=factors_str)

    try:
        response = client.chat.completions.create(
            model=DASHSCOPE_MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的刑事法官助手。"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        result_text = response.choices[0].message.content.strip()

        return parse_model_output(result_text, case_id)

    except Exception as e:
        print(f"❌ [ID: {case_id}] API调用失败: {str(e)}。使用备用区间 [6, 18]。")
        time.sleep(1)  # 避免API调用过快失败
        return [6, 18]


# --- 5. 主执行逻辑 (已修改) ---
def main():
    print("--- 开始执行 Task 2 刑期预测 (仅基于量刑情节) ---")

    # 检查输入文件是否存在
    if not os.path.exists(FILE_CONFIG.INPUT_FILE):
        print(f"❌ 错误：未找到 Task 1 结果文件 '{FILE_CONFIG.INPUT_FILE}'")
        return

    # 1. 计算总行数用于 tqdm
    print(f"正在计算 {FILE_CONFIG.INPUT_FILE} 的行数...")
    try:
        with open(FILE_CONFIG.INPUT_FILE, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"共 {total_lines} 条记录。")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 2. 遍历 Task 1 文件，调用 API 生成 Task 2 结果
    print(f"正在处理输入文件并写入 {FILE_CONFIG.OUTPUT_FILE}...")

    processed_count = 0
    with open(FILE_CONFIG.INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(FILE_CONFIG.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in, total=total_lines, desc="预测刑期"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                case_id = str(data['id'])
                answer1 = data['answer1']

                if not answer1:
                    print(f"⚠️ [ID: {case_id}] 警告：answer1 为空。跳过。")
                    continue

                # 调用 API 预测刑期 (使用修改后的函数)
                answer2 = predict_sentencing_range(answer1, case_id)

                # ==================================================
                # --- 新增的打印行：根据您的要求，打印每条结果 ---
                print(f"✅ [ID: {case_id}] 预测刑期: {answer2}")
                # ==================================================

                # 构建最终输出
                output_record = {
                    "id": data['id'],
                    "answer1": answer1,
                    "answer2": answer2
                }

                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1

            except Exception as e:
                print(f"❌ [ID: {case_id}] 处理行失败: {e}。原始行: {line.strip()}")

    print("\n--- 处理完成 ---")
    print(f"成功处理并写入 {processed_count} / {total_lines} 条记录。")
    print(f"最终提交文件已保存至: {FILE_CONFIG.OUTPUT_FILE}")


if __name__ == "__main__":
    main()
