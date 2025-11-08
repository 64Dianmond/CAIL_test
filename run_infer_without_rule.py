import json
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== 配置信息 ====================
DATA_PATH = "data/task6.jsonl"
OUTPUT_PATH = "result/submission_hybrid_rule_engine.jsonl"  # 新的输出文件

API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TEMPERATURE = 0.05
MAX_TOKENS = 8192

# ==================== Task1: 量刑情节识别 (LLM负责) ====================
# Prompt 保持不变，其设计已经很优秀
TASK1_SYSTEM_PROMPT = """你是专业的刑事司法辅助专家,精通《刑法》和《量刑指导意见》。你的任务是从案件事实中准确抽取**所有影响量刑的情节**。

**核心原则**:
1. 只抽取事实中明确存在的情节,不推测、不遗漏
2. 使用标准化术语,确保格式一致
3. "被抓获归案"不是量刑情节,不要提取
4. 数额、次数、档位等关键信息必须提取

**输出要求**:
- 仅输出JSON数组,每个元素是一个标准化标签
- 不要输出任何解释或分析
"""

TASK1_USER_TEMPLATE = """【标签规范】

**1. 数额类**(必提取,精确到元):
- 盗窃金额既遂XXX元、诈骗金额既遂XXX元
- 故意伤害造成经济损失XXX元

**2. 数额/情节档位**(根据金额或案情判断):
- 盗窃数额较大 / 盗窃数额巨大 / 盗窃数额特别巨大
- 诈骗数额较大 / 诈骗数额巨大 / 诈骗数额特别巨大

**3. 次数与频率**:
- [罪名]次数X次、多次[罪名]
- 示例: 盗窃次数3次、多次盗窃

**4. 特殊入罪/加重情形**:
- 盗窃: 扒窃、入户盗窃、携带凶器盗窃
- 诈骗: 电信网络诈骗、诈骗残疾人/老年人/未成年人财物
- 故意伤害: 持械伤害、伤害残疾人/老年人/未成年人

**5. 从宽情节**(按重要性排序):
- 自首、坦白、重大立功
- 当庭自愿认罪、认罪认罚
- 退赃、退赔、赔偿被害人、取得谅解、赔偿并取得谅解
- 初犯、偶犯

**6. 从重情节**:
- 累犯、前科、一般累犯

**7. 犯罪形态**:
- 未遂、中止、预备
- 主犯、从犯、胁从犯

**8. 伤害后果**:
- 轻伤一级、轻伤二级
- 重伤一级、重伤二级
- 致人死亡

**9. 其他法定情节**:
- 限制刑事责任能力
- 未成年人犯罪

---

【案件事实】
{fact}

---

【输出格式】
仅输出JSON数组,格式如下:
["标签1", "标签2", "标签3", ...]
/nothink"""


# ==================== Task2: 刑期预测 (规则引擎负责) ====================

class RuleEngineSentencer:
    """
    一个基于 sentencing guidelines 的确定性规则引擎。
    接收LLM提取的标签，通过精确计算输出刑期区间。
    """

    def __init__(self):
        # 定义数额阈值 (基于常见司法解释)
        self.THRESHOLDS = {
            "盗窃": {"较大": 3000, "巨大": 100000, "特别巨大": 500000},
            "诈骗": {"较大": 5000, "巨大": 50000, "特别巨大": 500000},
        }
        # 定义量刑起点 (lower, upper) in months
        self.BASE_SENTENCES = {
            "盗窃": {
                "较大": (6, 12), "多次": (6, 12), "入户": (6, 12), "携带凶器": (6, 12), "扒窃": (6, 12),
                "巨大": (36, 48),
                "特别巨大": (120, 144)
            },
            "诈骗": {"较大": (0, 12), "巨大": (36, 48), "特别巨大": (120, 144)},
            "故意伤害": {"轻伤二级": (6, 18), "轻伤一级": (12, 24), "重伤二级": (36, 48), "重伤一级": (48, 60)}
        }
        # 定义调节情节的百分比 (lower, upper)
        self.ADJUSTMENTS = {
            # 从宽
            "自首": (-0.4, -0.2), "坦白": (-0.2, -0.1), "当庭自愿认罪": (-0.1, 0),
            "认罪认罚": (-0.3, -0.1),  # 不与自首坦白等重复评价，逻辑中处理
            "立功": (-0.2, -0.1), "重大立功": (-0.5, -0.2),
            "退赃": (-0.3, -0.1), "退赔": (-0.3, -0.1),
            "取得谅解": (-0.4, -0.2),  # 包含赔偿并谅解
            "未成年人犯罪": (-0.5, -0.1),
            "从犯": (-0.5, -0.2), "未遂": (-0.5, -0.3),
            # 从重
            "累犯": (0.1, 0.4), "前科": (0, 0.1)
        }

    def _parse_labels(self, labels: List[str]) -> Dict[str, Any]:
        """将标签列表解析为结构化字典"""
        parsed = {"crime": None, "amount": 0, "injury_level": None, "factors": set()}

        for label in labels:
            # 提取罪名和金额
            m_theft = re.search(r"盗窃金额.*?([\d\.]+)", label)
            m_fraud = re.search(r"诈骗金额.*?([\d\.]+)", label)
            if m_theft:
                parsed["crime"] = "盗窃"
                parsed["amount"] = float(m_theft.group(1))
            elif m_fraud:
                parsed["crime"] = "诈骗"
                parsed["amount"] = float(m_fraud.group(1))

            # 提取伤害等级
            if "轻伤一级" in label:
                parsed["injury_level"] = "轻伤一级"
            elif "轻伤二级" in label:
                parsed["injury_level"] = "轻伤二级"
            elif "重伤一级" in label:
                parsed["injury_level"] = "重伤一级"
            elif "重伤二级" in label:
                parsed["injury_level"] = "重伤二级"

            # 添加其他情节
            for factor in self.ADJUSTMENTS.keys():
                if factor in label:
                    parsed["factors"].add(factor)

            # 添加特殊盗窃情节
            if "扒窃" in label: parsed["factors"].add("扒窃")
            if "入户盗窃" in label: parsed["factors"].add("入户")
            if "多次盗窃" in label: parsed["factors"].add("多次")

        # 如果没有从金额中识别出罪名，根据伤害等级判断
        if not parsed["crime"] and parsed["injury_level"]:
            parsed["crime"] = "故意伤害"

        # 兜底罪名识别
        if not parsed["crime"]:
            if any("盗窃" in l for l in labels):
                parsed["crime"] = "盗窃"
            elif any("诈骗" in l for l in labels):
                parsed["crime"] = "诈骗"
            elif any("故意伤害" in l for l in labels):
                parsed["crime"] = "故意伤害"

        return parsed

    def calculate_sentence(self, labels: List[str]) -> Tuple[int, int]:
        """核心计算函数"""
        parsed_info = self._parse_labels(labels)
        crime = parsed_info["crime"]

        if not crime:
            return (6, 24)  # 无法识别罪名，返回保守区间

        # 1. 确定基准刑
        base_lower, base_upper = 0, 0

        if crime in ["盗窃", "诈骗"]:
            amount = parsed_info["amount"]
            thresholds = self.THRESHOLDS[crime]
            level = None
            if amount >= thresholds["特别巨大"]:
                level = "特别巨大"
            elif amount >= thresholds["巨大"]:
                level = "巨大"
            elif amount >= thresholds["较大"]:
                level = "较大"

            # 特殊盗窃情节也可能决定量刑起点
            if crime == "盗窃":
                if "扒窃" in parsed_info["factors"] or "入户" in parsed_info["factors"] or "多次" in parsed_info[
                    "factors"]:
                    if level is None: level = "较大"  # 如果数额不够，但有特殊情节，按“较大”处理

            if level:
                base_lower, base_upper = self.BASE_SENTENCES[crime][level]
                # 根据数额在起点基础上增加刑期 (简化版)
                # 每超过标准一倍，增加2-4个月
                increase_times = (amount / thresholds[level]) - 1
                if increase_times > 0:
                    base_lower += increase_times * 2
                    base_upper += increase_times * 4

        elif crime == "故意伤害":
            level = parsed_info["injury_level"]
            if level and level in self.BASE_SENTENCES[crime]:
                base_lower, base_upper = self.BASE_SENTENCES[crime][level]
            else:  # 无法确定伤害等级，给一个轻伤的默认值
                base_lower, base_upper = self.BASE_SENTENCES[crime]["轻伤二级"]

        if base_lower == 0 and base_upper == 0:
            return (6, 24)  # 未能确定基准刑，返回保守区间

        # 2. 应用情节调节
        # 规则：从宽情节不重复评价，取最优的。例如有自首又有坦白，按自首算
        factors = parsed_info["factors"]
        final_factors = set()
        if "自首" in factors:
            final_factors.add("自首")
        elif "坦白" in factors:
            final_factors.add("坦白")

        if "取得谅解" in factors:  # "取得谅解" 通常包含了退赔
            final_factors.add("取得谅解")
        elif "退赔" in factors or "退赃" in factors:
            final_factors.add("退赔")

        # 添加其他不冲突的情节
        for f in ["认罪认罚", "立功", "重大立功", "未成年人犯罪", "从犯", "未遂", "累犯", "前科"]:
            if f in factors:
                final_factors.add(f)

        # 应用调节
        total_adj_lower, total_adj_upper = 0.0, 0.0
        for factor in final_factors:
            adj_range = self.ADJUSTMENTS.get(factor, (0, 0))
            total_adj_lower += adj_range[0]
            total_adj_upper += adj_range[1]

        final_lower = base_lower * (1 + total_adj_lower)
        final_upper = base_upper * (1 + total_adj_upper)

        # 累犯最低增加3个月
        if "累犯" in final_factors:
            final_lower = max(final_lower, base_lower + 3)
            final_upper = max(final_upper, base_upper + 3)

        # 3. 增加司法裁量缓冲区，确保覆盖并最小化区间
        # 对于较短刑期，固定buffer更有效；对于长刑期，百分比buffer更合适
        if final_upper < 36:
            buffer_months = 4
            final_lower = max(0, final_lower - buffer_months)
            final_upper += buffer_months
        else:
            buffer_percent = 0.15  # 15% 的浮动范围
            final_lower = max(0, final_lower * (1 - buffer_percent))
            final_upper = final_upper * (1 + buffer_percent)

        # 确保下限不大于上限，并取整
        lower = int(final_lower)
        upper = int(final_upper)
        if lower > upper:
            lower, upper = upper, lower  # swap

        return max(0, lower), max(lower + 1, upper)  # 确保区间至少为1个月


class LLMDrivenInferencer:
    """混合推理器：LLM提取 + 规则引擎计算"""

    def __init__(self, api_key: str = API_KEY, base_url: str = BASE_URL, model: str = MODEL_NAME):
        print(f"初始化 OpenAI 客户端 (模型: {model})")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.rule_engine = RuleEngineSentencer()  # 初始化规则引擎
        print("✓ 规则引擎已加载\n")

    def extract_labels(self, fact: str, max_retries: int = 3) -> List[str]:
        """Task1: 使用LLM提取量刑情节"""
        user_content = TASK1_USER_TEMPLATE.format(fact=fact)
        messages = [{"role": "system", "content": TASK1_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                )
                content = response.choices[0].message.content.strip()
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                return [line.strip().strip('"-,[] ') for line in content.split('\n')]
            except Exception as e:
                print(f"  Task1 LLM调用失败(尝试{attempt + 1}/{max_retries}): {e}")
        return ["LLM提取失败"]

    def process_dataset(self, data_path: str, output_path: str):
        """批量处理数据集"""
        print(f"开始处理数据集: {data_path}\n")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            print(f"共加载 {len(data)} 条数据\n")
        except FileNotFoundError:
            print(f"错误：找不到数据文件 {data_path}")
            return

        results = []
        for idx, item in enumerate(data, 1):
            print(f"{'=' * 60}")
            print(f"[{idx}/{len(data)}] ID={item['id']}")
            print(f"{'-' * 60}")

            # 步骤1: LLM提取情节
            print("Task1: [LLM] 正在提取量刑情节...")
            labels = self.extract_labels(item['fact'])
            print(f"✓ 提取到 {len(labels)} 个情节: {labels[:5]}...")

            # 步骤2: 规则引擎计算刑期
            print("\nTask2: [Rule Engine] 正在计算刑期区间...")
            lower, upper = self.rule_engine.calculate_sentence(labels)
            print(f"✓ 计算出的区间: [{lower}, {upper}] 月 (约 {lower / 12:.1f} - {upper / 12:.1f} 年)\n")

            results.append({"id": item["id"], "answer1": labels, "answer2": [lower, upper]})

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\n{'=' * 60}")
        print(f"✓ 处理完成! 结果已保存至: {output_path}")
        print(f"{'=' * 60}")


def main():
    print("\n" + "=" * 60)
    print("刑事案件量刑辅助系统 - [混合驱动版: LLM + 规则引擎]")
    print("=" * 60 + "\n")
    inferencer = LLMDrivenInferencer()
    inferencer.process_dataset(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()