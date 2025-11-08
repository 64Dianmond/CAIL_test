"""
主推理脚本 - v13 - 基于官方量刑指导意见的全面重构版

更新日志 (v13):
[知识库升级] 引入最高法官方量刑指导意见和详细罪名情节清单
[Task1重构] 使用规范化的情节提取提示词，包含详细的三分类标准
[Task2重构] 严格遵循"量刑起点→基准刑→宣告刑"三步法
[计算器优化] 使用官方调节比例进行精确计算
[v13.1修复] 修复区间倒挂问题,引入LLM自校正机制
"""

import json
import os
import re
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# --- 加载配置 ---
load_dotenv()

# --- 全局配置 ---
DATA_PATH = "data/task6.jsonl"
OUTPUT_PATH = "./result1/submission_v13_Official_Guidelines_235b.jsonl"
DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-235b-a22b-instruct-2507")
TEMPERATURE = 0.01
MAX_TOKENS = 8192

# ==================== [V13-OPTIMIZATION] 官方量刑知识库 ====================
OFFICIAL_SENTENCING_KNOWLEDGE = {
    "盗窃罪": {
        "法定刑幅度": {
            "数额较大": {"金额范围": [2000, 50000], "量刑起点": "一年以下有期徒刑、拘役", "量刑起点月数": [0, 12]},
            "数额巨大": {"金额范围": [50000, 400000], "量刑起点": "三年至四年有期徒刑", "量刑起点月数": [36, 48]},
            "数额特别巨大": {"金额范围": [400000, float('inf')], "量刑起点": "十年至十二年有期徒刑", "量刑起点月数": [120, 144]}
        },
        "构罪情节": [
            "盗窃数额较大", "盗窃数额巨大", "盗窃数额特别巨大",
            "多次盗窃", "入户盗窃", "携带凶器盗窃", "扒窃"
        ],
        "确定基准刑情节": [
            "曾因盗窃受过刑事处罚", "一年内曾因盗窃受过行政处罚",
            "组织、控制未成年人盗窃", "盗窃残疾人/孤寡老人/丧失劳动能力人财物",
            "在医院盗窃病人或其亲友财物", "盗窃救灾/抢险/防汛/优抚/扶贫/移民/救济款物",
            "因盗窃造成严重后果"
        ],
        "从重情节": {
            "累犯": {"调节比例": [0.10, 0.40], "最少增加月数": 3},
            "前科": {"调节比例": [0.0, 0.10]},
            "主犯": {"调节比例": [0.10, 0.30]}
        },
        "从轻情节": {
            "自首": {"调节比例": [0.0, 0.40]},
            "坦白": {"调节比例": [0.0, 0.20]},
            "当庭自愿认罪": {"调节比例": [0.0, 0.10]},
            "立功": {"调节比例": [0.0, 0.20]},
            "重大立功": {"调节比例": [0.20, 0.50]},
            "退赃": {"调节比例": [0.0, 0.30]},
            "积极赔偿并取得谅解": {"调节比例": [0.0, 0.40]},
            "积极赔偿但未取得谅解": {"调节比例": [0.0, 0.30]},
            "未赔偿但取得谅解": {"调节比例": [0.0, 0.20]},
            "从犯": {"调节比例": [0.20, 0.50]},
            "未遂": {"调节比例": [0.0, 0.50]},
            "未成年人(16-18岁)": {"调节比例": [0.10, 0.50]},
            "未成年人(12-16岁)": {"调节比例": [0.30, 0.60]},
            "认罪认罚": {"调节比例": [0.0, 0.30]},
            "羁押期间表现好": {"调节比例": [0.0, 0.10]}
        }
    },
    "诈骗罪": {
        "法定刑幅度": {
            "数额较大": {"金额范围": [5000, 50000], "量刑起点": "一年以下有期徒刑、拘役", "量刑起点月数": [0, 12]},
            "数额巨大": {"金额范围": [50000, 500000], "量刑起点": "三年至四年有期徒刑", "量刑起点月数": [36, 48]},
            "数额特别巨大": {"金额范围": [500000, float('inf')], "量刑起点": "十年至十二年有期徒刑", "量刑起点月数": [120, 144]}
        },
        "构罪情节": [
            "诈骗数额较大", "诈骗数额巨大", "诈骗数额特别巨大",
            "电信网络诈骗", "冒充国家机关工作人员诈骗"
        ],
        "确定基准刑情节": [
            "诈骗对象为老年人/残疾人/未成年人/丧失劳动能力人",
            "被害人因诈骗自杀/死亡/精神失常", "造成被害人家庭重大经济损失",
            "诈骗集团", "组织策划者", "骨干分子"
        ],
        "从重情节": {
            "累犯": {"调节比例": [0.10, 0.40], "最少增加月数": 3},
            "前科": {"调节比例": [0.0, 0.10]},
            "主犯": {"调节比例": [0.10, 0.30]},
            "电信网络诈骗": {"调节比例": [0.10, 0.30]}
        },
        "从轻情节": {
            "自首": {"调节比例": [0.0, 0.40]},
            "坦白": {"调节比例": [0.0, 0.20]},
            "当庭自愿认罪": {"调节比例": [0.0, 0.10]},
            "立功": {"调节比例": [0.0, 0.20]},
            "重大立功": {"调节比例": [0.20, 0.50]},
            "退赃": {"调节比例": [0.0, 0.30]},
            "积极赔偿并取得谅解": {"调节比例": [0.0, 0.40]},
            "积极赔偿但未取得谅解": {"调节比例": [0.0, 0.30]},
            "未赔偿但取得谅解": {"调节比例": [0.0, 0.20]},
            "从犯": {"调节比例": [0.20, 0.50]},
            "未成年人(16-18岁)": {"调节比例": [0.10, 0.50]},
            "认罪认罚": {"调节比例": [0.0, 0.30]}
        }
    },
    "故意伤害罪": {
        "法定刑幅度": {
            "轻伤": {"量刑起点": "二年以下有期徒刑、拘役", "量刑起点月数": [0, 24]},
            "重伤": {"量刑起点": "三年至五年有期徒刑", "量刑起点月数": [36, 60]},
            "重伤造成六级严重残疾": {"量刑起点": "十年至十三年有期徒刑", "量刑起点月数": [120, 156]}
        },
        "构罪情节": [
            "轻伤一级", "轻伤二级", "重伤一级", "重伤二级", "致人死亡"
        ],
        "确定基准刑情节": [
            "使用特别残忍手段", "造成严重残疾", "持械伤害", "伤害多人"
        ],
        "从重情节": {
            "累犯": {"调节比例": [0.10, 0.40], "最少增加月数": 3},
            "前科": {"调节比例": [0.0, 0.10]},
            "主犯": {"调节比例": [0.10, 0.30]}
        },
        "从轻情节": {
            "自首": {"调节比例": [0.0, 0.40]},
            "坦白": {"调节比例": [0.0, 0.20]},
            "当庭自愿认罪": {"调节比例": [0.0, 0.10]},
            "立功": {"调节比例": [0.0, 0.20]},
            "积极赔偿并取得谅解": {"调节比例": [0.0, 0.50]},
            "积极赔偿但未取得谅解": {"调节比例": [0.0, 0.30]},
            "未赔偿但取得谅解": {"调节比例": [0.0, 0.20]},
            "因民间纠纷引发": {"调节比例": [0.10, 0.30]},
            "被害人过错": {"调节比例": [0.10, 0.30]},
            "从犯": {"调节比例": [0.20, 0.50]},
            "未成年人(16-18岁)": {"调节比例": [0.10, 0.50]},
            "认罪认罚": {"调节比例": [0.0, 0.30]}
        }
    }
}

# ==================== [V13-OPTIMIZATION] Task1提示词 - 基于官方情节清单 ====================
TASK1_SYSTEM_PROMPT_V13 = """你是一位专业的刑事司法文书分析专家，需要从案件材料中精确提取量刑情节。

你的任务是从给定的案件文本中识别并提取所有与量刑相关的情节，按照以下三类进行分类：
1. **构罪情节**：决定是否构成犯罪，确定法定刑和量刑起点区间
2. **确定基准刑情节**：作为确定基准刑依据的情节
3. **一般量刑情节**：对既定基准刑进行具体调节的情节"""

TASK1_USER_TEMPLATE_V13 = """
【法律知识参考】
{knowledge_base}

【提取规则】
1. **精确匹配原则**：严格按照法律知识参考中的情节清单提取，不得臆测或扩大解释
2. **完整性原则**：提取所有出现的量刑情节，不遗漏
3. **数额精确原则**：
   - 盗窃罪：必须提取"盗窃金额既遂[X]元"、"盗窃次数[X]次"、"盗窃数额[较大/巨大/特别巨大]"
   - 诈骗罪：必须提取"诈骗金额[X]元"、"诈骗次数[X]次"、"诈骗数额[较大/巨大/特别巨大]"
   - 故意伤害罪：必须提取伤害等级
4. **标准化表述原则**：使用规范的法律术语
   - "当庭自愿认罪" 而非 "认罪态度好"
   - "前科" 而非 "有犯罪前科"
   - "自首" 而非 "主动投案"
5. **多重情节叠加原则**：如果同时存在多个情节，全部列出，注意区分：
   - 自首 vs 坦白 vs 当庭自愿认罪
   - 前科 vs 累犯
   - 主犯 vs 从犯

【Few-Shot示例】
案件事实：被告人王某在广州市白云区盗窃他人苹果手机一部，价值人民币45000元。王某系初犯，到案后坦白，已全部退赃。
提取结果：["盗窃金额既遂45000元", "盗窃数额较大", "盗窃次数1次", "坦白", "退赃", "初犯"]

案件事实：被告人李某盗窃人民币122274.87元。李某曾因犯抢劫罪于2013年被判处有期徒刑五年，2016年假释。因本案被抓获，到案后如实供述,当庭自愿认罪，已退赃。
提取结果：["盗窃金额既遂122274元", "盗窃数额巨大", "盗窃次数1次", "前科", "假释期间再犯罪", "坦白", "当庭自愿认罪", "退赃"]

【案件事实】
{fact}

【输出要求】
严格输出一个JSON格式的字符串数组，不要包含任何其他内容。示例格式：
["情节1", "情节2", "情节3"]
"""

# ==================== [V13-OPTIMIZATION] Task2提示词 - 基于官方量刑步骤 ====================
TASK2_SYSTEM_PROMPT_V13 = """你是一个严格遵循《最高人民法院量刑指导意见》的专业量刑AI。

你的任务是依据官方量刑步骤进行分析：
**第一步**：根据构罪情节确定量刑起点
**第二步**：根据其他犯罪事实确定基准刑
**第三步**：根据量刑情节调节基准刑，确定宣告刑

你必须严格使用官方规定的调节比例，并最终调用工具计算精确刑期。"""

TASK2_USER_TEMPLATE_V13 = """
【官方量刑知识】
{knowledge_base}

【案件已识别情节】
{labels_str}

【分析要求】
请按照以下步骤进行分析，并在分析后调用calculate_sentence_interval工具：

**步骤1：识别罪名和基础事实**
- 罪名是什么？
- 涉案金额或伤害等级是多少？
- 属于哪个法定刑幅度？

**步骤2：分析从重情节**
- 识别所有从重情节（累犯、前科、主犯等）
- 根据官方调节比例，综合确定总的从重比例

**步骤3：分析从轻情节**
- 识别所有从轻情节（自首、坦白、认罪认罚、退赃、谅解等）
- 注意：自首、坦白、当庭自愿认罪不能重复评价
- 注意：认罪认罚与其他从宽情节不能重复评价
- 根据官方调节比例，综合确定总的从轻比例

**步骤4：调用工具**
综合考虑以上分析，调用calculate_sentence_interval工具进行精确计算。

【注意事项】
1. 从重比例通常为0.0-0.4之间（累犯至少增加10%且不少于3个月）
2. 从轻比例通常为0.0-0.6之间（多个从宽情节可累加但不超过60%）
3. 认罪认罚的，已包含坦白、当庭认罪等情节，不重复评价
4. 严格使用官方规定的调节比例范围
"""

# ==================== [V13.1-NEW] 区间自校正提示词 ====================
INTERVAL_CORRECTION_PROMPT = """你是量刑专家,发现了一个严重问题:

【计算结果异常】
- 法定刑范围: {statutory_range}月
- 量刑起点: {sentencing_start}月
- 你之前分析的调节比例: 从重{aggravating}%, 从轻{mitigating}%
- 调节后刑期: {adjusted}月
- **计算出的区间: [{lower}, {upper}]月 ← 下限>上限,不合理!**

【问题原因分析】
当从轻情节过多(如60%)导致刑期大幅降低时,如果强制约束到法定刑范围,可能导致区间倒挂。

【你的任务】
重新审视案件的量刑情节,调整从重/从轻比例,使得:
1. 最终区间合理(下限<上限)
2. 符合案件实际情节
3. 在法定刑范围内或有合理偏离

【案件情节】
{labels_str}

请重新分析并调用calculate_sentence_interval工具,给出合理的区间。

**思考方向:**
- 是否从轻比例过高?多个从宽情节是否重复计算?
- 是否应该降低从轻比例(如从60%降到40%)?
- 或者从重比例是否偏低?
"""

# ==================== [V13-OPTIMIZATION] 精确刑期计算器 ====================
def calculate_sentence_interval_v13(
    crime_type: str,
    base_amount: int = 0,
    injury_level: str = "",
    aggravating_adjustment: float = 0.0,
    mitigating_adjustment: float = 0.0
) -> Dict[str, Any]:
    """
    [V13.2] 修复最低区间为0的问题
    当最低区间为0时，返回特殊标记，让LLM使用默认区间
    """
    knowledge = OFFICIAL_SENTENCING_KNOWLEDGE.get(crime_type)
    if not knowledge:
        return {"final_interval": [6, 12], "error": "Unknown crime type", "needs_correction": False}

    # 步骤1-3: 确定量刑起点和调节（保持原逻辑）
    statutory_lower, statutory_upper = 0, 0
    sentencing_start_point = 0

    if crime_type in ["盗窃罪", "诈骗罪"]:
        tiers = knowledge["法定刑幅度"]
        for level, info in tiers.items():
            if info["金额范围"][0] <= base_amount < info["金额范围"][1]:
                statutory_lower, statutory_upper = info["量刑起点月数"]
                sentencing_start_point = statutory_lower + (statutory_upper - statutory_lower) * 0.3
                break

    elif crime_type == "故意伤害罪":
        levels = knowledge["法定刑幅度"]
        if "轻伤" in injury_level:
            statutory_lower, statutory_upper = levels["轻伤"]["量刑起点月数"]
            sentencing_start_point = (statutory_lower + statutory_upper) * 0.4
        elif "重伤" in injury_level:
            statutory_lower, statutory_upper = levels["重伤"]["量刑起点月数"]
            sentencing_start_point = (statutory_lower + statutory_upper) * 0.5

    base_sentence = sentencing_start_point
    adjusted_sentence = base_sentence * (1.0 + aggravating_adjustment) * (1.0 - mitigating_adjustment)

    # 步骤4: 生成预测区间
    final_lower = int(adjusted_sentence - 3)
    final_upper = int(adjusted_sentence + 3)

    # ========== [关键修改] 检测最低区间为0的情况 ==========
    if final_lower <= 0:
        return {
            "final_interval": [0, 0],  # 特殊标记
            "use_llm_default": True,  # 标记需要使用LLM默认区间
            "sentencing_start_point": round(sentencing_start_point, 1),
            "adjusted_sentence": round(adjusted_sentence, 1),
            "statutory_range": [statutory_lower, statutory_upper],
            "needs_correction": False,
            "message": "最低区间为0，应使用模型默认区间"
        }

    # 步骤5-9: 正常的区间计算和校正逻辑（保持原代码）
    final_lower = max(0, final_lower)
    final_upper = max(final_lower + 2, final_upper)

    needs_correction = False
    if adjusted_sentence < statutory_lower * 0.5 or final_lower >= final_upper:
        needs_correction = True

    if needs_correction:
        return {
            "final_interval": [final_lower, final_upper],
            "sentencing_start_point": round(sentencing_start_point, 1),
            "base_sentence": round(base_sentence, 1),
            "adjusted_sentence": round(adjusted_sentence, 1),
            "statutory_range": [statutory_lower, statutory_upper],
            "needs_correction": True,
            "error": f"区间不合理: [{final_lower}, {final_upper}], 需要重新分析量刑情节"
        }

    if adjusted_sentence < statutory_lower * 0.7:
        final_lower = int(statutory_lower * 0.7)
        final_upper = int(statutory_lower * 0.9)
    elif adjusted_sentence > statutory_upper * 1.3:
        final_lower = int(statutory_upper * 1.1)
        final_upper = int(statutory_upper * 1.3)

    if final_upper - final_lower > 8:
        mid = (final_lower + final_upper) // 2
        final_lower = mid - 4
        final_upper = mid + 4

    if final_lower >= final_upper:
        final_upper = final_lower + 3

    return {
        "final_interval": [final_lower, final_upper],
        "sentencing_start_point": round(sentencing_start_point, 1),
        "base_sentence": round(base_sentence, 1),
        "adjusted_sentence": round(adjusted_sentence, 1),
        "statutory_range": [statutory_lower, statutory_upper],
        "needs_correction": False
    }



# 工具规范
SENTENCE_CALCULATOR_TOOL_V13 = {
    "type": "function",
    "function": {
        "name": "calculate_sentence_interval",
        "description": "根据罪名、基准事实和你分析的调节比例，严格按照官方量刑步骤计算最终刑期区间（单位：月）",
        "parameters": {
            "type": "object",
            "properties": {
                "crime_type": {"type": "string", "enum": ["盗窃罪", "诈骗罪", "故意伤害罪"]},
                "base_amount": {"type": "integer", "description": "涉案金额（元），用于盗窃罪和诈骗罪"},
                "injury_level": {"type": "string", "enum": ["轻伤二级", "轻伤一级", "重伤二级", "重伤一级", "致人死亡", ""], "description": "伤害等级，用于故意伤害罪"},
                "aggravating_adjustment": {"type": "number", "description": "综合所有从重情节的总调节比例（0-0.4，如累犯+前科可能是0.35）"},
                "mitigating_adjustment": {"type": "number", "description": "综合所有从轻情节的总调节比例（0-0.6，如自首+退赃+谅解可能是0.5）"}
            },
            "required": ["crime_type", "aggravating_adjustment", "mitigating_adjustment"]
        }
    }
}

# ==================== [V13-OPTIMIZATION] 主推理类 ====================
class LLMDrivenInferencer_v13:
    """v13: 基于官方量刑指导意见的全面重构版"""

    def __init__(self, api_key: str = DASHSCOPE_API_KEY, base_url: str = DASHSCOPE_BASE_URL, model: str = DASHSCOPE_MODEL_NAME):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        print(f"初始化客户端 (v13.1 - 区间自校正版 - {model})")

    def _call_llm_api(self, messages: List[dict], tools: List[dict] = None) -> Optional[Any]:
        try:
            kwargs = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
            if tools:
                kwargs["tools"] = tools
            completion = self.client.chat.completions.create(**kwargs)
            return completion.choices[0] if completion.choices else None
        except Exception as e:
            print(f"  API 调用失败: {e}")
            return None

    def _parse_json_response(self, text: str) -> Optional[List]:
        # 尝试多种解析方式
        # 方式1：直接查找JSON数组
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

        # 方式2：查找被代码块包裹的JSON
        match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        return None

    def _detect_crime_type(self, fact: str) -> str:
        """检测罪名"""
        if "故意伤害" in fact or "轻伤" in fact or "重伤" in fact:
            return "故意伤害罪"
        elif "诈骗" in fact:
            return "诈骗罪"
        else:
            return "盗窃罪"

    def _get_knowledge_for_task1(self, crime_type: str) -> str:
        """为Task1准备精简的知识"""
        knowledge = OFFICIAL_SENTENCING_KNOWLEDGE.get(crime_type, {})
        simplified = {
            "构罪情节示例": knowledge.get("构罪情节", []),
            "确定基准刑情节示例": knowledge.get("确定基准刑情节", []),
            "从重情节示例": list(knowledge.get("从重情节", {}).keys()),
            "从轻情节示例": list(knowledge.get("从轻情节", {}).keys())
        }
        return json.dumps(simplified, ensure_ascii=False, indent=2)

    def _get_knowledge_for_task2(self, crime_type: str) -> str:
        """为Task2准备详细的知识"""
        knowledge = OFFICIAL_SENTENCING_KNOWLEDGE.get(crime_type, {})
        return json.dumps(knowledge, ensure_ascii=False, indent=2)

    def extract_labels(self, fact: str, crime_type: str, max_retries: int = 3) -> List[str]:
        """Task1: 提取量刑情节"""
        knowledge = self._get_knowledge_for_task1(crime_type)
        user_prompt = TASK1_USER_TEMPLATE_V13.format(knowledge_base=knowledge, fact=fact)
        messages = [
            {"role": "system", "content": TASK1_SYSTEM_PROMPT_V13},
            {"role": "user", "content": user_prompt}
        ]
        for attempt in range(max_retries):
            choice = self._call_llm_api(messages)
            if not choice:
                continue

            response_text = choice.message.content
            labels = self._parse_json_response(response_text)

            if labels and isinstance(labels, list):
                print(f"  ✓ Task1成功提取情节: {labels}")
                return labels
            else:
                print(f"  ✗ Task1解析失败(尝试{attempt + 1}/{max_retries})")

        print("  ⚠ Task1失败，返回空列表")
        return []

    def predict_interval(self, fact: str, labels: List[str], crime_type: str, max_retries: int = 3) -> Tuple[
        int, int]:
        """Task2: 预测量刑区间（含自校正机制）"""
        knowledge = self._get_knowledge_for_task2(crime_type)
        labels_str = ", ".join(labels)

        user_prompt = TASK2_USER_TEMPLATE_V13.format(
            knowledge_base=knowledge,
            labels_str=labels_str
        )

        messages = [
            {"role": "system", "content": TASK2_SYSTEM_PROMPT_V13},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            choice = self._call_llm_api(messages, tools=[SENTENCE_CALCULATOR_TOOL_V13])
            if not choice:
                continue

            # 检查是否有工具调用
            if choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)

                # 调用计算器
                result = calculate_sentence_interval_v13(**args)

                if result.get("use_llm_default", False):
                    print(f"  ⚠ 检测到最低区间为0，使用模型默认区间")

                    # 不使用工具计算，直接让LLM给出区间
                    fallback_prompt = f"""
                刚才的工具计算显示最低刑期为0个月，这不符合法律规定。

                请你根据案件情节直接给出合理的量刑区间（单位：月）。

                案件情节：{labels_str}

                要求：
                1. 直接给出区间，格式：下限,上限（如：3,9）
                2. 下限不能为0，区间宽度小于12个月
                3. 符合{crime_type}的量刑规律

                请直接输出区间，不要解释。
                """
                    fallback_messages = [
                        {"role": "system", "content": "你是量刑专家，直接给出合理区间。"},
                        {"role": "user", "content": fallback_prompt}
                    ]

                    fallback_choice = self._call_llm_api(fallback_messages)
                    if fallback_choice:
                        response_text = fallback_choice.message.content.strip()
                        # 解析 "X,Y" 格式
                        match = re.search(r'(\d+)\s*,\s*(\d+)', response_text)
                        if match:
                            lower, upper = int(match.group(1)), int(match.group(2))
                            if lower > 0 and lower < upper:
                                print(f"  ✓ 使用LLM默认区间: [{lower}, {upper}]")
                                return lower, upper

                    # 兜底：使用保守区间
                    print(f"  ⚠ LLM未给出有效区间，使用保守兜底")
                    return 3, 9

                # [V13.1-KEY] 检查是否需要自校正
                if result.get("needs_correction", False):
                    print(f"  ⚠ 检测到区间异常，触发LLM自校正...")

                    # 构建校正提示
                    correction_prompt = INTERVAL_CORRECTION_PROMPT.format(
                        statutory_range=result["statutory_range"],
                        sentencing_start=result["sentencing_start_point"],
                        aggravating=args.get("aggravating_adjustment", 0) * 100,
                        mitigating=args.get("mitigating_adjustment", 0) * 100,
                        adjusted=result["adjusted_sentence"],
                        lower=result["final_interval"][0],
                        upper=result["final_interval"][1],
                        labels_str=labels_str
                    )

                    # 让LLM重新分析
                    correction_messages = [
                        {"role": "system", "content": TASK2_SYSTEM_PROMPT_V13},
                        {"role": "user", "content": correction_prompt}
                    ]

                    correction_choice = self._call_llm_api(correction_messages,
                                                           tools=[SENTENCE_CALCULATOR_TOOL_V13])

                    if correction_choice and correction_choice.message.tool_calls:
                        corrected_args = json.loads(correction_choice.message.tool_calls[0].function.arguments)
                        corrected_result = calculate_sentence_interval_v13(**corrected_args)

                        # 如果校正后仍不合理，使用保守兜底
                        if corrected_result.get("needs_correction", False):
                            print(f"  ⚠ 二次校正失败，使用保守兜底策略")
                            lower, upper = corrected_result["final_interval"]
                            if lower >= upper:
                                # 强制修正：以法定刑中位数为基准
                                statutory_lower, statutory_upper = corrected_result["statutory_range"]
                                mid = (statutory_lower + statutory_upper) // 2
                                lower = mid - 3
                                upper = mid + 3
                            return lower, upper
                        else:
                            print(f"  ✓ 自校正成功: {corrected_result['final_interval']}")
                            return tuple(corrected_result["final_interval"])

                # 正常情况：直接返回
                print(f"  ✓ Task2计算成功: {result['final_interval']}")
                return tuple(result["final_interval"])

            else:
                print(f"  ✗ Task2未调用工具(尝试{attempt + 1}/{max_retries})")

        # 兜底策略
        print("  ⚠ Task2失败，使用默认区间 [6, 12]")
        return 6, 12

    def infer_single_case(self, case_id: str, fact: str) -> dict:
        """推理单个案件"""
        print(f"\n{'=' * 60}")
        print(f"案件ID: {case_id}")
        print(f"事实摘要: {fact[:100]}...")

        # 步骤1: 检测罪名
        crime_type = self._detect_crime_type(fact)
        print(f"检测罪名: {crime_type}")

        # 步骤2: 提取情节
        labels = self.extract_labels(fact, crime_type)

        # 步骤3: 预测区间
        lower, upper = self.predict_interval(fact, labels, crime_type)

        return {
            "id": case_id,
            "crime": crime_type,
            "labels": labels,
            "term_of_imprisonment": {"imprisonment": f"{lower},{upper}"}
        }

# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("v13.1 - 基于官方量刑指导意见 + 区间自校正")
    print("=" * 60)

    # 初始化推理器
    inferencer = LLMDrivenInferencer_v13()

    # 读取数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        cases = [json.loads(line) for line in f]

    print(f"共加载 {len(cases)} 个案件\n")

    # 批量推理
    results = []
    for case in cases:
        result = inferencer.infer_single_case(case["id"], case["fact"])
        results.append(result)

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\n{'=' * 60}")
    print(f"推理完成！结果已保存至: {OUTPUT_PATH}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
