import json
import os
import re
from typing import List, Tuple, Optional, Dict, Any
# from openai import OpenAI # 移除对openai库的导入
import requests # 导入requests库
from dotenv import load_dotenv

load_dotenv()

# 配置路径
DATA_PATH = "data/task6.jsonl"
# 更新输出文件名以反映版本 (V4 - 规则版)
OUTPUT_PATH = "./result/submission_final_qwen3_rules.jsonl"

# --- New API 配置 ---
NEWAPI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-icjbwcFqx4XoGpA2pFGchvZGeM04DVShSDgZiadkaL1Anqzl")
NEWAPI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://45.150.227.150:32770")
NEWAPI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.5-pro")
NEWAPI_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

TEMPERATURE = 0.1
MAX_TOKENS = 32768
# (T2不再使用LLM, 移除T2_MAX_TOKENS)


# ==================== Task1: 量刑情节提取 (维持原版) ====================
# (经核对, 原T1提示词最符合 answer1 的 list[str] 输出格式, 故保留)
TASK1_SYSTEM_PROMPT = """你是专业的刑事司法辅助专家,精通《刑法》和《量刑指导意见》。你的任务是从案件事实中准确抽取**所有影响量刑的情节**。

**核心原则**:
1. 只抽取事实中明确存在的情节,不推测、不遗漏
2. 使用标准化术语,确保格式一致
3. "被抓获归案"不是量刑情节,不要提取
4. 数额、次数、档位等关键信息必须提取

**输出要求**:
- 仅输出JSON数组,每个元素是一个标准化标签
- 不要输出任何解释或分析"""

TASK1_USER_TEMPLATE = """【标签规范】

**1. 数额类**(必提取,精确到元):
- 盗窃金额既遂XXX元、诈骗金额既遂XXX元、抢劫金额XXX元
- 敲诈勒索金额XXX元、职务侵占金额XXX元、合同诈骗金额XXX元
- 信用卡诈骗金额XXX元、非法吸收公众存款金额XXX元

**2. 数额档位**(根据金额判断):
- [罪名]数额较大 / [罪名]数额巨大 / [罪名]数额特别巨大
- 示例:盗窃数额巨大、诈骗数额特别巨大

**3. 次数与频率**:
- [罪名]次数X次、多次[罪名]、[罪名]X次
- 示例:盗窃次数3次、多次盗窃

**4. 特殊入罪情形**:
- 盗窃:扒窃、入户盗窃、携带凶器盗窃
- 抢劫:入户抢劫、抢劫银行、持枪抢劫、在公共交通工具上抢劫
- 诈骗:电信网络诈骗
- 其他:多名被害人、被害人数X人

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

**8. 其他法定情节**:
- 限制刑事责任能力、完全无刑事责任能力
- 未成年人犯罪
- 致人重伤、致人死亡

**9. 特殊罪名情节**:
- 交通肇事:逃逸、因逃逸致人死亡
- 故意伤害:轻伤、重伤、致人死亡
- 危险驾驶:醉酒驾驶、追逐竞驶
- 毒品犯罪:毒品数量X克、毒品数量较大/大/特别大

---

【案件事实】
{fact}

---

【输出格式】
仅输出JSON数组,格式如下:
["标签1", "标签2", "标签3", ...]

示例:
["盗窃金额既遂105600元", "盗窃数额巨大", "盗窃次数2次", "退赔并取得谅解", "前科"]

请严格按照规范输出:"""

# ==================== Task2: 刑期预测 (V4 - 本地规则引擎) ====================
# (移除所有T2的Prompt)

class HybridInferencer:
    """
    大模型(T1)与规则引擎(T2)混合的量刑推理器
    V4 - 刑期预测完全由本地规则计算
    """

    def __init__(self, api_key: str = NEWAPI_API_KEY, base_url: str = NEWAPI_BASE_URL, model: str = NEWAPI_MODEL_NAME):
        print(f"初始化混合推理引擎 (V4 - T2 Rules Engine)")
        print(f"  Base URL: {base_url}")
        print(f"  Model (T1): {model}")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.chat_completions_url = f"{self.base_url}"

        # 测试连接 (仅T1需要)
        try:
            response_json = self._call_new_api_backend(
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10,
                temperature=0.1
            )
            if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                print(f"✓ API (T1) 连接成功\n")
            else:
                print(f"✗ API (T1) 连接失败: 收到无效响应\n")
        except Exception as e:
            print(f"✗ API (T1) 连接失败: {e}\n")

    def _call_new_api_backend(self, messages: List[dict], max_tokens: int, temperature: float, top_p: float = 0.95) -> Optional[dict]:
        """封装对New API服务器的HTTP POST请求 (仅用于 T1)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }

        try:
            response = requests.post(self.chat_completions_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"  HTTP错误: {http_err}, 响应内容: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"  请求发生错误: {req_err}")
        except Exception as e:
            print(f"  发生未知错误: {e}")
        return None

    def _detect_crime_type(self, fact: str) -> str:
        """根据案件事实简单判断核心罪名 (兜底用)"""
        fact_keywords = fact[:1000]

        if "盗窃" in fact_keywords or "扒窃" in fact_keywords:
            return "theft"
        if "诈骗" in fact_keywords:
            return "fraud"
        if "故意伤害" in fact_keywords or "轻伤" in fact_keywords or "重伤" in fact_keywords:
            return "injury"
        if "抢劫" in fact_keywords:
            return "robbery"
        if "交通肇事" in fact_keywords:
            return "traffic"
        return "general"

    def extract_labels(self, fact: str, max_retries: int = 3) -> List[str]:
        """Task1: 提取量刑情节 (使用LLM)"""
        messages = [
            {"role": "system", "content": TASK1_SYSTEM_PROMPT},
            {"role": "user", "content": TASK1_USER_TEMPLATE.format(fact=fact)}
        ]

        for attempt in range(max_retries):
            try:
                response_json = self._call_new_api_backend(
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=0.95
                )

                if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0].get("message", {}).get("content", "").strip()
                    labels = self._parse_json_array(content)

                    if labels:
                        labels = self._post_process_labels(labels, fact)
                        return labels
                else:
                    print(f"  Task1 API调用失败: 收到无效响应 (尝试{attempt + 1}/{max_retries})")

            except Exception as e:
                print(f"  Task1 API调用失败(尝试{attempt + 1}/{max_retries}): {e}")

        return self._fallback_extract(fact)

    # =======================================================================
    # Task 2: 本地规则引擎 (V4)
    # =======================================================================

    def _parse_amount_from_fact(self, fact: str) -> int:
        """兜底: 从事实中提取金额"""
        # 优先匹配 "XXX元"
        amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫|价值).*?([0-9]+(?:\.[0-9]+)?)\s*元', fact)
        if "万元" in fact and amount_match: # 修正: 如果有"万元", 优先匹配万元
             amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫|价值).*?([0-9]+(?:\.[0-9]+)?)\s*万元', fact)

        if not amount_match: # 如果没有 "元", 匹配 "万元"
            amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫|价值).*?([0-9]+(?:\.[0-9]+)?)\s*万元', fact)

        if amount_match:
            try:
                amount = float(amount_match.group(1))
                if "万元" in amount_match.group(0):
                    amount *= 10000
                return int(amount)
            except:
                pass
        return 0

    def _parse_labels(self, labels: List[str]) -> Dict[str, Any]:
        """将T1的标签列表解析为结构化字典"""
        info = {
            "crime_type": "unknown",
            "amount": 0,
            "is_recidivist": False, # 累犯
            "is_surrender": False, # 自首
            "is_confession": False, # 坦白
            "is_pleaded_guilty": False, # 认罪认罚
            "is_restitution": False, # 退赃/退赔
            "is_understanding": False, # 取得谅解
            "is_accomplice": False, # 从犯
            "is_attempt": False, # 未遂
            "is_minor": False, # 未成年
            "special_theft": [], # 扒窃, 入户盗窃, 携带凶器盗窃
            "injury_level": None, # 轻伤, 重伤
        }
        labels_str = " ".join(labels)

        if "盗窃" in labels_str or "扒窃" in labels_str:
            info["crime_type"] = "theft"
        elif "诈骗" in labels_str:
            info["crime_type"] = "fraud"
        elif "故意伤害" in labels_str or "轻伤" in labels_str or "重伤" in labels_str:
            info["crime_type"] = "injury"

        # 解析布尔值
        if "累犯" in labels_str: info["is_recidivist"] = True
        if "自首" in labels_str: info["is_surrender"] = True
        # 坦白和自首互斥
        if "坦白" in labels_str and not info["is_surrender"]: info["is_confession"] = True
        if "认罪认罚" in labels_str: info["is_pleaded_guilty"] = True
        if "退赃" in labels_str or "退赔" in labels_str: info["is_restitution"] = True
        if "取得谅解" in labels_str: info["is_understanding"] = True
        if "从犯" in labels_str: info["is_accomplice"] = True
        if "未遂" in labels_str: info["is_attempt"] = True
        if "未成年" in labels_str: info["is_minor"] = True

        # 解析盗窃专用
        if "扒窃" in labels_str: info["special_theft"].append("扒窃")
        if "入户盗窃" in labels_str: info["special_theft"].append("入户盗窃")
        if "携带凶器盗窃" in labels_str: info["special_theft"].append("携带凶器盗窃")

        # 解析金额
        for label in labels:
            match = re.search(r'(?:金额|数额)[^\d]*(\d+)', label)
            if match:
                info["amount"] = int(match.group(1))
                break

        # 解析伤害
        if "轻伤" in labels_str: info["injury_level"] = "light"
        if "重伤" in labels_str: info["injury_level"] = "heavy"

        return info

    def _calculate_theft_sentence(self, info: Dict[str, Any]) -> Tuple[int, int]:
        """
        T2核心: 盗窃罪刑期计算 (基于 "盗窃罪量刑建议" CoT)
        """
        amount = info["amount"]
        special_count = len(info["special_theft"])

        # 二、量刑起点
        base_range = (0, 0)
        # (贵州标准: 1000较大, 3万巨大, 30万特别巨大)
        if amount >= 300000:
            base_range = (120, 144) # 10-12 年
        elif amount >= 30000:
            base_range = (36, 48) # 3-4 年
        elif amount >= 1000 or special_count > 0:
            base_range = (1, 12) # 1 年以下 (取3-9月) -> (1, 12)
        else:
            return (0, 6) # 兜底

        # 三、调节基准刑 (核心计算)

        # 1. 数额调节 (加法)
        increase_add_lower = 0
        increase_add_upper = 0

        if base_range == (120, 144): # 特别巨大 (每 2万 +1月)
            increase_add_lower = (amount - 300000) / 20000
        elif base_range == (36, 48): # 巨大 (每 5000 +1月)
            increase_add_lower = (amount - 30000) / 5000
        elif base_range == (1, 12): # 较大 (每 1500 +1月)
            increase_add_lower = (amount - 1000) / 1500

        increase_add_upper = increase_add_lower # 数额调节是确定值

        # 2. 特殊情节调节 (加法)
        if special_count > 1:
            # (规则: 每增加一种, +1-3个月)
            increase_add_lower += (special_count - 1) * 1
            increase_add_upper += (special_count - 1) * 3

        # 计算调节后的基准刑 (范围)
        base_sentence_lower = base_range[0] + increase_add_lower
        base_sentence_upper = base_range[1] + increase_add_upper

        # 3. 百分比调节 (从重)
        increase_pct_lower = 0.0
        increase_pct_upper = 0.0

        if info["is_recidivist"]: # 累犯 (+10% to +40%)
            increase_pct_lower += 0.10
            increase_pct_upper += 0.40

        # 4. 百分比调节 (从轻) (累加)
        decrease_pct_lower = 0.0
        decrease_pct_upper = 0.0

        if info["is_surrender"]: # 自首 (规则2: -20% to -40%)
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.40

        if info["is_confession"]: # 坦白 (规则3: -20% 以下)
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.20

        if info["is_understanding"] and info["is_restitution"]: # 赔偿谅解 (规则9.1: -40% 以下)
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.40
        elif info["is_restitution"]: # 仅退赔 (规则9.2: -30% 以下)
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.30
        elif info["is_understanding"]: # 仅谅解 (规则9.3: -20% 以下)
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.20

        if info["is_accomplice"]: # 从犯 (规则6: -20% to -50%)
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.50

        if info["is_attempt"]: # 未遂 (规则4: -20% to -50%)
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.50

        if info["is_minor"]: # 未成年 (规则11.1: -10% to -50%)
            decrease_pct_lower += 0.10
            decrease_pct_upper += 0.50

        if info["is_pleaded_guilty"]: # 认罪认罚 (规则7)
            has_other = info["is_surrender"] or info["is_confession"] or info["is_restitution"]
            limit = 0.60 if has_other else 0.20 # (按盗窃罪规则 7.1/7.2)

            decrease_pct_lower += 0.0 # (认罪认罚至少减 0)
            decrease_pct_upper += limit # (最多减 limit)

        # 四、确定宣告刑
        # (基准刑 * (1 + 累加从重)) * (1 - 累加从轻)

        # 总从重系数
        total_increase_lower = 1 + increase_pct_lower
        total_increase_upper = 1 + increase_pct_upper

        # 总从轻系数 (封顶)
        total_decrease_lower = 1 - min(decrease_pct_upper, 0.9) # 下限 * (1 - 最大折扣)
        total_decrease_upper = 1 - min(decrease_pct_lower, 0.9) # 上限 * (1 - 最小折扣)

        final_lower = (base_sentence_lower * total_increase_lower) * total_decrease_lower
        final_upper = (base_sentence_upper * total_increase_upper) * total_decrease_upper

        # 5. 确保在法定刑档位内 (考虑减轻处罚)
        legal_min, legal_max = 0, 999
        if base_range == (120, 144): # 10-12 年
            legal_min, legal_max = 120, 180 # 法定刑 10-15 年
        elif base_range == (36, 48): # 3-4 年
            legal_min, legal_max = 36, 120 # 法定刑 3-10 年
        elif base_range == (1, 12): # 1 年以下
            legal_min, legal_max = 0, 36 # 法定刑 0-3 年

        # (如果从轻幅度大, 允许 *减轻* 到下一个档位, 所以 legal_min 不强制)
        final_lower = max(0, final_lower) # 下限不能低于0
        final_upper = min(legal_max, final_upper) # 上限不能超过本档位的法定最大值

        # 6. 清理和验证
        if final_lower > final_upper: # (如果下限跑到了上限之上, 说明折扣过大)
            final_lower, final_upper = final_upper, final_lower

        final_lower = max(0, final_lower)
        final_upper = max(final_lower, final_upper) # 确保 upper >= lower

        # (确保最小宽度)
        if final_upper - final_lower < 3:
            final_upper = final_lower + 3

        # (如果累犯, 确保最低刑期)
        if info["is_recidivist"]:
            final_lower = max(final_lower, 3) # (盗窃罪规则 3.3: 累犯一般不少于3个月)
            final_upper = max(final_upper, final_lower + 3)

        return (int(final_lower), int(final_upper))

    def _calculate_general_percentage_sentence(self, info: Dict[str, Any]) -> Tuple[int, int]:
        """
        T2通用: 诈骗罪、故意伤害罪等 (基于 "12.5量刑分析" 百分比)
        """

        # 1. 确定基准刑 (起点) 和 法定刑
        base_mid = 0 # (通用规则以基准刑中点为准)
        legal_min, legal_max = 0, 36 # 默认 0-3 年

        if info["crime_type"] == "fraud":
            amount = info["amount"]
            # (贵州标准: 3000较大, 3万巨大, 50万特别巨大)
            if amount >= 500000: # 特别巨大
                base_mid = 120 # 10 年
                legal_min, legal_max = 120, 180
            elif amount >= 30000: # 巨大
                base_mid = 36 # 3 年
                legal_min, legal_max = 36, 120
            elif amount >= 3000: # 较大
                base_mid = 12 # 1 年
                legal_min, legal_max = 0, 36
            else:
                base_mid = 6
        elif info["crime_type"] == "injury":
            if info["injury_level"] == "heavy":
                base_mid = 48 # 4 年 (重伤 3-10年)
                legal_min, legal_max = 36, 120
            else: # 轻伤
                base_mid = 12 # 1 年 (轻伤 0-3年)
                legal_min, legal_max = 0, 36
        else:
            # 未知罪名, 默认 0-3 年档
            base_mid = 12

        # 2. 累加从重百分比
        increase_pct_lower = 0.0
        increase_pct_upper = 0.0

        if info["is_recidivist"]: # 累犯
            increase_pct_lower += 0.10
            increase_pct_upper += 0.40

        # 3. 累加从轻百分比
        decrease_pct_lower = 0.0
        decrease_pct_upper = 0.0

        if info["is_surrender"]: # 自首 (-20% to -40%)
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.40

        if info["is_confession"]: # 坦白 (-20% 以下)
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.20

        if info["is_understanding"] and info["is_restitution"]: # 赔偿谅解
            decrease_pct_lower += 0.0
            decrease_pct_upper += 0.40
        elif info["is_restitution"]: # 仅退赔
            decrease_pct_lower += 0.10
            decrease_pct_upper += 0.30

        if info["is_accomplice"]: # 从犯
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.50

        if info["is_attempt"]: # 未遂
            decrease_pct_lower += 0.20
            decrease_pct_upper += 0.50

        if info["is_minor"]: # 未成年
            decrease_pct_lower += 0.10
            decrease_pct_upper += 0.50

        if info["is_pleaded_guilty"]: # 认罪认罚
            has_other = info["is_surrender"] or info["is_confession"] or info["is_restitution"]
            limit = 0.60 if has_other else 0.30 # (按 12.5 规则)

            decrease_pct_lower += 0.0
            decrease_pct_upper += limit

        # 4. 计算
        # (基准刑 * (1 + 累加从重)) * (1 - 累加从轻)

        final_lower = (base_mid * (1 + increase_pct_lower)) * (1 - min(decrease_pct_upper, 0.9))
        final_upper = (base_mid * (1 + increase_pct_upper)) * (1 - min(decrease_pct_lower, 0.9))

        # 5. 验证和清理
        if final_lower > final_upper:
            final_lower, final_upper = final_upper, final_lower

        # (通用规则下, 允许减轻处罚, 但上限不超过法定上限)
        final_lower = max(0, final_lower) # (允许低于 legal_min)
        final_upper = min(legal_max, final_upper)

        if final_lower > final_upper:
            final_lower = max(0, final_upper - 12) # 至少 12 个月宽度

        # (确保最小宽度)
        if final_upper - final_lower < 3:
            final_upper = final_lower + 3

        return (int(final_lower), int(final_upper))


    def predict_sentence(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """Task2: 预测刑期区间 (V4 - 本地规则计算)"""

        print("  [Task2] (V4 Rules Engine) 开始本地规则计算...")

        # 1. 解析 T1 (LLM) 的输出
        info = self._parse_labels(labels)

        # 2. 补充解析 (兜底)
        if info["crime_type"] == "unknown":
            info["crime_type"] = self._detect_crime_type(fact)

        if info["amount"] == 0:
            # (如果 T1 没提取到金额, T2 必须从 fact 提取)
            info["amount"] = self._parse_amount_from_fact(fact)

        print(f"  [Task2] 解析情节: 罪名={info['crime_type']}, 金额={info['amount']}, 累犯={info['is_recidivist']}, 自首={info['is_surrender']}, 认罪={info['is_pleaded_guilty']}")

        # 3. 路由到计算器
        if info["crime_type"] == "theft":
            print("  [Task2] 路由: 盗窃罪专用计算器")
            return self._calculate_theft_sentence(info)

        elif info["crime_type"] == "fraud" or info["crime_type"] == "injury":
            print(f"  [Task2] 路由: 通用百分比计算器 (罪名: {info['crime_type']})")
            return self._calculate_general_percentage_sentence(info)

        else:
            print("  [Task2] 路由: 通用兜底百分比计算器")
            # (如果 T1 没识别罪名, 但 T2 识别了, 也用通用)
            if info["crime_type"] == "unknown":
                info["crime_type"] = self._detect_crime_type(fact)
            return self._calculate_general_percentage_sentence(info)


    # =======================================================================
    # T1 的辅助函数 (不变)
    # =======================================================================

    def _parse_json_array(self, text: str) -> List[str]:
        """解析JSON数组"""
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if x]
            except:
                pass
        lines = [line.strip().strip('"-,[]') for line in text.split('\n') if line.strip()]
        filtered_lines = [line for line in lines if len(line) > 2 and not line.startswith(("请输出", "示例", "输出格式"))]
        return filtered_lines

    def _parse_json_interval(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """(V4中不再使用) 解析JSON区间"""
        match = re.search(r'\{\s*"lower"\s*:\s*(\d+)\s*,\s*"upper"\s*:\s*(\d+)\s*\}', text, re.DOTALL | re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
        numbers = re.findall(r'(\d+)', text)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None, None

    def _post_process_labels(self, labels: List[str], fact: str) -> List[str]:
        """后处理:补充T1关键标签"""
        labels_str = "".join(labels)

        if "金额" not in labels_str:
            amount = self._parse_amount_from_fact(fact)
            if amount > 0:
                crime_type = self._detect_crime_type(fact)
                if crime_type == "theft": crime = "盗窃"
                elif crime_type == "fraud": crime = "诈骗"
                else: crime = "犯罪"
                labels.insert(0, f"{crime}金额既遂{int(amount)}元")

        if "多次" in fact and not any("次数" in l for l in labels):
            count_match = re.search(r'(\d+)\s*次', fact)
            if count_match:
                crime = "盗窃" if "盗窃" in fact else "犯罪"
                labels.append(f"{crime}次数{count_match.group(1)}次")

        # (补充伤害等级)
        if ("故意伤害" in fact or "致人" in fact) and "轻伤" not in labels_str and "重伤" not in labels_str:
            if "轻伤" in fact: labels.append("轻伤")
            if "重伤" in fact: labels.append("重伤")

        return labels

    def _fallback_extract(self, fact: str) -> List[str]:
        """兜底:正则提取"""
        labels = []
        amount = self._parse_amount_from_fact(fact)
        if amount > 0:
            crime = "盗窃" if "盗窃" in fact else "诈骗"
            labels.append(f"{crime}金额既遂{int(amount)}元")

        keyword_map = {
            "自首": "自首", "坦白": "坦白", "累犯": "累犯", "前科": "前科",
            "退赔": "退赔", "退赃": "退赃", "谅解": "取得谅解",
            "认罪认罚": "认罪认罚", "未遂": "未遂", "从犯": "从犯",
            "轻伤": "轻伤", "重伤": "重伤", "扒窃": "扒窃", "入户": "入户盗窃"
        }
        for keyword, label in keyword_map.items():
            if keyword in fact:
                labels.append(label)
        return labels if labels else ["信息不足"]

    def _fallback_predict(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """(V4中不再使用, 已被规则引擎替代) 兜底:保守预测"""
        return (6, 12) # 默认6-12个月

    def _validate_interval(self, lower: int, upper: int, labels: List[str]) -> Tuple[int, int]:
        """(V4中不再使用, 逻辑已移入规则引擎)"""
        if lower > upper: lower, upper = upper, lower
        lower = max(0, lower)
        upper = max(0, upper)
        upper = min(240, upper)
        if upper - lower < 1: upper = lower + 3
        return lower, upper

    def process_dataset(self, data_path: str, output_path: str):
        """批量处理"""
        print(f"开始处理数据集: {data_path}\n")

        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        print(f"共 {len(data)} 条数据\n")

        results = []

        for idx, item in enumerate(data, 1):
            print(f"{'=' * 60}")
            print(f"[{idx}/{len(data)}] ID={item['id']}")
            print(f"{'=' * 60}")

            # Task1: 提取量刑情节 (LLM)
            print("Task1: 提取量刑情节 (LLM)...")
            labels = self.extract_labels(item['fact'])
            print(f"✓ 提取到 {len(labels)} 个情节:")
            for label in labels[:10]:
                print(f"  - {label}")
            if len(labels) > 10:
                print(f"  ... (还有 {len(labels) - 10} 个)")

            # Task2: 预测刑期 (Rules Engine)
            print("\nTask2: 预测刑期区间 (Rules Engine)...")
            lower, upper = self.predict_sentence(item['fact'], labels)
            print(f"✓ 预测区间: [{lower}, {upper}] 月")
            print(f"  (约 {lower // 12}年{lower % 12}月 - {upper // 12}年{upper % 12}月)\n")

            results.append({
                "id": item["id"],
                "answer1": labels,
                "answer2": [lower, upper]
            })

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\n{'=' * 60}")
        print(f"✓ 处理完成! 结果保存到: {output_path}")
        print(f"{'=' * 60}")


def main():
    print("\n" + "=" * 60)
    print("刑事案件量刑辅助系统 - 混合驱动版本 (V4 - T2 Rules Engine)")
    print("=" * 60 + "\n")

    inferencer = HybridInferencer()
    inferencer.process_dataset(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
