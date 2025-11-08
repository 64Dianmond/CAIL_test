"""
主推理脚本 - v4 - 规则计算版 (适配 qwen3-max)

更新日志 (v4):
- 遵照用户要求, Task2 (刑期预测) 完全重构为硬编码规则计算 (Rule-Based)
- 移除了 Task2 的所有 LLM Prompts 和 API 调用
- 新增 `_calculate_theft_sentence` (基于"盗窃罪量刑建议" - 增减月份法)
- 新增 `_calculate_general_sentence` (基于"12.5量刑分析" - 百分比调节法)
- `predict_sentence` 重写为规则分发器
- `_detect_crime_type` 优化以匹配数据集罪名 (盗窃, 诈骗, 伤害)
- Task1 (情节提取) 保持不变, 仍由 LLM (qwen3-max) 驱动
"""

import json
import os
import re
from typing import List, Tuple, Optional
from openai import OpenAI  # 导入OpenAI库
from dotenv import load_dotenv

load_dotenv()

# 配置路径
DATA_PATH = "data/task6.jsonl"
# 更新输出文件名以反映版本
OUTPUT_PATH = "./result/submission_final_qwen3_v4_rules_1.jsonl"

# --- DashScope API 配置 ---
DASHSCOPE_API_KEY = "sk-cfeee03df3e14e238a528b50d0d978e9"
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL_NAME = os.getenv("DASHSCOPE_MODEL", "qwen3-max")  # 设置为 qwen3-max

TEMPERATURE = 0.1
MAX_TOKENS = 32768
# MAX_TOKENS_TASK2 = 8192  # Task2 不再需要

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

# ==================== Task2: 刑期预测 (V4 - 规则计算版) ====================
# (Task2 Prompts 已移除, 将使用下面的硬编码规则)


class LLMDrivenInferencer:
    """大模型驱动(Task1) + 规则驱动(Task2)的量刑推理器"""

    def __init__(self, api_key: str = DASHSCOPE_API_KEY, base_url: str = DASHSCOPE_BASE_URL,
                 model: str = DASHSCOPE_MODEL_NAME):
        # 更新版本号
        print(f"初始化 DashScope 客户端 (v4 - 规则计算版 - {model})")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {model}")

        self.model = model
        # 初始化 OpenAI 客户端 (Task1 需要)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        if api_key and len(api_key) > 8:
            print(f"  API Key: {api_key[:4]}...{api_key[-4:]}")
        else:
            print("  API Key: (未提供或过短,请检查 .env 文件或环境变量)")

        # 测试连接 (Task1 需要)
        try:
            response_json = self._call_new_api_backend(
                messages=[{"role": "user", "content": "测试连接"}],
                max_tokens=10,
                temperature=0.1
            )
            if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                print(f"✓ API连接成功\n")
            else:
                print(f"✗ API连接失败: 收到无效响应\n")
        except Exception as e:
            print(f"✗ API连接失败: {e}\n")

    def _call_new_api_backend(self, messages: List[dict], max_tokens: int, temperature: float, top_p: float = 0.95) -> \
    Optional[dict]:
        """封装对DashScope API的HTTP POST请求 (Task1 使用)"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )
            return completion.model_dump()

        except Exception as e:
            print(f"  DashScope API (OpenAI lib) 调用失败: {e}")
            if hasattr(e, 'response'):
                try:
                    print(f"  HTTP错误: {e.status_code}, 响应内容: {e.response}")
                except:
                    print(f"  错误详情: {e}")
            elif hasattr(e, 'status_code'):
                print(f"  HTTP错误: {e.status_code}, 响应体: {e.body}")

            return None

    def _detect_crime_type(self, fact: str) -> str:
        """根据案件事实简单判断核心罪名 (适配竞赛数据集)"""
        fact_keywords = fact[:8000]  # 只检查开头, 提高效率

        # 优先判断盗窃
        if "盗窃" in fact_keywords or "扒窃" in fact_keywords:
            return "theft"
        # 其次判断诈骗
        if "诈骗" in fact_keywords:
            return "fraud"
        # 最后判断伤害 (基于用户提示)
        if "故意伤害" in fact_keywords or "轻伤" in fact_keywords:
            return "injury"

        # 兜底为通用
        print(f"  [Detect] 未识别到 '盗窃/诈骗/伤害', 降级到 general")
        return "general"

    def extract_labels(self, fact: str, max_retries: int = 3) -> List[str]:
        """Task1: 提取量刑情节 (使用优化版T1提示词)"""
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

    def predict_sentence(self, fact: str, labels: List[str], max_retries: int = 3) -> Tuple[int, int]:
        """Task2: 预测刑期区间 (V4 - 硬编码规则版)"""
        try:
            # 1. 解析结构化信息 (罪名, 金额, 次数等)
            details = self._get_crime_details(fact, labels)
            crime_type = details["crime_type"]

            print(f"  [Task2-Rules] 检测到罪名: {crime_type}")

            # 2. 动态选择计算器
            if crime_type == "theft":
                lower, upper = self._calculate_theft_sentence(details, labels)
            elif crime_type in ["fraud", "injury", "general"]: # 'general' 也使用通用百分比法
                lower, upper = self._calculate_general_sentence(details, labels)
            else:
                # 兜底使用通用百分比法
                print(f"  [Task2-Rules] 未知罪名 '{crime_type}', 降级到通用百分比计算")
                lower, upper = self._calculate_general_sentence(details, labels)

            # 3. 验证结果
            lower, upper = self._validate_interval(lower, upper, labels)
            print(f"  [Task2-Rules] 规则计算结果: [{int(lower)}, {int(upper)}]")
            return int(lower), int(upper)

        except Exception as e:
            print(f"  Task2 规则计算失败: {e}")
            # 兜底:保守估计
            return self._fallback_predict_static(fact, labels)

    def _get_crime_details(self, fact: str, labels: List[str]) -> dict:
        """(V4) 辅助函数: 解析结构化犯罪信息"""
        details = {
            "crime_type": self._detect_crime_type(fact),
            "amount": 0.0,
            "count": 1
        }

        # 1. 从标签解析
        for label in labels:
            if "金额既遂" in label:
                match = re.search(r'([0-9]+(?:\.[0-9]+)?)', label)
                if match:
                    details["amount"] = float(match.group(1))
            elif "次数" in label:
                match = re.search(r'([0-9]+)', label)
                if match:
                    details["count"] = int(match.group(1))

        # 2. 如果标签里没有, 再从fact里找一遍 (兜底)
        if details["amount"] == 0:
            amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
            if amount_match:
                amount = float(amount_match.group(1))
                if "万元" in amount_match.group(0):
                    amount *= 10000
                details["amount"] = amount

        return details

    def _calculate_theft_sentence(self, details: dict, labels: List[str]) -> Tuple[int, int]:
        """(V4) 盗窃罪刑期计算 (基于 "盗窃罪量刑建议" - 增减月份法)"""
        amount = details["amount"]
        labels_str = " ".join(labels)

        # --- 1. 确定量刑起点 (base_low, base_mid, base_high) ---
        base_low, base_mid, base_high = 0, 0, 0
        statutory_max = 36 # 默认3年以下

        has_special_circumstance = any(s in labels_str for s in ["入户盗窃", "携带凶器盗窃", "扒窃"]) or details["count"] >= 3

        if amount >= 300000:
            base_low, base_mid, base_high = 120, 132, 144 # 10-12年
            statutory_max = 180
        elif amount >= 30000:
            base_low, base_mid, base_high = 36, 42, 48 # 3-4年
            statutory_max = 120
        elif amount >= 1000 or has_special_circumstance:
            # 规则: "在一年以下有期徒刑、拘役幅度内" (1-12月)
            # 我们取 3-9 月作为中间范围
            base_low, base_mid, base_high = 3, 6, 9
            statutory_max = 36
        else:
            return 0, 6 # 不够罪或显著轻微

        # --- 2. 调节基准刑 ---

        # 2a. 月份调节 (adj_m_low, adj_m_mid, adj_m_high)
        adj_m_low, adj_m_mid, adj_m_high = 0, 0, 0

        if amount >= 300000: # 特别巨大: 每增加 2万元, +1个月
            extra_amount = amount - 300000
            adj_m = round(extra_amount / 20000)
            adj_m_low += adj_m; adj_m_mid += adj_m; adj_m_high += adj_m
        elif amount >= 30000: # 巨大: 每增加 5000 元, +1个月
            extra_amount = amount - 30000
            adj_m = round(extra_amount / 5000)
            adj_m_low += adj_m; adj_m_mid += adj_m; adj_m_high += adj_m
        elif amount >= 1000: # 较大: 每增加 1500 元, +1个月
            extra_amount = amount - 1000
            adj_m = round(extra_amount / 1500)
            adj_m_low += adj_m; adj_m_mid += adj_m; adj_m_high += adj_m

        # 特殊情形: "两种以上情形的, 每增加一种情形, +1至3个月"
        special_count = sum(1 for s in ["入户盗窃", "携带凶器盗窃", "扒窃"] if s in labels_str)
        if details["count"] >= 3:
            special_count += 1

        if special_count > 1:
            adj_m_low += (special_count - 1) * 1
            adj_m_mid += (special_count - 1) * 2
            adj_m_high += (special_count - 1) * 3

        # 2b. 百分比调节 (percent_heavy_*, percent_light_*)
        percent_heavy_low, percent_heavy_mid, percent_heavy_high = 0.0, 0.0, 0.0
        percent_light_low, percent_light_mid, percent_light_high = 0.0, 0.0, 0.0

        # (从重)
        if "累犯" in labels_str:
            percent_heavy_low += 0.10; percent_heavy_mid += 0.25; percent_heavy_high += 0.40 # +10%-40%
        elif "前科" in labels_str: # 累犯和前科不叠加
             percent_heavy_low += 0.0; percent_heavy_mid += 0.10; percent_heavy_high += 0.20 # +20% (盗窃罪规则)

        # (从轻)
        if "自首" in labels_str:
            # -30%以下 (0-30)
            percent_light_low += 0.0; percent_light_mid += 0.15; percent_light_high += 0.30
        elif "坦白" in labels_str: # 自首坦白不叠加
            # -20%以下 (0-20)
            percent_light_low += 0.0; percent_light_mid += 0.10; percent_light_high += 0.20

        if "认罪认罚" in labels_str:
             # -20%以下 (0-20)
            percent_light_low += 0.0; percent_light_mid += 0.10; percent_light_high += 0.20

        if "退赃" in labels_str or "退赔" in labels_str:
            if "谅解" in labels_str:
                # -40%以下 (0-40)
                percent_light_low += 0.0; percent_light_mid += 0.20; percent_light_high += 0.40
            else:
                # -30%以下 (0-30)
                percent_light_low += 0.0; percent_light_mid += 0.15; percent_light_high += 0.30

        if "未遂" in labels_str:
            # -30%至-50%
            percent_light_low += 0.30; percent_light_mid += 0.40; percent_light_high += 0.50

        if "从犯" in labels_str:
            # -30%至-40%
            percent_light_low += 0.30; percent_light_mid += 0.35; percent_light_high += 0.40

        if "初犯" in labels_str or "偶犯" in labels_str:
            # -10%以下 (0-10)
            percent_light_low += 0.0; percent_light_mid += 0.05; percent_light_high += 0.10

        # --- 3. 计算宣告刑 ---
        # (Base + Adjust_Months) * (1 + Percent_Heavy) * (1 - Percent_Light)

        # low 刑期: (low 基准 + low 月份) * (1 + low 从重) * (1 - high 从轻)
        final_low = (base_low + adj_m_low) * (1 + percent_heavy_low) * (1 - percent_light_high)

        # high 刑期: (high 基准 + high 月份) * (1 + high 从重) * (1 - low 从轻)
        final_high = (base_high + adj_m_high) * (1 + percent_heavy_high) * (1 - percent_light_low)

        # 确保在法定刑内
        final_low = max(0, min(final_low, statutory_max))
        final_high = max(0, min(final_high, statutory_max))

        return round(final_low), round(final_high)

    def _calculate_general_sentence(self, details: dict, labels: List[str]) -> Tuple[int, int]:
        """(V4) 通用刑期计算 (基于 "12.5量刑分析" - 百分比调节法)"""
        crime_type = details["crime_type"]
        amount = details["amount"]
        labels_str = " ".join(labels)

        # --- 1. 确定基准刑 (base_low, base_mid, base_high) ---
        base_low, base_mid, base_high = 0, 0, 0
        statutory_max = 36 # 默认3年

        if crime_type == "fraud":
            if amount >= 500000: # 特别巨大 (50万)
                base_low, base_mid, base_high = 120, 132, 144
                statutory_max = 180
            elif amount >= 30000: # 巨大 (3-50万)
                base_low, base_mid, base_high = 36, 48, 60
                statutory_max = 120
            elif amount >= 3000: # 较大 (3000-30000)
                base_low, base_mid, base_high = 6, 12, 18
                statutory_max = 36
            else:
                return 0, 6
        elif crime_type == "injury":
            if "重伤" in labels_str:
                base_low, base_mid, base_high = 36, 48, 60
                statutory_max = 120
            elif "轻伤" in labels_str:
                base_low, base_mid, base_high = 6, 12, 18
                statutory_max = 36
            else: # 默认轻伤
                base_low, base_mid, base_high = 6, 12, 18
                statutory_max = 36
        else: # general
            print(f"  [Task2-Rules] 'general' 罪名基准刑估算: 12-24 月")
            base_low, base_mid, base_high = 12, 18, 24
            statutory_max = 36

        # --- 2. 百分比调节 (来自 "12.5量刑分析") ---
        percent_heavy_low, percent_heavy_mid, percent_heavy_high = 0.0, 0.0, 0.0
        percent_light_low, percent_light_mid, percent_light_high = 0.0, 0.0, 0.0

        # (从重)
        if "累犯" in labels_str:
            percent_heavy_low += 0.10; percent_heavy_mid += 0.25; percent_heavy_high += 0.40 # +10%-40%
        elif "前科" in labels_str:
            percent_heavy_low += 0.0; percent_heavy_mid += 0.05; percent_heavy_high += 0.10 # +10%以下

        # (从轻)
        if "自首" in labels_str:
            percent_light_low += 0.20; percent_light_mid += 0.30; percent_light_high += 0.40 # -20%-40%
        elif "坦白" in labels_str:
            percent_light_low += 0.0; percent_light_mid += 0.10; percent_light_high += 0.20 # -20%以下

        if "当庭自愿认罪" in labels_str and "自首" not in labels_str and "坦白" not in labels_str:
            percent_light_low += 0.0; percent_light_mid += 0.05; percent_light_high += 0.10 # -10%以下

        if "认罪认罚" in labels_str:
            percent_light_low += 0.0; percent_light_mid += 0.15; percent_light_high += 0.30 # -30%以下

        if "退赃" in labels_str or "退赔" in labels_str:
            if "谅解" in labels_str:
                percent_light_low += 0.0; percent_light_mid += 0.20; percent_light_high += 0.40 # -40%以下
            else:
                percent_light_low += 0.10; percent_light_mid += 0.20; percent_light_high += 0.30 # -10%-30%

        if "未遂" in labels_str:
            percent_light_low += 0.20; percent_light_mid += 0.35; percent_light_high += 0.50 # -20%-50%

        if "从犯" in labels_str:
            percent_light_low += 0.20; percent_light_mid += 0.30; percent_light_high += 0.40 # -20%-40%

        if "初犯" in labels_str or "偶犯" in labels_str:
            percent_light_low += 0.0; percent_light_mid += 0.05; percent_light_high += 0.10 # -10%以下

        # --- 3. 计算宣告刑 ---
        # (基准刑 * (1 + 累加的从重比例)) * (1 - 累加的从轻比例)

        final_low = base_low * (1 + percent_heavy_low) * (1 - percent_light_high)
        final_high = base_high * (1 + percent_heavy_high) * (1 - percent_light_low)

        # 确保在法定刑内
        final_low = max(0, min(final_low, statutory_max))
        final_high = max(0, min(final_high, statutory_max))

        return round(final_low), round(final_high)

    def _parse_json_array(self, text: str) -> List[str]:
        """解析JSON数组 (Task1 使用)"""
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if x]
            except:
                pass
        lines = [line.strip().strip('"-,[]') for line in text.split('\n') if line.strip()]
        filtered_lines = [line for line in lines if
                          len(line) > 2 and not line.startswith(("请输出", "示例", "输出格式", "(请严格"))]
        return filtered_lines

    # _parse_json_interval 已移除

    def _post_process_labels(self, labels: List[str], fact: str) -> List[str]:
        """后处理:补充关键标签 (Task1 使用)"""
        labels_str = "".join(labels)

        if "金额" not in labels_str:
            amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
            if amount_match:
                amount = float(amount_match.group(1))
                if "万元" in amount_match.group(0):
                    amount *= 10000
                crime_type = self._detect_crime_type(fact)
                crime = "犯罪"
                if crime_type == "theft": crime = "盗窃"
                elif crime_type == "fraud": crime = "诈骗"
                labels.insert(0, f"{crime}金额既遂{int(amount)}元")

        if "多次" in fact and not any("次数" in l for l in labels):
            count_match = re.search(r'(\d+)\s*次', fact)
            if count_match:
                crime_type = self._detect_crime_type(fact)
                crime = "犯罪"
                if crime_type == "theft": crime = "盗窃"
                elif crime_type == "fraud": crime = "诈骗"
                labels.append(f"{crime}次数{count_match.group(1)}次")

        return labels

    def _validate_interval(self, lower: int, upper: int, labels: List[str]) -> Tuple[int, int]:
        """验证区间合理性 (Task2 使用)"""
        if lower > upper:
            lower, upper = upper, lower
        lower = max(0, lower)
        upper = max(0, upper)
        upper = min(240, upper) # 20年
        if upper - lower < 1:
            upper = lower + 3

        labels_str = "".join(labels)
        if "数额较大" in labels_str and "数额巨大" not in labels_str:
            upper = min(upper, 36) # 不超过3年

        return lower, upper

    def _fallback_extract(self, fact: str) -> List[str]:
        """兜底:正则提取 (Task1 使用)"""
        labels = []
        amount_match = re.search(r'(?:盗窃|诈骗|骗取).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
        if amount_match:
            amount = float(amount_match.group(1))
            if "万元" in amount_match.group(0):
                amount *= 10000
            crime = "盗窃" if "盗窃" in fact else "诈骗"
            labels.append(f"{crime}金额既遂{int(amount)}元")

        keyword_map = {"自首": "自首", "坦白": "坦白", "累犯": "累犯", "前科": "前科", "退赔": "退赔",
                       "退赃": "退赃", "谅解": "取得谅解", "认罪认罚": "认罪认罚", "未遂": "未遂", "从犯": "从犯"}
        for keyword, label in keyword_map.items():
            if keyword in fact:
                labels.append(label)
        return labels if labels else ["信息不足"]

    def _fallback_predict_static(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """(V4) 兜底:保守预测 (静态, 无API调用)"""
        labels_str = "".join(labels)
        if "数额特别巨大" in labels_str:
            return 120, 144
        elif "数额巨大" in labels_str or "重伤" in labels_str:
            return 48, 72
        elif "累犯" in labels_str:
            return 18, 30
        else:
            return 12, 24

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

            # Task2: 预测刑期 (Rules)
            print("\nTask2: 预测刑期区间 (Rules)...")
            # 调用重写后的 predict_sentence
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
    print(f"刑事案件量刑辅助系统 - (v4 - 规则计算版 - {DASHSCOPE_MODEL_NAME})")
    print("=" * 60 + "\n")

    inferencer = LLMDrivenInferencer()
    inferencer.process_dataset(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
