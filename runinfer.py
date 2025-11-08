"""
主推理脚本 - 完全由大模型驱动版本 (v3 - 专家规则版)

更新日志:
- Task1: 维持原状 (原T1提示词已针对 'answer1' 的 list[str] 格式优化)
- Task2 (General):
    - 注入了 "12.5量刑分析" 中的详细百分比调节规则，
    - 替换了原有的占位符规则。
- Task2 (Theft):
    - 完全替换为 "盗窃罪量刑建议" 的详细CoT(Chain-of-Thought)计算步骤，
    - 确保模型严格遵循 "量刑起点→调节基准刑→宣告刑" 的专家逻辑。
"""

import json
import os
import re
from typing import List, Tuple, Optional
# from openai import OpenAI # 移除对openai库的导入
import requests # 导入requests库
from dotenv import load_dotenv

load_dotenv()

# 配置路径
DATA_PATH = "data/task6.jsonl"
# 更新输出文件名以反映版本
OUTPUT_PATH = "./result/submission_final_gem.jsonl"

# --- New API 配置 ---
NEWAPI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-icjbwcFqx4XoGpA2pFGchvZGeM04DVShSDgZiadkaL1Anqzl")
NEWAPI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://45.150.227.150:32770")
NEWAPI_MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.5-pro")
NEWAPI_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"

TEMPERATURE = 0.1
MAX_TOKENS = 32768
MAX_TOKENS_TASK2 = 32768 # T2的Prompt现在更长, 但仍在token限制内


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

# ==================== Task2: 刑期预测 (V3 - 专家规则版) ====================

# 通用系统提示词 (适用于所有T2任务)
TASK2_SYSTEM_PROMPT = """你是资深刑事法官,精通《刑法》分则和《量刑指导意见(2024)》。你需要根据案件事实和已提取的量刑情节,**严格依据法律规定**预测合理的刑期区间。

**输出要求**:
- 必须输出JSON格式: {"lower": X, "upper": Y}
- X和Y是整数月份
- 区间宽度建议3-12个月(根据案情复杂度)
- 不要输出任何解释"""

# T2 模板 - V3 - 通用罪名 (基于 "12.5量刑分析" 规则)
TASK2_GENERAL_USER_TEMPLATE = """【案件事实】
{fact}

【已提取量刑情节】
{labels}

【任务】
根据上述事实和情节, 依据通用的**量刑指导意见（百分比调节法）**，推导被告人的宣告刑区间（以月为单位）。

【推理步骤 (Chain of Thought) - 供你内部参考】
1.  **确定罪名和法定刑档位**: 根据案件事实和情节, 确定罪名和基础法定刑区间 (参考下方的【关键参考】)。
2.  **确定基准刑 (起点)**: 在法定刑区间内，根据犯罪的基本事实（如诈骗金额、伤害程度）确定一个基准刑起点（月份）。
3.  **计算调节比例 (基于专家规则)**:
    * 严格审查【已提取量刑情节】列表, 并应用以下规则:
    
    **一、从轻、减轻或免除处罚的情节**
    * 未成年人犯罪:
        * 已满16周岁不满18周岁：减少基准刑10%–50%
    * 老年人犯罪:
        * 75周岁以上故意犯罪：减少基准刑40%以下
    * 精神病人犯罪:
        * 尚未完全丧失辨认或控制能力者：减少基准刑20%–40%
    * 犯罪预备:
        * 可比照既遂减少基准刑50%以下
    * 犯罪未遂:
        * 根据实行程度、损害结果等情况，减少基准刑20%–50%
    * 犯罪中止:
        * 造成损害的，减少30%–60%
    * 从犯:
        * 起次要作用的减少20%–40%，起辅助作用的减少30%–50%
    * 胁从犯:
        * 减少基准刑40%–60%
    * 自首:
        * 根据不同情形减少20%–40%
    * 坦白:
        * 如实供述罪行的减少20%以下；供述同种较重罪行的减少10%–30%
    * 当庭自愿认罪:
        * 减少基准刑10%以下（不与自首、坦白重复评价）
    * 立功:
        * 一般立功减少20%以下，重大立功减少20%–50%
    * 退赃、退赔:
        * 积极配合退赃退赔的，减少10%–30%
    * 赔偿谅解:
        * 积极赔偿并取得谅解的减少40%以下；赔偿未谅解的减少30%以下
    * 刑事和解:
        * 减少基准刑50%以下
    * 认罪认罚:
        * 一审认罪认罚减少30%以下；兼具其他从宽情节的可减少60%以下
    * 初犯、偶犯:
        * 较轻犯罪的初犯、偶犯减少10%以下

    **二、从重处罚的情节**
    * 累犯:
        * 增加基准刑10%–40%，一般不少于三个月
    * 前科:
        * 增加基准刑10%以下
    * 针对弱势人员犯罪:
        * 针对未成年人、老年人、残疾人、孕妇等犯罪，增加基准刑20%以下
    * 教唆未成年人犯罪:
        * 增加基准刑20%–40%
    * 毒品再犯:
        * 增加基准刑10%–30%
    * 拒不退赃、退赔可作为从重情节考虑

4.  **计算宣告刑区间**:
    * (基准刑 * (1 + 累加的从重比例)) * (1 - 累加的从轻比例)
    * 计算出一个**最低调节结果(lower)**和**最高调节结果(upper)**。
    * 确保结果在法定刑档位内。

【关键参考】
- 盗窃罪: 数额较大(1000-30000元):0-36个月; 数额巨大(3-30万):36-120个月; 数额特别巨大(30万以上):120-180个月
- 诈骗罪: 数额较大(3000-30000元):0-36个月; 数额巨大(3-50万):36-120个月; 数额特别巨大(50万以上):120-180个月
- 抢劫罪: 基本犯:36-120个月; 加重情节:120-180个月或无期
- 故意伤害罪: 轻伤:0-36个月; 重伤:36-120个月; 致人死亡:120-180个月
- 交通肇事罪: 一般:0-36个月; 逃逸或情节恶劣:36-84个月; 因逃逸致死:84-180个月

【输出格式】
仅输出JSON对象:
{{"lower": 下限月数, "upper": 上限月数}}

示例:
{{"lower": 36, "upper": 48}}

请输出:"""

# T2 模板 - V3 - 专用 (盗窃罪, 基于 "盗窃罪量刑建议" 专家规则)
TASK2_THEFT_USER_TEMPLATE = """【任务】
本案为**盗窃罪**。请根据下方提供的“案件事实”和“已提取量刑情节”, **严格按照盗窃罪的“量刑起点→调节基准刑→宣告刑”精确计算顺序**，计算最终的宣告刑区间（以月为单位）。

【推理步骤 (Chain of Thought) - 供你内部参考】
你必须严格遵循以下专家计算步骤：

##一、量刑情节规范化认定
(请核对【已提取量刑情节】列表, 确认以下情节是否存在)
（一）自首：分析到案经过。自动投案或电话通知传唤到案后如实供述的可认定。
（二）坦白：不具自首情节但如实供述。
（三）未遂：分析事实。
（五）退赃退赔：分析书证, 注意必须是“主动”退赃退赔。
（六）取得谅解：分析书证。
（七）累犯：刑罚执行完毕或赦免后五年内再犯应当判处有期徒刑以上刑罚之罪。
（八）认罪认罚：分析供述。

##二、量刑起点
（一）提取犯罪基本事实和手段：是否“二年内盗窃三次”、“入户盗窃”、“携带凶器盗窃”、“扒窃”。
（二）提取犯罪数额：
    * 1000元及以上：数额较大
    * 30000元及以上：数额巨大
    * 300000元及以上：数额特别巨大
（三）确定量刑起点：
    * 达到**数额较大**起点, 或有“二年内三次盗窃”、“入户盗窃”、“携带凶器盗窃”、“扒窃”的, 在 **1-12个月有期徒刑、拘役** 幅度内确定量刑起点 (通常取 3-9 个月)。
    * 达到**数额巨大**起点或者有其他严重情节的, 在 **36至48个月(三年至四年)** 有期徒刑幅度内确定量刑起点。
    * 达到**数额特别巨大**起点或者有其他特别严重情节的, 在 **120至144个月(十年至十二年)** 有期徒刑幅度内确定量刑起点。

##三、调节基准刑 (核心计算)
（一）从重情节
1.  犯罪数额、次数、手段等：
    * (1) 数额达到**较大**的, 每增加 1500 元, **+1个月**; 
    * (2) 数额达到**巨大**的, 每增加 5000 元, **+1个月**; 
    * (3) 数额达到**特别巨大**的, 每增加 2万元, **+1个月**;
    * (4) 具有“二年内盗窃三次”、“入户盗窃”、“携带凶器盗窃”、“扒窃”两种以上情形的, 每增加一种情形, **+1至3个月**。
2.  前科劣迹、危害后果等 (如有以下情形之一, +基准刑的20%):
    * (1) 曾因盗窃受过刑事处罚的;
    * (2) 一年内曾因盗窃受过行政处罚的;
    * (10) 流窜盗窃的;
    * (11) 为吸毒、赌博等违法犯罪活动而盗窃的;
    * (13) 盗窃数额达到起点, 同时又具有多次、携带凶器、入户、扒窃情形之一的。
3.  累犯：
    * **+基准刑的10%-40%** (且不少于3个月)。

（二）从轻情节
1.  自首：
    * (1) 犯罪事实、嫌疑人未被发觉, 主动投案: **-40%以下**;
    * (2) 事实或嫌疑人被发觉, 尚未讯问: **-30%以下**;
    * (3) 事实和嫌疑人均被发觉, 主动投案: **-20%以下**;
    * (8) 犯罪较轻的, 可 **-40%以上**。
2.  坦白：
    * (1) 如实供述自己罪行的: **-20%以下**;
    * (2) 如实供述同种较重罪行的: **-10%-30%**。
3.  未遂:
    * (1) 实行终了的未遂犯: **-20%至-40%** (根据损害);
    * (2) 未实行终了的未遂犯: **-30%至-50%** (根据损害)。
4.  从犯：
    * **-20%至-50%** (根据作用)。
5.  认罪认罚：
    * (1) 自愿认罪认罚: **-20%以下**;
    * (2) 同时具有自首、坦白、退赃退赔、赔偿谅解等情节的: **-60%以下**。
    * (注意: 认罪认罚与自首、坦白、退赃退赔等不作重复评价减幅, 但可叠加适用)。
6.  初犯、偶犯：
    * **-10%以下**。
7.  赔偿谅解：
    * (1) 积极赔偿且取得谅解: **-40%以下**;
    * (2) 积极赔偿但没有取得谅解: **-30%以下**;
    * (3) 没有赔偿但取得谅解: **-20%以下**。
8.  特殊人员：
    * (1) 已满16周岁不满18周岁: **-10%-50%**;
    * (2) 已满75周岁: **-40%以下**。

##四、确定宣告刑
在调节基准刑的基础上, 综合计算出最终的刑期 **区间 (lower, upper)**。
(调节步骤本身就是区间, 如-20%至-40%, 你需要计算最低和最高调节结果)

---
【案件事实】
{fact}

---
【已提取量刑情节】
{labels}

---
【输出格式】
严格仅输出JSON对象:
{{"lower": 下限月数, "upper": 上限月数}}

示例:
{{"lower": 40, "upper": 50}}

请输出:"""


class LLMDrivenInferencer:
    """大模型驱动的量刑推理器"""

    def __init__(self, api_key: str = NEWAPI_API_KEY, base_url: str = NEWAPI_BASE_URL, model: str = NEWAPI_MODEL_NAME):
        print(f"初始化 New API 客户端 (v3 - 专家规则版)")
        print(f"  Base URL: {base_url}")
        print(f"  Model: {model}")

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.chat_completions_url = f"{self.base_url}{NEWAPI_CHAT_COMPLETIONS_ENDPOINT}"

        # 测试连接
        try:
            # 模拟一个简单的请求来测试连接
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

    def _call_new_api_backend(self, messages: List[dict], max_tokens: int, temperature: float, top_p: float = 0.95) -> Optional[dict]:
        """封装对New API服务器的HTTP POST请求"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p, # 确保传递top_p参数
            "stream": False # 通常在非流式模式下调用
        }

        try:
            response = requests.post(self.chat_completions_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  # 检查HTTP状态码，如果不是2xx则抛出异常
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"  HTTP错误: {http_err}, 响应内容: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"  请求发生错误: {req_err}")
        except Exception as e:
            print(f"  发生未知错误: {e}")
        return None

    def _detect_crime_type(self, fact: str) -> str:
        """
        根据案件事实简单判断核心罪名。
        在竞赛数据集中,这通常足够。
        """
        fact_keywords = fact[:1000] # 只检查开头, 提高效率

        if "盗窃" in fact_keywords:
            return "theft"
        if "诈骗" in fact_keywords:
            return "fraud"
        if "故意伤害" in fact_keywords:
            return "injury"
        if "抢劫" in fact_keywords:
            return "robbery"
        if "交通肇事" in fact_keywords:
            return "traffic"

        # 兜底为通用
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
                    # 提取模型的content回答
                    content = response_json["choices"][0].get("message", {}).get("content", "").strip()
                    labels = self._parse_json_array(content)

                    if labels:
                        # 后处理:补充缺失的关键标签
                        labels = self._post_process_labels(labels, fact)
                        return labels
                else:
                    print(f"  Task1 API调用失败: 收到无效响应 (尝试{attempt + 1}/{max_retries})")

            except Exception as e:
                print(f"  Task1 API调用失败(尝试{attempt + 1}/{max_retries}): {e}")

        # 兜底:基于正则提取
        return self._fallback_extract(fact)

    def predict_sentence(self, fact: str, labels: List[str], max_retries: int = 3) -> Tuple[int, int]:
        """Task2: 预测刑期区间 (V3 - 动态选择专家模板)"""
        labels_str = "\n".join([f"- {label}" for label in labels])

        # 1. 动态选择提示词
        crime_type = self._detect_crime_type(fact)

        system_prompt = TASK2_SYSTEM_PROMPT # 系统提示词是通用的

        if crime_type == "盗窃":
            user_template = TASK2_THEFT_USER_TEMPLATE
            print(f"  [Task2] 检测到罪名: 盗窃罪 (theft) - 启用 V3 专用计算模板")
        else:
            user_template = TASK2_GENERAL_USER_TEMPLATE
            print(f"  [Task2] 检测到罪名: {crime_type} - 启用 V3 通用百分比模板")

        # 2. 格式化用户输入
        user_content = user_template.format(fact=fact, labels=labels_str)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # 3. API 调用与解析 (与之前相同)
        for attempt in range(max_retries):
            try:
                response_json = self._call_new_api_backend(
                    messages=messages,
                    max_tokens=MAX_TOKENS_TASK2, # 使用Task2的max_tokens
                    temperature=TEMPERATURE,
                    top_p=0.95
                )

                if response_json and "choices" in response_json and len(response_json["choices"]) > 0:
                    # 提取模型的content回答
                    content = response_json["choices"][0].get("message", {}).get("content", "").strip()
                    lower, upper = self._parse_json_interval(content)

                    if lower is not None and upper is not None:
                        # 合理性检查
                        lower, upper = self._validate_interval(lower, upper, labels)
                        return int(lower), int(upper)
                else:
                    print(f"  Task2 API调用失败: 收到无效响应 (尝试{attempt + 1}/{max_retries})")

            except Exception as e:
                print(f"  Task2 API调用失败(尝试{attempt + 1}/{max_retries}): {e}")

        # 兜底:保守估计
        return self._fallback_predict(fact, labels)

    def _parse_json_array(self, text: str) -> List[str]:
        """解析JSON数组"""
        # 提取JSON数组
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if x]
            except:
                pass

        # 尝试按行分割，如果不是有效的JSON数组
        lines = [line.strip().strip('"-,[]') for line in text.split('\n') if line.strip()]
        # 进一步过滤，确保是有效标签
        filtered_lines = [line for line in lines if len(line) > 2 and not line.startswith(("请输出", "示例", "输出格式", "(请严格"))]
        return filtered_lines

    def _parse_json_interval(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """解析JSON区间"""
        # 提取JSON对象
        # 修正正则表达式以更鲁棒地匹配JSON结构
        match = re.search(r'\{\s*"lower"\s*:\s*(\d+)\s*,\s*"upper"\s*:\s*(\d+)\s*\}', text, re.DOTALL | re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))

        # 提取数字对作为兜底，如果不是严格的JSON格式
        numbers = re.findall(r'(\d+)', text)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])

        return None, None

    def _post_process_labels(self, labels: List[str], fact: str) -> List[str]:
        """后处理:补充关键标签"""
        labels_str = "".join(labels)

        # 补充金额(如果缺失)
        if "金额" not in labels_str:
            amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢劫).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
            if amount_match:
                amount = float(amount_match.group(1))
                if "万元" in amount_match.group(0):
                    amount *= 10000

                crime_type = self._detect_crime_type(fact)
                if crime_type == "theft": crime = "盗窃"
                elif crime_type == "fraud": crime = "诈骗"
                else: crime = "犯罪"

                labels.insert(0, f"{crime}金额既遂{int(amount)}元")

        # 补充次数(如果有"多次"但没有具体次数)
        if "多次" in fact and not any("次数" in l for l in labels):
            count_match = re.search(r'(\d+)\s*次', fact)
            if count_match:
                crime_type = self._detect_crime_type(fact)
                if crime_type == "theft": crime = "盗窃"
                elif crime_type == "fraud": crime = "诈骗"
                else: crime = "犯罪"
                labels.append(f"{crime}次数{count_match.group(1)}次")

        return labels

    def _validate_interval(self, lower: int, upper: int, labels: List[str]) -> Tuple[int, int]:
        """验证区间合理性"""
        # S确保lower <= upper
        if lower > upper:
            lower, upper = upper, lower

        # 确保非负
        lower = max(0, lower)
        upper = max(0, upper)

        # 限制最大值(20年=240月)
        upper = min(240, upper)

        # 确保最小宽度
        if upper - lower < 1:
            upper = lower + 3

        # 确保不超过法定上限(根据关键词判断)
        labels_str = "".join(labels)
        if "数额较大" in labels_str and "数额巨大" not in labels_str:
            upper = min(upper, 36)

        return lower, upper

    def _fallback_extract(self, fact: str) -> List[str]:
        """兜底:正则提取"""
        labels = []

        # 提取金额
        amount_match = re.search(r'(?:盗窃|诈骗|骗取).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
        if amount_match:
            amount = float(amount_match.group(1))
            if "万元" in amount_match.group(0):
                amount *= 10000
            crime = "盗窃" if "盗窃" in fact else "诈骗"
            labels.append(f"{crime}金额既遂{int(amount)}元")

        # 提取关键情节
        keyword_map = {
            "自首": "自首",
            "坦白": "坦白",
            "累犯": "累犯",
            "前科": "前科",
            "退赔": "退赔",
            "退赃": "退赃",
            "谅解": "取得谅解",
            "认罪认罚": "认罪认罚",
            "未遂": "未遂",
            "从犯": "从犯"
        }

        for keyword, label in keyword_map.items():
            if keyword in fact:
                labels.append(label)

        return labels if labels else ["信息不足"]

    def _fallback_predict(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """兜底:保守预测"""
        labels_str = "".join(labels)

        # 根据关键词估算
        if "数额特别巨大" in labels_str or "抢劫" in fact:
            return 120, 144
        elif "数额巨大" in labels_str:
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

            # Task1: 提取量刑情节
            print("Task1: 提取量刑情节...")
            labels = self.extract_labels(item['fact'])
            print(f"✓ 提取到 {len(labels)} 个情节:")
            for label in labels[:10]:  # 只显示前10个
                print(f"  - {label}")
            if len(labels) > 10:
                print(f"  ... (还有 {len(labels) - 10} 个)")

            # Task2: 预测刑期
            print("\nTask2: 预测刑期区间...")
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
    print("刑事案件量刑辅助系统 - 大模型驱动版本 (v3 - 专家规则版)")
    print("=" * 60 + "\n")

    # 初始化LLMDrivenInferencer时，它会使用全局定义的 NEWAPI_XXX 变量
    inferencer = LLMDrivenInferencer()
    inferencer.process_dataset(DATA_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
