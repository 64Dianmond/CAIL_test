import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from cal1 import SentencingCalculator, SENTENCING_TOOLS, execute_tool_call

# 加载环境变量
load_dotenv()


class SentencingPredictor:
    """
    一个基于大型语言模型的法律量刑预测器。
    采用"提取-注入-计算"的三步混合法，并集成了权威的、分层的量刑计算规则。
    支持工具调用，使用专业计算器进行精确的刑期计算。
    """

    def __init__(self):
        """
        初始化客户端和模型配置。
        """
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model_name = os.getenv("OPENAI_MODEL", "qwen-max")
        self.temperature_task1 = 1.0  # Task1使用较高温度以增加多样性
        self.temperature_task2 = 0.1  # Task2使用较低温度以确保稳定性
        self.max_tokens = 32768

    def identify_crime_type(self, defendant_info, case_description):
        """
        增强版的罪名识别函数。
        优先从指控中识别,其次通过关键词匹配。
        """
        text = defendant_info + case_description
        text = text.replace(" ", "").replace("\n", "")

        # 1. 优先匹配指控罪名,这是最准确的方式
        charge_match = re.search(r'(因涉嫌|指控犯)(.*?)罪', text)
        if charge_match:
            crime = charge_match.group(2)
            if "盗窃" in crime: return "盗窃罪"
            if "故意伤害" in crime: return "故意伤害罪"
            if "诈骗" in crime: return "诈骗罪"

        # 2. 如果指控不明确,使用关键词作为备用方案
        theft_keywords = ["盗窃", "窃取", "扒窃", "盗走"]
        injury_keywords = ["故意伤害", "殴打", "打伤", "轻伤", "重伤"]
        fraud_keywords = ["诈骗", "骗取", "虚构事实"]

        if any(k in text for k in theft_keywords): return "盗窃罪"
        if any(k in text for k in injury_keywords): return "故意伤害罪"
        if any(k in text for k in fraud_keywords): return "诈骗罪"

        # 3. 默认回退,根据数据集的多数罪名来定,此处以盗窃罪为例
        return "盗窃罪"

    # def extract_region(self, defendant_info, case_description):
    #     """
    #     从案件信息中提取地区信息
    #     """
    #     text = defendant_info + case_description
    #     # 常见的地区关键词
    #     regions = ["北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林",
    #               "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    #               "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西",
    #               "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
    #               "香港", "澳门"]
    #
    #     # 常见的城市关键词
    #     cities = ["江门", "深圳", "广州", "珠海", "佛山", "东莞", "中山", "杭州",
    #              "宁波", "温州", "嘉兴", "绍兴", "台州", "义乌", "南京", "苏州",
    #              "无锡", "常州", "徐州", "济南", "青岛", "烟台", "潍坊", "大连",
    #              "沈阳", "哈尔滨", "长春", "成都", "西安", "武汉", "长沙", "福州",
    #              "厦门", "贵阳", "昆明", "南宁", "石家庄", "太原", "南昌", "合肥",
    #              "郑州", "海口", "乌鲁木齐", "呼和浩特", "银川", "西宁", "拉萨", "兰州"]
    #
    #     # 先尝试查找省份
    #     for region in regions:
    #         if region in text:
    #             return region
    #
    #     # 如果没有找到省份，尝试查找城市
    #     for city in cities:
    #         if city in text:
    #             return city
    #
    #     # 如果没有找到明确的地区，返回默认值
    #     return "default"

    def build_prompt_task1_authoritative(self, defendant_info, case_description):
        """
        构建权威版的量刑情节提取Prompt (Task 1)。
        Prompt内容严格依据官方量刑指导意见中的情节分类。
        """
        crime_type = self.identify_crime_type(defendant_info, case_description)

        prompt = f"""你是一位极其严谨的刑事法官,任务是依据《中华人民共和国刑法》及相关量刑指导意见,从案情中提取所有对量刑有影响的关键情节。
**特别注意:**
**重要提示:** 本任务将根据F1值进行评分，F1值是精确率(Precision)和召回率(Recall)的调和平均数。这意味着你需要在确保提取准确性的同时，尽可能完整地提取所有相关情节。错误提取会同时降低精确率和召回率，因此请优先确保准确性。
请明确标注是否存在"法定减轻处罚情节",这类情节包括:
- 自首
- 立功(重大立功)
- 未成年人犯罪
- 从犯
- 防卫过当
- 犯罪预备、中止、未遂

如果存在上述情节,请在输出中包含"法定减轻"这一标签。    

**案件信息:**
被告人信息:{defendant_info}
案情描述:{case_description}

**本案罪名(初步判断):** {crime_type}

**提取总要求:**
- **平衡准确性和完整性**: 优先确保提取情节的准确性，宁可少提取也不要提取错误的情节，因为错误提取会同时降低精确率和召回率。
- **处理边界情况**: 对于模糊或不确定的情节，只有在有明确证据支持时才提取，避免猜测性提取

---
**请按照以下分类和指引进行提取:**

**一、 犯罪构成与基本事实情节 (决定量刑起点和基准刑)**
- **犯罪数额/后果**: 必须明确为 **"盗窃/诈骗金额既遂XX元"** 或 **"故意伤害致X人轻伤/重伤X级"** 或 **"故意伤害致X人死亡"**。
- **数额/后果档次**: 必须明确标注 **"盗窃/诈骗数额较大/巨大/特别巨大"**。
- **犯罪手段/方式**: 提取特殊手段,如 **"入户盗窃"、"携带凶器盗窃"、"扒窃"、"电信网络诈骗"** 等。
- **犯罪次数**: 如 **"多次盗窃"**、**"多次诈骗"**。

**二、 法定从重、从轻、减轻处罚情节 (必须依法调节)**
- **累犯**: 重点核查被告人信息中的前科记录,判断是否构成累犯(一般为有期徒刑执行完毕或赦免以后,五年以内再犯应当判处有期徒刑以上刑罚之罪)。
- **自首**: 重点核查归案方式,如主动投案,或"形迹可疑,经盘问、教育后,主动交代了司法机关未掌握的罪行","在案发地等候处置"等均可能构成自首。
- **立功**: 是否有检举、揭发他人犯罪行为,经查证属实等情况。
- **未成年人犯罪**: 被告人犯罪时是否已满十四周岁不满十八周岁。
- **从犯/胁从犯**: 在共同犯罪中的作用。
- **犯罪预备/中止/未遂**。

**三、 酌定从重、从轻处罚情节 (可以酌情调节)**
- **坦白**: 被动归案后,如实供述自己罪行的。
- **认罪认罚**: 是否自愿如实供述自己的罪行,承认指控的犯罪事实,愿意接受处罚。
- **退赃/退赔/赔偿**: 是否退还赃款赃物,或赔偿被害人经济损失。必须量化,如 **"退赔XX元"**。
- **取得谅解**: 是否取得了被害人的书面或口头谅解。
- **前科**: 不构成累犯,但有犯罪记录的。
- **被害人过错** (主要适用于故意伤害罪): 案件起因是否由被害人过错引起。
- **多次犯罪**: 是否存在多次实施同种犯罪行为的情况，如"多次盗窃"、"多次诈骗"等。
- **其他**: 如诈骗残疾人、老年人等特定群体财物,属于酌情从重情节。

---
**输出格式:**
只输出一个JSON数组,包含所有提取到的情节字符串。不要任何解释或Markdown标记。

**重要原则:**
- 只提取案件事实中明确提及的情节，不要包含推测或假设的情节
- 宁可少提取也不要提取不准确的情节
- 确保每个提取的情节都有明确的事实依据
- 如果存在法定减轻情节，必须包含"法定减轻"标签

**示例({crime_type}):**
["盗窃金额既遂3631元", "盗窃数额较大", "扒窃", "累犯", "坦白"]

**另一个示例({crime_type}):**
["诈骗金额既遂35700元", "诈骗数额巨大", "多次盗窃", "自首", "退赔35570元", "认罪认罚", "法定减轻"]

**故意伤害罪示例:**
["故意伤害致1人死亡", "自首", "赔偿500000元", "取得谅解", "被害人过错", "法定减轻"]
"""
        return prompt

    def build_prompt_task2_with_tools(self, defendant_info, case_description, sentencing_factors):
        """
        构建支持工具调用的刑期预测Prompt (Task 2)。
        模型将使用计算器工具进行精确的刑期计算。
        """
        # 判断是否有法定减轻情节
        statutory_mitigation_keywords = [
            "自首", "立功", "重大立功",
            "未成年人", "已满十四周岁不满十八周岁",
            "从犯", "胁从犯",
            "犯罪中止", "犯罪未遂", "犯罪预备",
            "防卫过当", "避险过当",
            "七十五周岁", "75周岁"
        ]
        has_statutory = any(kw in str(sentencing_factors) for kw in statutory_mitigation_keywords)

        crime_type = self.identify_crime_type(defendant_info, case_description)
        factors_str = "\n- ".join(sentencing_factors)

        # 提取金额用于计算
        amount = None
        for factor in sentencing_factors:
            if "盗窃金额既遂" in factor or "诈骗金额既遂" in factor:
                try:
                    amount = float(re.search(r'(\d+\.?\d*)元', factor).group(1))
                except:
                    pass
                break

        prompt = f"""你是一位精通量刑计算的刑事法官。你必须使用提供的专业计算器工具来进行精确计算,不要自己估算数值。
 **重要约束条件:**
    1. 总调节减轻幅度原则上不得超过基准刑的50%(除非有法定减轻情节)
    2. 本案{'有' if has_statutory else '无'}法定减轻情节

**已认定的量刑情节:**
- {factors_str}

**你的任务:**
严格按照以下步骤使用工具进行计算:

**步骤1: 计算基准刑**
- 使用 `calculate_base_sentence` 工具
- 根据罪名类型、犯罪事实(金额/伤害等级)和案件地区计算基准刑

**步骤2: 分析和分层情节**
从上述情节中,识别:
- **第一层面情节(连乘)**: 未成年人、从犯、胁从犯、犯罪预备、犯罪中止、犯罪未遂
- **第二层面情节(加减)**: 累犯、自首、坦白、立功、认罪认罚、退赔、取得谅解、前科、多次盗窃、多次犯罪

根据最高人民法院及各地高级人民法院的量刑指导意见，为每个情节确定合适的调节比例:
- 未成年人: 0.5 (减半)
- 从犯: 0.8 (减20%)
- 累犯: 1.3 (增30%)
- 自首: 0.8 (减20%)
- 坦白: 0.9 (减10%)
- 立功: 0.8 (减20%)
- 认罪认罚: 0.9 (减10%)
- 退赃/退赔/取得谅解: 0.95 (减5%)
- 多次盗窃/多次犯罪: 1.2 (增20%)

**步骤3: 计算最终刑期**
- 使用 `calculate_layered_sentence_with_constraints` 工具
- 传入基准刑、罪名、金额、第一层面情节列表、第二层面情节列表和是否有法定减轻情节

**步骤4: 生成刑期区间**
- 使用 `months_to_range` 工具
- 将最终月数转换为合理区间(宽度6个月)

请按顺序调用工具,完成计算后,输出最终的刑期区间。如果刑期区间下限为0，请调整为1
"""
        return prompt

    def predict_task1_authoritative(self, defendant_info, case_description):
        """
        执行Task 1:提取量刑情节。
        """
        prompt = self.build_prompt_task1_authoritative(defendant_info, case_description)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "你是一位经验丰富的刑事法官,精通中国刑法量刑情节认定,对细节极其敏感。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature_task1,  # Task1使用较高温度
                max_tokens=self.max_tokens
            )
            result_text = response.choices[0].message.content.strip()

            # 使用正则表达式从文本中提取JSON数组
            json_match = re.search(r'\[.*?\]', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"警告 (Task 1): 未能在输出中找到JSON数组。返回: {result_text}")
                return ["盗窃数额较大"]  # Fallback
        except Exception as e:
            print(f"错误 (Task 1): API调用或JSON解析失败: {e}")
            return ["盗窃数额较大"]  # Fallback

    def predict_task2_with_tools(self, defendant_info, case_description, sentencing_factors):
        """
        执行Task 2:使用工具调用进行刑期预测。
        """
        if not sentencing_factors:
            sentencing_factors = ["犯罪情节较轻"]

        prompt = self.build_prompt_task2_with_tools(defendant_info, case_description, sentencing_factors)

        messages = [
            {"role": "system",
             "content": "你是一位刑事法官,必须使用提供的计算器工具进行精确计算,不要自己估算数值。请根据案件信息判断案件所在地区，如无法判断则使用默认标准。"},
            {"role": "user", "content": prompt}
        ]

        # 多轮对话处理工具调用
        max_iterations = 10
        final_range = None

        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=SENTENCING_TOOLS,
                    temperature=self.temperature_task2,  # Task2使用较低温度
                    max_tokens=self.max_tokens
                )

                assistant_message = response.choices[0].message

                # 如果没有工具调用,说明完成
                if not assistant_message.tool_calls:
                    content = assistant_message.content
                    print(f"  模型最终回复: {content}")

                    # 从最终响应中提取区间
                    json_match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', content)
                    if json_match:
                        final_range = [int(json_match.group(1)), int(json_match.group(2))]
                        break
                    elif final_range:  # 如果之前已经计算出了区间
                        break
                    else:
                        print(f"警告: 未找到刑期区间,使用默认值")
                        return [6, 12]

                # 添加助手消息
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                })

                # 执行工具调用
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"  🔧 调用工具: {function_name}")
                    print(f"     参数: {json.dumps(function_args, ensure_ascii=False)}")

                    # 执行工具
                    function_response = execute_tool_call(function_name, function_args)
                    print(f"     结果: {function_response}")

                    # 检查是否是最终的区间结果
                    if function_name == "months_to_range":
                        try:
                            result_data = json.loads(function_response)
                            if "range" in result_data:
                                final_range = result_data["range"]
                        except:
                            pass

                    # 添加工具响应
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_response
                    })

            except Exception as e:
                print(f"工具调用错误: {e}")
                return [6, 12]  # Fallback

        if final_range:
            return final_range
        else:
            print("警告: 达到最大迭代次数但未获得结果")
            return [6, 12]  # Fallback

    def process_all_data(self, preprocessed_data, output_file):
        """
        主处理流程:遍历所有数据,执行两阶段预测,并保存结果。
        """
        results = []

        for idx, item in enumerate(preprocessed_data):
            print(f"\n{'=' * 60}")
            print(f"处理第 {idx + 1}/{len(preprocessed_data)} 条数据 (ID: {item['id']})")
            print(f"{'=' * 60}")

            answer1, answer2 = [], []
            try:
                # 第一步:调用权威版 Task 1 预测,提取量刑情节
                print("\n【步骤1: 提取量刑情节】")
                answer1 = self.predict_task1_authoritative(
                    item['defendant_info'],
                    item['case_description']
                )
                print(f"✓ 提取到的情节: {answer1}")

                # 第二步:使用工具调用进行刑期预测
                print("\n【步骤2: 使用工具计算刑期】")
                answer2 = self.predict_task2_with_tools(
                    item['defendant_info'],
                    item['case_description'],
                    answer1,
                )
                print(f"✓ 预测刑期区间: {answer2}")

            except Exception as e:
                print(f"!!! 处理ID {item['id']} 时发生未知严重错误: {e}")
                answer1 = answer1 if answer1 else ["盗窃数额较大"]
                answer2 = answer2 if answer2 else [6, 12]

            result = {
                "id": item['id'],
                "answer1": answer1,
                "answer2": answer2
            }
            results.append(result)

            print(f"\n【最终结果】")
            print(f"  答案1 (情节提取): {answer1}")
            print(f"  答案2 (刑期预测): {answer2}")

            # 每处理10条数据保存一次,防止意外中断丢失进度
            if (idx + 1) % 10 == 0:
                print(f"\n--- 进度保存:已处理 {idx + 1} 条数据 ---")
                self._save_results(results, output_file)

        # 最终保存所有结果
        self._save_results(results, output_file)
        print(f"\n所有数据处理完成,结果已保存至: {output_file}")
        return results

    def process_fact_data(self, fact_data, output_file):
        """
        处理fact格式的数据（新格式）
        """
        results = []

        for idx, item in enumerate(fact_data):
            print(f"\n{'=' * 60}")
            print(f"处理第 {idx + 1}/{len(fact_data)} 条数据 (ID: {item['id']})")
            print(f"{'=' * 60}")

            answer1, answer2 = [], []
            try:
                # 第一步:调用权威版 Task 1 预测,提取量刑情节
                print("\n【步骤1: 提取量刑情节】")
                answer1 = self.predict_task1_authoritative(
                    "",  # 被告人信息为空
                    item['fact']  # 使用fact字段作为案情描述
                )
                print(f"✓ 提取到的情节: {answer1}")

                # 第二步:使用工具调用进行刑期预测
                print("\n【步骤2: 使用工具计算刑期】")
                # 从案件信息中提取地区
                answer2 = self.predict_task2_with_tools(
                    "",  # 被告人信息为空
                    item['fact'],  # 使用fact字段作为案情描述
                    answer1,
                )
                print(f"✓ 预测刑期区间: {answer2}")

            except Exception as e:
                print(f"!!! 处理ID {item['id']} 时发生未知严重错误: {e}")
                answer1 = answer1 if answer1 else ["盗窃数额较大"]
                answer2 = answer2 if answer2 else [6, 12]

            result = {
                "id": item['id'],
                "answer1": answer1,
                "answer2": answer2
            }
            results.append(result)

            print(f"\n【最终结果】")
            print(f"  答案1 (情节提取): {answer1}")
            print(f"  答案2 (刑期预测): {answer2}")

            # 每处理10条数据保存一次,防止意外中断丢失进度
            if (idx + 1) % 10 == 0:
                print(f"\n--- 进度保存:已处理 {idx + 1} 条数据 ---")
                self._save_results(results, output_file)

        # 最终保存所有结果
        self._save_results(results, output_file)
        print(f"\n所有数据处理完成,结果已保存至: {output_file}")
        return results

    def _save_results(self, results, output_file):
        """
        将结果以jsonl格式保存到文件。
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except IOError as e:
            print(f"错误:无法写入文件 {output_file}。请检查权限或路径。错误信息: {e}")


def load_preprocessed_data(preprocessed_file):
    """
    加载并验证预处理后的数据文件。
    """
    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(f"错误:预处理文件不存在: {preprocessed_file}\n请确保文件路径正确。")

    print(f"正在加载预处理数据: {preprocessed_file}")
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ 成功加载 {len(data)} 条预处理数据")
    return data


def load_fact_data(fact_file):
    """
    加载fact格式的数据文件。
    """
    if not os.path.exists(fact_file):
        raise FileNotFoundError(f"错误:数据文件不存在: {fact_file}\n请确保文件路径正确。")

    print(f"正在加载fact数据: {fact_file}")
    data = []
    with open(fact_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    print(f"✓ 成功加载 {len(data)} 条fact数据")
    return data


def main():
    """
    主函数:初始化并运行整个预测流程。
    """
    # 配置文件路径
    preprocessed_file = "extracted_info_fusai1.json"
    fact_file = "data/task6_fusai.jsonl"
    output_file = "submission_with_tools_fact_1114_2.jsonl"

    print("=" * 60)
    print(" 法律量刑预测系统 (工具调用版) ")
    print("=" * 60)

    # 检查fact文件是否存在
    if os.path.exists(fact_file):
        print(f"检测到fact格式数据文件: {fact_file}")
        try:
            fact_data = load_fact_data(fact_file)
        except Exception as e:
            print(f"\n加载fact数据时发生致命错误: {e}")
            return

        print("\n" + "=" * 60)
        print("开始模型预测...")
        print("=" * 60 + "\n")

        predictor = SentencingPredictor()
        results = predictor.process_fact_data(fact_data, output_file)

        print("\n" + "=" * 60)
        print("✓ 任务完成!")
        print(f"✓ 共处理 {len(results)} 条数据")
        print(f"✓ 结果已成功保存至: {output_file}")
        print("=" * 60)
        return

    # 如果没有fact文件，则尝试加载预处理文件
    try:
        preprocessed_data = load_preprocessed_data(preprocessed_file)
    except Exception as e:
        print(f"\n加载数据时发生致命错误: {e}")
        return

    print("\n" + "=" * 60)
    print("开始模型预测...")
    print("=" * 60 + "\n")

    predictor = SentencingPredictor()
    results = predictor.process_all_data(preprocessed_data, output_file)

    print("\n" + "=" * 60)
    print("✓ 任务完成!")
    print(f"✓ 共处理 {len(results)} 条数据")
    print(f"✓ 结果已成功保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()