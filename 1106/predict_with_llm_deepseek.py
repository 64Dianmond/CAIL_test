import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class SentencingPredictor:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.model_name = os.getenv("OPENAI_MODEL", "qwen3-max")
        self.temperature = 1  # 改回0.01以保证稳定性
        self.max_tokens = 8192

        # 罪名相关的量刑标准
        self.crime_standards = {
            "盗窃罪": {
                "法定刑": """
                - 数额较大(3000-30000元)或多次盗窃: 3年以下有期徒刑、拘役或管制
                - 数额巨大(30000-300000元)或其他严重情节: 3-10年有期徒刑
                - 数额特别巨大(300000元以上)或其他特别严重情节: 10年以上有期徒刑或无期徒刑
                """,
                "关键情节": [
                    "盗窃数额(既遂/未遂)",
                    "盗窃次数",
                    "数额档次(较大/巨大/特别巨大)",
                    "特殊方式(扒窃/入户盗窃/携带凶器盗窃)",
                    "自首",
                    "坦白",
                    "当庭自愿认罪/认罪认罚",
                    "退赔/赔偿损失",
                    "取得谅解",
                    "前科/累犯",
                    "从犯/主犯",
                    "未成年人",
                    "其他从轻/从重情节"
                ]
            },
            "故意伤害罪": {
                "法定刑": """
                - 致人轻伤: 3年以下有期徒刑、拘役或管制
                - 致人重伤: 3-10年有期徒刑
                - 致人死亡或以特别残忍手段致人重伤造成严重残疾: 10年以上有期徒刑、无期徒刑或死刑
                """,
                "关键情节": [
                    "伤害程度(致XX人轻伤一级/轻伤二级/重伤一级/重伤二级)",
                    "伤害后果/伤残等级",
                    "犯罪手段(是否残忍)",
                    "被害人过错",
                    "赔偿损失/积极赔偿",
                    "取得谅解",
                    "自首",
                    "坦白",
                    "当庭自愿认罪/认罪认罚",
                    "前科/累犯",
                    "从犯/主犯",
                    "未成年人",
                    "其他从轻/从重情节"
                ]
            },
            "诈骗罪": {
                "法定刑": """
                - 数额较大(3000-50000元): 3年以下有期徒刑、拘役或管制
                - 数额巨大(50000-500000元)或其他严重情节: 3-10年有期徒刑
                - 数额特别巨大(500000元以上)或其他特别严重情节: 10年以上有期徒刑或无期徒刑
                """,
                "关键情节": [
                    "诈骗数额(既遂/未遂)",
                    "诈骗次数",
                    "数额档次(较大/巨大/特别巨大)",
                    "诈骗手段(电信诈骗/网络诈骗/普通诈骗)",
                    "诈骗对象(老年人/学生等特殊群体)",
                    "自首",
                    "坦白",
                    "当庭自愿认罪/认罪认罚",
                    "退赔/退赃/赔偿损失",
                    "取得谅解",
                    "前科/累犯",
                    "从犯/主犯",
                    "未成年人",
                    "其他从轻/从重情节"
                ]
            }
        }

    def identify_crime_type(self, defendant_info, case_description):
        """识别罪名类型（增强版）"""
        text = defendant_info + case_description

        # 全文统一去空格
        text = text.replace(" ", "").replace("\n", "")

        # --- 盗窃罪 ---
        theft_keywords = [
            "盗窃", "窃取", "偷窃", "偷走", "盗走", "顺手牵羊", "撬锁入室",
            "扒窃", "偷拿", "盗得", "偷得", "偷取", "偷运", "盗入", "盗取"
        ]

        # --- 故意伤害罪 ---
        injury_keywords = [
            "故意伤害", "殴打", "打伤", "打架", "扭打", "持械伤人","厮打"
            "造成轻伤", "造成重伤", "轻伤", "重伤", "打击", "拳打脚踢", "刺伤", "砍伤"
        ]

        # --- 诈骗罪 ---
        fraud_keywords = [
            "诈骗", "骗取", "虚构事实", "隐瞒真相", "谎称", "骗得", "骗走",
            "以投资为名", "以买卖为名", "冒充他人", "冒充公司", "假借名义", "非法占有"
        ]

        # 判断逻辑（优先匹配更明确的罪名）
        if any(k in text for k in theft_keywords):
            return "盗窃罪"
        elif any(k in text for k in injury_keywords):
            return "故意伤害罪"
        elif any(k in text for k in fraud_keywords):
            return "诈骗罪"
        else:
            # 默认返回未知或盗窃罪
            return "盗窃罪"

    def build_prompt_task1(self, defendant_info, case_description):
        """构建子任务一的prompt：量刑情节识别"""
        # 识别罪名
        crime_type = self.identify_crime_type(defendant_info, case_description)
        crime_info = self.crime_standards.get(crime_type, self.crime_standards["盗窃罪"])

        key_factors = "\n".join([f"- {factor}" for factor in crime_info["关键情节"]])

        prompt = f"""你是一位专业的刑事法官，需要从案件中提取量刑情节。只提取有的量刑情节 ，没有的不需要提取。提取的量刑情节不需要解释。

本案罪名：{crime_type}

法定刑档次：
{crime_info["法定刑"]}

被告人信息：
{defendant_info}

案情描述：
{case_description}

请仔细分析案件事实，按照以下步骤提取量刑情节：

1. **构罪情节**（确定法定刑档次）：
   - 犯罪数额/犯罪后果
   - 数额档次或伤害程度
   - 特殊犯罪手段或方式

2. **基准刑情节**（确定基准刑）：
   - 既遂/未遂金额
   - 犯罪次数
   - 其他影响基准刑的情节

3. **调节情节**（调整基准刑）：
   - 自首、坦白、认罪认罚
   - 退赔、赔偿、取得谅解
   - 前科、累犯
   - 从犯、主犯
   - 未成年人、被害人过错
   - 其他从轻/从重情节

{crime_type}的关键量刑情节包括：
{key_factors}

**提取要求**：
- 每个情节要具体、量化，包含关键数字
- 区分既遂和未遂金额
- 明确标注数额档次（较大/巨大/特别巨大）或伤害程度
- 准确识别自首、坦白、认罪认罚等从轻情节
- 准确识别前科、累犯等从重情节

**输出格式**：
只输出JSON数组，每个元素是一个具体的量刑情节。

示例格式（{crime_type}）：
"""

        # 根据不同罪名给出示例
        if crime_type == "盗窃罪":
            prompt += '''["盗窃金额既遂3631元", "盗窃数额较大", "盗窃次数1次", "扒窃", "当庭自愿认罪", "前科"]'''
        elif crime_type == "故意伤害罪":
            prompt += '''["故意伤害致一人轻伤二级", "赔偿损失5000元", "取得谅解", "自首", "当庭认罪"]'''
        elif crime_type == "诈骗罪":
            prompt += '''["诈骗金额既遂50000元", "诈骗数额巨大", "诈骗次数3次", "电信诈骗", "认罪认罚", "退赔部分款项"]'''

        prompt += "\n\n只输出JSON数组，不要任何其他解释或说明。"

        return prompt

    def build_prompt_task2(self, defendant_info, case_description):
        """构建子任务二的prompt：刑期预测"""
        # 识别罪名
        crime_type = self.identify_crime_type(defendant_info, case_description)
        crime_info = self.crime_standards.get(crime_type, self.crime_standards["盗窃罪"])

        prompt = f"""你是一位专业的刑事法官，需要根据案件事实预测被告人的宣告刑期。

本案罪名：{crime_type}

法定刑档次：
{crime_info["法定刑"]}

被告人信息：
{defendant_info}

案情描述：
{case_description}

请按照严格的量刑步骤进行预测：

**第一步：确定法定刑档次**
"""

        # 根据罪名给出具体的档次判断标准
        if crime_type == "盗窃罪":
            prompt += """
- 数额3000-30000元或多次盗窃 → 3年以下(0-36个月)
- 数额30000-300000元 → 3-10年(36-120个月)
- 数额300000元以上 → 10年以上(120个月以上)
- 特殊方式(扒窃/入户/携带凶器)影响量刑起点
"""
        elif crime_type == "故意伤害罪":
            prompt += """
- 轻伤(一级/二级) → 3年以下(0-36个月)
- 重伤(一级/二级) → 3-10年(36-120个月)
- 死亡或特别残忍手段致重伤严重残疾 → 10年以上(120个月以上)
"""
        elif crime_type == "诈骗罪":
            prompt += """
- 数额3000-50000元 → 3年以下(0-36个月)
- 数额50000-500000元 → 3-10年(36-120个月)
- 数额500000元以上 → 10年以上(120个月以上)
- 电信网络诈骗可能从重处罚
"""

        prompt += """
**第二步：确定量刑起点**
根据法定刑档次和主要犯罪事实确定量刑起点。

**第三步：确定基准刑**
根据犯罪数额、次数、既遂未遂等情况调整基准刑。

**第四步：调整宣告刑**
综合考虑以下情节：
- 自首：可以从轻或减轻处罚(减少基准刑20%-40%)
- 坦白：可以从轻处罚(减少基准刑10%-20%)
- 认罪认罚：可以从宽处理(减少基准刑10%-30%)
- 取得谅解：可以从轻处罚(减少基准刑10%-20%)
- 积极退赔赔偿：可以从轻处罚(减少基准刑10%-30%)
- 前科：可以从重处罚(增加基准刑10%-20%)
- 累犯：应当从重处罚(增加基准刑10%-40%)
- 未成年人：应当从轻或减轻处罚(减少基准刑30%-50%)
- 从犯：应当从轻、减轻或免除处罚(减少基准刑20%-50%)

**第五步：考虑缓刑可能性**
- 3年以下有期徒刑，犯罪情节较轻，有悔罪表现，无再犯危险，可以宣告缓刑

**输出要求**：
1. 预测最终宣告刑期区间（单位：月）
2. 输出格式：[最小月数, 最大月数]
3. 区间范围应合理，通常相差3-12个月
4. 如果预测缓刑，刑期仍按实刑计算

**示例**：
- 盗窃5000元，自首，认罪，退赔 → [6, 10] (6-10个月)
- 盗窃50000元，前科，未退赔 → [36, 48] (3-4年)
- 故意伤害致轻伤二级，赔偿，谅解 → [6, 12] (6个月-1年)
- 诈骗80000元，自首，认罪认罚，部分退赔 → [24, 30] (2-2.5年)

只输出JSON数组格式的两个整数，不要任何其他解释或说明。
"""

        return prompt

    def predict_task1(self, defendant_info, case_description):
        """子任务一：量刑情节识别"""
        try:
            prompt = self.build_prompt_task1(defendant_info, case_description)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "你是一位经验丰富的刑事法官，精通盗窃罪、故意伤害罪、诈骗罪的量刑标准和情节认定。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            result = response.choices[0].message.content.strip()

            # 提取JSON数组
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)

            sentencing_factors = json.loads(result)
            return sentencing_factors

        except Exception as e:
            print(f"Task1预测出错: {e}")
            print(f"原始返回: {result if 'result' in locals() else 'N/A'}")
            return []

    def predict_task2(self, defendant_info, case_description):
        """子任务二：刑期预测"""
        try:
            prompt = self.build_prompt_task2(defendant_info, case_description)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "你是一位经验丰富的刑事法官，精通盗窃罪、故意伤害罪、诈骗罪的量刑标准和刑期计算。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            result = response.choices[0].message.content.strip()

            # 提取JSON数组
            json_match = re.search(r'\[\s*\d+\s*,\s*\d+\s*\]', result)
            if json_match:
                result = json_match.group(0)

            sentence_range = json.loads(result)
            return sentence_range

        except Exception as e:
            print(f"Task2预测出错: {e}")
            print(f"原始返回: {result if 'result' in locals() else 'N/A'}")
            return [0, 0]

    def process_all_data(self, preprocessed_data, output_file):
        """处理所有数据并生成提交文件"""
        results = []

        for idx, item in enumerate(preprocessed_data):
            print(f"处理第 {idx + 1}/{len(preprocessed_data)} 条数据 (ID: {item['id']})")

            # 预测子任务一
            answer1 = self.predict_task1(
                item['defendant_info'],
                item['case_description']
            )

            # 预测子任务二
            answer2 = self.predict_task2(
                item['defendant_info'],
                item['case_description']
            )

            result = {
                "id": item['id'],
                "answer1": answer1,
                "answer2": answer2
            }
            results.append(result)

            print(f"  答案1: {answer1}")
            print(f"  答案2: {answer2}\n")

            # 每处理5条保存一次
            if (idx + 1) % 5 == 0:
                self._save_results(results, output_file)

        # 最终保存
        self._save_results(results, output_file)
        print(f"所有数据处理完成，结果已保存至: {output_file}")

        return results

    def _save_results(self, results, output_file):
        """保存结果为jsonl格式"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


def load_preprocessed_data(preprocessed_file):
    """加载预处理后的数据"""
    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(f"预处理文件不存在: {preprocessed_file}\n"
                                f"请先运行预处理脚本生成该文件，或检查文件路径是否正确。")

    print(f"正在加载预处理数据: {preprocessed_file}")
    with open(preprocessed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ 成功加载 {len(data)} 条预处理数据")

    # 检查数据格式
    if data and isinstance(data, list):
        sample = data[0]
        required_fields = ['id', 'defendant_info', 'case_description']
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            raise ValueError(f"预处理数据缺少必要字段: {missing_fields}")
        print(f"✓ 数据格式验证通过")

    return data


def main():
    """
    主函数：直接读取预处理后的数据进行预测
    """
    # 配置文件路径
    preprocessed_file = "extracted_info1.json"  # 预处理后的文件
    output_file = "submission_ex_qwen3235_2.jsonl"  # 最终提交文件

    print("=" * 60)
    print("法律量刑预测系统")
    print("=" * 60)

    # 加载预处理数据
    try:
        preprocessed_data = load_preprocessed_data(preprocessed_file)
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示：如果还没有预处理数据，请先运行以下命令：")
        print("  python process.py")
        return
    except Exception as e:
        print(f"\n加载数据时出错: {e}")
        return

    # 开始模型预测
    print("\n" + "=" * 60)
    print("开始模型预测...")
    print("=" * 60)
    predictor = SentencingPredictor()
    results = predictor.process_all_data(preprocessed_data, output_file)

    print("\n" + "=" * 60)
    print("✓ 任务完成！")
    print(f"✓ 共处理 {len(results)} 条数据")
    print(f"✓ 结果已保存至: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
