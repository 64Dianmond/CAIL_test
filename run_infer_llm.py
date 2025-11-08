import json
import openai
from typing import Dict, List, Tuple
import time
import re


class SentencePredictionSystem:
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model_name = model_name

    def create_prompt(self, fact: str, sentencing_factors: List[str]) -> str:
        """根据最高检量刑指导意见创建提示词"""
        factors_text = "\n".join([f"- {factor}" for factor in sentencing_factors])

        prompt = f"""你是刑事法官，严格按照《最高人民法院 最高人民检察院关于常见犯罪的量刑指导意见》计算刑期。

# 第一步：确定量刑起点

## 盗窃罪（刑法第264条）
- 数额较大/三次盗窃/入户/携带凶器/扒窃：在**1年以下有期徒刑、拘役**幅度确定量刑起点（建议6-12个月）
- 数额巨大或其他严重情节：在**3-4年有期徒刑**幅度确定量刑起点（建议36-48个月）
- 数额特别巨大或其他特别严重情节：在**10-12年有期徒刑**幅度确定量刑起点（建议120-144个月）

数额标准参考：
- 数额较大：1000-3000元起点
- 数额巨大：3-10万元起点  
- 数额特别巨大：30-50万元起点

## 故意伤害罪（刑法第234条）
- 致一人轻伤：在**2年以下有期徒刑、拘役**幅度确定量刑起点（建议12-24个月）
- 致一人重伤：在**3-5年有期徒刑**幅度确定量刑起点（建议36-60个月）
- 以特别残忍手段致一人重伤，造成六级严重残疾：在**10-13年有期徒刑**幅度确定量刑起点（建议120-156个月）

## 诈骗罪（刑法第266条）
- 数额较大：在**1年以下有期徒刑、拘役**幅度确定量刑起点（建议6-12个月）
- 数额巨大或其他严重情节：在**3-4年有期徒刑**幅度确定量刑起点（建议36-48个月）
- 数额特别巨大或其他特别严重情节：在**10-12年有期徒刑**幅度确定量刑起点（建议120-144个月）

# 第二步：根据数额等增加刑罚量确定基准刑

- 盗窃/诈骗：超过起点数额的部分，每增加一定金额，递增刑期
  * 数额较大：每增加1000元，增加1个月
  * 数额巨大：每增加1万元，增加1个月
  * 数额特别巨大：每增加10万元，增加1个月

- 故意伤害：根据伤残等级、伤害后果严重程度增加刑期

# 第三步：应用量刑情节调节基准刑

## 从重情节（增加基准刑）
- **累犯**：+10%-40%（一般不少于3个月）
- **前科**（非过失、非未成年）：+10%以下
- **多次犯罪**：视具体次数增加
- **团伙/共同犯罪**：视地位作用确定

## 从轻减轻情节（减少基准刑）
- **自首**：-40%以下；犯罪较轻的-40%以上或免除
- **坦白**：-20%以下；供述同种较重罪行-10%-30%
- **认罪认罚**：-30%以下；有自首/坦白等-60%以下；犯罪较轻-60%以上或免除
- **当庭自愿认罪**：-10%以下
- **立功**：一般立功-20%以下；重大立功-20%-50%
- **退赃退赔**：-30%以下（抢劫等严重暴力犯罪从严）
- **赔偿+谅解**：-40%以下
- **赔偿未谅解**：-30%以下
- **仅谅解未赔偿**：-20%以下
- **刑事和解**：-50%以下；犯罪较轻-50%以上或免除
- **未遂**：-50%以下
- **从犯**：-20%-50%；犯罪较轻-50%以上或免除
- **初犯偶犯**：酌情从轻
- **羁押期间表现好**：-10%以下

注意：认罪认罚与自首、坦白、当庭认罪、退赃退赔、赔偿谅解、和解、羁押表现等**不重复评价**

# 案件信息

案件事实：
{fact}

量刑情节：
{factors_text}

# 计算要求

1. 先确定罪名和严重程度，选择正确的量刑起点区间
2. 根据具体数额/伤情在起点基础上增加刑罚量，得到基准刑
3. 识别所有量刑情节，计算调节比例：
   - 同向情节相加（多个从重相加，多个从轻相加）
   - 从重比例 - 从轻比例 = 总调节比例
4. 最终刑期 = 基准刑 × (1 + 总调节比例)
5. 考虑法定刑幅度限制，给出合理区间

# 输出格式

只输出两个整数（月份），逗号分隔，如：28,34

请计算：
"""
        return prompt

    def predict_sentence_range(self, fact: str, sentencing_factors: List[str],
                               max_retries: int = 3) -> Tuple[int, int]:
        """预测刑期区间"""
        prompt = self.create_prompt(fact, sentencing_factors)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是专业刑事法官，严格按照最高检量刑指导意见计算。只输出两个数字（月份），用逗号分隔。"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # 降低温度提高一致性
                    max_tokens=150,
                    top_p=0.9
                )

                result_text = response.choices[0].message.content.strip()
                print(f"\n模型原始输出: {result_text}")

                # 提取数字
                numbers = re.findall(r'\d+', result_text)
                if len(numbers) >= 2:
                    lower = int(numbers[0])
                    upper = int(numbers[1])

                    # 验证合理性
                    if 1 <= lower <= upper <= 180 and (upper - lower) <= 60:
                        return (lower, upper)
                    else:
                        print(f"尝试 {attempt + 1}: 刑期区间不合理 [{lower}, {upper}]")

            except Exception as e:
                print(f"尝试 {attempt + 1}: API调用失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        # 备用规则计算
        return self._calculate_by_rules(fact, sentencing_factors)

    def _calculate_by_rules(self, fact: str, sentencing_factors: List[str]) -> Tuple[int, int]:
        """根据量刑指导意见规则计算（备用方案）"""
        factors_text = " ".join(sentencing_factors).lower()

        # 第一步：确定量刑起点
        base_months = 12
        crime_type = ""

        # 识别罪名和严重程度
        if "盗窃" in factors_text:
            crime_type = "盗窃"
            if "特别巨大" in factors_text:
                base_months = 120  # 10-12年起点
            elif "巨大" in factors_text:
                base_months = 36  # 3-4年起点
            else:  # 数额较大
                base_months = 9  # 1年以下起点

        elif "诈骗" in factors_text:
            crime_type = "诈骗"
            if "特别巨大" in factors_text:
                base_months = 120
            elif "巨大" in factors_text:
                base_months = 36
            else:
                base_months = 9

        elif "伤害" in factors_text:
            crime_type = "伤害"
            if "死亡" in factors_text or "六级" in factors_text or "特别残忍" in factors_text:
                base_months = 132  # 10-13年起点
            elif "重伤" in factors_text:
                base_months = 48  # 3-5年起点
            else:  # 轻伤
                base_months = 18  # 2年以下起点

        # 第二步：根据数额增加刑罚量
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*万', factors_text)
        if amount_match and crime_type in ["盗窃", "诈骗"]:
            amount = float(amount_match.group(1))

            if "特别巨大" in factors_text:  # 假设50万起点
                extra = amount - 50
                if extra > 0:
                    base_months += int(extra / 10)  # 每10万增加1个月
            elif "巨大" in factors_text:  # 假设3万起点
                extra = amount - 3
                if extra > 0:
                    base_months += int(extra)  # 每1万增加1个月

        # 第三步：计算量刑情节调节比例
        adjustment = 0.0

        # 从重情节
        if "累犯" in factors_text:
            adjustment += 0.25  # +10%-40%，取中位
        if "前科" in factors_text and "累犯" not in factors_text:
            adjustment += 0.05  # +10%以下

        # 从轻情节（注意不重复评价）
        has_plea = "认罪认罚" in factors_text
        has_confession = "自首" in factors_text or "坦白" in factors_text or "如实供述" in factors_text
        has_restitution = "退赃" in factors_text or "退赔" in factors_text or "赔偿" in factors_text
        has_reconciliation = "谅解" in factors_text or "和解" in factors_text

        if has_plea:
            # 认罪认罚包含多个情节
            if has_confession and has_restitution and has_reconciliation:
                adjustment -= 0.60  # 有自首+退赃+谅解，-60%以下
            elif has_confession or (has_restitution and has_reconciliation):
                adjustment -= 0.45  # 有部分从宽情节，-30%-60%
            else:
                adjustment -= 0.25  # 仅认罪认罚，-30%以下
        else:
            # 单独评价各情节
            if "自首" in factors_text:
                adjustment -= 0.35  # -40%以下
            elif "坦白" in factors_text or "如实供述" in factors_text:
                adjustment -= 0.18  # -20%以下
            elif "当庭" in factors_text and "认罪" in factors_text:
                adjustment -= 0.08  # -10%以下

            if "退赃" in factors_text or "退赔" in factors_text:
                adjustment -= 0.20  # -30%以下

            if "赔偿" in factors_text and "谅解" in factors_text:
                adjustment -= 0.35  # -40%以下
            elif "赔偿" in factors_text:
                adjustment -= 0.25  # -30%以下
            elif "谅解" in factors_text:
                adjustment -= 0.15  # -20%以下

        # 其他情节
        if "立功" in factors_text:
            if "重大" in factors_text:
                adjustment -= 0.35  # -20%-50%
            else:
                adjustment -= 0.15  # -20%以下

        if "未遂" in factors_text:
            adjustment -= 0.45  # -50%以下

        if "从犯" in factors_text:
            adjustment -= 0.35  # -20%-50%

        if "初犯" in factors_text:
            adjustment -= 0.05  # 酌情从轻

        # 第四步：计算最终刑期
        final_months = int(base_months * (1 + adjustment))
        final_months = max(1, min(final_months, 180))  # 限制在1-180月

        # 生成合理区间（上下浮动2-4个月）
        range_width = 3
        lower = max(1, final_months - range_width)
        upper = min(180, final_months + range_width)

        print(f"\n规则计算详情:")
        print(f"  罪名: {crime_type}")
        print(f"  量刑起点: {base_months}月")
        print(f"  调节比例: {adjustment:+.1%}")
        print(f"  最终刑期: [{lower}, {upper}]月")

        return (lower, upper)

    def process_dataset(self, facts_file: str, submission_file: str, output_file: str):
        """处理整个数据集"""
        # 读取案件事实
        facts_dict = {}
        with open(facts_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                facts_dict[data['id']] = data['fact']

        # 读取量刑情节
        submissions = []
        with open(submission_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                submissions.append(data)

        total = len(submissions)
        print(f"共有 {total} 个案件需要处理\n")

        results = []
        for idx, submission in enumerate(submissions, 1):
            try:
                case_id = submission['id']
                sentencing_factors = submission['answer1']
                fact = facts_dict.get(case_id, "")

                print(f"\n{'=' * 60}")
                print(f"[{idx}/{total}] 案件 ID: {case_id}")
                print(f"量刑情节: {sentencing_factors}")

                # 预测刑期区间
                lower, upper = self.predict_sentence_range(fact, sentencing_factors)

                result = {
                    'id': case_id,
                    'answer1': sentencing_factors,
                    'answer2': [lower, upper]
                }
                results.append(result)

                print(f"✓ 预测刑期: [{lower}, {upper}]个月")

                # 每处理5个案件保存一次
                if idx % 1 == 0 or idx == total:
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for r in results:
                            out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    print(f"已保存前 {len(results)} 个案件")

                time.sleep(0.3)

            except Exception as e:
                print(f"✗ 处理失败: {e}")
                # 出错时使用规则计算
                lower, upper = self._calculate_by_rules(fact, sentencing_factors)
                results.append({
                    'id': case_id,
                    'answer1': sentencing_factors,
                    'answer2': [lower, upper]
                })

        print(f"\n{'=' * 60}")
        print(f"处理完成！共 {len(results)} 个案件")
        print(f"结果保存至: {output_file}")

        return results


if __name__ == "__main__":
    API_BASE = "http://45.150.227.150:32770/v1"
    API_KEY = "sk-UWIMnDHAeMzgBjs4OTW7tWN7ZcfmM6wGerisq45cHquU0cel"
    MODEL_NAME = "gemini-2.5-pro"

    system = SentencePredictionSystem(
        api_base=API_BASE,
        api_key=API_KEY,
        model_name=MODEL_NAME
    )

    system.process_dataset(
        facts_file='data/task6.jsonl',
        submission_file='submission_final.jsonl',
        output_file='submission_updated_ge1.jsonl'
    )
