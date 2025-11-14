"""
刑期计算器工具
提供精确的量刑计算功能，避免LLM直接进行数值计算
"""

import json
from typing import Dict, List, Union


class SentencingCalculator:
    """量刑计算器：用于精确计算刑期"""

    @staticmethod
    def calculate_base_sentence(crime_type: str, amount: float = None,
                                injury_level: str = None) -> int:
        """
        计算基准刑（单位：月）- 改进版
        采用"档位基准刑 + 超额累进"模式，参考大多数省份标准
        """
        if crime_type == "盗窃罪":
            # 全国统一标准（参考多数省份）
            # 数额较大: 3 000 – 30 000 元  ->  6 – 36 个月
            # 数额巨大: 30 000 – 300 000 元 -> 36 – 120 个月
            # 数额特别巨大: 300 000 以上 -> 120 – 180 个月
            if amount is None:
                return 12
            if amount < 2000:
                return 6  # 可能不构成犯罪或拘役
            elif amount < 30000:  # 数额较大档
                base = 6
                excess = amount - 2000
                additional = int(excess / 2000) * 1  # 每增加 2 000 加 1 个月
                return min(base + additional, 36)
            elif amount < 300000:  # 数额巨大档
                base = 36
                excess = amount - 30000
                additional = int(excess / 4000) * 1  # 每增加 5 000 加 1 个月
                return min(int(base + additional), 120)
            else:  # 数额特别巨大档
                base = 120
                excess = amount - 300000
                additional = int(excess / 50000) * 1  # 每增加 50 000 加 1 个月
                return min(int(base + additional), 180)

            # ---------- 诈骗罪 ----------
        elif crime_type == "诈骗罪":
            if amount is None:
                return 12
            if amount < 3000:
                return 6
            elif amount < 30000:  # 数额较大
                base = 6
                excess = amount - 3000
                additional = int(excess / 2000) * 1
                return min(base + additional, 36)
            elif amount < 500_000:  # 数额巨大（标准与盗窃不同）
                base = 36
                excess = amount - 30000
                additional = int(excess / 15000) * 1.5
                return min(int(base + additional), 120)
            else:  # 数额特别巨大
                base = 120
                excess = amount - 500000
                additional = int(excess / 100000) * 1
                return min(int(base + additional), 180)

            # ---------- 故意伤害罪 ----------
        elif crime_type == "故意伤害罪":
            injury_map = {
                "轻伤一级": 18,
                "轻伤二级": 12,
                "重伤一级": 72,
                "重伤二级": 48,
                "致人死亡": 120,
                "死亡": 120,
            }
            return injury_map.get(injury_level, 12)

            # 默认兜底
        return 12

    @staticmethod
    def calculate_layered_sentence_with_constraints(
            base_months: int,
            crime_type: str,
            amount: float,
            layer1_factors: List[Dict[str, Union[str, float]]],
            layer2_factors: List[Dict[str, Union[str, float]]],
            has_statutory_mitigation: bool = False,  # 是否有法定减轻情节
            injury_level: str = None  # 伤害等级（用于故意伤害罪）
    ) -> Dict[str, Union[float, str]]:
        """
        分层计算最终刑期 - 增强版,带约束条件
        """
        steps = []
        current_months = base_months
        steps.append(f"基准刑: {base_months}个月")

        # 确定法定刑档位范围(用于约束)
        legal_min, legal_max = SentencingCalculator._get_legal_range(
            crime_type, amount, injury_level
        )
        steps.append(f"本案法定刑档位: {legal_min}-{legal_max}个月")

        # 第一层面：连乘
        layer1_multiplier = 1.0
        for factor in layer1_factors:
            name = factor["name"]
            ratio = factor["ratio"]
            layer1_multiplier *= ratio
            steps.append(f"第一层面 - {name}: ×{ratio}")

        if layer1_factors:
            current_months = base_months * layer1_multiplier
            steps.append(f"第一层面结果: {current_months:.2f}个月")

        # 第二层面：加减
        layer2_adjustment = 0.0
        for factor in layer2_factors:
            name = factor["name"]
            ratio = factor["ratio"]
            adjustment = ratio - 1.0
            layer2_adjustment += adjustment
            steps.append(f"第二层面 - {name}: {'+' if adjustment > 0 else ''}{adjustment * 100:.0f}%")

        if layer2_factors:
            layer2_multiplier = 1.0 + layer2_adjustment
            temp_final = current_months * layer2_multiplier
            steps.append(f"第二层面初步结果: {temp_final:.2f}个月")
        else:
            temp_final = current_months

        # **关键约束1: 总减轻幅度不超过50%**
        total_reduction_ratio = (base_months - temp_final) / base_months
        if total_reduction_ratio > 0.5 and not has_statutory_mitigation:
            # 无法定减轻情节时,最多减50%
            adjusted_final = base_months * 0.5
            steps.append(f"⚠️ 约束调整: 总减轻幅度超过50%({total_reduction_ratio * 100:.1f}%)")
            steps.append(f"   调整至基准刑的50%: {adjusted_final:.2f}个月")
            temp_final = adjusted_final

        # **关键约束2: 只有在没有法定减轻情节的情况下，刑期不能低于档位最低刑期**
        # if temp_final < legal_min and not has_statutory_mitigation:
        #     steps.append(f"⚠️ 约束调整: 结果({temp_final:.2f}月)低于法定刑下限({legal_min}月)")
        #     steps.append(f"   调整至法定刑下限: {legal_min}个月")
        #     temp_final = legal_min
        # 有法定减轻情节的情况下，最低可以到1个月
        elif temp_final < 1:
            steps.append(f"⚠️ 约束调整: 结果({temp_final:.2f}月)低于1个月")
            steps.append(f"   调整至最低刑期: 1个月")
            temp_final = 1

        # **关键约束3: 不得超过本档法定刑上限**
        if temp_final > legal_max:
            steps.append(f"⚠️ 约束调整: 结果({temp_final:.2f}月)超过法定刑上限({legal_max}月)")
            steps.append(f"   调整至法定刑上限: {legal_max}个月")
            temp_final = legal_max

        final_months = round(temp_final, 2)

        return {
            "final_months": final_months,
            "base_months": base_months,
            "legal_range": [legal_min, legal_max],
            "total_reduction_ratio": round(total_reduction_ratio * 100, 1),
            "calculation_steps": steps,
            "constrained": temp_final != current_months * (1.0 + layer2_adjustment)
        }

    @staticmethod
    def _get_legal_range(crime_type: str, amount: float, injury_level: str = None) -> tuple:
        """
        获取法定刑档位的上下限
        """
        if crime_type == "盗窃罪":
            if amount < 30000:
                return (6, 36)  # 三年以下 = 6-36个月
            elif amount < 300000:
                return (36, 120)  # 三年以上十年以下 = 36-120个月
            else:
                return (120, 180)  # 十年以上 = 120-180个月(无期除外)

        elif crime_type == "诈骗罪":
            if amount < 30000:
                return (6, 36)
            elif amount < 500000:
                return (36, 120)
            else:
                return (120, 180)
        
        elif crime_type == "故意伤害罪":
            # 根据最高检相关解释和刑法规定确定故意伤害罪的法定刑范围
            injury_range_map = {
                # 轻伤（三年以下有期徒刑、拘役或者管制）
                "轻伤一级": (6, 36),   # 1年至3年
                "轻伤二级": (1, 36),   # 6个月至3年
                
                # 重伤（三年以上十年以下有期徒刑）
                "重伤一级": (72, 120), # 6年至10年
                "重伤二级": (36, 96),  # 3年至8年
                
                # 致人死亡或特别残忍手段致人重伤造成严重残疾（十年以上有期徒刑、无期徒刑或者死刑）
                "致人死亡": (120, 180), # 10年至15年
                "死亡": (120, 180)     # 10年至15年
            }
            # 默认返回较宽泛的范围
            return injury_range_map.get(injury_level, (1, 180))

        # 默认
        return (6, 120)

    @staticmethod
    def apply_factor(base_months: int, factor_name: str, factor_ratio: float) -> float:
        """
        应用单个情节调节因子

        Args:
            base_months: 基准月数
            factor_name: 情节名称
            factor_ratio: 调节比例（如0.5表示减少50%，1.3表示增加30%）

        Returns:
            调节后的月数
        """
        result = base_months * factor_ratio
        return round(result, 2)

    # @staticmethod
    # def calculate_layered_sentence(
    #         base_months: int,
    #         layer1_factors: List[Dict[str, Union[str, float]]],
    #         layer2_factors: List[Dict[str, Union[str, float]]]
    # ) -> Dict[str, Union[float, str]]:
    #     """
    #     分层计算最终刑期
    #
    #     Args:
    #         base_months: 基准刑（月）
    #         layer1_factors: 第一层面情节列表 [{"name": "未成年人", "ratio": 0.5}, ...]
    #         layer2_factors: 第二层面情节列表 [{"name": "累犯", "ratio": 0.3}, ...]
    #
    #     Returns:
    #         计算结果字典，包含最终月数和计算步骤
    #     """
    #     steps = []
    #     current_months = base_months
    #     steps.append(f"基准刑: {base_months}个月")
    #
    #     # 第一层面：连乘
    #     layer1_multiplier = 1.0
    #     for factor in layer1_factors:
    #         name = factor["name"]
    #         ratio = factor["ratio"]
    #         layer1_multiplier *= ratio
    #         steps.append(f"第一层面 - {name}: ×{ratio}")
    #
    #     if layer1_factors:
    #         current_months = base_months * layer1_multiplier
    #         steps.append(f"第一层面计算结果: {base_months} × {layer1_multiplier} = {current_months:.2f}个月")
    #
    #     # 第二层面：加减
    #     layer2_adjustment = 0.0
    #     for factor in layer2_factors:
    #         name = factor["name"]
    #         ratio = factor["ratio"]
    #         # ratio为正数表示从重（如1.3表示+30%），为负数表示从轻（如0.9表示-10%）
    #         adjustment = ratio - 1.0  # 转换为调节比例
    #         layer2_adjustment += adjustment
    #         steps.append(f"第二层面 - {name}: {'+' if adjustment > 0 else ''}{adjustment * 100:.0f}%")
    #
    #     if layer2_factors:
    #         layer2_multiplier = 1.0 + layer2_adjustment
    #         final_months = current_months * layer2_multiplier
    #         steps.append(f"第二层面计算结果: {current_months:.2f} × {layer2_multiplier} = {final_months:.2f}个月")
    #     else:
    #         final_months = current_months
    #
    #     return {
    #         "final_months": round(final_months, 2),
    #         "calculation_steps": steps,
    #         "formula": f"{base_months} × L1({layer1_multiplier}) × L2({1.0 + layer2_adjustment})"
    #     }

    @staticmethod
    def calculate_simple_adjustment(base_months: int, adjustment_percent: float) -> int:
        """
        简单的百分比调节计算

        Args:
            base_months: 基准月数
            adjustment_percent: 调节百分比（如30表示增加30%，-10表示减少10%）

        Returns:
            调节后的月数（整数）
        """
        multiplier = 1.0 + (adjustment_percent / 100.0)
        result = base_months * multiplier
        return round(result)

    @staticmethod
    def months_to_range(center_months: float, width: int = 6) -> List[int]:
        """
        将中心月数转换为刑期区间

        Args:
            center_months: 中心月数
            width: 区间宽度（默认6个月）

        Returns:
            [最小月数, 最大月数]
        """
        half_width = width // 2
        # 确保不低于1个月，同时正确处理浮点数
        min_months = max(1, round(center_months - half_width))
        max_months = max(1, round(center_months + half_width))
        return [min_months, max_months]

    @staticmethod
    def validate_legal_range(months: int, min_legal: int, max_legal: int) -> int:
        """
        验证并调整刑期是否在法定范围内

        Args:
            months: 计算出的月数
            min_legal: 法定最低月数
            max_legal: 法定最高月数

        Returns:
            调整后的合法月数
        """
        if months < min_legal:
            return min_legal
        elif months > max_legal:
            return max_legal
        return months


# 工具函数定义（OpenAI Function Calling格式）
SENTENCING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_base_sentence",
            "description": "根据罪名和犯罪事实计算基准刑",
            "parameters": {
                "type": "object",
                "properties": {
                    "crime_type": {
                        "type": "string",
                        "enum": ["盗窃罪", "诈骗罪", "故意伤害罪"],
                        "description": "罪名类型"
                    },
                    "amount": {
                        "type": "number",
                        "description": "犯罪金额（元），适用于盗窃罪和诈骗罪"
                    },
                    "injury_level": {
                        "type": "string",
                        "enum": ["轻伤一级", "轻伤二级", "重伤一级", "重伤二级", "致人死亡", "死亡"],
                        "description": "伤害等级，适用于故意伤害罪"
                    }
                },
                "required": ["crime_type"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "calculate_layered_sentence",
    #         "description": "根据分层量刑规则精确计算最终刑期。第一层面情节（未成年人、从犯等）使用连乘，第二层面情节（累犯、自首等）使用加减",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "base_months": {
    #                     "type": "integer",
    #                     "description": "基准刑月数"
    #                 },
    #                 "layer1_factors": {
    #                     "type": "array",
    #                     "items": {
    #                         "type": "object",
    #                         "properties": {
    #                             "name": {
    #                                 "type": "string",
    #                                 "description": "情节名称，如'未成年人'、'从犯'"
    #                             },
    #                             "ratio": {
    #                                 "type": "number",
    #                                 "description": "调节比例（乘数），如0.5表示减半，0.8表示减少20%"
    #                             }
    #                         },
    #                         "required": ["name", "ratio"]
    #                     },
    #                     "description": "第一层面情节列表（连乘因子）"
    #                 },
    #                 "layer2_factors": {
    #                     "type": "array",
    #                     "items": {
    #                         "type": "object",
    #                         "properties": {
    #                             "name": {
    #                                 "type": "string",
    #                                 "description": "情节名称，如'累犯'、'自首'、'坦白'"
    #                             },
    #                             "ratio": {
    #                                 "type": "number",
    #                                 "description": "调节比例（乘数），如1.3表示增加30%，0.9表示减少10%"
    #                             }
    #                         },
    #                         "required": ["name", "ratio"]
    #                     },
    #                     "description": "第二层面情节列表（加减因子）"
    #                 }
    #             },
    #             "required": ["base_months", "layer1_factors", "layer2_factors"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "months_to_range",
            "description": "将中心月数转换为刑期区间",
            "parameters": {
                "type": "object",
                "properties": {
                    "center_months": {
                        "type": "number",
                        "description": "中心月数"
                    },
                    "width": {
                        "type": "integer",
                        "description": "区间宽度（默认4个月）",
                        "default": 4
                    }
                },
                "required": ["center_months"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_layered_sentence_with_constraints",
            "description": "带约束条件的分层量刑计算,确保不超50%减轻且不低于法定刑下限",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_months": {"type": "integer"},
                    "crime_type": {"type": "string", "enum": ["盗窃罪", "诈骗罪", "故意伤害罪"]},
                    "amount": {"type": "number"},
                    "layer1_factors": {"type": "array", "items": {"type": "object"}},
                    "layer2_factors": {"type": "array", "items": {"type": "object"}},
                    "has_statutory_mitigation": {
                        "type": "boolean",
                        "description": "是否有法定减轻处罚情节(自首/立功/未成年人等)"
                    },
                    "injury_level": {
                        "type": "string",
                        "description": "伤害等级，适用于故意伤害罪",
                        "enum": ["轻伤一级", "轻伤二级","重伤一级", "重伤二级", "致人死亡", "死亡"]
                    }
                },
                "required": ["base_months", "crime_type", "amount", "layer1_factors", "layer2_factors"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_legal_range",
            "description": "验证并调整刑期是否在法定范围内",
            "parameters": {
                "type": "object",
                "properties": {
                    "months": {
                        "type": "integer",
                        "description": "计算出的月数"
                    },
                    "min_legal": {
                        "type": "integer",
                        "description": "法定最低月数"
                    },
                    "max_legal": {
                        "type": "integer",
                        "description": "法定最高月数"
                    }
                },
                "required": ["months", "min_legal", "max_legal"]
            }
        }
    }
]


def execute_tool_call(tool_name: str, tool_arguments: dict) -> str:
    """
    执行工具调用

    Args:
        tool_name: 工具名称
        tool_arguments: 工具参数

    Returns:
        执行结果的JSON字符串
    """
    calculator = SentencingCalculator()

    try:
        if tool_name == "calculate_base_sentence":
            result = calculator.calculate_base_sentence(**tool_arguments)
            return json.dumps({"base_months": result}, ensure_ascii=False)

        # elif tool_name == "calculate_layered_sentence":
        #     result = calculator.calculate_layered_sentence(**tool_arguments)
        #     return json.dumps(result, ensure_ascii=False)

        elif tool_name == "calculate_layered_sentence_with_constraints":
            result = calculator.calculate_layered_sentence_with_constraints(**tool_arguments)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "months_to_range":
            result = calculator.months_to_range(**tool_arguments)
            return json.dumps({"range": result}, ensure_ascii=False)

        elif tool_name == "validate_legal_range":
            result = calculator.validate_legal_range(**tool_arguments)
            return json.dumps({"validated_months": result}, ensure_ascii=False)

        else:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


if __name__ == "__main__":
    # 测试示例
    calc = SentencingCalculator()

    # 示例1：计算基准刑
    print("=== 示例1：计算基准刑 ===")
    base = calc.calculate_base_sentence("盗窃罪", amount=3631)
    print(f"基准刑: {base}个月\n")

    # 示例2：分层计算
    print("=== 示例2：分层计算 ===")
    result = calc.calculate_layered_sentence(
        base_months=100,
        layer1_factors=[
            {"name": "未成年人", "ratio": 0.5},
            {"name": "从犯", "ratio": 0.8}
        ],
        layer2_factors=[
            {"name": "累犯", "ratio": 1.3},
            {"name": "自首", "ratio": 0.9}
        ]
    )
    print(f"最终刑期: {result['final_months']}个月")
    print("计算步骤:")
    for step in result['calculation_steps']:
        print(f"  {step}")
    print()

    # 示例3：转换为区间
    print("=== 示例3：转换为区间 ===")
    range_result = calc.months_to_range(48, width=4)
    print(f"刑期区间: {range_result}\n")
