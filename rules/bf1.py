# project/rules/sentencing_rules.py
"""
量刑规则引擎 - 基于《量刑指导意见（试行）》
支持30种常见犯罪的量刑计算
目标：最窄区间策略 - 零宽命中时 Winkler Score = 0
优化内容：
1. 规则外部配置化：所有量刑起点范围、数额/数量阈值、调节比例等从 `rules_config.json` 加载。
2. 逻辑层级清晰化：严格遵循“量刑起点 -> 基准刑增加刑罚量 -> 量刑情节调节 -> 宣告刑”的步骤。
3. 常见量刑情节优先顺序和“不作重复评价”：尤其是认罪认罚与自首、坦白等情节的叠加逻辑。
4. 刑种和法定刑上限：区分有期徒刑、拘役、无期徒刑等，处理“依法应当判处无期徒刑的除外”等特殊情况。
5. 错误处理与日志：增加基础的错误处理，但为简洁起见，本例未引入完整的日志模块。
"""
import re
import json
import os
from typing import List, Tuple, Optional, Dict, Union

# 定义常量
# 这些常量从配置中加载，但为了类型提示和默认值，在这里先定义。
# 实际值会在__init__中被config覆盖
SENTENCE_TYPE_LIFE_IMPRISONMENT = "无期徒刑"
SENTENCE_TYPE_DEATH_PENALTY = "死刑"
MAX_YEARS_IMPRISONMENT = 15 # 有期徒刑最高15年，即180个月

class SentencingRulesEngine:
    """量刑规则引擎 - 优化版本"""

    def __init__(self, narrow_mode: str = "aggressive", config_path: str = None):
        self.narrow_mode = narrow_mode
        self.config = self._load_config(config_path)

        # 从配置中加载通用常量
        common_constants = self.config.get("COMMON_CONSTANTS", {})
        global SENTENCE_TYPE_LIFE_IMPRISONMENT
        global SENTENCE_TYPE_DEATH_PENALTY
        global MAX_YEARS_IMPRISONMENT

        SENTENCE_TYPE_LIFE_IMPRISONMENT = common_constants.get("SENTENCE_TYPE_LIFE_IMPRISONMENT", "无期徒刑")
        SENTENCE_TYPE_DEATH_PENALTY = common_constants.get("SENTENCE_TYPE_DEATH_PENALTY", "死刑")
        MAX_YEARS_IMPRISONMENT = common_constants.get("MAX_YEARS_IMPRISONMENT", 15)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载量刑规则配置文件"""
        if config_path is None:
            # 默认从当前文件所在目录的 rules_config.json 加载
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, "rules_config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"量刑规则配置文件未找到: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def detect_crime(self, fact: str, labels: List[str]) -> str:
        """识别罪名（优化优先级）"""
        text = fact + "".join(labels)

        # 优先级匹配，长的词组在前，避免误判
        # 这里的关键词可以进一步通过配置管理，但为简洁暂时硬编码
        crime_keywords = [
            ("集资诈骗", "集资诈骗罪"), ("信用卡诈骗", "信用卡诈骗罪"), ("合同诈骗", "合同诈骗罪"),
            ("非法吸收公众存款", "非法吸收公众存款罪"),
            ("走私、贩卖、运输、制造毒品", "毒品罪"), ("非法持有毒品", "非法持有毒品罪"),
            ("容留他人吸毒", "容留他人吸毒罪"),
            ("掩饰、隐瞒犯罪所得", "掩饰、隐瞒犯罪所得罪"),
            ("侵犯公民个人信息", "侵犯公民个人信息罪"), ("帮助信息网络犯罪活动", "帮助信息网络犯罪活动罪"),
            ("拒不执行判决、裁定", "拒不执行判决、裁定罪"),
            ("引诱、容留、介绍卖淫", "引诱、容留、介绍卖淫罪"),
            ("交通肇事", "交通肇事罪"), ("危险驾驶", "危险驾驶罪"), ("故意伤害", "故意伤害罪"),
            ("强奸", "强奸罪"), ("非法拘禁", "非法拘禁罪"), ("抢劫", "抢劫罪"),
            ("盗窃", "盗窃罪"), ("诈骗", "诈骗罪"), ("抢夺", "抢夺罪"),
            ("职务侵占", "职务侵占罪"), ("敲诈勒索", "敲诈勒索罪"), ("妨害公务", "妨害公务罪"),
            ("聚众斗殴", "聚众斗殴罪"), ("寻衅滋事", "寻衅滋事罪"), ("组织卖淫", "组织卖淫罪"),
            ("非法经营", "非法经营罪"), ("猥亵儿童", "猥亵儿童罪"), ("开设赌场", "开设赌场罪"),
        ]

        for keyword, crime_name in crime_keywords:
            if keyword in text:
                return crime_name

        # 兜底：如果没匹配到，返回一个通用罪名
        return "盗窃罪"

    def parse_amount(self, labels: List[str]) -> Optional[float]:
        """提取金额（元）"""
        for label in labels:
            match = re.search(r"(?:金额|数额|价值|所得|违法所得|赌资|涉案财物|非法经营数额).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)", label)
            if match:
                amount = float(match.group(1))
                if "万元" in match.group(0):
                    amount *= 10000
                return amount
        return None

    def parse_count(self, labels: List[str], keyword: str = "次数") -> int:
        """提取次数或人数"""
        for label in labels:
            match = re.search(rf"{keyword}[：:\s]*([0-9]+)\s*次", label)
            if match:
                return int(match.group(1))
        return 1

    def _get_threshold_bucket(self, crime_config: Dict, amount: float, labels_str: str, default_bucket: str = "较大") -> str:
        """根据金额和标签确定财产类犯罪的档位"""
        amount_thresholds = crime_config.get("amount_thresholds")
        if not amount_thresholds:
            return default_bucket # 如果没有金额阈值，直接返回默认档位

        # 优先通过标签匹配
        if "特别巨大" in labels_str: return "特别巨大"
        if "巨大" in labels_str: return "巨大"
        if "较大" in labels_str: return "较大"
        if "数额特别巨大" in labels_str: return "特别巨大" # 兼容非法吸收公众存款罪
        if "数额巨大" in labels_str: return "巨大"
        if "情节严重" in labels_str: return "严重" # 例如非法经营罪
        if "情节特别严重" in labels_str: return "特别严重"

        # 根据金额判断
        if amount >= amount_thresholds.get("特别巨大", float('inf')):
            return "特别巨大"
        elif amount >= amount_thresholds.get("巨大", float('inf')):
            return "巨大"
        elif amount >= amount_thresholds.get("较大", float('inf')):
            return "较大"
        return default_bucket # 如果金额未达到“较大”标准，返回默认档位

    def get_statutory_range(self, crime: str, amount: Optional[float], labels: List[str]) -> Tuple[
        str, Tuple[int, int], Optional[str]]:
        """
        确定量刑起点幅度
        返回: (档位, (量刑起点下限月, 量刑起点上限月), 特殊备注)
        """
        labels_str = "".join(labels)
        crime_config = self.config["CRIMES"].get(crime)
        if not crime_config:
            # 未知罪名的兜底处理
            return "一般", (1, 36), None # 默认1-36个月

        bucket = "基本" # 默认档位名

        # 根据罪名类型判断档位
        if "amount_thresholds" in crime_config: # 财产类犯罪
            amt = amount if amount is not None else 0
            bucket = self._get_threshold_bucket(crime_config, amt, labels_str, default_bucket="较大")
            if crime == "非法吸收公众存款罪" and bucket == "较大": bucket = "一般" # 适配其档位命名

        # 针对特定罪名根据标签确定档位
        for bucket_name in crime_config.get("sentencing_ranges", {}):
            if f"labels_for_{bucket_name}" in crime_config:
                if any(kw in labels_str for kw in crime_config[f"labels_for_{bucket_name}"]):
                    bucket = bucket_name
                    break # 找到最高优先级匹配的档位

        # 如果没有匹配到特定档位，使用默认档位
        if bucket not in crime_config["sentencing_ranges"]:
            # 对于有多个档位但未匹配到的情况，尝试寻找“基本”或“一般”档位
            if "基本" in crime_config["sentencing_ranges"]:
                bucket = "基本"
            elif "一般" in crime_config["sentencing_ranges"]:
                bucket = "一般"
            else: # 如果连基本/一般都没有，则取第一个定义的档位
                bucket = list(crime_config["sentencing_ranges"].keys())[0]


        range_conf = crime_config["sentencing_ranges"].get(bucket)
        if not range_conf:
            # 如果档位配置缺失，返回一个合理的默认值
            return bucket, (1, 36), f"量刑范围配置缺失，使用默认值1-36个月。档位: {bucket}"

        min_months = range_conf["min"]
        max_months = range_conf["max"]
        special_note = range_conf.get("special_note")

        # 对于像危险驾驶罪这种只有拘役的罪名，范围不能超过6个月
        if crime == "危险驾驶罪":
            max_months = min(max_months, 6)
            if min_months > max_months: # 确保下限不大于上限
                min_months = 1 # 拘役最低1个月

        return bucket, (min_months, max_months), special_note

    def _get_full_statutory_limit(self, crime: str, bucket: str) -> Union[int, str]:
        """
        获取完整法定刑上限（月或特殊标记）
        """
        crime_config = self.config["CRIMES"].get(crime)
        if not crime_config or "statutory_limits" not in crime_config:
            # 未知罪名或配置缺失时，返回一个保守的默认值（有期徒刑最高15年）
            return MAX_YEARS_IMPRISONMENT * 12

        limit = crime_config["statutory_limits"].get(bucket)

        if limit is None:
            # 如果具体档位没有定义，尝试返回该罪名的最高刑
            all_limits = [val for val in crime_config["statutory_limits"].values() if isinstance(val, int)]
            if all_limits:
                return max(all_limits)
            # 如果都是特殊标记，或者无配置，返回默认值
            return MAX_YEARS_IMPRISONMENT * 12

        if isinstance(limit, str): # 例如 "无期徒刑"
            return limit
        return limit

    def _calculate_base_months(self, crime: str, bucket: str, amount: Optional[float],
                               count: int, labels: List[str], statutory_start_range: Tuple[int, int]) -> float:
        """
        计算基准刑（月）
        在量刑起点幅度的基础上，根据其他影响犯罪构成的犯罪事实增加刑罚量。
        """
        min_stat, max_stat = statutory_start_range
        crime_config = self.config["CRIMES"].get(crime)
        if not crime_config:
            return (min_stat + max_stat) / 2 # 兜底

        base_months = (min_stat + max_stat) / 2 # 量刑起点的中点

        base_increase_factors = crime_config.get("base_increase_factors", {})
        labels_str = "".join(labels)

        # 财产犯罪数额超出阈值部分的增加刑罚量
        if amount is not None and "amount_exceed_threshold_ratio" in base_increase_factors:
            amount_config = base_increase_factors["amount_exceed_threshold_ratio"]
            thresholds = crime_config.get("amount_thresholds", {})
            current_threshold_value = 0

            # 找到当前档位对应的阈值下限
            if bucket == "特别巨大":
                current_threshold_value = thresholds.get("特别巨大", 0)
            elif bucket == "巨大":
                current_threshold_value = thresholds.get("巨大", 0)
            elif bucket == "较大":
                current_threshold_value = thresholds.get("较大", 0)

            if current_threshold_value > 0 and amount > current_threshold_value:
                exceed_ratio = (amount - current_threshold_value) / current_threshold_value
                # 增加刑罚量，初始增加比率加上每超出一定倍数再增加的比率
                increase_ratio = amount_config.get("initial_threshold_ratio", 0) + \
                                 exceed_ratio * amount_config.get("ratio_per_threshold_mult", 0)

                # 限制增加的上限，例如不超过当前档位上限的20-30%
                max_increase_base = (max_stat - min_stat) * 0.3 # 限制为当前档位区间的30%
                increase_months = min((max_stat - min_stat) * increase_ratio, max_increase_base)
                base_months += increase_months

        # 盗窃/诈骗/抢夺/敲诈勒索的次数增加（数额已作为起点，次数作为增加刑罚量）
        if crime in ["盗窃罪", "诈骗罪", "抢夺罪", "敲诈勒索罪"] and "times_over_three" in base_increase_factors:
            if count > 3: # 超过三次的次数作为增加刑罚量的事实
                times_config = base_increase_factors["times_over_three"]
                increase_months_per_time = times_config.get("per_time_months", 0)
                max_increase_ratio = times_config.get("max_increase_ratio", 0.20) # 限制增加幅度不超过20%

                increase_months = (count - 3) * increase_months_per_time
                base_months += min(increase_months, (max_stat - min_stat) * max_increase_ratio)

        # 针对特定罪名，根据其base_increase_factors进行定制化增加
        # 示例：交通肇事致人死亡/重伤人数增加刑罚量
        if crime == "交通肇事罪":
            if "death_count_over_one" in base_increase_factors and "死亡人数" in labels_str:
                death_count_config = base_increase_factors["death_count_over_one"]
                death_count_match = re.search(r"死亡人数([0-9]+)人", labels_str)
                if death_count_match:
                    num_deaths = int(death_count_match.group(1))
                    if num_deaths > 1:
                        base_months += (num_deaths - 1) * death_count_config.get("per_person_months", 0)
            if "injury_count_over_one" in base_increase_factors and "重伤人数" in labels_str:
                injury_count_config = base_increase_factors["injury_count_over_one"]
                injury_count_match = re.search(r"重伤人数([0-9]+)人", labels_str)
                if injury_count_match:
                    num_injuries = int(injury_count_match.group(1))
                    if num_injuries > 1:
                        base_months += (num_injuries - 1) * injury_count_config.get("per_person_months", 0)

        # 示例：抢劫罪的次数增加
        if crime == "抢劫罪" and "times_over_one" in base_increase_factors:
            rob_times = self.parse_count(labels, keyword="抢劫次数")
            if rob_times > 1:
                times_config = base_increase_factors["times_over_one"]
                increase_months = (rob_times - 1) * times_config.get("per_time_months", 0)
                base_months += min(increase_months, (max_stat - min_stat) * times_config.get("max_increase_ratio", 0.2))


        # 其他罪名根据自身特点添加增加刑罚量逻辑...
        # 例如毒品数量、开设赌场人数/赌资等

        return base_months

    def _apply_crime_specific_adjustments(self, base: float, crime: str, labels: List[str]) -> float:
        """应用特定罪名的基准刑调节情节"""
        labels_str = "".join(labels)
        crime_config = self.config["CRIMES"].get(crime)
        if not crime_config:
            return base

        adjustments = crime_config.get("specific_adjustments", [])
        for adj in adjustments:
            condition_met = False
            if isinstance(adj["condition_fact"], list): # 条件为多个关键词之一
                if any(kw in labels_str for kw in adj["condition_fact"]):
                    condition_met = True
            else: # 单个关键词
                if adj["condition_fact"] in labels_str:
                    condition_met = True

            if condition_met:
                if adj["type"] == "multiply":
                    # 对于范围调节，暂时取配置中的平均值，或根据实际情况可取更细致的判断
                    factor = adj.get("factor", 1.0) # 如果没有factor，默认为1
                    base *= factor
                elif adj["type"] == "add_months":
                    base += adj["months"]
        return base

    def _apply_general_circumstances(self, base: float, labels: List[str], crime: str, bucket: str) -> float:
        """
        应用通用的量刑情节调节
        严格遵循：特殊身份/行为 -> 法定减轻情节 -> 从宽情节 -> 从重情节
        """
        labels_str = "".join(labels)
        current = base
        has_statutory_reduction = False # 标记是否因法定情节可以突破法定最低刑

        common_circumstances_config = self.config["COMMON_CIRCUMSTANCES"]

        # === 阶段1：特殊身份/行为的法定调节（乘法调整，优先级最高）===
        # 这些情节通常直接乘以一个系数，且可能突破法定最低刑

        # 1. 未成年人犯罪
        if "已满十二周岁不满十六周岁" in labels_str:
            conf = common_circumstances_config.get("未成年人犯罪_12_16")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2 # 取中值
            has_statutory_reduction = True
        elif "已满十六周岁不满十八周岁" in labels_str:
            conf = common_circumstances_config.get("未成年人犯罪_16_18")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True

        # 2. 老年人犯罪
        if "七十五周岁" in labels_str: # 假设此标签包含75周岁信息
            conf = common_circumstances_config.get("老年人犯罪_75周岁")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True

        # 3. 又聋又哑的人或者盲人犯罪
        if any(kw in labels_str for kw in ["又聋又哑", "盲人"]):
            conf = common_circumstances_config.get("又聋又哑_盲人犯罪")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True

        # 4. 犯罪预备、未遂、中止
        if "未遂" in labels_str:
            conf = common_circumstances_config.get("未遂犯")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True
        # (犯罪预备、中止的逻辑类似，如果标签中有，需要添加)

        # 5. 从犯、胁从犯、教唆犯
        if "从犯" in labels_str:
            conf = common_circumstances_config.get("从犯")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True
        if "胁从犯" in labels_str:
            conf = common_circumstances_config.get("胁从犯")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True
        if "教唆犯" in labels_str: # 教唆犯如果是从犯地位，也从宽
            conf = common_circumstances_config.get("教唆犯")
            if conf: current *= (conf["factor_min"] + conf["factor_max"]) / 2
            has_statutory_reduction = True


        # === 阶段2：从宽情节（加法调整，处理认罪认罚不重复评价） ===
        total_lenient_ratio_non_rzrf = 0.0 # 非认罪认罚从宽比例累计

        # 1. 评估非认罪认罚的从宽情节
        if "自首" in labels_str:
            conf = common_circumstances_config.get("自首")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2

        if "坦白" in labels_str:
            if "避免特别严重后果" in labels_str:
                conf = common_circumstances_config.get("坦白_避免严重后果")
            elif "尚未掌握的同种较重罪行" in labels_str:
                conf = common_circumstances_config.get("坦白_同种较重罪行")
            else: # 默认一般坦白
                conf = common_circumstances_config.get("坦白_一般")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2

        if "当庭自愿认罪" in labels_str:
            conf = common_circumstances_config.get("当庭自愿认罪")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2

        # 赔偿、谅解、和解
        compensation_reduction_ratio = 0.0
        if "刑事和解" in labels_str:
            conf = common_circumstances_config.get("刑事和解")
            if conf: compensation_reduction_ratio = max(compensation_reduction_ratio, (conf["ratio_min"] + conf["ratio_max"]) / 2)
        elif "积极赔偿并取得谅解" in labels_str:
            conf = common_circumstances_config.get("积极赔偿_谅解")
            if conf: compensation_reduction_ratio = max(compensation_reduction_ratio, (conf["ratio_min"] + conf["ratio_max"]) / 2)
        elif "积极赔偿但没有取得谅解" in labels_str:
            conf = common_circumstances_config.get("积极赔偿_未谅解")
            if conf: compensation_reduction_ratio = max(compensation_reduction_ratio, (conf["ratio_min"] + conf["ratio_max"]) / 2)
        elif "取得谅解" in labels_str and "没有赔偿" in labels_str:
            conf = common_circumstances_config.get("未赔偿_谅解")
            if conf: compensation_reduction_ratio = max(compensation_reduction_ratio, (conf["ratio_min"] + conf["ratio_max"]) / 2)
        elif "退赃" in labels_str or "退赔" in labels_str: # 如果更具体的赔偿情节没有，则考虑退赃退赔
            conf = common_circumstances_config.get("退赃退赔")
            if conf: compensation_reduction_ratio = max(compensation_reduction_ratio, (conf["ratio_min"] + conf["ratio_max"]) / 2)
        total_lenient_ratio_non_rzrf += compensation_reduction_ratio


        # 立功
        if "重大立功" in labels_str:
            conf = common_circumstances_config.get("立功_重大")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2
        elif "立功" in labels_str:
            conf = common_circumstances_config.get("立功_一般")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2

        if "羁押期间表现好" in labels_str:
            conf = common_circumstances_config.get("羁押期间表现好")
            if conf: total_lenient_ratio_non_rzrf += (conf["ratio_min"] + conf["ratio_max"]) / 2

        # 2. 认罪认罚的特殊处理
        if "认罪认罚" in labels_str:
            rzrf_conf = common_circumstances_config.get("认罪认罚")
            if rzrf_conf:
                base_rzrf_lenient = rzrf_conf["base_ratio"]
                max_total_lenient = rzrf_conf["max_total_ratio"]

                # "不作重复评价"：取认罪认罚基础从宽与非认罪认罚从宽的较大值作为起点
                effective_lenient_ratio = max(base_rzrf_lenient, total_lenient_ratio_non_rzrf)

                # 在此基础上，可以根据认罪认罚协议的实际约定，额外增加从宽，但总从宽不得超过最大值
                # 这里的逻辑可以更精细，例如，如果自首+重大立功，认罪认罚可以给予更高从宽
                # 暂时简化为：如果其他情节已提供更多从宽，则以其他情节从宽为准，否则以认罪认罚基础从宽为准
                # 但最终从宽不能超过总上限
                current *= (1 - min(effective_lenient_ratio, max_total_lenient))
            else: # 如果认罪认罚配置缺失，则按非认罪认罚处理
                current *= (1 - min(total_lenient_ratio_non_rzrf, 0.70)) # 非认罪认罚从宽上限可自定义
        else:
            # 非认罪认罚案件，直接应用累加的从宽比例
            current *= (1 - min(total_lenient_ratio_non_rzrf, 0.70)) # 非认罪认罚从宽上限可自定义


        # === 阶段3：从重情节 ===
        total_aggravation_ratio = 0.0
        aggravation_months = 0 # 累犯有最低刑期要求

        if "累犯" in labels_str:
            conf = common_circumstances_config.get("累犯")
            if conf:
                total_aggravation_ratio += (conf["ratio_min"] + conf["ratio_max"]) / 2
                aggravation_months = conf.get("min_months", 0)
        elif "前科" in labels_str:  # 累犯和前科不并用，优先累犯
            conf = common_circumstances_config.get("有前科")
            if conf: total_aggravation_ratio += (conf["ratio_min"] + conf["ratio_max"]) / 2

        if "犯罪对象为弱势人员" in labels_str:
            conf = common_circumstances_config.get("犯罪对象弱势人员")
            if conf: total_aggravation_ratio += (conf["ratio_min"] + conf["ratio_max"]) / 2

        if "灾害期间故意犯罪" in labels_str:
            conf = common_circumstances_config.get("灾害期间故意犯罪")
            if conf: total_aggravation_ratio += (conf["ratio_min"] + conf["ratio_max"]) / 2

        current = current * (1 + total_aggravation_ratio) + aggravation_months


        # === 阶段4：法定刑幅度限制 ===
        # 根据是否具有法定减轻情节，确定最低刑限制
        min_statutory_floor = 0 # 如果有法定减轻，可以低于法定最低刑
        if not has_statutory_reduction:
            # 对于有期徒刑/拘役，最低刑期通常是1个月
            # 如果基准刑低于量刑起点的下限，则至少应达到量刑起点的下限
            # 或根据指导意见，一般不少于1个月，拘役为1个月
            min_statutory_floor = 1 if current > 0 else 0 # 至少1个月，或者如果计算出负值则为0


        max_statutory_limit = self._get_full_statutory_limit(crime, bucket) # 可能是月数，也可能是"无期徒刑"

        if isinstance(max_statutory_limit, str): # 如果是无期徒刑或死刑
            # 此时的current可能已经非常高，但有期徒刑最高15年 (180个月)
            # 如果计算结果超过15年，则视为可能判处无期徒刑或死刑
            if current > MAX_YEARS_IMPRISONMENT * 12:
                return MAX_YEARS_IMPRISONMENT * 12 # 宣告刑的月数上限仍是有期徒刑最高值
            else:
                current = max(current, min_statutory_floor)
                return current # 在有期徒刑范围内
        else: # 普通的月数上限
            current = max(current, min_statutory_floor)
            current = min(current, max_statutory_limit)
            return current


    def _snap_to_scale(self, months: float) -> int:
        """对齐刑期刻度"""
        # 可以将步长规则也配置化
        if months <= 6: # 拘役范围
            return max(0, round(months)) # 拘役按月算
        if months <= 36: # 1-3年有期徒刑
            step = 3 # 3个月为单位
        else: # 3年以上有期徒刑
            step = 6 # 6个月为单位
        return max(0, int(round(months / step) * step))

    def _calculate_confidence(self, crime: str, labels: List[str], amount: Optional[float], bucket: str) -> int:
        """
        计算置信度（0-5分）
        置信度逻辑可以根据实际效果调整，此处为示例
        """
        if not labels:
            return 0 # 无标签信息，置信度最低

        score = 2 # 基础分

        # 1. 关键信息完整度
        if crime != "盗窃罪" and self.detect_crime("", labels) != "盗窃罪": # 确保罪名识别稳定
            score += 1
        if amount is not None: # 金额信息
            score += 1

        # 2. 从宽从严情节数量
        key_circumstances_count = 0
        labels_str = "".join(labels)
        key_circumstances_keywords = ["自首", "坦白", "认罪认罚", "退赔", "退赃", "谅解", "立功", "累犯", "前科"]
        for kw in key_circumstances_keywords:
            if kw in labels_str:
                key_circumstances_count += 1

        if key_circumstances_count >= 3:
            score += 1
        elif key_circumstances_count == 0:
            score -= 1

        # 3. 标签数量
        if len(labels) > 8: # 标签过多可能意味着信息过载或噪声
            score += 1
        elif len(labels) < 3: # 标签过少可能意味着信息不足
            score -= 1

        # 4. 经验法则：高危/复杂罪名或情节
        if crime in ["毒品罪", "抢劫罪", "集资诈骗罪", "组织卖淫罪"] and "特别巨大" in bucket:
            score -= 1 # 复杂案件往往难以精确预测，降低置信度

        return max(0, min(score, 5))

    def _calculate_interval(self, point: float, crime: str, bucket: str,
                            labels: List[str], statutory_start_range: Tuple[int, int]) -> Tuple[int, int, str]:
        """
        生成预测区间。
        返回 (lower, upper, final_sentence_type)
        """
        labels_str = "".join(labels)
        min_stat_start, max_stat_start = statutory_start_range

        # 获取最终法定刑上限，可能是月数或特殊标记
        max_statutory_limit_final = self._get_full_statutory_limit(crime, bucket)

        # 如果最终刑期超出有期徒刑范围，则直接返回特殊刑种
        if isinstance(max_statutory_limit_final, str): # "无期徒刑" or "死刑"
            if point >= MAX_YEARS_IMPRISONMENT * 12 - 6: # 如果计算点接近或超过有期徒刑上限
                return MAX_YEARS_IMPRISONMENT * 12, MAX_YEARS_IMPRISONMENT * 12, max_statutory_limit_final # 给出有期徒刑最高刑的区间，并标记刑种
            else: # 如果虽然法定刑上限是无期，但计算点还在有期徒刑范围内
                 # 仍然按有期徒刑处理
                 pass

        snapped_point = self._snap_to_scale(point)
        confidence = self._calculate_confidence(crime, labels, self.parse_amount(labels), bucket)

        # 调整区间宽度映射表，可以将其配置化
        width_map_aggressive = {5: 0, 4: 1, 3: 2, 2: 3, 1: 4, 0: 6}
        width_map_balanced = {5: 1, 4: 2, 3: 3, 2: 4, 1: 6, 0: 9}
        width_map_conservative = {5: 2, 4: 3, 3: 6, 2: 9, 1: 12, 0: 18}

        if self.narrow_mode == "aggressive":
            width = width_map_aggressive.get(confidence, 3)
        elif self.narrow_mode == "balanced":
            width = width_map_balanced.get(confidence, 4)
        else:
            width = width_map_conservative.get(confidence, 6)

        lower = snapped_point - width
        upper = snapped_point + width

        # 应用刑种和法定刑限制
        final_sentence_type = "有期徒刑" # 默认

        # 拘役（1-6个月）
        if lower <= 6:
            if upper <= 6:
                final_sentence_type = "拘役"
            elif lower <= 0 and any(kw in labels_str for kw in ["轻微", "免除刑罚"]):
                final_sentence_type = "免予刑事处罚"
                return 0, 0, final_sentence_type # 免予刑事处罚区间为0,0
            else:
                 # 混合区间，比如下限拘役上限有期徒刑，通常认为是有期徒刑
                pass

        # 强制最低刑为1个月，除非免予刑事处罚
        lower = max(1, lower)
        if final_sentence_type == "免予刑事处罚":
            lower = 0

        # 如果有法定减轻情节（已在_apply_general_circumstances中处理并设置has_statutory_reduction），
        # 且计算出的点低于量刑起点下限，则可以低于起点
        has_statutory_reduction = any(kw in labels_str for kw in ["未遂", "从犯", "未成年人", "七十五周岁"])
        if not has_statutory_reduction: # 只有在没有法定减轻情节时，才确保下限不低于量刑起点下限
             lower = max(lower, min_stat_start)
        else: # 有法定减轻，但最低仍不能低于刑法总则规定的有期徒刑最低刑（通常6个月或免除）
            lower = max(lower, 0) # 如果可以免除，则最低可以为0


        # 最终上限不能超过法定刑上限
        if isinstance(max_statutory_limit_final, int): # 如果是月数
            upper = min(upper, max_statutory_limit_final)
        else: # 如果是无期徒刑/死刑，有期徒刑最高15年
            upper = min(upper, MAX_YEARS_IMPRISONMENT * 12)
            if snapped_point >= MAX_YEARS_IMPRISONMENT * 12 - 6: # 如果点在有期徒刑上限附近，则可能判无期
                final_sentence_type = max_statutory_limit_final


        if lower > upper: # 确保下限不大于上限
            lower = upper = snapped_point

        return int(lower), int(upper), final_sentence_type


    def _fallback_predict(self, fact: str) -> Tuple[int, int, str]:
        """
        API失败时的兜底预测（修复）
        从事实文本中提取关键信息进行简单推理
        """
        fallback_labels = []

        # 提取金额
        amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占|敲诈|抢夺).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
        if amount_match:
            amount = float(amount_match.group(1))
            if "万元" in amount_match.group(0):
                amount *= 10000

            # 推断罪名
            if "盗窃" in fact:
                fallback_labels.append(f"盗窃金额既遂{int(amount)}元")
            elif "诈骗" in fact:
                fallback_labels.append(f"诈骗金额既遂{int(amount)}元")
            elif "敲诈" in fact:
                fallback_labels.append(f"敲诈勒索金额既遂{int(amount)}元")

            # 根据金额粗略判断档位
            if amount >= 300000: fallback_labels.append("特别巨大")
            elif amount >= 30000: fallback_labels.append("巨大")
            elif amount >= 3000: fallback_labels.append("较大")


        # 提取关键情节
        circumstance_map = [
            ("自首", "自首"), ("坦白", "坦白"), ("累犯", "累犯"), ("前科", "前科"),
            ("退赔", "退赔"), ("退赃", "退赃"), ("谅解", "取得谅解"), ("认罪认罚", "认罪认罚"),
            ("未遂", "未遂"), ("从犯", "从犯"), ("未成年人", "未成年人"), ("七十五周岁", "七十五周岁")
        ]

        for keyword, label in circumstance_map:
            if keyword in fact:
                fallback_labels.append(label)

        # 如果完全提取失败，返回保守区间 (默认盗窃罪，较大，无情节)
        if not fallback_labels:
            return 6, 18, "有期徒刑" # 默认6-18个月有期徒刑

        # 使用提取的标签进行预测
        # 调用主预测逻辑，但会因为缺乏足够标签而得到较低置信度
        # 这里需要调整predict函数的返回值，以避免递归陷入死循环
        # 暂时返回一个安全区间的默认值，或者尝试进行一次简单预测
        crime = self.detect_crime(fact, fallback_labels)
        amount = self.parse_amount(fallback_labels) # 再次尝试从fallback_labels解析金额
        count = self.parse_count(fallback_labels)
        bucket, statutory_start_range, _ = self.get_statutory_range(crime, amount, fallback_labels)

        base = self._calculate_base_months(crime, bucket, amount, count, fallback_labels, statutory_start_range)
        base_adjusted = self._apply_crime_specific_adjustments(base, crime, fallback_labels)
        point = self._apply_general_circumstances(base_adjusted, fallback_labels, crime, bucket)

        lower, upper, final_sentence_type = self._calculate_interval(point, crime, bucket, fallback_labels, statutory_start_range)

        return lower, upper, final_sentence_type


    def predict(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """
        主预测函数
        """
        # API失败检测
        if not labels:
            lower, upper, _ = self._fallback_predict(fact)
            return lower, upper

        crime = self.detect_crime(fact, labels)
        amount = self.parse_amount(labels)
        count = self.parse_count(labels)
        bucket, statutory_start_range, special_note = self.get_statutory_range(crime, amount, labels)

        base = self._calculate_base_months(crime, bucket, amount, count, labels, statutory_start_range)
        base_adjusted = self._apply_crime_specific_adjustments(base, crime, labels)
        point = self._apply_general_circumstances(base_adjusted, labels, crime, bucket)

        lower, upper, _ = self._calculate_interval(point, crime, bucket, labels, statutory_start_range)
        return lower, upper

    def predict_with_details(self, fact: str, labels: List[str]) -> Dict:
        """
        详细预测
        """
        api_status = "success" if labels else "fallback"

        if not labels:
            lower, upper, final_sentence_type = self._fallback_predict(fact)
            return {
                "crime": "未知（兜底）",
                "amount": None,
                "count": 1,
                "bucket": "未知",
                "sentencing_start_range": (0, 36),
                "base_months": 12.0,
                "crime_specific_adjusted_base": 12.0,
                "final_adjusted_point": 12.0,
                "snapped_point": 12,
                "confidence": 0,
                "interval": [lower, upper],
                "width": upper - lower,
                "api_status": api_status,
                "final_sentence_type": final_sentence_type
            }

        crime = self.detect_crime(fact, labels)
        amount = self.parse_amount(labels)
        count = self.parse_count(labels)
        bucket, statutory_start_range, special_note = self.get_statutory_range(crime, amount, labels)

        base = self._calculate_base_months(crime, bucket, amount, count, labels, statutory_start_range)
        base_adjusted = self._apply_crime_specific_adjustments(base, crime, labels)
        point = self._apply_general_circumstances(base_adjusted, labels, crime, bucket)

        snapped_point = self._snap_to_scale(point)
        confidence = self._calculate_confidence(crime, labels, amount, bucket)
        lower, upper, final_sentence_type = self._calculate_interval(point, crime, bucket, labels, statutory_start_range)

        # 获取最终法定刑上限的显示值
        max_statutory_limit_for_display = self._get_full_statutory_limit(crime, bucket)

        return {
            "crime": crime,
            "amount": amount,
            "count": count,
            "bucket": bucket,
            "sentencing_start_range": statutory_start_range,
            "special_note": special_note,
            "full_statutory_limit": max_statutory_limit_for_display,
            "base_months": round(base, 2),
            "crime_specific_adjusted_base": round(base_adjusted, 2),
            "final_adjusted_point": round(point, 2),
            "snapped_point": snapped_point,
            "confidence": confidence,
            "interval": [lower, upper],
            "width": upper - lower,
            "api_status": api_status,
            "final_sentence_type": final_sentence_type
        }

# 便捷函数 (为了兼容赛方提供的 run_infer.py 可能的调用方式)
def predict_interval_months(fact: str, labels: List[str], narrow_mode: str = "aggressive", config_path: str = None) -> Tuple[int, int]:
    """
    便捷预测函数

    Args:
        fact: 案件事实描述
        labels: 量刑情节标签列表（可为空，会自动兜底）
        narrow_mode: "aggressive"(零宽) | "balanced"(±1月) | "conservative"(±2月)
        config_path: 量刑规则配置文件的路径，默认为rules/rules_config.json

    Returns:
        (下限月, 上限月)
    """
    engine = SentencingRulesEngine(narrow_mode=narrow_mode, config_path=config_path)
    return engine.predict(fact, labels)

def predict_with_confidence(fact: str, labels: List[str], config_path: str = None) -> Dict:
    """
    带详细信息的预测

    Args:
        fact: 案件事实描述
        labels: 量刑情节标签列表（可为空，会自动兜底）
        config_path: 量刑规则配置文件的路径，默认为rules/rules_config.json

    Returns:
        包含中间计算结果和api_status的字典
    """
    engine = SentencingRulesEngine(narrow_mode="aggressive", config_path=config_path)
    return engine.predict_with_details(fact, labels)

