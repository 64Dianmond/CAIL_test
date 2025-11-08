"""
量刑规则引擎 - 基于《量刑指导意见（试行）》2024版
支持30种常见犯罪的量刑计算
目标：最窄区间策略 - 零宽命中时 Winkler Score = 0
"""
import re
from typing import List, Tuple, Optional, Dict


class SentencingRulesEngine:
    """量刑规则引擎 - 完整版本"""

    # 各罪名数额标准
    THEFT_THRESHOLDS = {
        "较大_下限": 1000, "较大_上限": 30000,
        "巨大_下限": 30000, "巨大_上限": 300000,
        "特别巨大_下限": 300000,
    }

    FRAUD_THRESHOLDS = {
        "较大_下限": 3000, "较大_上限": 30000,
        "巨大_下限": 30000, "巨大_上限": 500000,
        "特别巨大_下限": 500000,
    }

    CONTRACT_FRAUD_THRESHOLDS = {
        "较大_下限": 20000, "较大_上限": 200000,
        "巨大_下限": 200000, "巨大_上限": 1000000,
        "特别巨大_下限": 1000000,
    }

    CREDIT_CARD_FRAUD_THRESHOLDS = {
        "较大_下限": 5000, "较大_上限": 50000,
        "巨大_下限": 50000, "巨大_上限": 500000,
        "特别巨大_下限": 500000,
    }

    EXTORTION_THRESHOLDS = {
        "较大_下限": 2000, "较大_上限": 30000,
        "巨大_下限": 30000, "巨大_上限": 300000,
        "特别巨大_下限": 300000,
    }

    EMBEZZLEMENT_THRESHOLDS = {
        "较大_下限": 30000, "较大_上限": 300000,
        "巨大_下限": 300000, "巨大_上限": 3000000,
        "特别巨大_下限": 3000000,
    }

    SNATCH_THRESHOLDS = {
        "较大_下限": 1000, "较大_上限": 30000,
        "巨大_下限": 30000, "巨大_上限": 300000,
        "特别巨大_下限": 300000,
    }

    def __init__(self, narrow_mode: str = "aggressive"):
        self.narrow_mode = narrow_mode

    def detect_crime(self, fact: str, labels: List[str]) -> str:
        """识别罪名"""
        text = fact + "".join(labels)

        # 优先级排序（避免误判）
        crime_keywords = [
            ("交通肇事", "交通肇事罪"),
            ("危险驾驶", "危险驾驶罪"),
            ("非法吸收公众存款", "非法吸收公众存款罪"),
            ("集资诈骗", "集资诈骗罪"),
            ("信用卡诈骗", "信用卡诈骗罪"),
            ("合同诈骗", "合同诈骗罪"),
            ("故意伤害", "故意伤害罪"),
            ("强奸", "强奸罪"),
            ("非法拘禁", "非法拘禁罪"),
            ("抢劫", "抢劫罪"),
            ("盗窃", "盗窃罪"),
            ("诈骗", "诈骗罪"),
            ("抢夺", "抢夺罪"),
            ("职务侵占", "职务侵占罪"),
            ("敲诈勒索", "敲诈勒索罪"),
            ("妨害公务", "妨害公务罪"),
            ("聚众斗殴", "聚众斗殴罪"),
            ("寻衅滋事", "寻衅滋事罪"),
            ("掩饰、隐瞒犯罪所得", "掩饰、隐瞒犯罪所得罪"),
            ("走私、贩卖、运输、制造毒品", "毒品罪"),
            ("非法持有毒品", "非法持有毒品罪"),
            ("容留他人吸毒", "容留他人吸毒罪"),
            ("引诱、容留、介绍卖淫", "引诱、容留、介绍卖淫罪"),
            ("组织卖淫", "组织卖淫罪"),
            ("非法经营", "非法经营罪"),
            ("猥亵儿童", "猥亵儿童罪"),
            ("侵犯公民个人信息", "侵犯公民个人信息罪"),
            ("帮助信息网络犯罪活动", "帮助信息网络犯罪活动罪"),
            ("开设赌场", "开设赌场罪"),
            ("拒不执行判决、裁定", "拒不执行判决、裁定罪"),
        ]

        for keyword, crime_name in crime_keywords:
            if keyword in text:
                return crime_name

        return "盗窃罪"  # 默认

    def parse_amount(self, labels: List[str]) -> Optional[float]:
        """提取金额（元）"""
        for label in labels:
            match = re.search(
                r"(?:盗窃|诈骗|侵占|敲诈|抢夺|非法吸收|集资|信用卡|合同|经营).*?金额.*?([0-9]+(?:\.[0-9]+)?)\s*元",
                label)
            if match:
                return float(match.group(1))
        return None

    def parse_count(self, labels: List[str]) -> int:
        """提取次数"""
        for label in labels:
            match = re.search(r"次数[：:\s]*([0-9]+)\s*次", label)
            if match:
                return int(match.group(1))
        return 1

    def get_statutory_range(self, crime: str, amount: Optional[float], labels: List[str]) -> Tuple[
        str, Tuple[int, int]]:
        """
        确定法定刑幅度
        返回: (档位, (下限月, 上限月))
        """
        labels_str = "".join(labels)

        # 财产犯罪
        if crime in ["盗窃罪", "诈骗罪", "抢夺罪"]:
            thresholds = self.THEFT_THRESHOLDS if crime == "盗窃罪" else \
                (self.FRAUD_THRESHOLDS if crime == "诈骗罪" else self.SNATCH_THRESHOLDS)

            if amount is None:
                if "特别巨大" in labels_str:
                    return "特别巨大", (120, 180)
                elif "巨大" in labels_str:
                    return "巨大", (36, 120)
                else:
                    return "较大", (0, 36)

            if amount >= thresholds["特别巨大_下限"]:
                return "特别巨大", (120, 180)
            elif amount >= thresholds["巨大_下限"]:
                return "巨大", (36, 120)
            else:
                return "较大", (0, 36)

        elif crime == "合同诈骗罪":
            if amount and amount >= self.CONTRACT_FRAUD_THRESHOLDS["特别巨大_下限"]:
                return "特别巨大", (120, 180)
            elif amount and amount >= self.CONTRACT_FRAUD_THRESHOLDS["巨大_下限"]:
                return "巨大", (36, 60)
            else:
                return "较大", (0, 12)

        elif crime == "信用卡诈骗罪":
            if amount and amount >= self.CREDIT_CARD_FRAUD_THRESHOLDS["特别巨大_下限"]:
                return "特别巨大", (120, 180)
            elif amount and amount >= self.CREDIT_CARD_FRAUD_THRESHOLDS["巨大_下限"]:
                return "巨大", (60, 72)
            else:
                return "较大", (0, 24)

        elif crime == "敲诈勒索罪":
            if amount and amount >= self.EXTORTION_THRESHOLDS["特别巨大_下限"]:
                return "特别巨大", (120, 144)
            elif amount and amount >= self.EXTORTION_THRESHOLDS["巨大_下限"]:
                return "巨大", (36, 60)
            else:
                return "较大", (0, 12)

        elif crime == "职务侵占罪":
            if amount and amount >= self.EMBEZZLEMENT_THRESHOLDS["特别巨大_下限"]:
                return "特别巨大", (120, 132)
            elif amount and amount >= self.EMBEZZLEMENT_THRESHOLDS["巨大_下限"]:
                return "巨大", (36, 48)
            else:
                return "较大", (0, 12)

        # 暴力犯罪
        elif crime == "抢劫罪":
            if "入户" in labels_str or "公共交通工具" in labels_str or "银行" in labels_str or \
                    "持枪" in labels_str or "致.*重伤" in labels_str:
                return "加重", (120, 156)
            else:
                return "基本", (36, 72)

        elif crime == "故意伤害罪":
            if "重伤" in labels_str and "严重残疾" in labels_str:
                return "严重", (120, 156)
            elif "重伤" in labels_str:
                return "重伤", (36, 60)
            else:
                return "轻伤", (0, 24)

        elif crime == "强奸罪":
            if "情节恶劣" in labels_str or "三人" in labels_str or "公共场所" in labels_str or \
                    "轮奸" in labels_str or "不满十周岁" in labels_str or "致.*重伤" in labels_str:
                return "加重", (120, 156)
            else:
                return "基本", (36, 72)

        elif crime == "非法拘禁罪":
            if "致.*死亡" in labels_str:
                return "致死", (120, 156)
            elif "致.*重伤" in labels_str:
                return "致重伤", (36, 60)
            else:
                return "基本", (0, 12)

        # 妨害社会管理秩序罪
        elif crime == "聚众斗殴罪":
            if "三次" in labels_str or "持械" in labels_str or "公共场所" in labels_str:
                return "加重", (36, 60)
            else:
                return "基本", (0, 24)

        elif crime == "寻衅滋事罪":
            if "纠集.*三次" in labels_str and "严重破坏" in labels_str:
                return "加重", (60, 84)
            else:
                return "基本", (0, 36)

        elif crime == "妨害公务罪":
            return "基本", (0, 24)

        # 金融犯罪
        elif crime == "非法吸收公众存款罪":
            if "特别严重" in labels_str:
                return "特别严重", (120, 144)
            elif "严重" in labels_str:
                return "严重", (36, 48)
            else:
                return "一般", (0, 12)

        elif crime == "集资诈骗罪":
            if "特别巨大" in labels_str:
                return "特别巨大", (120, 180)
            elif "巨大" in labels_str:
                return "巨大", (84, 108)
            else:
                return "较大", (36, 48)

        # 毒品犯罪
        elif crime in ["毒品罪", "非法持有毒品罪"]:
            if "数量大" in labels_str:
                return "数量大", (84, 108) if crime == "非法持有毒品罪" else (180, 240)
            elif "数量较大" in labels_str:
                return "数量较大", (36, 48) if crime == "非法持有毒品罪" else (84, 96)
            else:
                return "少量", (0, 36)

        elif crime == "容留他人吸毒罪":
            return "基本", (0, 12)

        # 组织卖淫类
        elif crime == "组织卖淫罪":
            if "情节严重" in labels_str:
                return "严重", (120, 156)
            else:
                return "一般", (60, 120)

        elif crime == "引诱、容留、介绍卖淫罪":
            if "情节严重" in labels_str:
                return "严重", (60, 84)
            else:
                return "一般", (0, 24)

        # 其他罪名
        elif crime == "交通肇事罪":
            if "因逃逸致.*死亡" in labels_str:
                return "逃逸致死", (84, 120)
            elif "逃逸" in labels_str or "特别恶劣" in labels_str:
                return "逃逸", (36, 60)
            else:
                return "一般", (0, 24)

        elif crime == "危险驾驶罪":
            return "基本", (1, 6)

        elif crime in ["掩饰、隐瞒犯罪所得罪", "非法经营罪", "侵犯公民个人信息罪", "帮助信息网络犯罪活动罪"]:
            if "特别严重" in labels_str:
                return "特别严重", (36, 48 if crime == "侵犯公民个人信息罪" else 60)
            else:
                return "一般", (0, 12)

        elif crime == "猥亵儿童罪":
            if "多人" in labels_str or "多次" in labels_str or "聚众" in labels_str or "公共场所" in labels_str or \
                    "伤害" in labels_str or "恶劣" in labels_str:
                return "加重", (60, 84)
            else:
                return "一般", (12, 36)

        elif crime == "开设赌场罪":
            if "情节严重" in labels_str:
                return "严重", (60, 72)
            else:
                return "一般", (0, 24)

        elif crime == "拒不执行判决、裁定罪":
            if "特别严重" in labels_str:
                return "特别严重", (36, 48)
            else:
                return "严重", (0, 12)

        # 默认
        return "一般", (0, 36)

    def calculate_base_months(self, crime: str, bucket: str, amount: Optional[float],
                              count: int, labels: List[str], statutory: Tuple[int, int]) -> float:
        """计算基准刑（月）"""
        min_stat, max_stat = statutory

        # 财产犯罪基准刑计算
        if crime in ["盗窃罪", "诈骗罪", "抢夺罪"]:
            if bucket == "较大":
                if amount:
                    ratio = (amount - 1000) / 29000
                    base = 6 + ratio * 12
                    base = max(6, min(18, base))
                else:
                    base = 12
                if count > 1:
                    base += min((count - 1) * 1.5, 6)

            elif bucket == "巨大":
                if amount:
                    ratio = (amount - 30000) / 270000
                    base = 40 + ratio * 40
                    base = max(36, min(84, base))
                else:
                    base = 60
                if count > 1:
                    base += min((count - 1) * 2.5, 10)

            else:  # 特别巨大
                if amount:
                    ratio = min((amount - 300000) / 300000, 1.0)
                    base = 130 + ratio * 20
                    base = max(120, min(156, base))
                else:
                    base = 135
                if count > 1:
                    base += min((count - 1) * 3.5, 14)

        # 其他罪名简化处理：量刑起点+少量增量
        else:
            base = (min_stat + max_stat) / 2  # 中位数

            # 根据金额调整
            if amount and amount > 100000:
                base += (max_stat - min_stat) * 0.2

            # 根据次数调整
            if count > 1:
                base += min((count - 1) * 2, (max_stat - min_stat) * 0.15)

        return base

    def apply_circumstances(self, base: float, labels: List[str], statutory: Tuple[int, int]) -> float:
        """应用量刑情节调节"""
        labels_str = "".join(labels)
        current = base
        has_statutory_reduction = False

        # 法定减轻
        if "未遂" in labels_str:
            current *= 0.5
            has_statutory_reduction = True
        if "从犯" in labels_str:
            current *= 0.65
            has_statutory_reduction = True
        if "限制刑事责任能力" in labels_str:
            current *= 0.75

        # 从宽情节（累加上限60%）
        total_lenient = 0.0

        if "自首" in labels_str:
            total_lenient += 0.40
        elif "坦白" in labels_str:
            total_lenient += 0.20 if "重大" not in labels_str else 0.30

        if "当庭自愿认罪" in labels_str and "认罪认罚" not in labels_str:
            total_lenient += 0.10

        compensation = 0.0
        if "赔偿并取得谅解" in labels_str or "退赔并取得谅解" in labels_str:
            compensation = 0.40
        elif "取得谅解" in labels_str:
            compensation = 0.30
        elif "退赔" in labels_str or "退赃" in labels_str:
            compensation = 0.30
        total_lenient += compensation

        if "认罪认罚" in labels_str:
            if total_lenient > 0.30:
                total_lenient = min(total_lenient, 0.60)
            else:
                total_lenient += 0.30

        total_lenient = min(total_lenient, 0.60)
        current *= (1 - total_lenient)

        # 从重情节
        aggravation_ratio = 0.0
        aggravation_months = 0

        if "累犯" in labels_str:
            aggravation_ratio += 0.20
            aggravation_months = 3
        if "前科" in labels_str:
            aggravation_ratio += 0.10

        current = current * (1 + aggravation_ratio) + aggravation_months

        # 法定幅度限制
        min_stat, max_stat = statutory
        if not has_statutory_reduction:
            current = max(current, min_stat)
        else:
            current = max(current, 0)
        current = min(current, max_stat)

        return current

    def snap_to_scale(self, months: float) -> int:
        """对齐刑期刻度"""
        if months < 36:
            step = 3
        else:
            step = 6
        snapped = round(months / step) * step
        return max(0, int(snapped))

    def calculate_confidence(self, crime: str, labels: List[str], amount: Optional[float], bucket: str) -> int:
        """计算置信度（0-5分）"""
        score = 0
        labels_str = "".join(labels)

        if amount is not None:
            score += 2
            if bucket == "巨大" and 50000 < amount < 250000:
                score += 1
            elif bucket == "较大" and 5000 < amount < 25000:
                score += 1

        count = self.parse_count(labels)
        if count <= 2:
            score += 1

        key_circumstances = sum([
            "自首" in labels_str,
            "坦白" in labels_str,
            "认罪认罚" in labels_str,
            "退赔" in labels_str or "谅解" in labels_str,
        ])
        if 1 <= key_circumstances <= 2:
            score += 1

        if ("累犯" in labels_str and "前科" in labels_str) or len(labels) > 8:
            score -= 1

        return max(0, min(score, 5))

    def calculate_interval(self, point: float, crime: str, bucket: str,
                           labels: List[str], statutory: Tuple[int, int]) -> Tuple[int, int]:
        """生成预测区间"""
        labels_str = "".join(labels)
        min_stat, max_stat = statutory
        has_reduction = any(kw in labels_str for kw in ["未遂", "从犯"])

        snapped = self.snap_to_scale(point)
        confidence = self.calculate_confidence(crime, labels, self.parse_amount(labels), bucket)

        # 根据置信度决定宽度
        if self.narrow_mode == "aggressive":
            if confidence >= 5:
                width = 0
            elif confidence == 4:
                width = 1
            elif confidence == 3:
                width = 2
            else:
                width = 3
        elif self.narrow_mode == "balanced":
            if confidence >= 4:
                width = 1
            elif confidence == 3:
                width = 2
            else:
                width = 3
        else:  # conservative
            if confidence >= 4:
                width = 2
            else:
                width = 3

        lower = snapped - width
        upper = snapped + width

        # 法定幅度截断
        if not has_reduction:
            lower = max(lower, min_stat)
        else:
            lower = max(lower, 0)
        upper = min(upper, max_stat)

        if lower > upper:
            lower = upper = snapped

        return int(lower), int(upper)

    def predict(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """主预测函数"""
        # API失败检测
        if not labels:
            return self._fallback_predict(fact)

        crime = self.detect_crime(fact, labels)
        amount = self.parse_amount(labels)
        count = self.parse_count(labels)
        bucket, statutory = self.get_statutory_range(crime, amount, labels)

        base = self.calculate_base_months(crime, bucket, amount, count, labels, statutory)
        point = self.apply_circumstances(base, labels, statutory)
        lower, upper = self.calculate_interval(point, crime, bucket, labels, statutory)

        return lower, upper

    def _fallback_predict(self, fact: str) -> Tuple[int, int]:
        """API失败时的兜底预测"""
        # 简单正则提取
        fallback_labels = []

        # 提取金额
        amount_match = re.search(r'(?:盗窃|诈骗|骗取|侵占).*?([0-9]+(?:\.[0-9]+)?)\s*(?:元|万元)', fact)
        if amount_match:
            amount = float(amount_match.group(1))
            if "万元" in amount_match.group(0):
                amount *= 10000

            if "盗窃" in fact:
                fallback_labels.append(f"盗窃金额既遂{int(amount)}元")
            elif "诈骗" in fact:
                fallback_labels.append(f"诈骗金额既遂{int(amount)}元")

        # 提取关键情节
        if "自首" in fact:
            fallback_labels.append("自首")
        if "坦白" in fact:
            fallback_labels.append("坦白")
        if "累犯" in fact:
            fallback_labels.append("累犯")
        if "退赔" in fact or "退赃" in fact:
            fallback_labels.append("退赔")
        if "谅解" in fact:
            fallback_labels.append("取得谅解")

        # 如果仍为空，返回保守区间
        if not fallback_labels:
            return 6, 18  # 保守兜底

        return self.predict(fact, fallback_labels)

    def predict_with_details(self, fact: str, labels: List[str]) -> Dict:
        """详细预测"""
        if not labels:
            lower, upper = self._fallback_predict(fact)
            return {
                "crime": "未知",
                "amount": None,
                "count": 1,
                "bucket": "未知",
                "statutory_range": (0, 36),
                "base_months": 12,
                "adjusted_point": 12,
                "snapped_point": 12,
                "confidence": 0,
                "interval": [lower, upper],
                "width": upper - lower,
                "api_status": "fallback"
            }

        crime = self.detect_crime(fact, labels)
        amount = self.parse_amount(labels)
        count = self.parse_count(labels)
        bucket, statutory = self.get_statutory_range(crime, amount, labels)

        base = self.calculate_base_months(crime, bucket, amount, count, labels, statutory)
        point = self.apply_circumstances(base, labels, statutory)
        snapped = self.snap_to_scale(point)
        confidence = self.calculate_confidence(crime, labels, amount, bucket)
        lower, upper = self.calculate_interval(point, crime, bucket, labels, statutory)

        return {
            "crime": crime,
            "amount": amount,
            "count": count,
            "bucket": bucket,
            "statutory_range": statutory,
            "base_months": round(base, 2),
            "adjusted_point": round(point, 2),
            "snapped_point": snapped,
            "confidence": confidence,
            "interval": [lower, upper],
            "width": upper - lower,
            "api_status": "success"
        }


# 便捷函数
def predict_interval_months(fact: str, labels: List[str], narrow_mode: str = "aggressive") -> Tuple[int, int]:
    """便捷预测函数"""
    engine = SentencingRulesEngine(narrow_mode=narrow_mode)
    return engine.predict(fact, labels)


def predict_with_confidence(fact: str, labels: List[str]) -> Dict:
    """带详细信息的预测"""
    engine = SentencingRulesEngine(narrow_mode="aggressive")
    return engine.predict_with_details(fact, labels)
