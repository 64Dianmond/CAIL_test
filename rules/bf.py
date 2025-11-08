"""
量刑规则引擎
基于“两高《常见犯罪的量刑指导意见（试行）》”及相关司法解释
"""

import re
from typing import List, Tuple, Optional


class SentencingRulesEngine:
    """量刑规则引擎"""

    # 盗窃罪数额标准（全国幅度，单位：元）
    THEFT_THRESHOLDS = {
        "较大_下限": 3000,
        "较大_上限": 30000,
        "巨大_下限": 30000,
        "巨大_上限": 300000,
        "特别巨大_下限": 300000,
    }

    # 诈骗罪数额标准（全国幅度，单位：元）
    FRAUD_THRESHOLDS = {
        "较大_下限": 3000,
        "较大_上限": 30000,
        "巨大_下限": 30000,
        "巨大_上限": 500000,
        "特别巨大_下限": 500000,
    }

    # 量刑情节调节比例（按指导意见）
    CIRCUMSTANCE_RATIOS = {
        "自首": -0.40,
        "坦白": -0.20,
        "坦白_重大": -0.30,
        "当庭自愿认罪": -0.10,
        "认罪认罚": -0.30,
        "认罪认罚_综合": -0.60,  # 与其他情节综合时上限
        "退赃": -0.30,
        "退赔": -0.30,
        "赔偿并取得谅解": -0.40,
        "取得谅解": -0.30,
        "累犯_下限": 0.10,
        "累犯_上限": 0.40,
        "累犯_最低月": 3,
        "前科": 0.10,
        "未遂": -0.50,
        "从犯": -0.50,
        "限制刑事责任能力": -0.30,
    }

    def __init__(self):
        pass

    # ========== 信息提取部分 ==========

    def detect_crime(self, fact: str, labels: List[str]) -> str:
        """识别罪名"""
        fact_lower = fact.lower()
        labels_str = "".join(labels).lower()
        if "盗窃" in fact or "盗窃" in labels_str:
            return "盗窃"
        elif "诈骗" in fact or "诈骗" in labels_str:
            return "诈骗"
        else:
            # 默认按盗窃处理（保守）
            return "盗窃"

    def parse_amount(self, labels: List[str]) -> Optional[float]:
        """从标签中提取金额（元）"""
        for label in labels:
            match = re.search(r"(?:盗窃|诈骗)金额(?:既遂)?[：:]?\s*([0-9]+(?:\.[0-9]+)?)\s*元", label)
            if match:
                return float(match.group(1))
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*元", label)
            if match and ("金额" in label or "盗窃" in label or "诈骗" in label):
                return float(match.group(1))
        return None

    def parse_count(self, labels: List[str]) -> int:
        """提取作案次数"""
        for label in labels:
            match = re.search(r"(?:盗窃|诈骗)?次数[：:]?\s*([0-9]+)\s*次", label)
            if match:
                return int(match.group(1))
        return 1  # 默认1次

    # ========== 法定幅度与数额分档 ==========

    def get_amount_bucket(self, crime: str, amount: Optional[float], labels: List[str]) -> Tuple[str, Tuple[int, int]]:
        """确定数额档位与法定刑幅度"""
        thresholds = self.THEFT_THRESHOLDS if crime == "盗窃" else self.FRAUD_THRESHOLDS

        special_circumstances = any(
            kw in "".join(labels) for kw in ["扒窃", "入户", "携带凶器", "多次盗窃"]
        )

        # 无金额时按情形判断
        if amount is None:
            if special_circumstances:
                return "较大", (0, 36)
            labels_str = "".join(labels)
            if "特别巨大" in labels_str:
                return "特别巨大", (120, 180)
            elif "巨大" in labels_str:
                return "巨大", (36, 120)
            else:
                return "较大", (0, 36)

        # 按金额判断
        if amount >= thresholds["特别巨大_下限"]:
            return "特别巨大", (120, 180)
        elif amount >= thresholds["巨大_下限"]:
            return "巨大", (36, 120)
        elif amount >= thresholds["较大_下限"] or special_circumstances:
            return "较大", (0, 36)
        else:
            return "较大", (0, 36)

    # ========== 基准刑计算 ==========

    def calculate_base_months(self, bucket: str, amount: Optional[float], count: int, labels: List[str]) -> float:
        """计算基准刑（月）"""
        if bucket == "较大":
            if amount:
                base = 6 + (amount - 3000) / (30000 - 3000) * 12
                base = max(6, min(18, base))
            else:
                base = 12
            if count > 1:
                base += min((count - 1) * 2, 6)

        elif bucket == "巨大":
            if amount:
                base = 36 + (amount - 30000) / (300000 - 30000) * 48
                base = max(36, min(84, base))
            else:
                base = 60
            if count > 1:
                base += min((count - 1) * 3, 12)

        else:  # 特别巨大
            if amount:
                base = 120 + min((amount - 300000) / 200000 * 24, 24)
                base = max(120, min(156, base))
            else:
                base = 132
            if count > 1:
                base += min((count - 1) * 4, 18)

        labels_str = "".join(labels)
        if "多名被害人" in labels_str:
            base += 2
        if any(kw in labels_str for kw in ["救灾", "抢险", "优抚"]):
            base += 3

        return base

    # ========== 量刑情节调整 ==========

    def apply_circumstances(self, base: float, labels: List[str], bucket: str, statutory: Tuple[int, int]) -> float:
        """应用量刑情节调节（从轻/从重/减轻等）"""
        labels_str = "".join(labels)
        reduction_factor = 0.0
        has_statutory_reduction = False

        # --- 法定减轻 ---
        if "未遂" in labels_str:
            reduction_factor += 0.50
            has_statutory_reduction = True
        if "从犯" in labels_str:
            reduction_factor += 0.50
            has_statutory_reduction = True
        if "限制刑事责任能力" in labels_str:
            reduction_factor += 0.30

        current = base * (1 - min(reduction_factor, 0.70))

        # --- 一般从宽情节 ---
        lenient_reduction = 0.0
        if "自首" in labels_str:
            lenient_reduction += 0.40
        elif "坦白" in labels_str:
            lenient_reduction += 0.30 if "重大" in labels_str else 0.20
        if "当庭自愿认罪" in labels_str:
            lenient_reduction += 0.10

        has_compensation = False
        if any(kw in labels_str for kw in ["赔偿并取得谅解", "退赔并取得谅解"]):
            lenient_reduction += 0.40
            has_compensation = True
        elif "取得谅解" in labels_str:
            lenient_reduction += 0.30
            has_compensation = True
        elif any(kw in labels_str for kw in ["退赔", "退赃"]):
            lenient_reduction += 0.30
            has_compensation = True

        if "认罪认罚" in labels_str:
            if has_compensation or "自首" in labels_str or "坦白" in labels_str:
                lenient_reduction = min(lenient_reduction, 0.60)
            else:
                lenient_reduction += 0.30
            lenient_reduction = min(lenient_reduction, 0.60)

        current *= (1 - lenient_reduction)

        # --- 从重情节 ---
        aggravation_factor = 0.0
        aggravation_months = 0
        if "累犯" in labels_str:
            aggravation_factor += 0.25
            aggravation_months = max(aggravation_months, 3)
        if "前科" in labels_str:
            aggravation_factor += 0.10

        current = current * (1 + aggravation_factor) + aggravation_months

        # --- 法定幅度回切 ---
        min_statutory, max_statutory = statutory
        if not has_statutory_reduction:
            current = max(current, min_statutory)
        current = min(current, max_statutory)
        return current

    # ========== 区间生成工具函数 ==========

    def _snap_to_common_months(self, months: float, bucket: str) -> int:
        """将点估值对齐到法院常见的量刑刻度"""
        step = 3 if months < 36 else 6
        snapped = int(round(months / step) * step)
        return max(0, snapped)

    def calculate_interval(
        self, point: float, bucket: str, labels: List[str], statutory: Tuple[int, int]
    ) -> Tuple[int, int]:
        """生成量刑区间 [下限, 上限]"""
        labels_str = "".join(labels)
        min_statutory, max_statutory = statutory

        # --- 置信度 ---
        confidence = 0
        if self.parse_amount(labels) is not None:
            confidence += 2
        if self.parse_count(labels) <= 2:
            confidence += 1
        if any(kw in labels_str for kw in ["认罪认罚", "当庭自愿认罪"]):
            confidence += 1
        if "次数" in labels_str and "次" in labels_str:
            confidence -= 1
        if "累犯" in labels_str and "前科" in labels_str:
            confidence -= 1
        if len(labels) > 7:
            confidence -= 1

        has_statutory_reduction = any(kw in labels_str for kw in ["未遂", "从犯"])
        snapped = self._snap_to_common_months(point, bucket)

        if not has_statutory_reduction:
            snapped = max(snapped, min_statutory)
        snapped = min(snapped, max_statutory)

        if confidence >= 3:
            lower, upper = snapped, snapped
        elif confidence == 2:
            lower, upper = snapped - 1, snapped + 1
        else:
            lower, upper = snapped - 2, snapped + 2

        if not has_statutory_reduction:
            lower = max(lower, min_statutory)
        else:
            lower = max(lower, 0)
        upper = min(upper, max_statutory)

        return int(round(lower)), int(round(upper))

    # ========== 主入口函数 ==========

    def predict(self, fact: str, labels: List[str]) -> Tuple[int, int]:
        """主预测函数，返回(下限月, 上限月)"""
        crime = self.detect_crime(fact, labels)
        amount = self.parse_amount(labels)
        count = self.parse_count(labels)
        bucket, statutory = self.get_amount_bucket(crime, amount, labels)
        base = self.calculate_base_months(bucket, amount, count, labels)
        point = self.apply_circumstances(base, labels, bucket, statutory)
        lower, upper = self.calculate_interval(point, bucket, labels, statutory)
        return lower, upper


# ========== 便捷调用函数 ==========

def predict_interval_months(fact: str, labels: List[str]) -> Tuple[int, int]:
    """便捷预测函数"""
    engine = SentencingRulesEngine()
    return engine.predict(fact, labels)
