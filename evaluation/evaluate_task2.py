"""
子任务二：刑期区间预测 Winkler Score 评测
支持年份和月份的自动转换
"""
import json
import re
from typing import List, Dict, Tuple, Union


class Task2Evaluator:
    """刑期区间预测评测器"""

    def __init__(self, alpha: float = 0.2, s0: float = 10.0):
        """
        Args:
            alpha: 置信度参数（越小惩罚越大）
            s0: 转化灵敏度参数
        """
        self.alpha = alpha
        self.s0 = s0

    def parse_sentence_to_months(self, sentence: Union[str, int, float, List]) -> float:
        """
        将刑期转换为月份

        支持格式:
        - "4年" -> 48
        - "3年6个月" -> 42
        - "6个月" -> 6
        - 48 -> 48
        - [36, 60] -> 48 (取中点)

        Args:
            sentence: 刑期（多种格式）

        Returns:
            月份数
        """
        # 如果是列表/元组（区间），取中点
        if isinstance(sentence, (list, tuple)):
            return sum(sentence) / len(sentence)

        # 如果已经是数字，直接返回（假设是月份）
        if isinstance(sentence, (int, float)):
            return float(sentence)

        # 字符串格式解析
        sentence_str = str(sentence).strip()

        # 匹配 "X年Y个月" 或 "X年" 或 "Y个月"
        total_months = 0.0

        # 匹配年份
        year_match = re.search(r'(\d+(?:\.\d+)?)\s*年', sentence_str)
        if year_match:
            years = float(year_match.group(1))
            total_months += years * 12

        # 匹配月份
        month_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:个)?月', sentence_str)
        if month_match:
            months = float(month_match.group(1))
            total_months += months

        # 如果都没匹配到，尝试纯数字
        if total_months == 0:
            try:
                total_months = float(sentence_str)
            except:
                raise ValueError(f"无法解析刑期: {sentence}")

        return total_months

    def winkler_score(self, lower: float, upper: float, true_value: float) -> float:
        """
        计算 Winkler Score

        Args:
            lower: 预测区间下限（月）
            upper: 预测区间上限（月）
            true_value: 真实值（月）

        Returns:
            Winkler score（越小越好）
        """
        width = upper - lower

        if lower <= true_value <= upper:
            # 真实值在区间内
            return width
        elif true_value < lower:
            # 真实值在区间下方
            return width + (2 / self.alpha) * (lower - true_value)
        else:  # true_value > upper
            # 真实值在区间上方
            return width + (2 / self.alpha) * (true_value - upper)

    def winkler_to_score(self, winkler: float) -> float:
        """
        将 Winkler Score 转化为得分（0-1之间，越大越好）
        """
        return 1.0 / (1.0 + winkler / self.s0)

    def evaluate(self, pred_file: str, gt_file: str) -> Dict:
        """
        评测整个数据集

        Args:
            pred_file: 预测结果文件路径（月份格式）
            gt_file: 真实标签文件路径（年份格式或月份格式）

        Returns:
            评测结果字典
        """
        # 读取预测结果
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = {item['id']: item for item in (json.loads(line) for line in f)}

        # 读取真实标签
        with open(gt_file, 'r', encoding='utf-8') as f:
            ground_truths = {item['id']: item for item in (json.loads(line) for line in f)}

        # 逐样本计算
        results = []
        total_winkler = 0.0
        total_score = 0.0
        parse_errors = []

        for item_id in sorted(predictions.keys()):
            if item_id not in ground_truths:
                print(f"警告: ID {item_id} 在真实标签中不存在")
                continue

            try:
                # 预测区间（假设已是月份）
                pred_interval = predictions[item_id].get('answer2', [0, 0])
                if isinstance(pred_interval, list) and len(pred_interval) == 2:
                    lower, upper = float(pred_interval[0]), float(pred_interval[1])
                else:
                    raise ValueError(f"预测格式错误: {pred_interval}")

                # 真实值（可能是年份格式，需要转换）
                true_raw = ground_truths[item_id].get('answer', None)
                if true_raw is None:
                    true_raw = ground_truths[item_id].get('answer2', None)

                if true_raw is None:
                    print(f"警告: ID {item_id} 缺少真实标签")
                    continue

                # 转换为月份
                true_months = self.parse_sentence_to_months(true_raw)

                # 计算 Winkler Score
                winkler = self.winkler_score(lower, upper, true_months)
                score = self.winkler_to_score(winkler)

                # 判断是否覆盖真实值
                is_covered = lower <= true_months <= upper

                results.append({
                    'id': item_id,
                    'pred_lower_months': lower,
                    'pred_upper_months': upper,
                    'pred_lower_years': round(lower / 12, 2),
                    'pred_upper_years': round(upper / 12, 2),
                    'true_raw': str(true_raw),
                    'true_months': round(true_months, 2),
                    'true_years': round(true_months / 12, 2),
                    'interval_width_months': round(upper - lower, 2),
                    'interval_width_years': round((upper - lower) / 12, 2),
                    'is_covered': is_covered,
                    'deviation': round(abs(true_months - (lower + upper) / 2), 2),
                    'winkler_score': round(winkler, 4),
                    'normalized_score': round(score, 4)
                })

                total_winkler += winkler
                total_score += score

            except Exception as e:
                parse_errors.append({
                    'id': item_id,
                    'error': str(e),
                    'pred': predictions[item_id].get('answer2'),
                    'true': ground_truths[item_id].get('answer')
                })
                print(f"错误: ID {item_id} 解析失败 - {e}")

        n_samples = len(results)

        if n_samples == 0:
            return {
                'error': '没有有效样本',
                'parse_errors': parse_errors
            }

        # 计算平均指标
        avg_winkler = total_winkler / n_samples
        avg_score = total_score / n_samples

        # 统计覆盖率
        n_covered = sum(1 for r in results if r['is_covered'])
        coverage_rate = n_covered / n_samples

        # 统计区间宽度
        avg_width = sum(r['interval_width_months'] for r in results) / n_samples

        return {
            'task': 'task2_sentence_interval_prediction',
            'metric': 'Winkler Score',
            'n_samples': n_samples,
            'n_parse_errors': len(parse_errors),
            'alpha': self.alpha,
            's0': self.s0,
            'avg_winkler_score': round(avg_winkler, 4),
            'avg_normalized_score': round(avg_score, 4),
            'coverage_rate': round(coverage_rate, 4),
            'n_covered': n_covered,
            'avg_interval_width_months': round(avg_width, 2),
            'avg_interval_width_years': round(avg_width / 12, 2),
            'score': round(avg_score, 4),  # 总分
            'per_sample_results': results,
            'parse_errors': parse_errors if parse_errors else None
        }

    def save_results(self, results: Dict, output_file: str):
        """保存评测结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 子任务二评测结果已保存到: {output_file}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='子任务二：刑期区间预测评测（支持年份/月份自动转换）')
    parser.add_argument('--pred', required=True, help='预测结果文件路径（月份格式）')
    parser.add_argument('--gt', required=True, help='真实标签文件路径（年份或月份格式）')
    parser.add_argument('--output', default='output/task2_evaluation.json', help='输出文件路径')
    parser.add_argument('--alpha', type=float, default=0.2, help='置信度参数')
    parser.add_argument('--s0', type=float, default=10.0, help='转化灵敏度参数')

    args = parser.parse_args()

    evaluator = Task2Evaluator(alpha=args.alpha, s0=args.s0)
    results = evaluator.evaluate(args.pred, args.gt)

    if 'error' in results:
        print(f"\n错误: {results['error']}")
        if results.get('parse_errors'):
            print("\n解析错误详情:")
            for err in results['parse_errors']:
                print(f"  ID {err['id']}: {err['error']}")
        return

    print("\n" + "="*60)
    print("子任务二：刑期区间预测评测结果")
    print("="*60)
    print(f"样本数量: {results['n_samples']}")
    print(f"覆盖率: {results['coverage_rate']:.2%} ({results['n_covered']}/{results['n_samples']})")
    print(f"平均区间宽度: {results['avg_interval_width_months']:.1f} 月 ({results['avg_interval_width_years']:.2f} 年)")
    print(f"平均 Winkler Score: {results['avg_winkler_score']:.4f}")
    print(f"平均标准化得分: {results['avg_normalized_score']:.4f}")
    print(f"最终得分: {results['score']:.4f}")

    if results.get('n_parse_errors', 0) > 0:
        print(f"\n警告: {results['n_parse_errors']} 个样本解析失败")

    print("="*60)

    evaluator.save_results(results, args.output)


if __name__ == '__main__':
    main()
