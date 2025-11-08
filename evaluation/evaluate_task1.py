"""
子任务一：量刑情节识别 F1 评测
"""
import json
from typing import List, Dict, Set, Tuple


class Task1Evaluator:
    """量刑情节识别评测器"""

    def __init__(self):
        pass

    def normalize_label(self, label: str) -> str:
        """
        标准化标签（处理格式差异）
        例如：将 "盗窃金额既遂 105600 元" 和 "盗窃金额既遂105600元" 视为相同
        """
        # 去除多余空格
        label = " ".join(label.split())
        # 转小写
        label = label.lower()
        return label

    def calculate_f1(self, pred_labels: List[str], true_labels: List[str]) -> Tuple[float, float, float]:
        """
        计算单个样本的 Precision, Recall, F1

        Args:
            pred_labels: 预测的标签列表
            true_labels: 真实的标签列表

        Returns:
            (precision, recall, f1)
        """
        # 标准化
        pred_set = set(self.normalize_label(l) for l in pred_labels)
        true_set = set(self.normalize_label(l) for l in true_labels)

        if len(pred_set) == 0 and len(true_set) == 0:
            return 1.0, 1.0, 1.0

        if len(pred_set) == 0 or len(true_set) == 0:
            return 0.0, 0.0, 0.0

        # 计算交集
        intersection = pred_set & true_set

        # Precision = TP / (TP + FP)
        precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0

        # F1 = 2 * P * R / (P + R)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return precision, recall, f1

    def evaluate(self, pred_file: str, gt_file: str) -> Dict:
        """
        评测整个数据集

        Args:
            pred_file: 预测结果文件路径
            gt_file: 真实标签文件路径

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
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for item_id in sorted(predictions.keys()):
            if item_id not in ground_truths:
                print(f"警告: ID {item_id} 在真实标签中不存在")
                continue

            pred_labels = predictions[item_id].get('answer1', [])
            true_labels = ground_truths[item_id].get('answer1', [])

            precision, recall, f1 = self.calculate_f1(pred_labels, true_labels)

            results.append({
                'id': item_id,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_count': len(pred_labels),
                'true_count': len(true_labels)
            })

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n_samples = len(results)

        # 计算平均指标
        avg_precision = total_precision / n_samples if n_samples > 0 else 0.0
        avg_recall = total_recall / n_samples if n_samples > 0 else 0.0
        avg_f1 = total_f1 / n_samples if n_samples > 0 else 0.0

        return {
            'task': 'task1_circumstances_recognition',
            'metric': 'F1',
            'n_samples': n_samples,
            'avg_precision': round(avg_precision, 4),
            'avg_recall': round(avg_recall, 4),
            'avg_f1': round(avg_f1, 4),
            'score': round(avg_f1, 4),  # 总分 = F1
            'per_sample_results': results
        }

    def save_results(self, results: Dict, output_file: str):
        """保存评测结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ 子任务一评测结果已保存到: {output_file}")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='子任务一：量刑情节识别评测')
    parser.add_argument('--pred', required=True, help='预测结果文件路径')
    parser.add_argument('--gt', required=True, help='真实标签文件路径')
    parser.add_argument('--output', default='output/task1_evaluation.json', help='输出文件路径')

    args = parser.parse_args()

    evaluator = Task1Evaluator()
    results = evaluator.evaluate(args.pred, args.gt)

    print("\n" + "=" * 60)
    print("子任务一：量刑情节识别评测结果")
    print("=" * 60)
    print(f"样本数量: {results['n_samples']}")
    print(f"平均 Precision: {results['avg_precision']:.4f}")
    print(f"平均 Recall: {results['avg_recall']:.4f}")
    print(f"平均 F1: {results['avg_f1']:.4f}")
    print(f"最终得分: {results['score']:.4f}")
    print("=" * 60)

    evaluator.save_results(results, args.output)


if __name__ == '__main__':
    main()
