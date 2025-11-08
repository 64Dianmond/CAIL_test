"""
统一评测入口：同时评测两个子任务
"""
import json
import os
from evaluate_task1 import Task1Evaluator
from evaluate_task2 import Task2Evaluator


def evaluate_all(pred_file: str, gt_file: str, output_dir: str = 'output'):
    """
    评测两个子任务

    Args:
        pred_file: 预测结果文件路径
        gt_file: 真实标签文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("开始评测")
    print("=" * 60)

    # 子任务一
    print("\n[1/2] 评测子任务一：量刑情节识别...")
    task1_evaluator = Task1Evaluator()
    task1_results = task1_evaluator.evaluate(pred_file, gt_file)
    task1_output = os.path.join(output_dir, 'task1_evaluation.json')
    task1_evaluator.save_results(task1_results, task1_output)

    # 子任务二
    print("\n[2/2] 评测子任务二：刑期区间预测...")
    task2_evaluator = Task2Evaluator(alpha=0.2, s0=10.0)
    task2_results = task2_evaluator.evaluate(pred_file, gt_file)
    task2_output = os.path.join(output_dir, 'task2_evaluation.json')
    task2_evaluator.save_results(task2_results, task2_output)

    # 汇总结果
    summary = {
        'task1': {
            'name': '量刑情节识别',
            'metric': 'F1',
            'score': task1_results['score'],
            'details': {
                'precision': task1_results['avg_precision'],
                'recall': task1_results['avg_recall'],
                'f1': task1_results['avg_f1']
            }
        },
        'task2': {
            'name': '刑期区间预测',
            'metric': 'Winkler Score',
            'score': task2_results['score'],
            'details': {
                'coverage_rate': task2_results['coverage_rate'],
                'avg_winkler': task2_results['avg_winkler_score'],
                'avg_normalized_score': task2_results['avg_normalized_score']
            }
        },
        'overall': {
            'task1_score': task1_results['score'],
            'task2_score': task2_results['score'],
            'average_score': (task1_results['score'] + task2_results['score']) / 2
        }
    }

    # 保存汇总
    summary_output = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n" + "=" * 60)
    print("评测完成 - 汇总结果")
    print("=" * 60)
    print(f"\n子任务一（量刑情节识别）:")
    print(f"  F1 Score: {task1_results['score']:.4f}")
    print(f"  Precision: {task1_results['avg_precision']:.4f}")
    print(f"  Recall: {task1_results['avg_recall']:.4f}")

    print(f"\n子任务二（刑期区间预测）:")
    print(f"  Winkler Score: {task2_results['score']:.4f}")
    print(f"  覆盖率: {task2_results['coverage_rate']:.2%}")
    print(f"  平均 Winkler: {task2_results['avg_winkler_score']:.4f}")

    print(f"\n整体:")
    print(f"  平均得分: {summary['overall']['average_score']:.4f}")

    print(f"\n结果已保存到:")
    print(f"  - {task1_output}")
    print(f"  - {task2_output}")
    print(f"  - {summary_output}")
    print("=" * 60 + "\n")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description='统一评测两个子任务')
    parser.add_argument('--pred', required=True, help='预测结果文件路径')
    parser.add_argument('--gt', required=True, help='真实标签文件路径')
    parser.add_argument('--output-dir', default='output', help='输出目录')

    args = parser.parse_args()

    evaluate_all(args.pred, args.gt, args.output_dir)


if __name__ == '__main__':
    main()
