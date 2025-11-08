import json
import os
from pathlib import Path


def filter_cases_by_accusation(input_file, output_file, target_accusations, mode='a'):
    """
    筛选指定罪名的案件

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        target_accusations: 目标罪名列表
        mode: 写入模式 ('w' 表示覆盖, 'a' 表示追加)
    """
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return 0

    count = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
                open(output_file, mode, encoding='utf-8') as f_out:

            for line_num, line in enumerate(f_in, 1):
                try:
                    # 去除空行
                    line = line.strip()
                    if not line:
                        continue

                    # 解析JSON
                    data = json.loads(line)

                    # 检查是否包含meta和accusation字段
                    if 'meta' in data and 'accusation' in data['meta']:
                        accusations = data['meta']['accusation']

                        # 检查是否包含目标罪名
                        if any(acc in target_accusations for acc in accusations):
                            # 写入输出文件
                            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                            count += 1

                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 - 文件: {input_file}, 行: {line_num}, 错误: {e}")
                except Exception as e:
                    print(f"处理错误 - 文件: {input_file}, 行: {line_num}, 错误: {e}")

        print(f"处理完成: {input_file} -> 找到 {count} 条匹配记录")
        return count

    except Exception as e:
        print(f"文件处理失败: {input_file}, 错误: {e}")
        return 0


def main():
    # 定义目标罪名
    target_accusations = ['盗窃', '故意伤害', '诈骗']

    # 定义输出文件
    output_file = 'exercise_contest/filtered_cases.jsonl'

    # 定义要处理的文件列表
    files_to_process = [
        'exercise_contest/data_train.json',
        'exercise_contest/data_test.json',
        'exercise_contest/data_valid.json',
        'first_stage/test.json'
    ]

    # 如果输出文件已存在,先删除
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"已删除旧的输出文件: {output_file}\n")

    # 统计总数
    total_count = 0

    # 处理每个文件
    for i, file_path in enumerate(files_to_process):
        print(f"\n正在处理 [{i + 1}/{len(files_to_process)}]: {file_path}")

        # 第一个文件使用写入模式,后续文件使用追加模式
        mode = 'w' if i == 0 else 'a'
        count = filter_cases_by_accusation(file_path, output_file, target_accusations, mode)
        total_count += count

    print(f"\n{'=' * 60}")
    print(f"全部处理完成!")
    print(f"目标罪名: {', '.join(target_accusations)}")
    print(f"总共筛选出: {total_count} 条案件记录")
    print(f"输出文件: {output_file}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
