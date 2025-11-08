import json


def convert_to_jsonl(input_file, output_file):
    """
    读取包含多个 JSON 对象的文本文件，将其转换为 jsonl 格式。

    输入格式（每行一个）:
    {"id": 1, "crime": "...", "labels": [...], "term_of_imprisonment": {"imprisonment": "25,32"}}

    输出格式（每行一个）:
    {"id": 1, "answer1": [...], "answer2": [25, 32]}
    """

    try:
        # 使用 'utf-8' 编码打开输入和输出文件，以正确处理中文字符
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:

            line_number = 0
            for line in infile:
                line_number += 1
                line = line.strip()  # 去除行首尾的空白字符

                if not line:  # 跳过空行
                    continue

                try:
                    # 1. 将每行解析为
                    data = json.loads(line)

                    # 2. 提取并转换刑期
                    imprisonment_str = data.get('term_of_imprisonment', {}).get('imprisonment', '0,0')
                    imprisonment_parts = imprisonment_str.split(',')

                    # 确保有两个部分，并能转换为整数
                    if len(imprisonment_parts) == 2:
                        term_start = int(imprisonment_parts[0])
                        term_end = int(imprisonment_parts[1])
                        imprisonment_list = [term_start, term_end]
                    else:
                        print(f"警告：第 {line_number} 行刑期格式错误，使用 [0, 0] 代替: {imprisonment_str}")
                        imprisonment_list = [0, 0]

                    # 3. 构建新的字典
                    output_data = {
                        "id": data.get('id'),
                        "answer1": data.get('labels', []),
                        "answer2": imprisonment_list
                    }

                    # 4. 将新字典转换为 JSON 字符串并写入输出文件
                    # ensure_ascii=False 确保中文按原样写入
                    json.dump(output_data, outfile, ensure_ascii=False)
                    outfile.write('\n')  # 添加换行符，符合 jsonl 格式

                except json.JSONDecodeError:
                    print(f"错误：第 {line_number} 行不是有效的 JSON，已跳过。")
                except (KeyError, TypeError, ValueError) as e:
                    print(f"错误：处理第 {line_number} 行数据时出错: {e}，已跳过。")

            print(f"转换完成！已成功写入 {output_file}")

    except FileNotFoundError:
        print(f"错误：未找到输入文件 '{input_file}'。")
    except Exception as e:
        print(f"发生意外错误: {e}")


# --- 脚本执行 ---
if __name__ == "__main__":
    input_filename = "./result1/submission_v13_Official_Guidelines_deepseek.jsonl"
    output_filename = "submission_v13_Official_Guidelines_deepseek.jsonl"

    convert_to_jsonl(input_filename, output_filename)
