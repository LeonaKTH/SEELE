import json


def convert_to_jsonl_and_fix(input_path, output_path, column_to_fix="event_list"):
    """
    读取一个包含单一JSON数组的文件，将其转换为JSON Lines (.jsonl) 格式，
    并在转换过程中修复指定列，确保其始终为列表。
    """
    print("--- 开始将JSON数组转换为JSON Lines格式 ---")
    print(f"读取文件: {input_path}")
    print(f"准备写入文件: {output_path}")
    print(f"需要修复的字段: '{column_to_fix}'")

    try:
        # 1. 一次性读取整个JSON数组文件
        with open(input_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # 确保文件顶层是一个列表
        if not isinstance(all_data, list):
            print(f"错误：文件 {input_path} 的内容不是一个JSON数组/列表。请检查原始文件。")
            return

        # 2. 逐条处理并写入为JSON Lines格式
        record_count = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in all_data:
                # 跳过非字典类型的无效记录
                if not isinstance(record, dict):
                    continue

                # 3. 修复指定列的格式，确保是列表
                if column_to_fix in record and not isinstance(record[column_to_fix], list):
                    # 如果字段存在但不是列表，将其包装成列表
                    record[column_to_fix] = [record[column_to_fix]]

                # 4. 将处理后的单条记录（字典）转换为JSON字符串，并添加换行符写入新文件
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                record_count += 1

        print(f"\n处理完成！总共转换了 {record_count} 条记录。")
        print(f"已成功生成JSON Lines格式文件: {output_path}")
        print("--- 转换成功 ---")

    except json.JSONDecodeError as e:
        print(f"\n错误：JSON解析失败！请确保文件 {input_path} 是一个有效的、包含单个数组的JSON格式。")
        print(f"错误详情: {e}")
    except FileNotFoundError:
        print(f"\n错误：找不到输入文件 {input_path}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")


# --- 使用方法 ---

# 1. 指定你的原始输入文件路径
input_file = r'D:\NLP\SEELE-main\SEELE-main\data\FNDEE\FNDEE_test_2500_.json'

# 2. 指定输出文件的路径
output_file = r'D:\NLP\SEELE-main\SEELE-main\data\FNDEE\FNDEE_test_2500_corrected.json'

# 3. 再次确认需要修正的列名！根据我们之前的分析，"event_list" 是最可能的。
#    如果还有问题，请打开原始JSON文件，确认这个名字是否正确。
column_name_to_fix = "event_list"

# 4. 运行转换函数
convert_to_jsonl_and_fix(input_file, output_file, column_to_fix=column_name_to_fix)