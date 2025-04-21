import csv

# 定义提示词字符串，包含您提供的例子
prompt_str = (
    '对于网络上出现的热点事件，网友会对这一事件自由的发表自己的观点，但是网友发表的观点评论可能会有直接或间接冒犯到个人或集体的，帮我对以下输入的网友评论进行解析，并判断是友善的还是恶意的。\n文本:'
)


def classify_text(text, client):
    # 将提示词和待分类的文本拼接
    prompt = prompt_str + f'"{text}"'+'如果是恶意的就输出1,如果是友善的就输出0。仅需要输出0或者1'

    response = client.chat.completions.create(
        model='Qwen/Qwen2.5-7B-Instruct',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
        max_tokens=5
    )

    output = ''
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content is not None:
            output += content
    return output.strip()


def standardize_output(output):
    output = output.strip().lower()  # 去除空格并转换为小写
    print(output)
    if '1' in output or '恶意' in output or 'negative' in output:
        return 1
    elif '0' in output or '友善' in output or 'positive' in output:
        return 0
    else:
        return None  # 无法识别的输出


def analyze_sentiment_and_update_csv(csv_file, client, output_csv_file):
    # 打开输入和输出文件
    with open(csv_file, 'r', encoding='gbk') as infile, open(output_csv_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 手动指定列索引：假设第 3 列为标签，第 4 列为文本
        for row in reader:
            text = row[3]  # 文本内容（假设为第 4 列）

            # 进行情感分类
            predicted_output = classify_text(text, client)
            predicted_label = standardize_output(predicted_output)

            # 在原始行末尾添加预测结果
            row.append(predicted_label)
            writer.writerow(row)
            print(f"Processed: {text}, Predicted: {predicted_label}")

# 初始化OpenAI客户端（请确保替换为您的API密钥）
from openai import OpenAI

client = OpenAI(
    api_key="sk-qfllasqpccngetzdtocpvzvuwkuapbbzdstwqjtgvrdjghuu",  # 请替换为您的API密钥
    base_url="https://api.siliconflow.cn/v1"
)

# 指定CSV文件路径和输出文件路径
csv_file_path = r'C:\Users\zxnb\Desktop\总数据.csv'
output_file_path = 'merged_file_with_predictions.csv'

# 调用分析函数
analyze_sentiment_and_update_csv(csv_file_path, client, output_file_path)
