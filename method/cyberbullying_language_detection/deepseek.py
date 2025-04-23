import pandas as pd
from openai import OpenAI
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-****",
    base_url="https://api.deepseek.com"
)

# 读取原始数据
input_file = "总数据.csv"
output_file = "总数据_已打标签.csv"

df = pd.read_csv(input_file, encoding='gbk')
df.columns.values[2] = "true_label"  # 第三列是真实标签
df.columns.values[3] = "content"     # 第四列是文本内容

# 如果输出文件已存在，读取它以避免重复打标签
processed_ids = set()
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file, encoding="utf-8-sig")
    processed_ids = set(processed_df.index.tolist())

# 打标签函数
def classify_text(text):
    prompt = (f"对于网络上出现的热点事件，网友会对这一事件自由的发表自己的观点，但是网友发表的观点评论可能会有直接或间接冒犯到个人或集体的，帮我对以下输入的网友评论进行解析，并判断是友善的还是恶意的。"
              f"友善的就输出0，恶意的就输出1.仅需要输出0或1，不要做解释。\n\n文本：{text}\n\n回答：")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个善于识别网络恶意言论的助手"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        result = response.choices[0].message.content.strip()
        print(f"模型输出：{result}")
        return int(result) if result in ["0", "1"] else None
    except Exception as e:
        print(f"错误处理文本: {text}\n错误信息: {e}")
        return None

# 逐行处理并立即写入
for idx, row in df.iloc[11540:].iterrows():
    if idx in processed_ids:
        continue  # 已处理过，跳过

    text = row["content"]
    label = classify_text(text)

    # 构造包含原始信息的新行 DataFrame
    new_row = row.copy()
    new_row["predicted_label"] = label
    new_row_df = pd.DataFrame([new_row])

    # 写入文件
    write_header = not os.path.exists(output_file)
    new_row_df.to_csv(output_file, mode='a', index=False, header=write_header, encoding="utf-8-sig")

    print(f"✅ 已处理第 {idx} 行，预测标签: {label}")

    # 加载当前写入数据，计算性能指标
    try:
        temp_df = pd.read_csv(output_file, encoding="utf-8-sig")
        temp_df = temp_df.dropna(subset=["predicted_label"])
        y_true = temp_df["true_label"].astype(int)
        y_pred = temp_df["predicted_label"].astype(int)

        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0,average='weighted')
        rec = recall_score(y_true, y_pred, zero_division=0,average='weighted')
        f1 = f1_score(y_true, y_pred, zero_division=0,average='weighted')

        print(f"📊 当前指标 - ACC: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    except Exception as e:
        print(f"⚠️ 无法计算指标: {e}")

    time.sleep(1)
