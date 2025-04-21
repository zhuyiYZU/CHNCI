import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 定义保存结果图的文件夹
output_dir = r"C:\Users\zxnb\Desktop\NLinear_res"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹，如果不存在
error_file_path = os.path.join(output_dir, "error_metrics.txt")

# 定义 NLinear 模型
class NLinear(nn.Module):
    def __init__(self, input_size=6, output_size=1):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()  # 增加非线性激活层

    def forward(self, src):
        # src: (batch_size, time_steps, 1)
        src = src.squeeze(-1)  # 去掉最后一个维度，变成 (batch_size, time_steps)
        output = self.linear(src)  # (batch_size, output_size)
        return self.activation(output)  # 通过激活函数映射为非线性输出

# 定义滑动窗口函数
time_steps = 6  # 使用过去6小时的数据预测未来的趋势
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 初始化 NLinear 模型
model = NLinear(input_size=time_steps, output_size=1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练事件和文件夹路径
train_file_paths = {
    "遭教官体罚进ICU的14岁女孩离世": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\遭教官体罚进ICU的14岁女孩离世.csv",
    "徐州多人占铁轨拍照逼停火车头(2)": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\徐州多人占铁轨拍照逼停火车头(2).csv",
    "武磊回应遭球迷辱骂": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\武磊回应遭球迷辱骂.csv",
    "小区业主": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\小区业主.csv",
    "韩国冠军被霸凌": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\董宇辉.csv",
}
data_folder = r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res"

# 获取测试事件路径（排除训练事件）
train_files_set = set(train_file_paths.values())
test_file_paths = {
    f: os.path.join(data_folder, f) 
    for f in os.listdir(data_folder) 
    if f.endswith('.csv') and os.path.join(data_folder, f) not in train_files_set
}

# 训练 NLinear 模型
for event_name, file_path in train_file_paths.items():
    df = pd.read_csv(file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])

    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data)

    # 生成训练数据
    X, y = create_sequences(scaled_data, time_steps)
    X = torch.from_numpy(X).float().unsqueeze(-1)  # (batch_size, time_steps, 1)
    y = torch.from_numpy(y).float()  # (batch_size, 1)

    # 模型训练
    for epoch in range(20):
        total_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            src = X[i].unsqueeze(0)  # (1, time_steps, 1)
            src = src.squeeze(-1)  # (1, time_steps)
            y_pred = model(src)
            single_loss = loss_function(y_pred, y[i].unsqueeze(0))
            single_loss.backward()
            optimizer.step()
            total_loss += single_loss.item()
        print(f'Epoch {epoch+1}/20 | Average Loss for {event_name}: {total_loss/len(X):.6f}')

# 测试部分
mae_list = []
mse_list = []
rmse_list = []

for test_event_name, test_file_path in test_file_paths.items():
    print(f"\nEvaluating on test event: {test_event_name}")
    df = pd.read_csv(test_file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])

    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaled_data = scaler.transform(hourly_data)

    # 创建测试数据
    X_test = create_sequences(scaled_data, time_steps)[0]
    X_test = torch.from_numpy(X_test).float().unsqueeze(-1)  # (batch_size, time_steps, 1)

    # 预测
    predictions = []
    with torch.no_grad():
        for seq in X_test:
            src = seq.unsqueeze(0)  # (1, time_steps, 1)
            src = src.squeeze(-1)  # (1, time_steps)
            prediction = model(src).item()
            predictions.append(prediction)

    # 反标准化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(scaled_data[time_steps:])

    # 计算误差指标
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 保存误差指标到文件
    with open(error_file_path, "a") as f:
        f.write(f"Event: {test_event_name}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("="*30 + "\n")

    # 可视化预测结果并保存
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")
    plt.xlabel("Time Steps")
    plt.ylabel("Malicious Comments Count")
    plt.legend()

    # 保存图像
    save_path = os.path.join(output_dir, f"{test_event_name}_prediction.png")
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")

# 打印平均误差
print(f"Average MAE: {sum(mae_list)/len(mae_list):.4f}")
print(f"Average MSE: {sum(mse_list)/len(mse_list):.4f}")
print(f"Average RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
