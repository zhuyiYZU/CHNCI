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
output_dir = r"C:\Users\zxnb\Desktop\DLinear_res"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹，如果不存在
error_file_path = os.path.join(output_dir, "error_metrics.txt")


class DLinear(nn.Module):
    def __init__(self, seq_len=6, pred_len=1):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 趋势项和残差项分别建模
        self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self.linear_residual = nn.Linear(self.seq_len, self.pred_len)

    def decompose(self, x):
        """
        时间序列分解为趋势项和残差项
        x: shape (batch_size, seq_len)
        """
        moving_avg = x.mean(dim=1, keepdim=True).repeat(1, self.seq_len)
        trend = moving_avg
        residual = x - trend
        return trend, residual

    def forward(self, x):
        """
        x: shape (batch_size, seq_len)
        返回预测值 shape (batch_size, pred_len)
        """
        trend, residual = self.decompose(x)
        trend_output = self.linear_trend(trend)
        residual_output = self.linear_residual(residual)
        return trend_output + residual_output
# ----------------------------------------------

# 定义滑动窗口函数
time_steps = 6  # 使用过去6小时的数据预测未来趋势
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 初始化 DLinear 模型
model = DLinear(seq_len=time_steps, pred_len=1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练事件和路径
train_file_paths = {
    "遭教官体罚进ICU的14岁女孩离世": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\遭教官体罚进ICU的14岁女孩离世.csv",
    "徐州多人占铁轨拍照逼停火车头(2)": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\徐州多人占铁轨拍照逼停火车头(2).csv",
    "武磊回应遭球迷辱骂": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\武磊回应遭球迷辱骂.csv",
    "小区业主": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\小区业主.csv",
    "董宇辉": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\董宇辉.csv",
}
data_folder = r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res"

# 获取测试事件路径（排除训练事件）
train_files_set = set(train_file_paths.values())
test_file_paths = {
    f: os.path.join(data_folder, f) 
    for f in os.listdir(data_folder) 
    if f.endswith('.csv') and os.path.join(data_folder, f) not in train_files_set
}

# 训练 DLinear 模型
for event_name, file_path in train_file_paths.items():
    df = pd.read_csv(file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])

    # 取恶意评论标签为1的数据，按小时聚合
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data)

    # 生成训练数据序列
    X, y = create_sequences(scaled_data, time_steps)
    X = torch.from_numpy(X).float()  # (batch_size, time_steps)
    y = torch.from_numpy(y).float()  # (batch_size, 1)

    # 训练
    for epoch in range(20):
        total_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            src = X[i].unsqueeze(0)  # (1, time_steps)
            y_pred = model(src)      # (1, pred_len=1)
            loss = loss_function(y_pred, y[i].unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
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

    # 取恶意评论标签为1的数据，按小时聚合
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaled_data = scaler.transform(hourly_data)

    # 创建测试数据序列
    X_test = create_sequences(scaled_data, time_steps)[0]
    X_test = torch.from_numpy(X_test).float()  # (batch_size, time_steps)

    predictions = []
    with torch.no_grad():
        for seq in X_test:
            src = seq.unsqueeze(0)  # (1, time_steps)
            pred = model(src).item()
            predictions.append(pred)

    # 反标准化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(scaled_data[time_steps:])

    # 误差指标
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    # 写入误差文件
    with open(error_file_path, "a") as f:
        f.write(f"Event: {test_event_name}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("="*30 + "\n")

    # 画图保存
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")
    plt.xlabel("时间步（小时）")
    plt.ylabel("恶意评论数量")
    plt.legend()
    plt.title(test_event_name + " 预测结果")
    save_path = os.path.join(output_dir, f"{test_event_name}_prediction.png")
    plt.savefig(save_path)
    plt.close()
    print(f"预测图已保存到 {save_path}")

# 平均误差打印
print(f"Average MAE: {sum(mae_list)/len(mae_list):.4f}")
print(f"Average MSE: {sum(mse_list)/len(mse_list):.4f}")
print(f"Average RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
