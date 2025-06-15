import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 定义保存结果图的文件夹
output_dir = r"./autoformer"
os.makedirs(output_dir, exist_ok=True)
error_file_path = os.path.join(output_dir, "error_metrics.txt")

# 使用 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Autoformer 核心模块**
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, nhead):
        super(AutoCorrelation, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 计算 Auto-Correlation Attention
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = self.softmax(Q @ K.transpose(-2, -1) / (x.size(-1) ** 0.5))
        return attn_weights @ V

# **Autoformer 模型架构**
class Autoformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=8, num_encoder_layers=3, dim_feedforward=128, dropout=0.1, seq_len=6):
        super(Autoformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.encoder_layer = nn.ModuleList([
            nn.Sequential(
                AutoCorrelation(d_model, nhead),
                nn.Linear(d_model, d_model),
                nn.ReLU()
            ) for _ in range(num_encoder_layers)
        ])
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src)  # (batch_size, time_steps, d_model)
        for layer in self.encoder_layer:
            src = layer(src)  # 自相关模块
        output = self.fc(src[:, -1, :])  # 直接取最后一个时间步
        return output

# 定义滑动窗口函数
time_steps = 6  # 使用过去6小时的数据预测未来的趋势
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 初始化模型
model = Autoformer(d_model=64, nhead=8, num_encoder_layers=3).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和测试数据路径
train_file_paths = {
    "遭教官体罚进ICU的14岁女孩离世": r"/home/ubuntu/zx/data/遭教官体罚进ICU的14岁女孩离世.csv",
    "徐州多人占铁轨拍照逼停火车头(2)": r"/home/ubuntu/zx/data/徐州多人占铁轨拍照逼停火车头(2).csv",
    "武磊回应遭球迷辱骂": r"/home/ubuntu/zx/data/武磊回应遭球迷辱骂.csv",
    "小区业主": r"/home/ubuntu/zx/data/小区业主.csv",
    "董宇辉": r"/home/ubuntu/zx/data/董宇辉.csv",
}
data_folder = r"/home/ubuntu/zx/data"

# 获取测试事件路径（排除训练事件）
train_files_set = set(train_file_paths.values())
test_file_paths = {
    f: os.path.join(data_folder, f) 
    for f in os.listdir(data_folder) 
    if f.endswith('.csv') and os.path.join(data_folder, f) not in train_files_set
}

# 训练 Autoformer 模型
scaler = MinMaxScaler()
for event_name, file_path in train_file_paths.items():
    df = pd.read_csv(file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # Remove invalid times

    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaled_data = scaler.fit_transform(hourly_data)

    # 生成训练数据
    X, y = create_sequences(scaled_data, time_steps)
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    # 创建 DataLoader
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/20 | Average Loss for {event_name}: {total_loss/len(train_loader):.6f}')
# 计算误差并保存结果
mae_list = []
mse_list = []
rmse_list = []
all_predictions = []

# 在测试事件上进行预测并计算误差指标
for test_event_name, test_file_path in test_file_paths.items():
    print(f"\nEvaluating on test event: {test_event_name}")
    df = pd.read_csv(test_file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # Remove invalid times

    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaled_data = scaler.transform(hourly_data)

    # 创建测试数据
    X_test = create_sequences(scaled_data, time_steps)[0]
    X_test = torch.from_numpy(X_test).float().to(device)

    # 预测
    predictions = []
    with torch.no_grad():
        for seq in X_test:
            src = seq.unsqueeze(0).to(device)  # 增加维度适应 Transformer
            prediction = model(src)
            predictions.append(prediction.item())

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

# 打印平均误差指标
print(f"Average MAE: {sum(mae_list)/len(mae_list):.4f}")
print(f"Average MSE: {sum(mse_list)/len(mse_list):.4f}")
print(f"Average RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
