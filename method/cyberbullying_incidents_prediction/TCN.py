import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
output_dir = r"C:\Users\zxnb\Desktop\tcn"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹，如果不存在
error_file_path = os.path.join(output_dir, "error_metrics.txt")

# TCN Model Definition
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=(kernel_size-1) * dilation_size, dilation=dilation_size),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(dropout)
            ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size=1, num_channels=[128, 128, 128], kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2))  # [batch_size, num_channels, seq_len]
        o = self.linear(y1[:, :, -1])  # Use the last time step for prediction
        return o

# Define Train and Test Event Paths
train_file_paths = {
    "遭教官体罚进ICU的14岁女孩离世": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\遭教官体罚进ICU的14岁女孩离世.csv",
    "徐州多人占铁轨拍照逼停火车头(2)": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\徐州多人占铁轨拍照逼停火车头(2).csv",
    "武磊回应遭球迷辱骂": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\武磊回应遭球迷辱骂.csv",
    "小区业主": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\小区业主.csv",
    "韩国冠军被霸凌": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\董宇辉.csv",
}
data_folder = r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res"

# Get Test Files by Excluding Train Files
train_files_set = set(train_file_paths.values())  # Set of train file paths
test_file_paths = {
    f: os.path.join(data_folder, f) 
    for f in os.listdir(data_folder) 
    if f.endswith('.csv') and os.path.join(data_folder, f) not in train_files_set
}

# Sliding Window Sequence Function
time_steps = 6  # Set time steps
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Initialize TCN Model
model = TCNModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the Model
for event_name, file_path in train_file_paths.items():
    df = pd.read_csv(file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # Remove invalid times

    # Data Preprocessing
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('h', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data)

    # Create Training Sequences
    X, y = create_sequences(scaled_data, time_steps)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Model Training
    for epoch in range(20):
        total_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            src = X[i].unsqueeze(0)  # Add batch dimension
            y_pred = model(src)
            single_loss = loss_function(y_pred, y[i])
            single_loss.backward()
            optimizer.step()
            total_loss += single_loss.item()
        print(f'Epoch {epoch+1}/10 | Average Loss for {event_name}: {total_loss/len(X):.6f}')
mae_list = []
mse_list = []
rmse_list = []
mape_list = []
r2_list = []
all_predictions = []
# Testing and Evaluating the Model
for test_event_name, test_file_path in test_file_paths.items():
    print(f"\nEvaluating on test event: {test_event_name}")
    df = pd.read_csv(test_file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # Remove invalid times

    # Data Preprocessing
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('h', on='time').size().fillna(0).to_frame(name='malicious_count')
    scaled_data = scaler.transform(hourly_data)

    # Create Test Sequences
    X_test = create_sequences(scaled_data, time_steps)[0]
    X_test = torch.from_numpy(X_test).float()

    # Prediction
    predictions = []
    with torch.no_grad():
        for seq in X_test:
            src = seq.unsqueeze(0)  # Add batch dimension
            prediction = model(src)
            predictions.append(prediction.item())

    # Inverse Transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(scaled_data[time_steps:])
    all_predictions.extend(
        [
            {"Event": test_event_name, "True Value": tv[0], "Prediction": pred[0]} 
            for tv, pred in zip(true_values, predictions)
        ]
    )
    # Calculate Error Metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    with open(error_file_path, "a") as f:
        f.write(f"Event: {test_event_name}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("="*30 + "\n")

    # Plotting Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")
    plt.xlabel("Time Steps")
    plt.ylabel("Malicious Comments Count")
    plt.title(f"True Values vs Predictions for {test_event_name}")
    plt.legend()
    # Save Plot
    save_path = os.path.join(output_dir, f"{test_event_name}_prediction.png")
    plt.savefig(save_path)  # Save figure to path
    print(f"Prediction plot saved to {save_path}")
all_predictions_df = pd.DataFrame(all_predictions)
all_predictions_csv_path = os.path.join(output_dir, "all_predictions.csv")
all_predictions_df.to_csv(all_predictions_csv_path, index=False, encoding='gbk')
print(f"All predictions saved to {all_predictions_csv_path}")

# 打印平均误差指标
print(f"Average MAE: {sum(mae_list)/len(mae_list):.4f}")
print(f"Average MSE: {sum(mse_list)/len(mse_list):.4f}")
print(f"Average RMSE: {sum(rmse_list)/len(rmse_list):.4f}")