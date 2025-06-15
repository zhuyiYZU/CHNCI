import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题
output_dir = r"C:\Users\zxnb\Desktop\result_LSTM"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹，如果不存在
error_file_path = os.path.join(output_dir, "error_metrics.txt")
# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 定义训练事件和文件夹路径
train_file_paths = {
    "遭教官体罚进ICU的14岁女孩离世": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\遭教官体罚进ICU的14岁女孩离世.csv",
    "徐州多人占铁轨拍照逼停火车头(2)": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\徐州多人占铁轨拍照逼停火车头(2).csv",
    "武磊回应遭球迷辱骂": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\武磊回应遭球迷辱骂.csv",
    "小区业主": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\小区业主.csv",
    "董宇辉": r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res\董宇辉.csv",
}
data_folder = r"C:\Users\zxnb\Desktop\数据集(最终版)\res\res"

# 获取测试事件路径（排除训练事件）
train_files_set = set(train_file_paths.values())  # 将训练事件路径存储为集合
test_file_paths = {
    f: os.path.join(data_folder, f) 
    for f in os.listdir(data_folder) 
    if f.endswith('.csv') and os.path.join(data_folder, f) not in train_files_set
}

# 定义滑动窗口函数
time_steps = 6  # 设定时间步
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X).reshape(-1, time_steps, 1), np.array(y)

# 初始化模型
model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for event_name, file_path in train_file_paths.items():
    df = pd.read_csv(file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # 移除无效时间

    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    # total_numbers = len(df)

    #     # 计算每小时恶意评论比例
    # hourly_data = hourly_data / total_numbers
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hourly_data)

    # 生成训练数据
    X, y = create_sequences(scaled_data, time_steps)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # 模型训练
    for epoch in range(20):
        total_loss = 0
        for i in range(len(X)):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(X[i])
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
# 在测试事件上进行预测并计算误差指标
for test_event_name, test_file_path in test_file_paths.items():
    print(f"\nEvaluating on test event: {test_event_name}")
    df = pd.read_csv(test_file_path, encoding='gbk')
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['time'])  # 移除无效时间
    # 数据预处理
    event_data = df[df['label3'] == 1].sort_values(by='time')
    

    hourly_data = event_data.resample('H', on='time').size().fillna(0).to_frame(name='malicious_count')
    # total_numbers = len(df)

    #     # 计算每小时恶意评论比例
    # hourly_data = hourly_data / total_numbers
    # print(hourly_data)

    scaled_data = scaler.transform(hourly_data)

    # 创建测试数据
    X_test = create_sequences(scaled_data, time_steps)[0]
    X_test = torch.from_numpy(X_test).float()

    # 预测
    predictions = []
    print(predictions)
    with torch.no_grad():
        for seq in X_test:
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            prediction = model(seq)
            predictions.append(prediction.item())

    # 反标准化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    true_values = scaler.inverse_transform(scaled_data[time_steps:])
    all_predictions.extend(
        [
            {"Event": test_event_name, "True Value": tv[0], "Prediction": pred[0]} 
            for tv, pred in zip(true_values, predictions)
        ]
    )
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
    with open(error_file_path, "a") as f:
        f.write(f"Event: {test_event_name}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("="*30 + "\n")
    # 可视化预测结果
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", color="blue")
    plt.plot(predictions, label="Predictions", color="orange")
    plt.xlabel("Time Steps")
    plt.ylabel("Malicious Comments Count")
    plt.legend()
    # 保存图像
    save_path = os.path.join(output_dir, f"{test_event_name}_prediction.png")
    plt.savefig(save_path)  # 保存图像到指定路径
    print(f"Prediction plot saved to {save_path}")
all_predictions_df = pd.DataFrame(all_predictions)
all_predictions_csv_path = os.path.join(output_dir, "all_predictions.csv")
all_predictions_df.to_csv(all_predictions_csv_path, index=False, encoding='gbk')
print(f"All predictions saved to {all_predictions_csv_path}")

# 打印平均误差指标
print(f"Average MAE: {sum(mae_list)/len(mae_list):.4f}")
print(f"Average MSE: {sum(mse_list)/len(mse_list):.4f}")
print(f"Average RMSE: {sum(rmse_list)/len(rmse_list):.4f}")
