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
from torch.cuda.amp import GradScaler,autocast
# 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

# 定义保存结果图的文件夹
output_dir = r"./Informer"
os.makedirs(output_dir, exist_ok=True)  # 创建文件夹，如果不存在
error_file_path = os.path.join(output_dir, "error_metrics.txt")

# 使用 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Informer 模型架构定义
class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor  # 控制稀疏度
        self.scale = 1 / math.sqrt(self.d_head)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K, sample_k, n_top):
        # Q, K shape: [batch, heads, length, d_head]
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # 采样部分K点进行近似
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (sample_k,), device=Q.device)
        K_sample = K_expand[:, :, :, index_sample, :]

        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)  # [B,H,L_Q,sample_k]

        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)  # 重要性度量
        M_top = torch.topk(M, n_top, dim=-1)[1]  # 选出重要的 query 索引
        return M_top

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.shape
        H = self.n_heads
        d_head = self.d_head

        Q = self.q_linear(Q).view(B, L, H, d_head).transpose(1,2)  # [B,H,L,d_head]
        K = self.k_linear(K).view(B, -1, H, d_head).transpose(1,2)  # [B,H,L_K,d_head]
        V = self.v_linear(V).view(B, -1, H, d_head).transpose(1,2)  # [B,H,L_V,d_head]

        # ProbSparse Attention 关键步骤
        u = int(math.ceil(math.log(L)))  # sample_k = c*log(L)
        u = max(u * self.factor, 1)
        u = int(u)
        n_top = u  # 选取前n_top个 query
        M_top = self._prob_QK(Q, K, sample_k=u, n_top=n_top)  # [B,H,n_top]

        # 只计算 top query 对应的 attention
        Q_reduce = torch.gather(Q, 2, M_top.unsqueeze(-1).expand(-1, -1, -1, d_head))  # [B,H,n_top,d_head]
        scores = torch.matmul(Q_reduce, K.transpose(-2,-1)) * self.scale  # [B,H,n_top,L_K]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # [B,H,n_top,L_K]
        context = torch.matmul(attn, V)  # [B,H,n_top,d_head]

        # 将 sparse 计算结果重新填充回全部 query 位置（简化版中仅返回 top query的结果）
        # 实际完整实现会有补全机制，这里为了示例略去

        # 这里先做个简单的聚合
        context_mean = context.mean(dim=2)  # [B,H,d_head]

        context_mean = context_mean.transpose(1,2).contiguous().view(B, -1)  # [B, H*d_head]
        out = self.out(context_mean.unsqueeze(1))  # [B,1,d_model]

        return out.squeeze(1)  # [B, d_model]

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Informer(nn.Module):
    def __init__(self, input_size=1, d_model=64, n_heads=8, num_layers=2, dim_feedforward=256, dropout=0.1, seq_len=24, pred_len=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def forward(self, x):
        # x shape: [B, seq_len, input_size]
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        out = self.proj(x[:, -self.pred_len:, :]).squeeze(-1)  # 预测最后 pred_len 个时间步
        return out

# 定义滑动窗口函数
time_steps = 6  # 使用过去6小时的数据预测未来的趋势
def create_sequences(data, time_steps=6):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 初始化模型
model = Informer(input_size=1, d_model=64, n_heads=8, num_layers=2, dim_feedforward=256, dropout=0.1, seq_len=24, pred_len=1).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练事件和文件夹路径
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

# 训练 Informer 模型
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

    # 使用混合精度训练
    scaler_ = GradScaler()
    
    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 混合精度训练
            with autocast():
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
            
            scaler_.scale(loss).backward()
            scaler_.step(optimizer)
            scaler_.update()
            
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
