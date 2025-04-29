import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import time

# -----------------------------
# 自定义数据集：WindPowerDataset
# （增加 unit_size 参数，确保滑动窗口采样时不跨单元）
# -----------------------------
class WindPowerDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, unit_size=None):
        """
        data: numpy数组，形状 (总时刻数, 特征数)
        seq_len: 输入序列长度
        pred_len: 预测时步数
        unit_size: 若非 None，则要求滑动窗口完全落在同一单元内（单位大小为 unit_size 行）
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unit_size = unit_size
        self.samples = []
        self.indices = []
        # 使用滑动窗口生成样本，每个样本返回 (X, y, idx)
        for i in range(len(data) - seq_len - pred_len + 1):
            if unit_size is not None:
                start_unit = i // unit_size
                end_unit = (i + seq_len + pred_len - 1) // unit_size
                if start_unit != end_unit:
                    # 如果窗口跨越了不同单元，则舍弃该样本
                    continue
            X = data[i:i + seq_len]
            y = data[i + seq_len:i + seq_len + pred_len]
            self.samples.append((X, y))
            # 记录预测窗口在原始数据中的起始位置
            self.indices.append(i + seq_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), self.indices[idx]

# -----------------------------
# 定义领域专属输入层
# -----------------------------
class SourceInputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SourceInputLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return torch.relu(self.fc(x))

class TargetInputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TargetInputLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return torch.relu(self.fc(x))

# -----------------------------
# 定义 TCN 基本单元
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        # 为保持输出序列长度不变，padding 设置为 (kernel_size - 1)*dilation
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              dilation=dilation, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

# -----------------------------
# 定义共享 TCN + 自注意力模块（替代原 LSTM 共享部分）
# -----------------------------
class SharedTCNAttention(nn.Module):
    def __init__(self, in_channels):
        """
        in_channels: 输入通道数，应与输入层映射后的维度一致（例如 32）
        """
        super(SharedTCNAttention, self).__init__()
        # -- 第一层 --
        self.tcn1 = TCNBlock(in_channels, 128, kernel_size=3, dilation=1)
        self.mha1 = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(128)
        # -- 第二层 --
        self.tcn2 = TCNBlock(128, 64, kernel_size=3, dilation=1)
        self.mha2 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(64)
        # -- 第三层 --
        self.tcn3 = TCNBlock(64, 32, kernel_size=3, dilation=4)
        self.mha3 = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(32)

    def forward(self, x):
        # 输入 x: (B, seq_len, in_channels)
        # 第一层：先转置为 (B, in_channels, seq_len)
        x_t = x.transpose(1, 2)               # (B, in_channels, seq_len)
        out1 = self.tcn1(x_t)                 # (B, 128, seq_len)
        out1 = out1.transpose(1, 2)           # (B, seq_len, 128)
        attn1, _ = self.mha1(out1, out1, out1)  # (B, seq_len, 128)
        out1 = self.norm1(out1 + attn1)         # 残差连接 + LayerNorm

        # 第二层
        out1_t = out1.transpose(1, 2)         # (B, 128, seq_len)
        out2 = self.tcn2(out1_t)              # (B, 64, seq_len)
        out2 = out2.transpose(1, 2)           # (B, seq_len, 64)
        attn2, _ = self.mha2(out2, out2, out2)  # (B, seq_len, 64)
        out2 = self.norm2(out2 + attn2)

        # 第三层
        out2_t = out2.transpose(1, 2)         # (B, 64, seq_len)
        out3 = self.tcn3(out2_t)              # (B, 32, seq_len)
        out3 = out3.transpose(1, 2)           # (B, seq_len, 32)
        attn3, _ = self.mha3(out3, out3, out3)  # (B, seq_len, 32)
        out3 = self.norm3(out3 + attn3)

        # 全局平均池化：对时间维度求均值，获得全局特征 (B, 32)
        pooled = out3.mean(dim=1)
        return pooled

# -----------------------------
# 定义目标域模型（多步预测）
# -----------------------------
class TargetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_len):
        """
        input_dim: 目标域原始输入维度（例如 6）
        hidden_dim: 输入层映射后的维度（保持与源域一致，如 32）
        pred_len: 预测时步数（例如 8）
        """
        super(TargetModel, self).__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.input_layer = TargetInputLayer(input_dim, hidden_dim)
        self.shared_tcn = SharedTCNAttention(hidden_dim)
        self.output_layer = nn.Linear(32, pred_len * input_dim)

    def forward(self, x):
        B, seq_len, _ = x.shape
        x = x.view(-1, x.shape[-1])
        x = self.input_layer(x)
        x = x.view(B, seq_len, -1)
        features = self.shared_tcn(x)
        out = self.output_layer(features)
        out = out.view(B, self.pred_len, self.input_dim)
        return out

# -----------------------------
# 训练和验证函数（基于 DataLoader）
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X, y, _ in dataloader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y, _ in dataloader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(dataloader.dataset)

# -----------------------------
# 测试函数：计算反归一化后的指标
# -----------------------------
def test_model(model, dataloader, scaler, device):
    model.eval()
    preds_all = []
    truths_all = []
    indices_all = []
    with torch.no_grad():
        for X, y, idx in dataloader:
            X = X.to(device)
            outputs = model(X)
            preds_all.append(outputs.cpu().numpy())
            truths_all.append(y.cpu().numpy())
            indices_all.extend(idx)
    preds_all = np.concatenate(preds_all, axis=0)  # (样本数, pred_len, feature_dim)
    truths_all = np.concatenate(truths_all, axis=0)
    shape = preds_all.shape
    preds_flat = preds_all.reshape(-1, shape[-1])
    truths_flat = truths_all.reshape(-1, shape[-1])
    preds_inv = scaler.inverse_transform(preds_flat).reshape(shape)
    truths_inv = scaler.inverse_transform(truths_flat).reshape(shape)
    mse = mean_squared_error(truths_inv.flatten(), preds_inv.flatten())
    mae = mean_absolute_error(truths_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(truths_inv.flatten(), preds_inv.flatten())
    print(f"Test Metrics - R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    return preds_inv, truths_inv, indices_all

# -----------------------------
# 主程序
# -----------------------------
def main():
    start_time = time.time()  # 记录开始时间
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 24   # 输入序列长度
    pred_len = 4   # 预测时步数

    # ========== 目标域数据处理（以 216 行为单元） ==========
    df_target = pd.read_excel('../nignxia_6_windfarms.xlsx', index_col=0)
    # 选取每隔4行的样本，并取前2160行数据
    df_target = df_target.iloc[0:-1:4, :].iloc[0:4320, :]
    time_index_full = df_target.index  # 完整时间索引

    # 转换为 numpy 数组，取前6个特征
    dataset_2 = df_target.to_numpy()
    target_data = dataset_2[:, :6]

    scalers_target = []
    target_data_scaled = np.zeros_like(target_data)
    for i in range(target_data.shape[1]):
        sc = StandardScaler()
        col_scaled = sc.fit_transform(target_data[:, i].reshape(-1, 1))
        scalers_target.append(sc)
        target_data_scaled[:, i] = col_scaled.flatten()

    # 以 216 行作为一个单元；不足一个单元的末尾数据舍弃
    unit_size = 36
    num_full_units = len(df_target) // unit_size
    num_rows_to_use = num_full_units * unit_size
    df_target_used = df_target.iloc[:num_rows_to_use, :]
    time_index_used = df_target_used.index

    # 重塑为 (单元数, unit_size, 特征数) 及对应时间索引 (单元数, unit_size)
    data_units = target_data_scaled.reshape(num_full_units, unit_size, -1)
    time_units = np.array(time_index_used).reshape(num_full_units, unit_size)

    # 随机打乱单元，并按 70%/20%/10% 划分
    indices = np.arange(num_full_units)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_count = int(0.7 * num_full_units)
    val_count = int(0.2 * num_full_units)
    test_count = num_full_units - train_count - val_count
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    train_units = data_units[train_indices]
    val_units = data_units[val_indices]
    test_units = data_units[test_indices]

    time_train_units = time_units[train_indices]
    time_val_units = time_units[val_indices]
    time_test_units = time_units[test_indices]

    # 将各集合内的单元重塑为二维数据（样本数, 特征数），时间索引同步展开
    train_data = train_units.reshape(-1, target_data.shape[1])
    val_data = val_units.reshape(-1, target_data.shape[1])
    test_data = test_units.reshape(-1, target_data.shape[1])

    time_index_train = time_train_units.reshape(-1)
    time_index_val = time_val_units.reshape(-1)
    time_index_test = time_test_units.reshape(-1)

    print("目标域训练数据 shape:", train_data.shape)
    print("目标域验证数据 shape:", val_data.shape)
    print("目标域测试数据 shape:", test_data.shape)

    # 构造数据集时传入 unit_size 参数，确保滑动窗口采样时不跨单元边界
    train_dataset = WindPowerDataset(train_data, seq_len, pred_len, unit_size=unit_size)
    val_dataset = WindPowerDataset(val_data, seq_len, pred_len, unit_size=unit_size)
    test_dataset = WindPowerDataset(test_data, seq_len, pred_len, unit_size=unit_size)

    target_loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    target_loader_val = DataLoader(val_dataset, batch_size=32, shuffle=False)
    target_loader_test = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ========== 目标域模型训练 ==========
    target_model = TargetModel(input_dim=6, hidden_dim=32, pred_len=pred_len).to(device)
    optimizer_target = torch.optim.Adam(target_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    epochs = 500

    print("Training target domain model...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(target_model, target_loader_train, optimizer_target, criterion, device)
        if (epoch + 1) % 20 == 0:
            val_loss = evaluate_model_epoch(target_model, target_loader_val, criterion, device)
            print(f"Target Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    # ========== 目标域模型测试 ==========
    print("Evaluating target domain model on test set...")
    # 这里使用 scalers_target[0] 对第1个特征进行反归一化（可根据需要分别处理各特征）
    preds_inv, truths_inv, indices_test = test_model(target_model, target_loader_test, scalers_target[0], device)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time
    print(f"程序总运行时间: {elapsed_time:.2f} 秒")
    # -----------------------------
    # 保存目标域预测结果及对应时间（以第1个风电场为例）
    # -----------------------------
    time_list = []
    original_list = []
    prediction_list = []
    # test_dataset 中每个样本的索引为相对于 test_data 的起始位置，对应原始时间为 time_index_test[idx + k]
    for i in range(len(preds_inv)):
        idx_sample = test_dataset.indices[i]  # 预测窗口在 test_data 中的起始索引
        for k in range(pred_len):
            abs_idx = idx_sample + k
            time_list.append(time_index_test[abs_idx])
            original_list.append(truths_inv[i, k, 0])
            prediction_list.append(preds_inv[i, k, 0])

    df_output = pd.DataFrame({
        "时间": time_list,
        "原始值": original_list,
        "预测值": prediction_list
    })
    df_output = df_output.drop_duplicates(subset=["时间"], keep="last")
    output_path = "24tcn+注意力4时间测试.csv"
    df_output.to_csv(output_path, index=False)
    print(f"目标域预测结果及对应时间已保存至 {output_path}")

if __name__ == "__main__":
    main()
