import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# matplotlib 配置
# -----------------------------
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20


# -----------------------------
# 自定义数据集类（和之前相同，确保滑动窗口不跨单元）
# -----------------------------
class WindPowerDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, unit_size=None):
        """
        data: (T, feature_dim) 的数据
        seq_len: 使用过去多少个时刻作为输入
        pred_len: 预测未来多少个时刻
        unit_size: 若非 None，则要求滑动窗完全落在同一单元内（单位大小为 unit_size 行）
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unit_size = unit_size

        # 预先计算所有合法的起始索引，确保采样窗口 [i, i+seq_len+pred_len) 完全落在同一单元内
        self.valid_indices = []
        total_len = len(data)
        for i in range(total_len - seq_len - pred_len + 1):
            if unit_size is not None:
                start_unit = i // unit_size
                end_unit = (i + seq_len + pred_len - 1) // unit_size
                if start_unit != end_unit:
                    # 窗口跨越不同单元，舍弃该采样
                    continue
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.data[real_idx: real_idx + self.seq_len]
        y = self.data[real_idx + self.seq_len: real_idx + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),  # (seq_len, feature_dim)
            torch.tensor(y, dtype=torch.float32),  # (pred_len, feature_dim)
            real_idx  # 用于后续对应时间
        )


# -----------------------------
# 定义 LSTM 多步预测模型
# -----------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_dim)
        返回: (batch_size, pred_len)
        """
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_dim)
        out = out[:, -1, :]  # 取最后时刻的输出
        out = self.fc(out)  # (batch_size, pred_len)
        return out


# -----------------------------
# LSTM 训练函数
# -----------------------------
def train_lstm(model, train_loader, val_loader, device, num_epochs=50, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for x, y, _ in train_loader:
            x = x.to(device)  # (batch_size, seq_len, feature_dim)
            # y: (batch_size, pred_len, feature_dim)，只关心第 0 风电场 => shape (batch_size, pred_len)
            y = y[..., 0].to(device)  # (batch_size, pred_len)

            pred = model(x)  # (batch_size, pred_len)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)

        # 验证集损失
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, _ in val_loader:
                x_val = x_val.to(device)
                y_val = y_val[..., 0].to(device)
                pred_val = model(x_val)
                val_loss = criterion(pred_val, y_val)
                epoch_val_loss += val_loss.item()

        epoch_val_loss /= len(val_loader)

        train_loss_list.append(epoch_train_loss)
        val_loss_list.append(epoch_val_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    return train_loss_list, val_loss_list


# -----------------------------
# 评估函数 (打印 RMSE, MAE, R2)
# -----------------------------
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            y = y[..., 0].to(device)  # (batch_size, pred_len)
            pred = model(x)  # (batch_size, pred_len)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    return rmse, mae, r2


# -----------------------------
# 绘制 预测 vs 实际 (仅第0风电场)
# -----------------------------
def plot_predictions_vs_actuals(
        model, test_loader, device,
        scalers_target, max_values=None, min_values=None
):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            # y: (batch_size, pred_len, feature_dim) => 取第0列
            y = y[..., 0].cpu().numpy()  # (batch_size, pred_len)
            pred = model(x).cpu().numpy()  # (batch_size, pred_len)

            # 对每条样本做反标准化
            for i in range(len(pred)):
                # (pred_len,) -> (pred_len,1)
                p_inv = scalers_target[0].inverse_transform(pred[i].reshape(-1, 1))
                l_inv = scalers_target[0].inverse_transform(y[i].reshape(-1, 1))
                if max_values is not None and min_values is not None:
                    p_inv = np.clip(p_inv, np.array(min_values).min(), np.array(max_values).max())
                all_preds.extend(p_inv.flatten())
                all_labels.extend(l_inv.flatten())

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f"Predict Result - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 绘制前 60 个时刻
    plt.figure(figsize=(12, 5))
    plt.plot(all_labels[:60], label='actually', linestyle='-', linewidth=3, marker='o', markersize=8)
    plt.plot(all_preds[:60], label='prediction', linestyle='--', linewidth=3, marker='x', markersize=8)
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Power (MW)", fontsize=20)
    plt.title(f"Prediction vs Actually Value (R2={r2:.2f})", fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 保存目标域预测结果 (仅第0风电场) + 时间
# -----------------------------
def save_target_predictions_with_time(
        model, test_loader, device,
        scalers_target, time_index, seq_len, pred_len
):
    model.eval()
    results = {}

    with torch.no_grad():
        for x, y, indices in test_loader:
            x = x.to(device)
            # y: (batch_size, pred_len, feature_dim) => (batch_size, pred_len) 只取第0列
            y_true = y[..., 0].numpy()
            y_pred = model(x).cpu().numpy()

            for i in range(len(indices)):
                idx = indices[i].item()
                for k in range(pred_len):
                    t = time_index[idx + seq_len + k]
                    true_val = y_true[i, k]
                    pred_val = y_pred[i, k]
                    # 反标准化
                    true_val_inv = scalers_target[0].inverse_transform([[true_val]])[0, 0]
                    pred_val_inv = scalers_target[0].inverse_transform([[pred_val]])[0, 0]
                    error = abs(true_val_inv - pred_val_inv)
                    # 如果同一时刻有多条预测，选误差更小的
                    if t not in results or error < results[t][0]:
                        results[t] = (error, true_val_inv, pred_val_inv)

    sorted_times = sorted(results.keys())
    time_list = []
    original_list = []
    prediction_list = []
    for t in sorted_times:
        time_list.append(t)
        original_list.append(results[t][1])
        prediction_list.append(results[t][2])

    df_out = pd.DataFrame({"time": time_list, "acc": original_list, "dann": prediction_list})
    df_out.to_csv("24lstm12核心.csv", index=False)
    print("目标域预测结果及对应时间已保存至 24lstm12核心.csv")


# -----------------------------
# 主程序：仅使用目标域数据训练 LSTM
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = 24  # 过去 24 个时刻
    pred_len = 12  # 预测未来 8 个时刻
    unit_size = 36  # 每个单元大小
    num_epochs = 500
    batch_size = 32

    # ========== 读取目标域数据 ==========
    df_target = pd.read_excel('../nignxia_6_windfarms.xlsx', index_col=0)
    # 例如，每隔4行取一行，并取前4320行数据
    df_target = df_target.iloc[0:-1:4, :].iloc[0:4320, :]
    time_index_full = df_target.index

    # 以36行为一个单元（不足单元的末尾数据将舍弃）
    num_full_units = len(df_target) // unit_size
    num_rows_to_use = num_full_units * unit_size
    df_target_used = df_target.iloc[:num_rows_to_use, :]
    time_index_used = df_target_used.index

    # 转换为 numpy，并取前6列
    dataset_2 = df_target_used.to_numpy()
    target_data = dataset_2[:, :6]  # shape: (N, 6)

    # 标准化
    scalers_target = []
    target_data_scaled = np.zeros_like(target_data)
    for i in range(target_data.shape[1]):
        sc = StandardScaler()
        col_scaled = sc.fit_transform(target_data[:, i].reshape(-1, 1))
        scalers_target.append(sc)
        target_data_scaled[:, i] = col_scaled.flatten()

    # 重塑成 (num_full_units, unit_size, feature_dim)
    data_units = target_data_scaled.reshape(num_full_units, unit_size, -1)
    time_units = np.array(time_index_used).reshape(num_full_units, unit_size)

    # 随机打乱单元
    indices = np.arange(num_full_units)
    np.random.seed(42)
    np.random.shuffle(indices)

    # 按 70% / 20% / 10% 划分训练、验证、测试
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

    # 将各集合展开为二维
    train_data = train_units.reshape(-1, target_data.shape[1])
    val_data = val_units.reshape(-1, target_data.shape[1])
    test_data = test_units.reshape(-1, target_data.shape[1])

    time_index_train = time_train_units.reshape(-1)
    time_index_val = time_val_units.reshape(-1)
    time_index_test = time_test_units.reshape(-1)

    print("目标域训练数据 shape:", train_data.shape)
    print("目标域验证数据 shape:", val_data.shape)
    print("目标域测试数据 shape:", test_data.shape)

    # 构建 Dataset / Dataloader
    train_dataset = WindPowerDataset(train_data, seq_len, pred_len, unit_size=unit_size)
    val_dataset = WindPowerDataset(val_data, seq_len, pred_len, unit_size=unit_size)
    test_dataset = WindPowerDataset(test_data, seq_len, pred_len, unit_size=unit_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 定义并训练 LSTM ==========
    print("Training LSTM model on target domain data...")
    input_dim = 6  # 输入特征维度
    hidden_dim = 64  # LSTM隐藏层大小，可自行调整
    num_layers = 2  # LSTM层数，可自行调整

    model = LSTMPredictor(input_dim, hidden_dim, num_layers, pred_len).to(device)
    train_lstm(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-3)

    # ========== 在验证集上评估，可自行调参 ==========
    rmse_val, mae_val, r2_val = evaluate_model(model, val_loader, device)
    print(f"Validation - RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}")

    # ========== 在测试集上最终评估 ==========
    print("Evaluating model on target test set...")
    rmse_tgt, mae_tgt, r2_tgt = evaluate_model(model, test_loader, device)
    print(f"Test Set - RMSE: {rmse_tgt:.4f}, MAE: {mae_tgt:.4f}, R2: {r2_tgt:.4f}")

    # ========== 绘制预测 vs 实际 (仅第0风电场) ==========
    min_values = np.min(dataset_2[:, 0:1], axis=0)
    max_values = np.max(dataset_2[:, 0:1], axis=0)
    print("Plotting predictions vs actual for target domain (first wind farm) using test set...")
    plot_predictions_vs_actuals(
        model, test_loader, device,
        scalers_target, max_values=max_values, min_values=min_values
    )

    # ========== 保存目标域预测结果及时间 (仅第0风电场) ==========
    save_target_predictions_with_time(
        model, test_loader, device,
        scalers_target, time_index_test,
        seq_len, pred_len
    )


if __name__ == "__main__":
    main()
