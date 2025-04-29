import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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
# 自定义数据集类（防止滑动窗口跨单元）
# -----------------------------
class WindPowerDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, unit_size=None):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.unit_size = unit_size

        self.valid_indices = []
        total_len = len(data)
        for i in range(total_len - seq_len - pred_len + 1):
            if unit_size is not None:
                start_unit = i // unit_size
                end_unit = (i + seq_len + pred_len - 1) // unit_size
                if start_unit != end_unit:
                    continue
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.data[real_idx : real_idx + self.seq_len]
        y = self.data[real_idx + self.seq_len : real_idx + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            real_idx
        )


# -----------------------------
# 源域输入层
# -----------------------------
class SourceInputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SourceInputLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# 目标域输入层
# -----------------------------
class TargetInputLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TargetInputLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# TCN Block
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               dilation=dilation, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 如果输入通道不等于输出通道，则用1x1卷积
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


# -----------------------------
# 特征提取器 (TCN + MultiheadAttention)
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, proj_dim=128, kernel_size=3, dropout=0.2):
        super(FeatureExtractor, self).__init__()
        # 投影层
        self.proj = nn.Linear(input_dim, proj_dim)

        # Block1: TCN+MHA
        self.tcn1 = TCNBlock(proj_dim, 128, kernel_size=kernel_size, dilation=1, dropout=dropout)
        self.mha1 = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(128)

        # Block2: TCN+MHA
        self.tcn2 = TCNBlock(128, 64, kernel_size=kernel_size, dilation=2, dropout=dropout)
        self.mha2 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(64)

        # Block3: TCN+MHA
        self.tcn3 = TCNBlock(64, 32, kernel_size=kernel_size, dilation=4, dropout=dropout)
        self.mha3 = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(32)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        bsz, seq_len, in_dim = x.shape
        x = x.view(bsz*seq_len, in_dim)
        x = self.proj(x)
        x = x.view(bsz, seq_len, -1)

        x = x.permute(0, 2, 1)  # (B, C, seq_len)
        out1 = self.tcn1(x)     # (B, 128, seq_len)
        out1 = out1.permute(0, 2, 1)  # (B, seq_len, 128)
        attn1, _ = self.mha1(out1, out1, out1)
        out1 = self.norm1(out1 + attn1)

        out2 = out1.permute(0, 2, 1)  # (B, 128, seq_len)
        out2 = self.tcn2(out2)        # (B, 64, seq_len)
        out2 = out2.permute(0, 2, 1)  # (B, seq_len, 64)
        attn2, _ = self.mha2(out2, out2, out2)
        out2 = self.norm2(out2 + attn2)

        out3 = out2.permute(0, 2, 1)  # (B, 64, seq_len)
        out3 = self.tcn3(out3)        # (B, 32, seq_len)
        out3 = out3.permute(0, 2, 1)  # (B, seq_len, 32)
        attn3, _ = self.mha3(out3, out3, out3)
        out3 = self.norm3(out3 + attn3)

        return out3[:, -1, :]  # (B, 32)


# -----------------------------
# 回归器
# -----------------------------
class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


# -----------------------------
# 源域训练 (不带 block-mask)
# -----------------------------
def train_source_model(
    source_loader, source_loader_val,
    source_input_layer, feature_extractor, source_regressor,
    device, pred_len, lr=1e-3, num_epochs=50
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(source_input_layer.parameters()) +
        list(feature_extractor.parameters()) +
        list(source_regressor.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        source_input_layer.train()
        feature_extractor.train()
        source_regressor.train()

        total_loss = 0.0
        for x, y, _ in source_loader:
            x, y = x.to(device), y.to(device)
            x_in = source_input_layer(x)            # (batch, seq_len, 32)
            feat = feature_extractor(x_in)          # (batch, 32)
            pred = source_regressor(feat).view(y.size(0), pred_len, -1)  # (batch, pred_len, 30)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(source_loader)

        # 验证
        source_input_layer.eval()
        feature_extractor.eval()
        source_regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, _ in source_loader_val:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_in_val = source_input_layer(x_val)
                feat_val = feature_extractor(x_in_val)
                pred_val = source_regressor(feat_val).view(y_val.size(0), pred_len, -1)
                loss_val = criterion(pred_val, y_val)
                val_loss += loss_val.item()
        val_loss = val_loss / len(source_loader_val)

        print(f"[Source Training] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 保存 feature_extractor
    torch.save(feature_extractor.state_dict(), "feature_extractor_src.pth")
    print("FeatureExtractor weights saved to feature_extractor_src.pth")


# -----------------------------
# 目标域训练
# -----------------------------
def train_target_model(
    train_loader, val_loader, test_loader,
    target_input_layer, feature_extractor, target_regressor,
    device, pred_len, lr=1e-3, num_epochs=50
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        list(target_input_layer.parameters()) +
        list(feature_extractor.parameters()) +
        list(target_regressor.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        target_input_layer.train()
        feature_extractor.train()
        target_regressor.train()

        total_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            # y: (batch, pred_len, 1) => 只关注第0列
            x_in = target_input_layer(x)
            feat = feature_extractor(x_in)
            pred = target_regressor(feat).view(y.size(0), pred_len, -1)
            loss = criterion(pred, y[..., 0:1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # 验证
        target_input_layer.eval()
        feature_extractor.eval()
        target_regressor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val, _ in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                x_in_val = target_input_layer(x_val)
                feat_val = feature_extractor(x_in_val)
                pred_val = target_regressor(feat_val).view(y_val.size(0), pred_len, -1)
                loss_val = criterion(pred_val, y_val[..., 0:1])
                val_loss += loss_val.item()
        val_loss = val_loss / len(val_loader)

        print(f"[Target Training] Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 测试集评估
    evaluate_model(test_loader, target_input_layer, feature_extractor, target_regressor, device)


# -----------------------------
# 评估函数（打印 RMSE, MAE, R2）
# -----------------------------
def evaluate_model(test_loader, input_layer, feature_extractor, regressor, device):
    input_layer.eval()
    feature_extractor.eval()
    regressor.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            x_in = input_layer(x)
            feat = feature_extractor(x_in)
            pred = regressor(feat).view(y.size(0), -1)
            # y只关心第0列
            label = y[..., 0].view(y.size(0), -1)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    print(f"Test Set - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


# -----------------------------
# (可选) 测试集预测值可视化/保存
# -----------------------------
def plot_predictions_vs_actuals(
    loader,
    input_layer,
    feature_extractor,
    regressor,
    device,
    scalers_target,
    seq_len,
    pred_len,
    time_index,
    max_values=None,
    min_values=None
):
    input_layer.eval()
    feature_extractor.eval()
    regressor.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y, indices in loader:
            x = x.to(device)
            y = y[..., 0].to(device)  # 只关心第0列
            x_in = input_layer(x)
            feat = feature_extractor(x_in)
            pred = regressor(feat).view(y.size(0), -1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 反标准化并展开
    preds_flat = []
    labels_flat = []
    for i in range(len(all_preds)):
        p = all_preds[i].reshape(-1, 1)
        l = all_labels[i].reshape(-1, 1)
        p_inv = scalers_target[0].inverse_transform(p)
        l_inv = scalers_target[0].inverse_transform(l)
        if max_values is not None and min_values is not None:
            p_inv = np.clip(p_inv, min_values[0], max_values[0])
        preds_flat.extend(p_inv.flatten())
        labels_flat.extend(l_inv.flatten())

    preds_flat = np.array(preds_flat)
    labels_flat = np.array(labels_flat)

    rmse = np.sqrt(mean_squared_error(labels_flat, preds_flat))
    mae = mean_absolute_error(labels_flat, preds_flat)
    r2 = r2_score(labels_flat, preds_flat)

    print(f"Final Test Predict - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 只画前60个点
    plt.figure(figsize=(12, 5))
    plt.plot(labels_flat[:60], label='actually', linestyle='-', linewidth=3, marker='o', markersize=8)
    plt.plot(preds_flat[:60], label='prediction', linestyle='--', linewidth=3, marker='x', markersize=8)
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Power (MW)", fontsize=20)
    plt.title(f"Prediction vs Actually Value (R2={r2:.2f})", fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def save_target_predictions_with_time(
    loader,
    input_layer,
    feature_extractor,
    regressor,
    device,
    scalers_target,
    time_index,
    seq_len,
    pred_len,
    filename="24普通迁移1核心.csv"
):
    input_layer.eval()
    feature_extractor.eval()
    regressor.eval()
    results = {}

    with torch.no_grad():
        for x, y, indices in loader:
            x = x.to(device)
            y = y[..., 0].cpu().numpy()  # (batch, pred_len)
            x_in = input_layer(x)
            feat = feature_extractor(x_in)
            pred = regressor(feat).cpu().numpy()  # (batch, pred_len)

            for i in range(len(indices)):
                idx = indices[i].item()
                for k in range(pred_len):
                    t = time_index[idx + seq_len + k]
                    true_val = y[i, k]
                    pred_val = pred[i, k]
                    true_val_inv = scalers_target[0].inverse_transform([[true_val]])[0, 0]
                    pred_val_inv = scalers_target[0].inverse_transform([[pred_val]])[0, 0]
                    error = abs(true_val_inv - pred_val_inv)
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
    df_out.to_csv(filename, index=False)
    print(f"目标域预测结果及对应时间已保存至 {filename}")


# -----------------------------
# 主函数：完整演示
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 24
    pred_len = 1
    unit_size = 36

    # ========== 源域数据处理 ==========
    source_df = pd.read_csv('../All_Power_Data.csv', index_col=0)
    source_df = source_df.iloc[0:300000, :]
    source_data = source_df.to_numpy()  # (T,30)，假设源域有30个风电场

    # 标准化
    scalers_source = []
    source_data_scaled = np.zeros_like(source_data)
    for i in range(source_data.shape[1]):
        sc = StandardScaler()
        col_scaled = sc.fit_transform(source_data[:, i].reshape(-1, 1))
        scalers_source.append(sc)
        source_data_scaled[:, i] = col_scaled.flatten()

    num_full_units_source = len(source_data_scaled) // unit_size
    use_len_source = num_full_units_source * unit_size
    source_data_used = source_data_scaled[:use_len_source, :]
    data_units_source = source_data_used.reshape(num_full_units_source, unit_size, -1)

    np.random.seed(42)
    indices_src = np.arange(num_full_units_source)
    np.random.shuffle(indices_src)
    train_count_src = int(0.7 * num_full_units_source)
    val_count_src = num_full_units_source - train_count_src

    train_units_src = data_units_source[indices_src[:train_count_src]]
    val_units_src   = data_units_source[indices_src[train_count_src:]]

    source_train_data = train_units_src.reshape(-1, source_data.shape[1])
    source_val_data   = val_units_src.reshape(-1, source_data.shape[1])

    ds_source_train = WindPowerDataset(source_train_data, seq_len, pred_len, unit_size)
    ds_source_val   = WindPowerDataset(source_val_data,   seq_len, pred_len, unit_size)
    loader_source   = DataLoader(ds_source_train, batch_size=32, shuffle=True)
    loader_source_val= DataLoader(ds_source_val,  batch_size=32, shuffle=False)

    # ========== 目标域数据处理 ==========
    df_target = pd.read_excel('../nignxia_6_windfarms.xlsx', index_col=0)
    df_target = df_target.iloc[0:-1:4, :].iloc[0:4320, :]
    target_data_full = df_target.to_numpy()[:, :6]  # (T,6)

    scalers_target = []
    target_data_scaled = np.zeros_like(target_data_full)
    for i in range(target_data_full.shape[1]):
        sc = StandardScaler()
        col_scaled = sc.fit_transform(target_data_full[:, i].reshape(-1, 1))
        scalers_target.append(sc)
        target_data_scaled[:, i] = col_scaled.flatten()

    num_full_units_tgt = len(target_data_scaled) // unit_size
    use_len_tgt = num_full_units_tgt * unit_size
    tgt_data_used = target_data_scaled[:use_len_tgt, :]
    data_units_tgt = tgt_data_used.reshape(num_full_units_tgt, unit_size, -1)

    np.random.seed(42)
    indices_tgt = np.arange(num_full_units_tgt)
    np.random.shuffle(indices_tgt)
    train_cnt_tgt = int(0.7 * num_full_units_tgt)
    val_cnt_tgt = int(0.2 * num_full_units_tgt)
    test_cnt_tgt = num_full_units_tgt - train_cnt_tgt - val_cnt_tgt

    train_units_tgt = data_units_tgt[indices_tgt[:train_cnt_tgt]]
    val_units_tgt   = data_units_tgt[indices_tgt[train_cnt_tgt:train_cnt_tgt+val_cnt_tgt]]
    test_units_tgt  = data_units_tgt[indices_tgt[train_cnt_tgt+val_cnt_tgt:]]

    train_data_tgt = train_units_tgt.reshape(-1, target_data_full.shape[1])
    val_data_tgt   = val_units_tgt.reshape(-1, target_data_full.shape[1])
    test_data_tgt  = test_units_tgt.reshape(-1, target_data_full.shape[1])

    ds_tgt_train = WindPowerDataset(train_data_tgt, seq_len, pred_len, unit_size)
    ds_tgt_val   = WindPowerDataset(val_data_tgt,   seq_len, pred_len, unit_size)
    ds_tgt_test  = WindPowerDataset(test_data_tgt,  seq_len, pred_len, unit_size)

    loader_tgt_train = DataLoader(ds_tgt_train, batch_size=32, shuffle=True)
    loader_tgt_val   = DataLoader(ds_tgt_val,   batch_size=32, shuffle=False)
    loader_tgt_test  = DataLoader(ds_tgt_test,  batch_size=32, shuffle=False)

    # 生成对应的时间索引 (示例)
    # 假设你有 df_target.index => time_index_full
    # 这里把拆分后对应到 test_data_tgt  => time_index_test
    time_units_tgt = np.array(df_target.index[:use_len_tgt]).reshape(num_full_units_tgt, unit_size)
    time_test_units = time_units_tgt[indices_tgt[train_cnt_tgt+val_cnt_tgt:]]
    time_index_test = time_test_units.reshape(-1)

    # ========== 源域模型 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_input_layer = SourceInputLayer(30, 32).to(device)
    feature_extractor = FeatureExtractor(input_dim=32, proj_dim=128, kernel_size=3, dropout=0.2).to(device)
    source_regressor = Regressor(input_dim=32, output_dim=pred_len * 30).to(device)

    print("===== (A) Train on Source Domain =====")
    train_source_model(
        loader_source, loader_source_val,
        source_input_layer, feature_extractor, source_regressor,
        device, pred_len=pred_len, lr=1e-3, num_epochs=50
    )

    # ========== 在目标域上微调 ==========
    # 重新加载 FeatureExtractor
    feature_extractor_tgt = FeatureExtractor(input_dim=32, proj_dim=128, kernel_size=3, dropout=0.2).to(device)
    feature_extractor_tgt.load_state_dict(torch.load("feature_extractor_src.pth"))
    print("Loaded FeatureExtractor from source domain.")

    target_input_layer = TargetInputLayer(6, 32).to(device)
    target_regressor = Regressor(input_dim=32, output_dim=pred_len).to(device)

    print("===== (B) Fine-tune on Target Domain =====")
    train_target_model(
        loader_tgt_train, loader_tgt_val, loader_tgt_test,
        target_input_layer, feature_extractor_tgt, target_regressor,
        device, pred_len=pred_len, lr=1e-3, num_epochs=50
    )

    # ========== 测试集可视化 & 保存 ==========

    # 取原始目标域第0列的 min/max
    min_values = np.min(target_data_full[:, 0:1], axis=0)
    max_values = np.max(target_data_full[:, 0:1], axis=0)

    print("Plotting predictions vs actual on target test set...")
    plot_predictions_vs_actuals(
        loader=loader_tgt_test,
        input_layer=target_input_layer,
        feature_extractor=feature_extractor_tgt,
        regressor=target_regressor,
        device=device,
        scalers_target=scalers_target,
        seq_len=seq_len,
        pred_len=pred_len,
        time_index=time_index_test,
        max_values=max_values,
        min_values=min_values
    )

    save_target_predictions_with_time(
        loader=loader_tgt_test,
        input_layer=target_input_layer,
        feature_extractor=feature_extractor_tgt,
        regressor=target_regressor,
        device=device,
        scalers_target=scalers_target,
        time_index=time_index_test,
        seq_len=seq_len,
        pred_len=pred_len,
        filename="24普通迁移1核心.csv"
    )


if __name__ == "__main__":
    main()
