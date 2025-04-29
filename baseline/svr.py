import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# matplotlib 配置
# -----------------------------
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 20


# -----------------------------
# 自定义数据集类（加入 unit_size 参数，确保滑动窗口不跨单元）
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
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            real_idx  # 返回原始数据中的起始索引（可用于后续对应时间）
        )


# -----------------------------
# 将 WindPowerDataset 数据转为 (X, Y) 形式，方便 SVR 训练
# 这里仅取第 0 列风电场作为目标变量，并做多步输出
# -----------------------------
def dataset_to_svr_xy(dataset):
    """
    输入: dataset(WindPowerDataset)
    输出:
        X: (N, seq_len * feature_dim)
        Y: (N, pred_len)
        indices: 对应每个样本的起始索引
    """
    all_X = []
    all_Y = []
    all_indices = []
    for i in range(len(dataset)):
        x, y, idx = dataset[i]
        # x shape: (seq_len, feature_dim)
        # y shape: (pred_len, feature_dim)，但只关心第 0 列
        x_np = x.numpy().reshape(-1)               # Flatten: seq_len*feature_dim
        y_np = y.numpy()[:, 0]                     # 只取第0列 (pred_len,)
        all_X.append(x_np)
        all_Y.append(y_np)
        all_indices.append(idx)
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)  # shape: (N, pred_len)
    return all_X, all_Y, all_indices


# -----------------------------
# 评估函数 (打印 RMSE, MAE, R2)
# -----------------------------
def evaluate_model(X_data, Y_data, svr_model):
    """
    X_data: (N, seq_len * feature_dim)
    Y_data: (N, pred_len)
    svr_model: MultiOutputRegressor 包装的 SVR
    """
    preds = svr_model.predict(X_data)  # shape: (N, pred_len)
    preds = preds.reshape(-1)          # Flatten
    labels = Y_data.reshape(-1)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return rmse, mae, r2


# -----------------------------
# 绘制 预测 vs 实际 (仅第0风电场)
# - 这里需要额外的时间索引和标准化逆变换
# -----------------------------
def plot_predictions_vs_actuals(
    X_data, Y_data, svr_model, indices,
    scalers_target, max_values=None, min_values=None,
    seq_len=24, pred_len=8, time_index=None
):
    """
    X_data: (N, seq_len * feature_dim)
    Y_data: (N, pred_len)
    svr_model: 训练好的SVR
    indices: 每个样本对应的起始索引
    scalers_target: 存储目标域每列的StandardScaler
    max_values, min_values: 第0列的最大最小值 (用于截断, 可选)
    seq_len, pred_len: 滑动窗口长度
    time_index: 对应的时间索引(一维)
    """
    preds = svr_model.predict(X_data)  # (N, pred_len)

    # 仅对第0列风电场做反标准化（因为Y_data仅代表第0列的多步输出）
    preds_inv_list = []
    labels_inv_list = []

    for i in range(len(preds)):
        p = preds[i].reshape(-1, 1)  # (pred_len,1)
        l = Y_data[i].reshape(-1, 1)

        # 反标准化
        p_inv = scalers_target[0].inverse_transform(p)
        l_inv = scalers_target[0].inverse_transform(l)

        if max_values is not None and min_values is not None:
            p_inv = np.clip(p_inv, np.array(min_values).min(), np.array(max_values).max())

        preds_inv_list.append(p_inv.flatten())
        labels_inv_list.append(l_inv.flatten())

    preds_inv = np.concatenate(preds_inv_list, axis=0)   # 总长度 = N*pred_len
    labels_inv = np.concatenate(labels_inv_list, axis=0)

    rmse = np.sqrt(mean_squared_error(labels_inv, preds_inv))
    mae = mean_absolute_error(labels_inv, preds_inv)
    r2 = r2_score(labels_inv, preds_inv)

    print(f"Predict Result - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 绘制前60个时刻的数据(仅示例)
    plt.figure(figsize=(12, 5))
    plt.plot(labels_inv[:60], label='actually', linestyle='-', linewidth=3, marker='o', markersize=8)
    plt.plot(preds_inv[:60], label='prediction', linestyle='--', linewidth=3, marker='x', markersize=8)
    plt.xlabel("Time Steps", fontsize=20)
    plt.ylabel("Power (MW)", fontsize=20)
    plt.title(f"Prediction vs Actually Value (R2={r2:.2f})", fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 修改后的保存函数
# - 不再接收 loader，改为接收 (X_data, Y_data, indices)
# - 若同一时刻 t 多次预测，则保留“最后一次”预测
# -----------------------------
def save_target_predictions_with_time(
        X_data,
        Y_data,
        svr_model,
        indices,
        scalers_target,
        time_index,
        seq_len,
        pred_len,
        filename="24svr12核心.csv"
):
    """
    X_data: (N, seq_len * feature_dim)
    Y_data: (N, pred_len)
    svr_model: 已训练的SVR模型
    indices: 每个样本在原始数据中的起始索引
    scalers_target: 列表[StandardScaler(...), ...], 用于反标准化
    time_index: 所有行对应的时间索引(一维)
    seq_len, pred_len: 用于定位测试集每条样本对应的预测时间
    filename: 输出文件名
    """
    # 先用SVR预测
    preds = svr_model.predict(X_data)  # (N, pred_len)

    results = {}
    N = len(X_data)
    for i in range(N):
        idx = indices[i]
        for k in range(pred_len):
            t = time_index[idx + seq_len + k]
            true_val = Y_data[i, k]
            pred_val = preds[i, k]

            # 只对第0列风电场做反标准化
            true_val_inv = scalers_target[0].inverse_transform([[true_val]])[0, 0]
            pred_val_inv = scalers_target[0].inverse_transform([[pred_val]])[0, 0]

            # 误差(可仅作参考)
            error = abs(true_val_inv - pred_val_inv)

            # 直接覆盖 => 保留最后一次预测
            results[t] = (error, true_val_inv, pred_val_inv)

    # 按时间排序并输出CSV
    sorted_times = sorted(results.keys())
    time_list, original_list, prediction_list = [], [], []
    for t in sorted_times:
        time_list.append(t)
        original_list.append(results[t][1])
        prediction_list.append(results[t][2])

    df_out = pd.DataFrame({"time": time_list, "acc": original_list, "dann": prediction_list})
    df_out.to_csv(filename, index=False)
    print(f"预测结果已保存至 {filename}")


# -----------------------------
# 主程序：仅使用目标域数据训练SVR
# -----------------------------
def main():
    # ---------------------
    # 参数设置
    # ---------------------
    seq_len = 24  # 输入过去 24 个时刻
    pred_len = 12  # 预测未来 8 个时刻
    unit_size = 36  # 每个单元大小（与之前相同）

    # ========== 读取目标域数据 ==========
    df_target = pd.read_excel('../nignxia_6_windfarms.xlsx', index_col=0)
    # 例如，每隔4行取一行，并取前4320行数据（0:4320）
    df_target = df_target.iloc[0:-1:4, :].iloc[0:4320, :]
    time_index_full = df_target.index  # 保存完整时间索引

    # 以36行为一个单元（不足单元的末尾数据将舍弃）
    num_full_units = len(df_target) // unit_size
    num_rows_to_use = num_full_units * unit_size
    df_target_used = df_target.iloc[:num_rows_to_use, :]
    time_index_used = df_target_used.index

    # 转换为 numpy 数组，并取前6列数据
    dataset_2 = df_target_used.to_numpy()
    target_data = dataset_2[:, :6]  # (N, 6)

    # 对目标域数据逐列标准化
    scalers_target = []
    target_data_scaled = np.zeros_like(target_data)
    for i in range(target_data.shape[1]):
        sc = StandardScaler()
        col_scaled = sc.fit_transform(target_data[:, i].reshape(-1, 1))
        scalers_target.append(sc)
        target_data_scaled[:, i] = col_scaled.flatten()

    # (单元数, unit_size, 特征数)
    data_units = target_data_scaled.reshape(num_full_units, unit_size, -1)
    time_units = np.array(time_index_used).reshape(num_full_units, unit_size)

    # 随机打乱
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

    # 将各集合内的单元展开为二维数据
    train_data = train_units.reshape(-1, target_data.shape[1])
    val_data = val_units.reshape(-1, target_data.shape[1])
    test_data = test_units.reshape(-1, target_data.shape[1])

    time_index_train = time_train_units.reshape(-1)
    time_index_val = time_val_units.reshape(-1)
    time_index_test = time_test_units.reshape(-1)

    print("目标域训练数据 shape:", train_data.shape)
    print("目标域验证数据 shape:", val_data.shape)
    print("目标域测试数据 shape:", test_data.shape)

    # 构建 Dataset
    train_dataset = WindPowerDataset(train_data, seq_len, pred_len, unit_size=unit_size)
    val_dataset = WindPowerDataset(val_data, seq_len, pred_len, unit_size=unit_size)
    test_dataset = WindPowerDataset(test_data, seq_len, pred_len, unit_size=unit_size)

    # 转成可用于SVR的 (X, Y)
    X_train, Y_train, idx_train = dataset_to_svr_xy(train_dataset)
    X_val,   Y_val,   idx_val   = dataset_to_svr_xy(val_dataset)
    X_test,  Y_test,  idx_test  = dataset_to_svr_xy(test_dataset)

    # ========== 训练 SVR (多输出) ==========
    print("Training SVR model on target domain data...")
    svr_model = MultiOutputRegressor(
        SVR(kernel='rbf', C=1.0, epsilon=0.1)
    )
    svr_model.fit(X_train, Y_train)

    # 验证集
    rmse_val, mae_val, r2_val = evaluate_model(X_val, Y_val, svr_model)
    print(f"Validation - RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}")

    # 测试集
    print("Evaluating model on target test set...")
    rmse_tgt, mae_tgt, r2_tgt = evaluate_model(X_test, Y_test, svr_model)
    print(f"Test Set - RMSE: {rmse_tgt:.4f}, MAE: {mae_tgt:.4f}, R2: {r2_tgt:.4f}")

    # ========== 绘制预测 vs 实际 (仅第0风电场) ==========
    min_values = np.min(dataset_2[:, 0:1], axis=0)
    max_values = np.max(dataset_2[:, 0:1], axis=0)
    print("Plotting predictions vs actual for target domain (first wind farm) using test set...")
    plot_predictions_vs_actuals(
        X_data=X_test,
        Y_data=Y_test,
        svr_model=svr_model,
        indices=idx_test,
        scalers_target=scalers_target,
        max_values=max_values,
        min_values=min_values,
        seq_len=seq_len,
        pred_len=pred_len,
        time_index=time_index_test
    )

    # ========== 保存预测结果(仅第0风电场) ==========
    save_target_predictions_with_time(
        X_data=X_test,
        Y_data=Y_test,
        svr_model=svr_model,
        indices=idx_test,
        scalers_target=scalers_target,
        time_index=time_index_test,
        seq_len=seq_len,
        pred_len=pred_len
    )


if __name__ == "__main__":
    main()
