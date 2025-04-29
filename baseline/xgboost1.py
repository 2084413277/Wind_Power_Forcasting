import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# 需要先安装 xgboost: pip install xgboost
from xgboost import XGBRegressor

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
        x = self.data[real_idx : real_idx + self.seq_len]
        y = self.data[real_idx + self.seq_len : real_idx + self.seq_len + self.pred_len]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            real_idx
        )


# -----------------------------
# 将 WindPowerDataset 数据转为 (X, Y) 形式，方便 XGBoost 训练
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
        x_np = x.numpy().reshape(-1)  # Flatten: seq_len*feature_dim
        y_np = y.numpy()[:, 0]        # 只取第0列 (pred_len,)
        all_X.append(x_np)
        all_Y.append(y_np)
        all_indices.append(idx)
    all_X = np.array(all_X)
    all_Y = np.array(all_Y)  # shape: (N, pred_len)
    return all_X, all_Y, all_indices


# -----------------------------
# 评估函数 (打印 RMSE, MAE, R2)
# -----------------------------
def evaluate_model(X_data, Y_data, model):
    """
    X_data: (N, seq_len * feature_dim)
    Y_data: (N, pred_len)
    model: MultiOutputRegressor 包装的 XGBRegressor
    """
    preds = model.predict(X_data)  # (N, pred_len)
    preds = preds.reshape(-1)      # Flatten
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
    X_data, Y_data, model, indices,
    scalers_target, max_values=None, min_values=None,
    seq_len=24, pred_len=8, time_index=None
):
    """
    X_data: (N, seq_len * feature_dim)
    Y_data: (N, pred_len)
    model: 训练好的 XGBoost (多输出包装)
    indices: 每个样本对应的起始索引
    scalers_target: 存储目标域每列的StandardScaler
    max_values, min_values: 第0列的最大最小值 (用于截断, 可选)
    seq_len, pred_len: 滑动窗口长度
    time_index: 对应的时间索引(一维)
    """
    preds = model.predict(X_data)  # (N, pred_len)

    preds_inv_list = []
    labels_inv_list = []

    for i in range(len(preds)):
        p = preds[i].reshape(-1, 1)    # (pred_len,1)
        l = Y_data[i].reshape(-1, 1)   # (pred_len,1)

        # 反标准化
        p_inv = scalers_target[0].inverse_transform(p)
        l_inv = scalers_target[0].inverse_transform(l)
        if max_values is not None and min_values is not None:
            p_inv = np.clip(p_inv, np.array(min_values).min(), np.array(max_values).max())

        preds_inv_list.append(p_inv.flatten())
        labels_inv_list.append(l_inv.flatten())

    preds_inv = np.concatenate(preds_inv_list, axis=0)   # (N*pred_len,)
    labels_inv = np.concatenate(labels_inv_list, axis=0)

    rmse = np.sqrt(mean_squared_error(labels_inv, preds_inv))
    mae = mean_absolute_error(labels_inv, preds_inv)
    r2 = r2_score(labels_inv, preds_inv)

    print(f"Predict Result - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # 仅演示绘制前60条时刻的数据
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
# 保存目标域预测结果 (仅第0风电场) + 时间
# - 这里改成“最后一次预测覆盖之前的预测”
# -----------------------------
def save_target_predictions_with_time(
    X_data, Y_data, model, indices,
    scalers_target, time_index, seq_len, pred_len
):
    """
    将预测值与真实值和对应时间对齐后保存。
    这里若同一时刻 t 被多次预测到，就用“最后一次预测”覆盖前面结果。
    """
    preds = model.predict(X_data)  # (N, pred_len)

    results = {}
    # 逐样本
    for i in range(len(preds)):
        idx = indices[i]  # 滑动窗口起始位置
        for k in range(pred_len):
            t = time_index[idx + seq_len + k]
            true_val = Y_data[i][k]
            pred_val = preds[i][k]

            # 反标准化（只预测第0列）
            true_val_inv = scalers_target[0].inverse_transform([[true_val]])[0, 0]
            pred_val_inv = scalers_target[0].inverse_transform([[pred_val]])[0, 0]

            # 直接覆盖 => 同一个 t 若出现多次，就保存“最后一次”
            error = abs(true_val_inv - pred_val_inv)
            results[t] = (error, true_val_inv, pred_val_inv)

    # 其余按时间排序并写CSV
    sorted_times = sorted(results.keys())
    time_list = []
    original_list = []
    prediction_list = []
    for t in sorted_times:
        time_list.append(t)
        original_list.append(results[t][1])
        prediction_list.append(results[t][2])

    df_out = pd.DataFrame({"time": time_list, "acc": original_list, "dann": prediction_list})
    df_out.to_csv("24xgboost2.csv", index=False)  # 文件名随意
    print("目标域预测结果及对应时间已保存至 24xgboost2.csv")


# -----------------------------
# 主程序：仅使用目标域数据训练 XGBoost
# -----------------------------
def main():
    # ---------------------
    # 参数设置
    # ---------------------
    seq_len = 24   # 输入过去 24 个时刻
    pred_len = 2   # 预测未来 8 个时刻
    unit_size = 36 # 每个单元大小

    # ========== 读取目标域数据 ==========
    df_target = pd.read_excel('../nignxia_6_windfarms.xlsx', index_col=0)
    df_target = df_target.iloc[0:-1:4, :].iloc[0:4320, :]
    time_index_full = df_target.index

    # 以36行为一个单元（不足单元的末尾数据将舍弃）
    num_full_units = len(df_target) // unit_size
    num_rows_to_use = num_full_units * unit_size
    df_target_used = df_target.iloc[:num_rows_to_use, :]
    time_index_used = df_target_used.index

    # 转换为 numpy 数组，并取前6列数据
    dataset_2 = df_target_used.to_numpy()
    target_data = dataset_2[:, :6]

    # 对目标域数据进行逐列标准化
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
    val_units   = data_units[val_indices]
    test_units  = data_units[test_indices]

    time_train_units = time_units[train_indices]
    time_val_units   = time_units[val_indices]
    time_test_units  = time_units[test_indices]

    # 将各集合内的单元展开为二维数据
    train_data = train_units.reshape(-1, target_data.shape[1])
    val_data   = val_units.reshape(-1, target_data.shape[1])
    test_data  = test_units.reshape(-1, target_data.shape[1])

    time_index_train = time_train_units.reshape(-1)
    time_index_val   = time_val_units.reshape(-1)
    time_index_test  = time_test_units.reshape(-1)

    print("目标域训练数据 shape:", train_data.shape)
    print("目标域验证数据 shape:", val_data.shape)
    print("目标域测试数据 shape:", test_data.shape)

    # 构建 Dataset / Dataloader
    train_dataset = WindPowerDataset(train_data, seq_len, pred_len, unit_size=unit_size)
    val_dataset   = WindPowerDataset(val_data, seq_len, pred_len, unit_size=unit_size)
    test_dataset  = WindPowerDataset(test_data, seq_len, pred_len, unit_size=unit_size)

    # 转成可用于 XGBoost 的 (X, Y)
    X_train, Y_train, idx_train = dataset_to_svr_xy(train_dataset)
    X_val,   Y_val,   idx_val2  = dataset_to_svr_xy(val_dataset)
    X_test,  Y_test,  idx_test  = dataset_to_svr_xy(test_dataset)

    # ========== 训练 XGBRegressor (多输出) ==========
    print("Training XGBoost model on target domain data...")
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor

    xgb_model = MultiOutputRegressor(
        XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42
        )
    )
    xgb_model.fit(X_train, Y_train)

    # 验证集评估
    rmse_val, mae_val, r2_val = evaluate_model(X_val, Y_val, xgb_model)
    print(f"Validation - RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}")

    # 测试集评估
    print("Evaluating model on target test set...")
    rmse_tgt, mae_tgt, r2_tgt = evaluate_model(X_test, Y_test, xgb_model)
    print(f"Test Set - RMSE: {rmse_tgt:.4f}, MAE: {mae_tgt:.4f}, R2: {r2_tgt:.4f}")

    # ========== 绘制预测 vs 实际 (仅第0风电场) ==========
    min_values = np.min(dataset_2[:, 0:1], axis=0)
    max_values = np.max(dataset_2[:, 0:1], axis=0)
    print("Plotting predictions vs actual for target domain (first wind farm) using test set...")
    plot_predictions_vs_actuals(
        X_data=X_test,
        Y_data=Y_test,
        model=xgb_model,
        indices=idx_test,
        scalers_target=scalers_target,
        max_values=max_values,
        min_values=min_values,
        seq_len=seq_len,
        pred_len=pred_len,
        time_index=time_index_test
    )

    # ========== 保存目标域预测结果及时间 (仅第0风电场) ==========
    save_target_predictions_with_time(
        X_data=X_test,
        Y_data=Y_test,
        model=xgb_model,
        indices=idx_test,
        scalers_target=scalers_target,
        time_index=time_index_test,
        seq_len=seq_len,
        pred_len=pred_len
    )


if __name__ == "__main__":
    main()
