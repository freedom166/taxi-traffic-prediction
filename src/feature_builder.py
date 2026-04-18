"""
特征矩阵构建模块
构建时间序列特征矩阵 X（过去n个时隙）和预测目标 y（下一个时隙）
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import LOOKBACK_WINDOW, PREDICT_HORIZON, RANDOM_SEED

np.random.seed(RANDOM_SEED)


class TimeSeriesFeatureBuilder:
    """时间序列特征构建类"""

    def __init__(self, lookback=LOOKBACK_WINDOW, predict_horizon=PREDICT_HORIZON):
        self.lookback = lookback
        self.predict_horizon = predict_horizon
        self.scaler = StandardScaler()

    def select_grid(self, df, grid_id=None, top_n=5):
        """
        选择要预测的网格
        可以选择指定网格ID，或选择流量最大的前N个网格
        """
        if grid_id:
            df_grid = df[df['grid_id'] == grid_id].copy()
            print(f"选择网格: {grid_id}, 数据量: {len(df_grid)}")
        else:
            # 按流量排序，选择流量最大的网格
            traffic_by_grid = df.groupby('grid_id')['traffic_volume'].sum().sort_values(ascending=False)
            top_grids = traffic_by_grid.head(top_n).index.tolist()
            df_grid = df[df['grid_id'].isin(top_grids)].copy()
            print(f"选择前{top_n}个流量最大的网格: {top_grids}")

        return df_grid

    def create_sequences(self, data, target_col='avg_speed'):
        """
        创建时间序列样本
        输入X: 过去lookback个时隙的特征
        输出y: 未来predict_horizon个时隙的目标值
        """
        X, y = [], []

        # 按时间排序
        data = data.sort_values('time_slot_start').reset_index(drop=True)

        values = data[target_col].values

        for i in range(len(values) - self.lookback - self.predict_horizon + 1):
            # 过去lookback个时隙的值
            X.append(values[i:i + self.lookback])
            # 未来predict_horizon个时隙的值（取平均值）
            y.append(np.mean(values[i + self.lookback:i + self.lookback + self.predict_horizon]))

        return np.array(X), np.array(y)

    def add_time_features(self, df):
        """
        添加时间特征作为辅助输入
        """
        df = df.copy()

        # 时间特征（周期性编码）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def build_features(self, df_aggregated, grid_id=None, target_col='avg_speed'):
        """
        构建完整的特征矩阵
        """
        print("=" * 50)
        print("开始构建特征矩阵...")
        print("=" * 50)

        # 1. 选择目标网格
        df_grid = self.select_grid(df_aggregated, grid_id)

        # 2. 添加时间特征
        df_grid = self.add_time_features(df_grid)

        # 3. 创建序列
        X, y = self.create_sequences(df_grid, target_col)

        # 4. 如果需要使用多特征，扩展X的维度
        # 这里先使用单变量时间序列

        print(f"特征矩阵形状: X={X.shape}, y={y.shape}")

        return X, y, df_grid

    def train_val_test_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        划分训练集、验证集、测试集
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(f"\n数据集划分:")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
        print(f"  测试集: {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def normalize_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        数据标准化
        """
        # 重塑X用于标准化
        original_shape_X = X_train.shape
        X_train_flat = X_train.reshape(-1, 1)
        X_val_flat = X_val.reshape(-1, 1)
        X_test_flat = X_test.reshape(-1, 1)

        # 拟合标准化器
        X_scaled_flat = self.scaler.fit_transform(X_train_flat)
        X_val_scaled_flat = self.scaler.transform(X_val_flat)
        X_test_scaled_flat = self.scaler.transform(X_test_flat)

        # 重塑回原始形状
        X_train_scaled = X_scaled_flat.reshape(original_shape_X)
        X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
        X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)

        # 对y进行标准化
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        print(f"\n数据标准化完成")

        return (X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled,
                X_test_scaled, y_test_scaled,
                self.scaler, y_scaler)


if __name__ == "__main__":
    # 测试特征构建
    import pandas as pd
    from config import DATA_PROCESSED_DIR

    # 加载聚合数据
    df_agg = pd.read_csv(f"{DATA_PROCESSED_DIR}/aggregated_traffic_data.csv")
    df_agg['time_slot_start'] = pd.to_datetime(df_agg['time_slot_start'])

    # 构建特征
    builder = TimeSeriesFeatureBuilder(lookback=12, predict_horizon=1)
    X, y, df_grid = builder.build_features(df_agg, target_col='avg_speed')

    # 划分数据集
    X_train, y_train, X_val, y_val, X_test, y_test = builder.train_val_test_split(X, y)

    print(f"\n最终特征形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")