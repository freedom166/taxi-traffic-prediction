"""
LSTM神经网络模型用于交通预测
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LSTM_CONFIG, FIGURES_DIR, METRICS_DIR, RANDOM_SEED
from feature_builder import TimeSeriesFeatureBuilder

# 设置随机种子
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class LSTMPredictor:
    """LSTM交通预测模型"""

    def __init__(self, input_shape, config=LSTM_CONFIG):
        """
        初始化LSTM模型

        Parameters:
        -----------
        input_shape : tuple, 输入形状 (lookback, features)
        config : dict, 模型配置参数
        """
        self.input_shape = input_shape
        self.config = config
        self.model = None
        self.history = None

    def build_model(self):
        """
        构建LSTM网络结构
        """
        model = Sequential()

        # 输入层
        model.add(Input(shape=self.input_shape))

        # 第一层LSTM
        model.add(LSTM(
            units=self.config['lstm_units'][0],
            return_sequences=True,
            activation='tanh'
        ))
        model.add(Dropout(self.config['dropout_rate']))

        # 第二层LSTM
        model.add(LSTM(
            units=self.config['lstm_units'][1],
            return_sequences=False,
            activation='tanh'
        ))
        model.add(Dropout(self.config['dropout_rate']))

        # 输出层
        model.add(Dense(units=1))

        # 编译模型
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        print(model.summary())

        return model

    def prepare_lstm_data(self, X, y):
        """
        为LSTM准备数据格式
        LSTM期望输入形状: (samples, timesteps, features)
        """
        # X已经是 (samples, timesteps) 格式
        # 需要重塑为 (samples, timesteps, features=1)
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y

    def train(self, X_train, y_train, X_val, y_val):
        """
        训练LSTM模型
        """
        # 准备数据
        X_train_lstm, y_train_lstm = self.prepare_lstm_data(X_train, y_train)
        X_val_lstm, y_val_lstm = self.prepare_lstm_data(X_val, y_val)

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'{METRICS_DIR}/lstm_best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]

        # 训练模型
        print("=" * 50)
        print("开始训练LSTM模型...")
        print("=" * 50)

        self.history = self.model.fit(
            X_train_lstm, y_train_lstm,
            validation_data=(X_val_lstm, y_val_lstm),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def predict(self, X):
        """
        使用训练好的模型进行预测
        """
        X_lstm, _ = self.prepare_lstm_data(X, np.zeros(len(X)))
        predictions = self.model.predict(X_lstm, verbose=0)
        return predictions.flatten()

    def plot_training_history(self):
        """
        绘制训练过程中的损失曲线
        """
        if self.history is None:
            print("模型尚未训练")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('LSTM Training Loss', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE曲线
        axes[1].plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        axes[1].plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('MAE', fontsize=12)
        axes[1].set_title('LSTM Training MAE', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/lstm_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()

    def evaluate(self, y_true, y_pred, y_scaler=None):
        """
        评估模型性能
        """
        # 如果使用了标准化，需要反标准化
        if y_scaler is not None:
            y_true_original = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_original = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_original = y_true
            y_pred_original = y_pred

        mae = mean_absolute_error(y_true_original, y_pred_original)
        mse = mean_squared_error(y_true_original, y_pred_original)
        mape = np.mean(np.abs((y_true_original - y_pred_original) / (y_true_original + 1e-8))) * 100

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape
        }

        print("\n" + "=" * 50)
        print("LSTM模型评估结果:")
        print("=" * 50)
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return metrics

    def plot_predictions(self, y_true, y_pred, title="LSTM Predictions vs True Values"):
        """
        绘制预测值与真实值对比图
        """
        plt.figure(figsize=(14, 6))

        # 只显示前200个点以便清晰展示
        n_display = min(200, len(y_true))

        plt.plot(y_true[:n_display], label='True Values', linewidth=2, alpha=0.7)
        plt.plot(y_pred[:n_display], label='Predictions', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step (15-min intervals)', fontsize=12)
        plt.ylabel('Average Speed (km/h)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/lstm_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_lstm_experiment():
    """
    运行完整的LSTM实验流程
    """
    print("\n" + "=" * 60)
    print("LSTM交通预测实验")
    print("=" * 60)

    # 1. 加载数据
    df_agg = pd.read_csv(f"{DATA_PROCESSED_DIR}/aggregated_traffic_data.csv")
    df_agg['time_slot_start'] = pd.to_datetime(df_agg['time_slot_start'])
    print(f"加载聚合数据: {len(df_agg)} 条记录")

    # 2. 构建特征
    builder = TimeSeriesFeatureBuilder(lookback=12, predict_horizon=1)
    X, y, df_grid = builder.build_features(df_agg, target_col='avg_speed')

    # 3. 划分数据集
    X_train, y_train, X_val, y_val, X_test, y_test = builder.train_val_test_split(X, y)

    # 4. 标准化
    (X_train_scaled, y_train_scaled,
     X_val_scaled, y_val_scaled,
     X_test_scaled, y_test_scaled,
     X_scaler, y_scaler) = builder.normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # 5. 构建LSTM模型
    lstm = LSTMPredictor(input_shape=(X_train_scaled.shape[1], 1))
    lstm.build_model()

    # 6. 训练模型
    lstm.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)

    # 7. 绘制训练曲线
    lstm.plot_training_history()

    # 8. 预测
    y_pred_scaled = lstm.predict(X_test_scaled)

    # 9. 评估
    metrics = lstm.evaluate(y_test_scaled, y_pred_scaled, y_scaler)

    # 10. 绘制预测对比图
    y_test_original = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    lstm.plot_predictions(y_test_original, y_pred_original)

    return metrics, y_test_original, y_pred_original


if __name__ == "__main__":
    from config import DATA_PROCESSED_DIR

    metrics, y_true, y_pred = run_lstm_experiment()