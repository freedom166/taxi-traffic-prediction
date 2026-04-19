"""
ARIMA统计模型用于交通预测
成员2负责
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 动态添加项目根目录到路径，确保能导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import ARIMA_CONFIG, FIGURES_DIR, METRICS_DIR
except ImportError:
    # 如果没有config文件，使用默认配置
    FIGURES_DIR = 'outputs/figures'
    METRICS_DIR = 'outputs/metrics'
    ARIMA_CONFIG = {
        'max_p': 5,
        'max_q': 5,
        'max_d': 2,
        'seasonal': False,
        'seasonal_period': 12
    }


class ARIMAPredictor:
    """ARIMA交通预测模型"""

    def __init__(self, config=ARIMA_CONFIG):
        """
        初始化ARIMA模型

        Parameters:
        -----------
        config : dict, 模型配置参数
        """
        self.config = config
        self.model = None
        self.results = None
        self.best_order = None

    def build_model(self, order):
        """
        构建ARIMA模型结构
        
        Parameters:
        -----------
        order : tuple, (p, d, q) 参数
        """
        self.model = ARIMA(endog=None, order=order)
        self.best_order = order
        print(f"构建ARIMA模型: order={order}")
        return self.model

    def _grid_search(self, data):
        """
        使用网格搜索寻找最优参数 (p, d, q)
        """
        print("=" * 50)
        print("开始ARIMA参数自动搜索 (Grid Search)...")
        print("=" * 50)
        
        best_aic = float("inf")
        best_order = (0, 0, 0)
        
        # 简单的差分阶数确定逻辑
        d_range = range(self.config['max_d'] + 1)
        
        for d in d_range:
            for p in range(self.config['max_p'] + 1):
                for q in range(self.config['max_q'] + 1):
                    try:
                        # 尝试拟合模型
                        model_temp = ARIMA(data, order=(p, d, q))
                        results_temp = model_temp.fit()
                        
                        if results_temp.aic < best_aic:
                            best_aic = results_temp.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        
        self.best_order = best_order
        print(f"最优参数找到: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")
        print(f"最小AIC: {best_aic:.2f}")
        return best_order

    def train(self, X_train, y_train):
        """
        训练ARIMA模型
        
        Note: ARIMA通常处理单变量时间序列。
        这里我们将 y_train 作为目标序列进行拟合。
        """
        print("=" * 50)
        print("开始训练ARIMA模型...")
        print("=" * 50)

        # 确保数据是Series格式，且索引连续
        if isinstance(y_train, np.ndarray):
            train_series = pd.Series(y_train.flatten())
        else:
            train_series = y_train

        # 如果尚未确定最优参数，则进行搜索
        if self.best_order is None:
            self._grid_search(train_series)

        # 使用最优参数训练最终模型
        print(f"使用最优参数 {self.best_order} 进行最终训练...")
        self.model = ARIMA(train_series, order=self.best_order)
        self.results = self.model.fit()
        
        print("ARIMA模型训练完成")
        # print(self.results.summary()) # 摘要信息过多，默认注释
        
        return self.results

    def predict(self, X_test, y_test):
        """
        使用训练好的模型进行预测
        
        Parameters:
        -----------
        X_test : np.array, 测试集特征 (ARIMA通常不需要X，但为了接口统一保留)
        y_test : np.array, 测试集真实值 (用于确定预测步数)
        """
        if self.results is None:
            print("模型尚未训练，请先调用 train()")
            return None

        steps = len(y_test)
        print(f"预测未来 {steps} 个时间步...")
        
        # 进行动态预测
        pred_result = self.results.forecast(steps=steps)
        
        return pred_result.values

    def plot_diagnostics(self):
        """
        绘制模型诊断图 (残差分析)
        """
        if self.results is None:
            print("模型尚未训练")
            return

        plt.figure(figsize=(15, 10))
        # 使用 statsmodels 内置的绘图功能
        self.results.plot_diagnostics(figsize=(15, 10))
        plt.suptitle("ARIMA Model Diagnostics", fontsize=16)
        plt.tight_layout()
        
        if not os.path.exists(FIGURES_DIR):
            os.makedirs(FIGURES_DIR)
            
        plt.savefig(f'{FIGURES_DIR}/arima_diagnostics.png', dpi=150, bbox_inches='tight')
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
        # 防止除以0
        mape = np.mean(np.abs((y_true_original - y_pred_original) / (y_true_original + 1e-8))) * 100

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape
        }

        print("\n" + "=" * 50)
        print("ARIMA模型评估结果:")
        print("=" * 50)
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return metrics

    def plot_predictions(self, y_true, y_pred, title="ARIMA Predictions vs True Values"):
        """
        绘制预测值与真实值对比图
        """
        plt.figure(figsize=(14, 6))

        # 只显示前200个点以便清晰展示
        n_display = min(200, len(y_true))

        plt.plot(y_true[:n_display], label='True Values', linewidth=2, alpha=0.7)
        plt.plot(y_pred[:n_display], label='Predictions', linewidth=2, alpha=0.7)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Target Value', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if not os.path.exists(FIGURES_DIR):
            os.makedirs(FIGURES_DIR)
            
        plt.savefig(f'{FIGURES_DIR}/arima_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()


def run_arima_experiment():
    """
    运行完整的ARIMA实验流程
    """
    import sys
    import os
    from contextlib import redirect_stdout
    
    # 保存训练日志
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'outputs', 'arima_model.output')
    
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            print("\n" + "=" * 60)
            print("ARIMA交通预测实验")
            print("=" * 60)

            # 1. 加载数据 (复用LSTM的数据加载逻辑)
            try:
                from config import DATA_PROCESSED_DIR
                df_agg = pd.read_csv(f"{DATA_PROCESSED_DIR}/aggregated_traffic_data.csv")
                df_agg['time_slot_start'] = pd.to_datetime(df_agg['time_slot_id'], format='%Y%m%d_%H%M')
                print(f"加载聚合数据: {len(df_agg)} 条记录")
            except Exception as e:
                print(f"数据加载失败: {e}")
                return

            # 2. 构建特征 (复用LSTM的特征构建，保证输入一致)
            try:
                from feature_builder import TimeSeriesFeatureBuilder
                builder = TimeSeriesFeatureBuilder(lookback=12, predict_horizon=1)
                X, y, df_grid = builder.build_features(df_agg, target_col='avg_speed')
            except ImportError:
                print("未找到 feature_builder，请确保文件存在")
                return

            # 3. 划分数据集
            X_train, y_train, X_val, y_val, X_test, y_test = builder.train_val_test_split(X, y)

            # 4. 标准化
            (X_train_scaled, y_train_scaled,
             X_val_scaled, y_val_scaled,
             X_test_scaled, y_test_scaled,
             X_scaler, y_scaler) = builder.normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)

            # 5. 构建ARIMA模型
            # ARIMA通常不需要X输入，我们主要关注y序列的预测
            arima = ARIMAPredictor()
            # 注意：ARIMA类不需要build_model步骤，train内部会处理

            # 6. 训练模型
            # 这里传入y_train_scaled，ARIMA将学习这个序列的统计规律
            arima.train(X_train_scaled, y_train_scaled)

            # 7. 预测
            y_pred_scaled = arima.predict(X_test_scaled, y_test_scaled)

            # 8. 评估
            metrics = arima.evaluate(y_test_scaled, y_pred_scaled, y_scaler)

            # 9. 绘制预测对比图
            y_test_original = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
            y_pred_original = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            arima.plot_predictions(y_test_original, y_pred_original)
            
            # 10. 绘制诊断图
            # 注意：诊断图是基于训练数据的残差
            arima.plot_diagnostics()

    return metrics, y_test_original, y_pred_original


if __name__ == "__main__":
    metrics, y_true, y_pred = run_arima_experiment()
