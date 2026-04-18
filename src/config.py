"""
配置文件 - 统一管理所有参数
"""
import os
from dotenv import load_dotenv

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw') # 需要替换为真实的原始数据路径
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

load_dotenv(os.path.join(BASE_DIR, '.env'))

# 数据库配置
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '')
DB_NAME = os.getenv('DB_NAME', 'taxi_gps')
DB_PORT = int(os.getenv('DB_PORT', 3306))

# 确保目录存在
for dir_path in [DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据聚合配置
TIME_INTERVAL_MINUTES = 15  # 时间片长度（分钟）
TIME_INTERVAL_SECONDS = TIME_INTERVAL_MINUTES * 60

# 空间网格配置
GRID_SIZE = 0.01  # 网格大小（经纬度约1km）
LON_MIN, LON_MAX = 85.830965, 119.017763
LAT_MIN, LAT_MAX = 11.924277, 43.196307

# 时间序列配置
LOOKBACK_WINDOW = 12  # 过去n个时隙（12*15min=3小时）
PREDICT_HORIZON = 1   # 预测下一个时隙

# 训练配置
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# LSTM模型参数
LSTM_CONFIG = {
    'lstm_units': [64, 32],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}

# 随机种子
RANDOM_SEED = 42