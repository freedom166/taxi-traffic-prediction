# 模型输入输出格式说明文档

## 1. 数据格式概述

本文档详细说明LSTM和GRU神经网络模型的输入输出格式，以及数据处理流程。

## 2. 输入数据格式

### 2.1 原始数据格式

原始数据来源于聚合后的交通数据文件 `aggregated_traffic_data.csv`，包含以下字段：

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| grid_id | string | 网格ID，格式为 "lon_lat" |
| time_slot_id | string | 时间片ID，格式为 "YYYYMMDD_HHMM" |
| avg_speed | float | 平均速度 (km/h) |
| traffic_volume | int | 交通流量 |
| sample_count | int | 样本数量 |
| hour | int | 小时 (0-23) |
| day_of_week | int | 星期几 (0=Monday, 6=Sunday) |
| is_weekend | int | 是否周末 (0=否, 1=是) |

### 2.2 特征矩阵构建

使用 `TimeSeriesFeatureBuilder` 类构建时间序列特征矩阵：

- **输入参数**：
  - `lookback`：过去的时间步数，默认12（即3小时）
  - `predict_horizon`：预测的时间步数，默认1
  - `grid_id`：可选，指定网格ID，默认选择流量最大的网格
  - `target_col`：目标列，默认"avg_speed"

- **输出**：
  - `X`：特征矩阵，形状为 (samples, lookback)
  - `y`：目标向量，形状为 (samples,)
  - `df_grid`：处理后的网格数据

### 2.3 数据标准化

使用 `StandardScaler` 对数据进行标准化处理：

- **输入**：训练集、验证集、测试集
- **输出**：标准化后的数据和标准化器对象

### 2.4 LSTM/GRU模型输入格式

LSTM和GRU模型期望的输入形状为：

- **形状**：(samples, timesteps, features)
- **说明**：
  - `samples`：样本数量
  - `timesteps`：时间步数，默认为12
  - `features`：特征数量，默认为1（仅使用速度数据）

## 3. 输出数据格式

### 3.1 模型预测输出

- **形状**：(samples,)
- **类型**：float
- **描述**：预测的平均速度值 (km/h)

### 3.2 模型评估输出

模型评估返回以下指标：

| 指标 | 描述 | 单位 |
| :--- | :--- | :--- |
| MAE | 平均绝对误差 | km/h |
| MSE | 均方误差 | (km/h)² |
| MAPE | 平均绝对百分比误差 | % |

### 3.3 保存的文件

模型训练过程中会保存以下文件：

1. **模型权重文件**：
   - LSTM：`results/metrics/lstm_best_model.keras`
   - GRU：`results/metrics/gru_best_model.keras`

2. **可视化文件**：
   - 训练历史图：`results/figures/lstm_training_history.png` 和 `results/figures/gru_training_history.png`
   - 预测对比图：`results/figures/lstm_predictions.png` 和 `results/figures/gru_predictions.png`

3. **训练日志**：
   - 标准输出中包含训练过程的详细日志

## 4. 模型配置

### 4.1 LSTM/GRU模型参数

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| lstm_units | LSTM/GRU单元数 | [64, 32] |
| dropout_rate | Dropout率 | 0.2 |
| learning_rate | 学习率 | 0.001 |
| batch_size | 批量大小 | 32 |
| epochs | 训练轮数 | 100 |
| early_stopping_patience | 早停 patience | 10 |

### 4.2 数据处理参数

| 参数 | 描述 | 默认值 |
| :--- | :--- | :--- |
| lookback | 过去时间步数 | 12 |
| predict_horizon | 预测时间步数 | 1 |
| train_ratio | 训练集比例 | 0.7 |
| val_ratio | 验证集比例 | 0.15 |
| test_ratio | 测试集比例 | 0.15 |

## 5. 运行流程

1. **数据加载**：读取 `aggregated_traffic_data.csv` 文件
2. **特征构建**：使用 `TimeSeriesFeatureBuilder` 构建时间序列特征
3. **数据划分**：将数据划分为训练集、验证集和测试集
4. **数据标准化**：对数据进行标准化处理
5. **模型构建**：构建LSTM或GRU模型
6. **模型训练**：训练模型并保存最佳权重
7. **模型评估**：在测试集上评估模型性能
8. **结果可视化**：绘制训练历史和预测对比图

## 6. 示例代码

### 6.1 运行LSTM模型

```python
from src.models.lstm_model import run_lstm_experiment

metrics, y_true, y_pred = run_lstm_experiment()
```

### 6.2 运行GRU模型

```python
from src.models.gru_model import run_gru_experiment

metrics, y_true, y_pred = run_gru_experiment()
```

## 7. 注意事项

1. 确保 `aggregated_traffic_data.csv` 文件存在于 `data/processed/` 目录
2. 模型训练可能需要较长时间，取决于硬件性能
3. 可以通过调整模型参数来优化性能
4. 预测结果的准确性可能受到数据质量和特征选择的影响
