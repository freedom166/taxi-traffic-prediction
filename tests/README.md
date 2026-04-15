# 测试目录

此目录用于存放项目的单元测试和集成测试。

## 测试文件结构
- `test_data_loader.py` - 数据加载模块测试
- `test_feature_builder.py` - 特征构建模块测试
- `test_models/` - 模型测试目录
  - `test_arima_model.py` - ARIMA模型测试
  - `test_svr_model.py` - SVR模型测试
  - `test_rf_model.py` - 随机森林模型测试
  - `test_lstm_model.py` - LSTM模型测试
- `test_evaluation.py` - 评估模块测试
- `test_visualization.py` - 可视化模块测试

## 运行测试
```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_data_loader.py

# 运行测试并显示详细输出
python -m pytest tests/ -v

# 运行测试并生成覆盖率报告
python -m pytest tests/ --cov=src
```

## 测试数据
测试数据应放在 `tests/test_data/` 目录中，使用小型数据集进行测试。