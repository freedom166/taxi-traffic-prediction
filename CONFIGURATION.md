# 项目配置说明

## 环境配置

### 1. Python环境
- Python版本: >= 3.8
- 推荐使用虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
1. 将原始GPS数据文件放入 `data/raw/` 目录
2. 数据格式要求：
   - CSV格式
   - 包含时间戳、经纬度、速度等字段
   - 建议文件名：`gps_data.csv`

### 3. 项目结构验证
运行以下命令验证项目结构：

```bash
python run_project.py
```

## 开发配置

### 1. 代码规范
- 使用PEP 8代码风格
- 添加必要的注释
- 函数和类需要有文档字符串

### 2. 模块导入规范
```python
# 标准库
import os
import sys
import datetime

# 第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 项目模块
from src.data_loader import load_data
from src.feature_builder import build_features
```

### 3. 数据路径配置
在代码中使用相对路径：

```python
# 正确的方式
RAW_DATA_PATH = "data/raw/gps_data.csv"
PROCESSED_DATA_PATH = "data/processed/aggregated_data.pkl"

# 避免使用绝对路径
```

## 协作配置

### 1. Git工作流
```bash
# 克隆项目
git clone <repository-url>
cd exp4-taxi-traffic-prediction

# 创建功能分支
git checkout -b feature/your-feature-name

# 提交更改
git add .
git commit -m "描述你的更改"

# 推送到远程
git push origin feature/your-feature-name
```

### 2. 文件命名规范
- Python文件: `snake_case.py`
- 数据文件: `descriptive_name.csv`
- 配置文件: `CONFIGURATION.md`
- 文档文件: `README.md`

### 3. 避免提交的文件
- 大型数据文件（>100MB）
- 个人配置文件
- 临时文件
- 虚拟环境文件

## 测试配置

### 1. 单元测试
在 `tests/` 目录中添加测试文件：

```python
# tests/test_data_loader.py
import unittest
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        # 测试代码
        pass
```

### 2. 运行测试
```bash
python -m pytest tests/
```

## 问题排查

### 1. 常见问题
1. **导入错误**: 确保在项目根目录运行代码
2. **数据路径错误**: 检查文件是否存在，路径是否正确
3. **依赖缺失**: 重新运行 `pip install -r requirements.txt`

### 2. 获取帮助
1. 查看 `README.md` 和 `CONFIGURATION.md`
2. 查看 `project_quick_reference.md`
3. 联系对应模块的负责成员