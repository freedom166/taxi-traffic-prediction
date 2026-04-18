# 实验4：基于出租车GPS数据的交通预测

## 项目简介

本项目为实验4课程作业，旨在通过对出租车GPS数据进行处理与分析，实现交通流量/速度的预测任务。项目涵盖了数据预处理、特征构建、多种预测模型的实现与评估，并比较不同模型在交通预测任务上的性能表现。

## 主要任务

1. **数据预处理与特征构建**
    - 将原始GPS数据按15分钟时间片和空间单元（路段ID/网格ID）进行聚合
    - 绘制指定路段/网格在一周内的平均速度变化曲线
    - 构建时间序列特征矩阵X（过去n个时隙）和预测目标y（下一个时隙）

2. **交通预测模型实现**
    - 实现至少4种不同的预测方法（含1种神经网络模型）
    - 包括：ARIMA、SVR、随机森林/XGBoost、LSTM/GRU

3. **模型评估与比较**
    - 计算各模型的MAE、MSE、MAPE指标
    - 绘制预测值与真实值对比图
    - 汇总性能比较表格

4. **实验报告撰写**
    - 包含实验目标、内容、步骤、结果分析、问题总结、分工说明

## 项目结构

```plaintext
exp4-taxi-traffic-prediction/
│
├── data/ # 原始数据与处理后数据
│ ├── raw/ # 原始GPS数据
│ └── processed/ # 聚合后的数据集
│
├── src/ # 源代码
│ ├── data_loader.py # 数据加载与预处理
│ ├── feature_builder.py # 特征矩阵构建
│ ├── models/ # 预测模型
│ │ ├── arima_model.py
│ │ ├── svr_model.py
│ │ ├── rf_model.py
│ │ └── lstm_model.py
│ ├── evaluation.py # 评估指标计算
│ └── visualization.py # 可视化工具
│
├── results/ # 实验结果
│ ├── figures/ # 对比图、曲线图
│ ├── metrics/ # MAE/MSE/MAPE结果
│ └── comparison_table.xlsx # 性能比较表
│
├── reports/ # 实验报告
│ ├── Experiment_4_Report.pdf
│ └── Experiment_4_Report.docx
│
├── requirements.txt # 依赖库列表
├── README.md # 项目说明
└── Team_Division.md # 分工文档
```

## 环境依赖

```markdown
Python >= 3.8

主要依赖库：

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels (ARIMA)
- tensorflow / pytorch (LSTM)
- xgboost (可选)
```

安装命令：

```bash
pip install -r requirements.txt
```

## 数据说明

- **数据来源**：出租车GPS轨迹数据
- **时间片长度**：15分钟
- **空间单元**：路段ID 或 网格ID
- **预测目标**：交通流量 或 平均速度

## 模型列表

| 模型             | 类型     | 负责成员 |
|:---------------|:-------|:----:|
| ARIMA          | 时间序列模型 | 成员2  |
| SVR            | 支持向量回归 | 成员3  |
| 随机森林 / XGBoost | 树模型集成  | 成员4  |
| LSTM / GRU     | 神经网络模型 | 成员1  |

## 评估指标

- **MAE**（平均绝对误差）
- **MSE**（均方误差）
- **MAPE**（平均绝对百分比误差）

## 运行说明

1. 克隆仓库

```bash
git clone https://github.com/RestRegular/exp4-taxi-traffic-prediction.git
cd exp4-taxi-traffic-prediction
```

2. 准备数据
    - 将原始GPS数据放入 `data/raw/` 目录
    - 运行数据预处理脚本

3. 训练与评估模型

```bash

# 示例：运行LSTM模型

python src/models/lstm_model.py

# 运行评估

python src/evaluation.py
```

## 团队成员与分工

| 成员  | 工作量 | 主要职责               |
|:---:|:---:|:-------------------|
| 成员1 | 21% | 项目统筹、数据预处理、LSTM模型  |
| 成员2 | 20% | 特征构建、ARIMA模型       |
| 成员3 | 20% | SVR模型、评估指标计算       |
| 成员4 | 20% | 随机森林/XGBoost、对比图绘制 |
| 成员5 | 19% | 实验报告撰写、结果分析        |

详细分工请参阅 [`Team_Division.md`](./Team_Division_Plan.md)

## 实验结果摘要

_（实验完成后填写）_

| 模型           | MAE | MSE | MAPE |
|:-------------|:---:|:---:|:----:|
| ARIMA        |  -  |  -  |  -   |
| SVR          |  -  |  -  |  -   |
| 随机森林/XGBoost |  -  |  -  |  -   |
| LSTM/GRU     |  -  |  -  |  -   |

## 注意事项

- 所有成员需使用统一的数据集，确保结果可比性
- 代码需有清晰注释，便于整合与调试
- 实验结果与图表及时同步给报告撰写成员
