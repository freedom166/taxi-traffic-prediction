# 处理数据目录

此目录用于存储处理后的数据集，包括：
- 按15分钟时间片聚合的数据
- 按网格ID/路段ID聚合的数据
- 特征矩阵X和目标向量y
- 训练集、验证集、测试集划分

**当前文件需要包含对各数据集进行详细描述**

## 1. [aggregated_traffic_data.csv](./aggregated_traffic_data.csv)

### **样例数据**
```csv
grid_id,time_slot_id,avg_speed,traffic_volume,sample_count,hour,day_of_week,is_weekend
3336_1176,20170305_2115,4.052985,8,134,21,6,1
3421_1220,20170305_0300,24.7,1,11,3,6,1
3341_1179,20170305_0300,27.975,4,28,3,6,1
3348_1151,20170301_0215,26.574263,80,509,2,2,0
3345_1146,20170305_2300,33.8,2,3,23,6,1
```

### **数据字段**
| 字段名              | 字段描述  | 数据类型     |
|:-----------------|:------|:---------|
| `grid_id`        | 网格ID  | `string` |
| `time_slot_id`   | 时间片ID | `string` |
| `avg_speed`      | 平均速度  | `float`  |
| `traffic_volume` | 交通量   | `int`    |
| `sample_count`   | 样本数量  | `int`    |
| `hour`           | 小时    | `int`    |
| `day_of_week`    | 星期几   | `int`    |
| `is_weekend`     | 是否周末  | `int`    |
