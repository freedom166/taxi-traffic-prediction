# query_bounds.py - 查询数据库中的经纬度范围
import pandas as pd
from sqlalchemy import create_engine

from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS

# 数据库连接（请确认密码正确）
engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

query = """
    SELECT 
        MIN(longitude) as min_lon, 
        MAX(longitude) as max_lon,
        MIN(latitude) as min_lat, 
        MAX(latitude) as max_lat 
    FROM taxi_gps_data
"""

df_sample = pd.read_sql(query, engine)

print("=" * 50)
print("数据库经纬度范围查询结果")
print("=" * 50)
print(f"经度范围: [{df_sample['min_lon'].iloc[0]:.6f}, {df_sample['max_lon'].iloc[0]:.6f}]")
print(f"纬度范围: [{df_sample['min_lat'].iloc[0]:.6f}, {df_sample['max_lat'].iloc[0]:.6f}]")
print("=" * 50)

# 查询数据量
count_query = "SELECT COUNT(*) as total FROM taxi_gps_data"
df_count = pd.read_sql(count_query, engine)
print(f"总数据量: {df_count['total'].iloc[0]:,} 条记录")