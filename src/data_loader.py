"""
使用 Roal 框架批流一体能力处理交通大数据
"""

import asyncio
import os
import csv
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from roal import Pipeline, task, TaskContext, DataBoundary, setup_logging
from roal.shuffle import ShuffleStorageConfig

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, TEMP_DIR


# ==================== 数据模型 ====================

@dataclass
class GPSData:
    """GPS原始数据"""
    record_date: str
    record_time: str
    taxi_id: str
    longitude: float
    latitude: float
    speed: float
    altitude: int = 0
    passenger: int = 0
    gps_type: int = 0


@dataclass
class GPSDataWithGrid:
    """带网格信息的GPS数据"""
    record_date: str
    record_time: str
    taxi_id: str
    longitude: float
    latitude: float
    speed: float
    altitude: int
    passenger: int
    gps_type: int
    grid_lon: int
    grid_lat: int
    grid_id: str
    datetime: datetime
    time_slot_start: datetime
    time_slot_id: str

    def to_dict(self):
        return self.__dict__


@dataclass
class AggregatedData:
    """聚合后的数据"""
    grid_id: str
    time_slot_id: str
    avg_speed: float
    traffic_volume: int
    sample_count: int
    hour: int
    day_of_week: int
    is_weekend: int = 0

    def to_dict(self):
        return self.__dict__


# ==================== 配置常量 ====================

@dataclass
class LocalConfig:
    """配置类"""
    data_folder: str = "D:\\文件\\临时文件\\交通大数据\\gps_datas" # DATA_RAW_DIR
    output_file: str = os.path.join(DATA_PROCESSED_DIR, "aggregated_traffic_data.csv")
    shuffle_spill_dir: str = os.path.join(TEMP_DIR, "shuffle_temp")
    file_encoding: str = 'gbk'
    time_interval_minutes: int = 15
    grid_size: float = 0.01
    lon_min: float = 73.0  # 中国最西端
    lon_max: float = 135.0  # 中国最东端
    lat_min: float = 18.0  # 中国最南端
    lat_max: float = 53.0  # 中国最北端
    error_threshold: int = 1000
    log_level: str = "INFO"
    max_files: int = 0
    max_shuffle_memory_limit_mb: int = 256 # MB


# ==================== 辅助函数 ====================

def validate_data(row):
    """验证数据行是否有效"""
    if len(row) < 10:
        return False

    try:
        import re
        # 检查时间日期格式
        if not re.match(r'^\d{8}$', row[0]):
            return False
        if not re.match(r'^\d{6}$', row[1]):
            return False

        # 检查出租车ID
        if not row[3] or len(str(row[3]).strip()) == 0:
            return False

        # 检查经纬度范围
        longitude = float(row[4])
        latitude = float(row[5])
        if longitude < -180 or longitude > 180:
            return False
        if latitude < -90 or latitude > 90:
            return False

        # 检查速度范围
        speed = float(row[6])
        if speed < 0 or speed > 200:
            return False

        return True
    except (ValueError, IndexError):
        return False


def parse_datetime(date_str: str, time_str: str) -> datetime:
    """解析日期时间"""
    # 处理可能的格式：20220101 和 083000
    if len(date_str) == 8:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    if len(time_str) == 6:
        time_str = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")


def calculate_time_slot(dt: datetime, interval_minutes: int = 15) -> datetime:
    """计算时间片起始时间"""
    minutes = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=minutes, second=0, microsecond=0)


def parse_hour(time_slot_id: str) -> int:
    """解析小时"""
    # time_slot_id 格式：20220101_0800
    try:
        time_part = time_slot_id.split('_')[1]
        return int(time_part[:2])
    except Exception:
        print(f"时间格式错误: {time_slot_id}")
        raise


def parse_day_of_week(time_slot_id: str) -> int:
    """解析星期几（0=Monday, 6=Sunday）"""
    date_part = time_slot_id.split('_')[0]
    dt = datetime.strptime(date_part, "%Y%m%d")
    return dt.weekday()


# ==================== Task 定义 ====================

@task(name="file_scanner", workers=2)
async def file_scanner(_, ctx: TaskContext):
    """文件扫描 Task：扫描指定目录下的所有文件，产出文件路径"""
    import os
    import re

    data_folder = ctx.config_get("data_folder", LocalConfig.data_folder)
    max_files = ctx.config_get("max_files", LocalConfig.max_files)
    sample_step = ctx.config_get("sample_step", 4)
    ctx.logger.info(f"开始扫描目录: {data_folder}")

    all_files = []

    for root, dirs, files in os.walk(data_folder):
        for d in sorted(dirs):
            if re.match(r'^\d+$', d):
                date_folder = os.path.join(root, d)
                day_files = sorted(
                    f for f in os.listdir(date_folder) if f.endswith('.TXT')
                )
                sampled = day_files[::sample_step]
                for f in sampled:
                    file_path = os.path.join(date_folder, f)
                    all_files.append(file_path)
                    ctx.logger.debug(f"找到文件: {file_path}")

    if max_files > 0:
        all_files = all_files[:max_files]

    for file_path in all_files:
        yield file_path

    ctx.metrics.counter("files_scanned").inc(len(all_files))
    ctx.logger.info(f"扫描完成，共找到 {len(all_files)} 个文件 (sample_step={sample_step})")


@task(name="file_data_loader", workers=2)
async def file_data_loader(file_path: str, ctx: TaskContext):
    """数据加载 Task：逐行读取文件内容"""
    ctx.logger.info(f"开始加载文件: {file_path}")
    line_count = 0
    file_encoding = ctx.config_get("file_encoding", LocalConfig.file_encoding)

    try:
        with open(file_path, 'r', encoding=file_encoding, errors='ignore', buffering=1024*1024) as f:
            for line in f:
                line_count += 1
                if line_count % 10000 == 0:
                    ctx.logger.debug(f"文件 {file_path} 已加载 {line_count} 行")
                yield line.strip()

        ctx.metrics.counter("lines_loaded").inc(line_count)
        ctx.logger.info(f"文件加载完成: {file_path}, 共 {line_count} 行")
    except Exception as e:
        ctx.logger.error(f"文件加载失败: {file_path}, 错误: {e}")
        ctx.metrics.counter("file_load_errors").inc()
        raise


@task(name="data_processor", workers=4)
async def data_processor(line: str, ctx: TaskContext):
    """数据处理 Task：清洗和验证数据"""
    if not line:
        ctx.metrics.counter("empty_lines").inc()
        return None

    ctx.metrics.counter("total").inc()
    row = line.split(',')

    if not validate_data(row):
        ctx.metrics.counter("invalid").inc()
        return None

    try:
        # 获取配置的范围限制
        lon_min = ctx.config_get("lon_min", LocalConfig.lon_min)
        lon_max = ctx.config_get("lon_max", LocalConfig.lon_max)
        lat_min = ctx.config_get("lat_min", LocalConfig.lat_min)
        lat_max = ctx.config_get("lat_max", LocalConfig.lat_max)

        longitude = float(row[4])
        latitude = float(row[5])
        speed = float(row[6])

        # 范围检查
        if longitude < lon_min or longitude > lon_max:
            ctx.metrics.counter("longitude_out_of_range").inc()
            return None
        if latitude < lat_min or latitude > lat_max:
            ctx.metrics.counter("latitude_out_of_range").inc()
            return None
        if speed < 0 or speed > 200:
            ctx.metrics.counter("speed_out_of_range").inc()
            return None

        ctx.metrics.counter("valid").inc()

        # 记录速度分布
        if speed > 0:
            if speed < 30:
                ctx.metrics.counter("speed_low").inc()
            elif speed < 60:
                ctx.metrics.counter("speed_medium").inc()
            else:
                ctx.metrics.counter("speed_high").inc()

        return GPSData(
            record_date=row[0],
            record_time=row[1],
            taxi_id=str(row[3]).strip(),
            longitude=longitude,
            latitude=latitude,
            speed=speed,
            altitude=int(row[7]) if len(row) > 7 else 0,
            passenger=int(row[8]) if len(row) > 8 else 0,
            gps_type=int(row[9]) if len(row) > 9 else 0,
        )
    except Exception as e:
        ctx.logger.error(f"数据处理失败: {line[:50]}..., 错误: {e}")
        ctx.metrics.counter("processing_errors").inc()
        return None


@task(name="grid_assigner", workers=4)
async def grid_assigner(data: GPSData, ctx: TaskContext):
    """网格分配 Task：计算网格并设置 Shuffle Key"""
    if data is None:
        return None

    try:
        # 获取配置
        lon_min = ctx.config_get("lon_min", LocalConfig.lon_min)
        lat_min = ctx.config_get("lat_min", LocalConfig.lat_min)
        grid_size = ctx.config_get("grid_size", LocalConfig.grid_size)
        time_interval = ctx.config_get("time_interval_minutes", LocalConfig.time_interval_minutes)

        # 计算网格索引
        grid_lon = int((data.longitude - lon_min) // grid_size)
        grid_lat = int((data.latitude - lat_min) // grid_size)
        grid_id = f"{grid_lon}_{grid_lat}"

        # 解析时间
        datetime_obj = parse_datetime(data.record_date, data.record_time)
        time_slot_start = calculate_time_slot(datetime_obj, time_interval)
        time_slot_id = time_slot_start.strftime("%Y%m%d_%H%M")

        # 设置 Shuffle Key：相同网格+时间片的数据将被路由到同一个 Reducer
        shuffle_key = f"{grid_id}#{time_slot_id}"
        ctx.set_shuffle_key(shuffle_key)

        ctx.metrics.counter("grid_assigned").inc()

        return GPSDataWithGrid(
            record_date=data.record_date,
            record_time=data.record_time,
            taxi_id=data.taxi_id,
            longitude=data.longitude,
            latitude=data.latitude,
            speed=data.speed,
            altitude=data.altitude,
            passenger=data.passenger,
            gps_type=data.gps_type,
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            grid_id=grid_id,
            datetime=datetime_obj,
            time_slot_start=time_slot_start,
            time_slot_id=time_slot_id,
        )
    except Exception as e:
        ctx.logger.error(f"网格分配失败: {e}")
        ctx.metrics.counter("grid_assignment_errors").inc()
        return None


@task(name="data_aggregator", workers=2, task_type="reduce")
async def data_aggregator(key_values: tuple, ctx: TaskContext):
    """
    数据聚合 Task（Reduce 阶段）
    接收相同 (grid_id, time_slot) 的所有数据，进行最终聚合
    """
    key, values = key_values

    speed_sum = 0.0
    taxi_ids = set()  # 全局去重
    sample_count = 0

    # 解析 key 获取元信息
    grid_id, time_slot_id = key.rsplit('#', 1)

    ctx.logger.debug(f"开始聚合 key: {key}")

    # 流式处理数据，不会一次性加载全部
    async for data in values:
        if data is not None:
            speed_sum += data.speed
            taxi_ids.add(data.taxi_id)
            sample_count += 1

    if sample_count == 0:
        ctx.logger.info(f"key {key} 无有效数据")
        return None

    # 计算聚合结果
    avg_speed = round(speed_sum / sample_count if sample_count > 0 else 0.0, 6)
    traffic_volume = len(taxi_ids)  # 精确去重

    # 计算时间特征
    hour = parse_hour(time_slot_id)
    day_of_week = parse_day_of_week(time_slot_id)
    is_weekend = 1 if day_of_week >= 5 else 0

    ctx.metrics.counter("aggregated_keys").inc()
    ctx.metrics.counter("aggregated_records").inc(sample_count)

    ctx.logger.debug(f"key {key} 聚合完成: avg_speed={avg_speed:.2f}, volume={traffic_volume}, count={sample_count}")

    return AggregatedData(
        grid_id=grid_id,
        time_slot_id=time_slot_id,
        avg_speed=avg_speed,
        traffic_volume=traffic_volume,
        sample_count=sample_count,
        hour=hour,
        day_of_week=day_of_week,
        is_weekend=is_weekend,
    )


_sink_buffer = []
_sink_header_written = False


@task(name="data_sink", workers=1)
async def data_sink(data: AggregatedData, ctx: TaskContext):
    """数据输出 Task：将聚合结果批量写入 CSV"""
    global _sink_buffer, _sink_header_written

    if data is None:
        return

    output_file = ctx.config_get("output_file", LocalConfig.output_file)

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    _sink_buffer.append(data.to_dict())

    if len(_sink_buffer) >= 100:
        try:
            with open(output_file, 'a', newline='', encoding='utf-8') as f:
                fieldnames = list(_sink_buffer[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not _sink_header_written:
                    writer.writeheader()
                    _sink_header_written = True
                writer.writerows(_sink_buffer)

            ctx.metrics.counter("records_exported").inc(len(_sink_buffer))
            ctx.logger.debug(f"批量导出 {len(_sink_buffer)} 条聚合记录")
            _sink_buffer = []
        except Exception as e:
            ctx.logger.error(f"CSV 写入失败: {e}")
            ctx.metrics.counter("sink_errors").inc()
            raise


def flush_sink():
    """将缓冲区中剩余记录写入 CSV"""
    global _sink_buffer, _sink_header_written

    if not _sink_buffer:
        return

    output_file = LocalConfig.output_file

    try:
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(_sink_buffer[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not _sink_header_written:
                writer.writeheader()
                _sink_header_written = True
            writer.writerows(_sink_buffer)

        print(f"最终 flush 导出 {len(_sink_buffer)} 条聚合记录")
        _sink_buffer = []
    except Exception as e:
        print(f"CSV flush 写入失败: {e}")
        raise


# ==================== Pipeline 组装 ====================

async def main():
    """主函数：组装并运行 Pipeline"""
    import os

    if not os.path.exists(LocalConfig.data_folder):
        print(f"错误：数据目录不存在: {LocalConfig.data_folder}")
        return

    if os.path.exists(LocalConfig.output_file):
        os.remove(LocalConfig.output_file)
        print(f"已删除旧的输出文件: {LocalConfig.output_file}")

    setup_logging(log_level=LocalConfig.log_level)

    pipeline = Pipeline(
        "TrafficGPSDataLoader",
        boundary=DataBoundary.FILE_SET,
    )

    # 定义处理流程
    pipeline.source(file_scanner) \
        .then(file_data_loader) \
        .then(data_processor) \
        .then(grid_assigner) \
        .shuffle(
            key_extractor="shuffle_key",  # 使用 grid_assigner 中设置的 key
            partitioner="hash",
            num_partitions=20,  # 分区数，影响 Reduce 并发度
            storage_config=ShuffleStorageConfig(
                backend="hybrid",           # 混合存储：内存 + 磁盘
                memory_limit_mb=LocalConfig.max_shuffle_memory_limit_mb,        # 256MB 内存缓冲（达到后溢写磁盘）
                spill_dir=LocalConfig.shuffle_spill_dir,
                compress=True,
                cleanup_on_complete=True
            )
        ) \
        .reduce(
            data_aggregator,
            workers=2,                      # Reduce 并发数
        ) \
        .sink(data_sink)

    # 配置各个 Task
    pipeline.update_config("file_scanner", {
        "data_folder": LocalConfig.data_folder,
        "max_files": LocalConfig.max_files
    })

    pipeline.update_config("file_data_loader", {
        "file_encoding": LocalConfig.file_encoding
    })

    pipeline.update_config("grid_assigner", {
        "lon_min": LocalConfig.lon_min,
        "lon_max": LocalConfig.lon_max,
        "lat_min": LocalConfig.lat_min,
        "lat_max": LocalConfig.lat_max,
        "grid_size": LocalConfig.grid_size,
        "time_interval_minutes": LocalConfig.time_interval_minutes
    })

    pipeline.update_config("data_sink", {
        "output_file": LocalConfig.output_file
    })

    # 注册完成回调
    def on_complete(_):
        print(f"\nPipeline '{pipeline.name}' 执行完成!")

    pipeline.on_complete(on_complete)

    print(f"Pipeline '{pipeline.name}' 已构建:")
    print(f"  - 数据目录: {LocalConfig.data_folder}")
    print(f"  - 输出文件: {LocalConfig.output_file}")
    print(f"  - 网格大小: {LocalConfig.grid_size}°")
    print(f"  - 时间间隔: {LocalConfig.time_interval_minutes} 分钟")
    print()
    print("开始处理数据...")

    # 运行 Pipeline
    await pipeline.run_async()

    # flush 缓冲区中剩余的记录
    flush_sink()

    # 打印最终统计
    print("\n数据处理完成!")
    print(f"输出文件: {LocalConfig.output_file}")


if __name__ == "__main__":
    asyncio.run(main())