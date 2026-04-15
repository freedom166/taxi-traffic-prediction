#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4项目主运行脚本
用于协调整个项目的运行流程
"""

import os
import sys

def print_project_info():
    """打印项目信息"""
    print("=" * 60)
    print("实验4：基于出租车GPS数据的交通预测")
    print("=" * 60)
    print("\n项目结构说明：")
    print("1. data/ - 数据目录（原始数据和处理后数据）")
    print("2. src/ - 源代码目录（数据处理、模型、评估等）")
    print("3. results/ - 实验结果目录（图表、指标、比较表）")
    print("4. reports/ - 实验报告目录")
    print("\n团队成员分工：")
    print("- 成员1: 项目统筹、数据预处理、LSTM模型")
    print("- 成员2: 特征构建、ARIMA模型")
    print("- 成员3: SVR模型、评估指标计算")
    print("- 成员4: 随机森林/XGBoost、对比图绘制")
    print("- 成员5: 实验报告撰写、结果分析")
    print("\n详细分工请参考 Team_Division_Plan.md")

def check_dependencies():
    """检查依赖库"""
    print("\n检查项目依赖...")
    try:
        import numpy
        import pandas
        import matplotlib
        import sklearn
        print("✓ 基础依赖库已安装")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖库: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    """主函数"""
    print_project_info()
    
    if not check_dependencies():
        print("\n请先安装依赖库再继续")
        return
    
    print("\n项目运行选项：")
    print("1. 查看项目结构")
    print("2. 运行数据预处理")
    print("3. 运行特征构建")
    print("4. 运行模型训练")
    print("5. 运行模型评估")
    print("6. 生成可视化图表")
    print("7. 退出")
    
    choice = input("\n请选择操作 (1-7): ")
    
    if choice == "1":
        print("\n当前项目结构：")
        os.system("tree . /F" if os.name == "nt" else "tree .")
    elif choice == "2":
        print("\n运行数据预处理...")
        print("请执行: python src/data_loader.py")
    elif choice == "3":
        print("\n运行特征构建...")
        print("请执行: python src/feature_builder.py")
    elif choice == "4":
        print("\n运行模型训练...")
        print("请选择模型：")
        print("  a) ARIMA模型: python src/models/arima_model.py")
        print("  b) SVR模型: python src/models/svr_model.py")
        print("  c) 随机森林模型: python src/models/rf_model.py")
        print("  d) LSTM模型: python src/models/lstm_model.py")
    elif choice == "5":
        print("\n运行模型评估...")
        print("请执行: python src/evaluation.py")
    elif choice == "6":
        print("\n生成可视化图表...")
        print("请执行: python src/visualization.py")
    elif choice == "7":
        print("\n退出程序")
    else:
        print("\n无效选择")

if __name__ == "__main__":
    main()