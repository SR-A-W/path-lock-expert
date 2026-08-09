# 数据集处理脚本包
# Step 1: prepare_think_data.py    — 下载源数据集，转换为 /think 格式
# Step 2: generate_no_think_data.py — 通过 vLLM API 生成 /no_think 回答
# Step 3: filter_no_think_data.py   — 过滤 /no_think 数据
# Step 4: merge_hybrid_dataset.py   — 合并最终训练集
