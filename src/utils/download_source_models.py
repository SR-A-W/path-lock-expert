#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""源模型下载辅助工具

用于下载和验证源模型文件。
目前为 stub 模块，源模型已手动下载到 models/source/ 目录。

源模型：
- Qwen2.5-7B-Instruct: models/source/Qwen2.5-7B-Instruct/
- DeepSeek-R1-Distill-Qwen-7B: models/source/DeepSeek-R1-Distill-Qwen-7B/
"""

import os
from pathlib import Path


def download_model(model_name: str, output_dir: str = "./models/source") -> str:
    """下载模型到本地（stub 实现）
    
    Args:
        model_name: 模型名称（如 "Qwen/Qwen2.5-7B-Instruct"）
        output_dir: 输出目录
        
    Returns:
        模型本地路径
    """
    model_short_name = model_name.split("/")[-1]
    local_path = os.path.join(output_dir, model_short_name)
    
    if os.path.exists(local_path):
        print(f"模型已存在: {local_path}")
        return local_path
    
    raise NotImplementedError(
        f"自动下载暂未实现。请手动下载模型到 {local_path}\n"
        f"例如: huggingface-cli download {model_name} --local-dir {local_path}"
    )


def verify_model(model_path: str) -> bool:
    """验证模型文件完整性
    
    检查 config.json 和 safetensors 文件是否存在。
    
    Args:
        model_path: 模型路径
        
    Returns:
        验证是否通过
    """
    required_files = ["config.json"]
    
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            print(f"❌ 缺少: {f}")
            return False
    
    # 检查 safetensors 文件
    has_safetensors = (
        os.path.exists(os.path.join(model_path, "model.safetensors")) or
        os.path.exists(os.path.join(model_path, "model.safetensors.index.json"))
    )
    
    if not has_safetensors:
        print(f"❌ 缺少 safetensors 文件")
        return False
    
    print(f"✅ 模型验证通过: {model_path}")
    return True
