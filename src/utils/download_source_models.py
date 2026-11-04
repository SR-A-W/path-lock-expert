#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for downloading and verifying source models

Downloads and verifies source model files.
Currently a stub: source models are downloaded manually into models/source/.

Source models:
- Qwen2.5-7B-Instruct: models/source/Qwen2.5-7B-Instruct/
- DeepSeek-R1-Distill-Qwen-7B: models/source/DeepSeek-R1-Distill-Qwen-7B/
"""

import os
from pathlib import Path


def download_model(model_name: str, output_dir: str = "./models/source") -> str:
    """Download a model locally (stub implementation)
    
    Args:
        model_name: model name (e.g. "Qwen/Qwen2.5-7B-Instruct")
        output_dir: output directory
        
    Returns:
        local path of the model
    """
    model_short_name = model_name.split("/")[-1]
    local_path = os.path.join(output_dir, model_short_name)
    
    if os.path.exists(local_path):
        print(f"Model already exists: {local_path}")
        return local_path
    
    raise NotImplementedError(
        f"Automatic download is not implemented. Please download the model to {local_path} manually\n"
        f"e.g.: huggingface-cli download {model_name} --local-dir {local_path}"
    )


def verify_model(model_path: str) -> bool:
    """Check model file integrity
    
    Checks that config.json and the safetensors files exist.
    
    Args:
        model_path: model path
        
    Returns:
        whether verification passed
    """
    required_files = ["config.json"]
    
    for f in required_files:
        if not os.path.exists(os.path.join(model_path, f)):
            print(f"❌ Missing: {f}")
            return False
    
    # Check safetensors files
    has_safetensors = (
        os.path.exists(os.path.join(model_path, "model.safetensors")) or
        os.path.exists(os.path.join(model_path, "model.safetensors.index.json"))
    )
    
    if not has_safetensors:
        print(f"❌ Missing safetensors files")
        return False
    
    print(f"✅ Model verification passed: {model_path}")
    return True
