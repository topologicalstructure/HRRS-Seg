import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import time
import torch
import numpy as np

import segearth_segmentor
import custom_datasets

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope  # 引入作用域初始化工具
from mmseg.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark SegEarth Model Inference Speed & FPS')
    parser.add_argument('--config', default='./configs/cfg_openearthmap.py', help='Config file path')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of images to evaluate')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup iterations')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 Config 并初始化作用域
    print(f"[1/4] Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)

    # 初始化作用域为 'mmseg'，让注册表能正确定位到 OpenEarthMapDataset
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # 确保 batch_size 为 1 以准确测量单张图像延迟
    cfg.test_dataloader.batch_size = 1

    print("[2/4] Building DataLoader...")
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # 2. 构建模型
    print("[3/4] Building SegEarth model...")
    model = MODELS.build(cfg.model)
    model = model.to(args.device)
    model.eval()

    total_iters = args.warmup + args.num_samples
    durations = []

    print(f"[4/4] Starting Benchmark (Warmup: {args.warmup}, Test Samples: {args.num_samples})...")
    print("-" * 65)

    with torch.no_grad():
        for idx, data_batch in enumerate(test_dataloader):
            if idx >= total_iters:
                break

            # 预热阶段
            if idx < args.warmup:
                _ = model.test_step(data_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"Warmup step [{idx + 1}/{args.warmup}] completed.")
                continue

            # 正式测量阶段
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 执行模型推理与后处理
            _ = model.test_step(data_batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            step_time = end_time - start_time
            durations.append(step_time)

            curr_sample_idx = idx - args.warmup + 1
            print(f"Sample [{curr_sample_idx:>3}/{args.num_samples:>3}] | "
                  f"Latency: {step_time * 1000:6.2f} ms | "
                  f"Instant FPS: {1.0 / step_time:5.2f}")

    # 3. 统计并输出结果
    durations = np.array(durations)
    avg_latency = np.mean(durations)
    median_latency = np.median(durations)
    std_latency = np.std(durations)
    min_latency = np.min(durations)
    max_latency = np.max(durations)
    fps = 1.0 / avg_latency

    print("=" * 65)
    print("                      Benchmark Results                        ")
    print("=" * 65)
    print(f" Evaluated Samples  : {len(durations)}")
    print(f" Average Latency    : {avg_latency * 1000:.2f} ms (± {std_latency * 1000:.2f} ms)")
    print(f" Median Latency     : {median_latency * 1000:.2f} ms")
    print(f" Min / Max Latency  : {min_latency * 1000:.2f} ms / {max_latency * 1000:.2f} ms")
    print(f" Overall FPS        : {fps:.2f} frames/s")
    print("=" * 65)


if __name__ == '__main__':
    main()