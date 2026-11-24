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
from mmengine.registry import init_default_scope
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

    # Load the config and initialize the scope.
    print(f"[1/4] Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)

    # Initialize the mmseg scope so that the OpenEarthMap dataset is registered.
    init_default_scope(cfg.get('default_scope', 'mmseg'))

    # Use batch size 1 for per-image latency measurement.
    cfg.test_dataloader.batch_size = 1

    print("[2/4] Building DataLoader...")
    test_dataloader = Runner.build_dataloader(cfg.test_dataloader)

    # Build the model.
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

            # Warm-up iterations.
            if idx < args.warmup:
                _ = model.test_step(data_batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"Warmup step [{idx + 1}/{args.warmup}] completed.")
                continue

            # Measured iterations.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run model inference and post-processing.
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

    # Print benchmark statistics.
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