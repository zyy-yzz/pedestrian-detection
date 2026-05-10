"""
Evaluation script for Enhanced YOLOv8 night-time pedestrian detection.

Computes mAP@0.5, mAP@0.5:0.95, precision, recall, miss rate, FPS.
"""

import os
import sys
import yaml
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import ConfusionMatrix, ap_per_class

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str) -> dict:
    with open(config_path, 'r')  as f:
        return yaml.safe_load(f)


def evaluate(model_path: str, config_path: str, device: str = '0')  -> dict:
    """
    Run full evaluation suite on the test set.

    Returns dict with all metrics.
    """
    cfg = load_config(config_path)
    eval_cfg = cfg['evaluation']
    data_cfg = cfg['data']

    # Load model
    model = YOLO(model_path)

    # Run validation
    results = model.val(
        data=str(Path(data_cfg['data_dir']) / 'data.yaml'),
        imgsz=cfg['model']['img_size'],
        batch=cfg['training']['batch_size'],
        conf=eval_cfg['conf_threshold'],
        iou=eval_cfg['iou_threshold'],
        device=device,
        split='test',
        verbose=True,
    )

    # Extract metrics
    metrics = {
        'mAP_0.5': float(results.box.map50),
        'mAP_0.5_0.95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }

    LOGGER.info(f"mAP@0.5:      {metrics['mAP_0.5']:.4f}")
    LOGGER.info(f"mAP@0.5:0.95: {metrics['mAP_0.5_0.95']:.4f}")
    LOGGER.info(f"Precision:    {metrics['precision']:.4f}")
    LOGGER.info(f"Recall:       {metrics['recall']:.4f}")

    # Measure FPS
    fps = benchmark_fps(model, cfg['model']['img_size'], device)
    metrics['fps'] = fps
    LOGGER.info(f"FPS:          {fps:.1f}")

    # Save metrics
    metrics_path = Path('runs/night_ped_detection') / 'eval_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f'Metrics saved to {metrics_path}')

    return metrics


def benchmark_fps(model: YOLO, img_size: int, device: str, num_iters: int = 500) -> float:
    """Measure inference FPS using model.predict() on synthetic input."""
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    model.model.eval()

    # Warmup
    for _ in range(30):
        with torch.inference_mode():
            _ = model.predict(dummy, verbose=False)

    # Benchmark
    if device != 'cpu':
        torch.cuda.synchronize()
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model.predict(dummy, verbose=False)
        if device != 'cpu':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_latency = np.mean(times) * 1000  # ms
    fps = 1000.0 / avg_latency

    LOGGER.info(f'Average latency: {avg_latency:.1f} ms, FPS: {fps:.1f}')
    return fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Enhanced YOLOv8')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--config', type=str,
                        default='model_training/configs/train_config.yaml',
                        help='Path to evaluation config')
    parser.add_argument('--device', type=str, default='0',
                        help='Device to run evaluation on')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run FPS benchmark only')

    args = parser.parse_args()

    if args.benchmark:
        model = YOLO(args.model)
        fps = benchmark_fps(model, 640, args.device)
        print(f'FPS: {fps:.1f}')
    else:
        evaluate(args.model, args.config, args.device)
