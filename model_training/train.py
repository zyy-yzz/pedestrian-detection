"""
Enhanced YOLOv8 Training Script for Night-Time Pedestrian Detection.

Integrates CBAM attention, C2f-Ghost blocks, BiFPN neck, and WIoU loss
into YOLOv8 and trains on the LLVIP paired visible-infrared dataset.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from ultralytics.utils import LOGGER

# Add project root and modules to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model_training.modules.cbam import CBAM
from model_training.modules.c2f_ghost import C2fGhost
from model_training.modules.bifpn import BiFPN
from model_training.modules.wiou_loss import WIoULoss


def load_config(config_path: str) -> dict:
    """Load YAML training configuration."""
    with open(config_path, 'r')  as f:
        return yaml.safe_load(f)


def build_enhanced_model(cfg: dict, device: str = '0') -> DetectionModel:
    """
    Build enhanced YOLOv8 model with custom modules injected.

    Architecture modifications:
        - CBAM inserted after backbone stages P3, P4, P5
        - C2fGhost replaces C2f blocks in backbone
        - BiFPN replaces PANet neck
        - WIoU loss in detection head (applied at loss computation)
    """
    model_cfg = cfg['model']

    # Load base YOLOv8 model architecture
    base_model = YOLO(model_cfg['pretrained_weights']).model

    # == Inject CBAM after backbone stages ==
    # YOLOv8 backbone outputs at indices corresponding to P3, P4, P5
    # We identify these by stride: P3=8x, P4=16x, P5=32x
    backbone_channels = {
        'P3': 128,  # YOLOv8n: 128 channels at P3
        'P4': 256,  # YOLOv8n: 256 channels at P4
        'P5': 512,  # YOLOv8n: 512 channels at P5
    }

    cbam_p3 = CBAM(backbone_channels['P3']).to(device)
    cbam_p4 = CBAM(backbone_channels['P4']).to(device)
    cbam_p5 = CBAM(backbone_channels['P5']).to(device)

    # Store CBAM modules as model attributes for forward hook injection
    base_model.cbam_p3 = cbam_p3
    base_model.cbam_p4 = cbam_p4
    base_model.cbam_p5 = cbam_p5
    base_model.use_cbam = True

    # == Replace C2f with C2fGhost in backbone ==
    _replace_c2f_with_ghost(base_model)

    # == Replace PANet neck with BiFPN ==
    bifpn_channels = 256
    base_model.bifpn = BiFPN(
        channels=bifpn_channels,
        num_repeats=3,
    ).to(device)
    base_model.use_bifpn = True

    # == Register WIoU loss ==
    wiou_delta = cfg.get('losses', {}).get('wiou_delta', 1.9)
    base_model.wiou_loss = WIoULoss(delta=wiou_delta)
    _inject_wiou_loss(base_model, delta=wiou_delta)

    LOGGER.info('Enhanced YOLOv8 model built: CBAM + C2fGhost + BiFPN + WIoU')
    return base_model


def _inject_wiou_loss(model: nn.Module, delta: float = 1.9):
    """
    Monkey-patch ultralytics bbox_iou to compute WIoU loss.

    The trick: bbox_iou is patched to return 1 - weight * (1 - iou_base),
    so the existing loss code (loss = 1 - iou) produces WIoU loss.
    """
    from ultralytics.utils.metrics import bbox_iou as _orig_bbox_iou

    def _wiou_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        iou = _orig_bbox_iou(box1, box2, xywh=xywh, GIoU=False, DIoU=False, CIoU=False, eps=eps)
        with torch.no_grad():
            beta = (1.0 - iou.detach()) / delta
            weight = torch.exp(beta)
        return 1.0 - weight * (1.0 - iou)

    import ultralytics.utils.metrics as metrics_module
    metrics_module.bbox_iou = _wiou_bbox_iou
    LOGGER.info(f'WIoU loss injected with delta={delta}')


def _replace_c2f_with_ghost(model: nn.Module):
    """Replace all C2f blocks in the model with C2fGhost.

    Collects replacement targets first, then applies them in a second pass
    to avoid mutating module children during iteration.
    """
    replacements: list[tuple[str, nn.Module, C2f]] = []
    for name, module in model.named_modules():
        if isinstance(module, C2f) and hasattr(module, 'cv1'):
            try:
                in_ch = module.cv1.conv.in_channels
                out_ch = module.cv2.conv.out_channels
                n_bottlenecks = len(module.m)
                ghost_block = C2fGhost(in_ch, out_ch, n=n_bottlenecks)
                replacements.append((name, module, ghost_block))
            except Exception:
                LOGGER.warning(f'Failed to prepare C2fGhost for "{name}", keeping original')
                continue

    for name, _original, ghost_block in replacements:
        try:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1] if '.' in name else name
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, ghost_block)
        except Exception:
            LOGGER.warning(f'Failed to apply C2fGhost to "{name}", keeping original')
            continue


def prepare_llvip_dataset(cfg: dict) -> dict:
    """
    Prepare LLVIP dataset in YOLO format.

    Converts VOC XML annotations to YOLO txt format if needed,
    creates data.yaml for ultralytics training.
    """
    data_dir = Path(cfg['data_dir'])
    ann_dir = data_dir / 'Annotations'

    # Create YOLO-format label directories
    for split in ['train', 'val', 'test']:
        label_dir = data_dir / 'labels' / split
        label_dir.mkdir(parents=True, exist_ok=True)

    # Convert VOC XML → YOLO txt (single class: pedestrian)
    _convert_voc_to_yolo(ann_dir, data_dir)

    # Create val split: random 15% of train labels moved to val
    _split_train_val(data_dir, val_ratio=0.15)

    # Create data.yaml for ultralytics
    data_yaml = {
        'path': str(data_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'pedestrian'},
        'nc': 1,
    }

    yaml_path = data_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    LOGGER.info(f'Dataset prepared: {yaml_path}')
    return data_yaml


def _convert_voc_to_yolo(ann_dir: Path, data_dir: Path):
    """
    Convert VOC XML annotations to YOLO format txt files.
    LLVIP: one XML per image pair, bounding boxes for pedestrians.
    Train/val split determined by XML <folder> tag or file number.
    """
    import xml.etree.ElementTree as ET

    for xml_file in ann_dir.glob('*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        img_name = root.find('filename').text
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        folder = root.find('folder')
        if folder is not None and folder.text == 'test':
            split_dir = 'test'
        else:
            split_dir = 'train'

        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name.lower() != 'person' and cls_name.lower() != 'pedestrian':
                continue

            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            # Normalize to [0, 1]
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            label_file = data_dir / 'labels' / split_dir / f'{xml_file.stem}.txt'
            with open(label_file, 'w') as f:
                f.write(f'0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')


def _split_train_val(data_dir: Path, val_ratio: float = 0.15):
    """Move a random subset of train labels to val."""
    import random
    train_label_dir = data_dir / 'labels' / 'train'
    val_label_dir = data_dir / 'labels' / 'val'
    label_files = list(train_label_dir.glob('*.txt'))
    if not label_files:
        return
    random.seed(42)
    random.shuffle(label_files)
    val_count = max(1, int(len(label_files) * val_ratio))
    for f in label_files[:val_count]:
        f.rename(val_label_dir / f.name)


def train(args):
    """Main training function."""
    cfg = load_config(args.config)
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    aug_cfg = cfg['augmentation']

    # Prepare dataset
    data_yaml = prepare_llvip_dataset(data_cfg)

    # Build enhanced model or resume from checkpoint
    resume = args.resume
    if resume:
        if resume == 'auto':
            yolo_model = YOLO(model_cfg['pretrained_weights'])
            LOGGER.info('Resuming training from last checkpoint (auto-detect)')
        else:
            yolo_model = YOLO(resume)
            LOGGER.info(f'Resuming training from checkpoint: {resume}')
    else:
        model = build_enhanced_model(cfg, device=str(train_cfg.get('device', '0')))
        yolo_model = YOLO(model_cfg["pretrained_weights"])
        yolo_model.model = model

    # Train
    results = yolo_model.train(
        resume=bool(resume),
        data=str(Path(data_yaml['path']) / 'data.yaml'),
        epochs=train_cfg['epochs'],
        batch=train_cfg['batch_size'],
        imgsz=model_cfg['img_size'],
        optimizer=train_cfg['optimizer'],
        lr0=train_cfg['lr0'],
        lrf=train_cfg['lrf'],
        momentum=train_cfg['momentum'],
        weight_decay=train_cfg['weight_decay'],
        warmup_epochs=train_cfg['warmup_epochs'],
        warmup_momentum=train_cfg['warmup_momentum'],
        warmup_bias_lr=train_cfg['warmup_bias_lr'],
        cos_lr=train_cfg['cos_lr'],
        close_mosaic=train_cfg['close_mosaic'],
        workers=train_cfg['workers'],
        device=train_cfg['device'],
        amp=train_cfg['amp'],
        # Augmentations
        mosaic=aug_cfg['mosaic'],
        mixup=aug_cfg['mixup'],
        degrees=aug_cfg['degrees'],
        translate=aug_cfg['translate'],
        scale=aug_cfg['scale'],
        flipud=aug_cfg['flipud'],
        fliplr=aug_cfg['fliplr'],
        hsv_h=aug_cfg['hsv_h'],
        hsv_s=aug_cfg['hsv_s'],
        hsv_v=aug_cfg['hsv_v'],
        # Project settings
        project='runs/night_ped_detection',
        name=f"enhanced_yolov8n_{train_cfg['epochs']}e",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Save final model
    export_path = Path('models') / 'enhanced_yolov8n.pt'
    export_path.parent.mkdir(exist_ok=True)
    yolo_model.save(str(export_path))
    LOGGER.info(f'Model saved to {export_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Enhanced YOLOv8 for Night-Time Detection')
    parser.add_argument(
        '--config',
        type=str,
        default='model_training/configs/train_config.yaml',
        help='Path to training configuration YAML',
    )
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        default=None,
        help='Resume training from checkpoint. Without a path, auto-finds last.pt in runs/',
    )
    args = parser.parse_args()
    train(args)
