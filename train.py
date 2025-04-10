EPOCHS = 100
MOSAIC = 0.7
OPTIMIZER = 'AdamW'
MOMENTUM = 0.85
LR0 = 0.001
LRF = 0.1
SINGLE_CLS = False
IMGSZ = 640     # Reduced from 1024 to save memory
BATCH = 8       # Reduced from 16 to save memory

import argparse
from ultralytics import YOLO
import os
import sys
import torch

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    parser.add_argument('--batch', type=int, default=BATCH, help='Batch size')
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Select device and set memory allocation
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu':
        # Configure PyTorch CUDA memory allocation
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    print(f"Using device: {device}")
    
    # Load YOLOv8x model
    model = YOLO('yolov8x.pt')

    # Advanced training configuration with memory optimization
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=device,
        single_cls=args.single_cls,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=0.01,
        imgsz=args.imgsz,
        batch=args.batch,
        cos_lr=True,           # Cosine LR scheduler
        close_mosaic=10,       # Disable mosaic in final epochs
        mosaic=args.mosaic,    # Mosaic augmentation
        hsv_h=0.015,          # HSV augmentation
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.2,        # Translation augmentation
        scale=0.9,           # Scaling augmentation
        fliplr=0.5,          # Horizontal flip
        flipud=0.1,          # Vertical flip
        perspective=0.0005,   # Perspective augmentation
        mixup=0.15,          # Mixup augmentation
        copy_paste=0.1,      # Copy-paste augmentation
        cache=True,          # Cache images for faster training
        amp=True,            # Mixed precision training
        rect=False,          # Rectangular training
        overlap_mask=True,   # Mask overlap
        mask_ratio=4,        # Mask downsample ratio
        degrees=0.0,         # Image rotation (+/- deg)
        shear=0.0,          # Image shear (+/- deg)
        warmup_epochs=5,     # Warmup epochs
        warmup_momentum=0.8, # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias lr
        box=7.5,            # Box loss gain
        cls=0.5,            # Cls loss gain
        dfl=1.5,            # DFL loss gain
        plots=True,         # Save plots during training
        save=True,          # Save trained model
        save_period=10,     # Save checkpoint every x epochs
        project='runs/train',
        name='yolov8x_optimized'
    )

    print("Training completed successfully!")


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''