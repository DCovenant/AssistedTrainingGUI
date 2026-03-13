"""Profile DataLoader and identify bottleneck."""

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from backend.fine_tuning.training import COCOCropDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

project_root = Path(__file__).parent.parent.parent
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
train_json = str(project_root / "ml" / "data" / "coco" / "coco_train.json")
train_dataset = COCOCropDataset(train_json, processor)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True,
    num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=True
)

# Warm up
for batch_idx, (pixel_values, labels) in enumerate(train_loader):
    if batch_idx >= 2:
        break

# Profile
start = time.time()
for batch_idx, (pixel_values, labels) in enumerate(train_loader):
    if batch_idx >= 10:
        break

    # Measure data transfer
    transfer_start = time.time()
    pixel_values = pixel_values.to(device)
    labels = labels.to(device)
    torch.cuda.synchronize()
    transfer_time = time.time() - transfer_start

    print(f"Batch {batch_idx}: Transfer {transfer_time*1000:.1f}ms, "
          f"Pixel shape {pixel_values.shape}, Labels {labels.shape}")

avg_time = (time.time() - start) / 10
print(f"\nAvg time per batch: {avg_time*1000:.1f}ms")
print(f"Throughput: {32*10/avg_time:.0f} samples/sec")
