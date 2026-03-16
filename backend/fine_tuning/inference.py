"""
Inference module for CLIP-based object detection.

Sliding window approach:
1. Slide a window across the image at multiple scales
2. Classify each window crop with the trained CLIP model
3. Keep high-confidence predictions
4. Remove overlapping boxes with Non-Max Suppression (NMS)
"""

import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from backend.fine_tuning.training import CLIPDetector
from backend.fine_tuning.background_generator import WINDOW_SIZES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "openai/clip-vit-base-patch32"


def load_model(model_path: str) -> tuple[CLIPDetector, CLIPProcessor, list[dict]]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    num_classes = checkpoint['num_classes']
    categories = checkpoint['categories']

    clip_model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    model = CLIPDetector(clip_model, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (Dev Acc: {checkpoint['dev_accuracy']:.1f}%, "
          f"Epoch: {checkpoint['epoch']})")

    return model, processor, categories


def sliding_window(image_width: int, image_height: int,
                   window_sizes: list[tuple[int, int]],
                   stride_ratio: float = 0.5) -> list[tuple[int, int, int, int]]:
    """
    Generate sliding window positions across the image.

    Args:
        image_width, image_height: Image dimensions
        window_sizes: List of (width, height) window sizes to scan
        stride_ratio: How much to move each step (0.5 = 50% overlap)

    Returns:
        List of (x, y, width, height) window positions
    """
    windows = []
    for win_w, win_h in window_sizes:
        stride_x = max(1, int(win_w * stride_ratio))
        stride_y = max(1, int(win_h * stride_ratio))

        for y in range(0, image_height - win_h + 1, stride_y):
            for x in range(0, image_width - win_w + 1, stride_x):
                windows.append((x, y, win_w, win_h))

    return windows


def nms(predictions: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """
    Non-Max Suppression: remove overlapping boxes, keep highest confidence.

    If two boxes overlap more than iou_threshold, keep only the one
    with higher confidence.
    """
    if not predictions:
        return []

    # Sort by confidence (highest first)
    preds = sorted(predictions, key=lambda p: p['confidence'], reverse=True)
    keep = []

    while preds:
        best = preds.pop(0)
        keep.append(best)

        # Remove boxes that overlap too much with the best one
        remaining = []
        for pred in preds:
            if _iou(best, pred) < iou_threshold:
                remaining.append(pred)
        preds = remaining

    return keep


def _iou(a: dict, b: dict) -> float:
    """Calculate Intersection over Union between two boxes."""
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['width'], ay1 + a['height']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['width'], by1 + b['height']

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = a['width'] * a['height']
    area_b = b['width'] * b['height']
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def run_inference(image_path: str, model_path: str,
                  confidence_threshold: float = 0.6,
                  batch_size: int = 64) -> list[dict]:
    """
    Run object detection on a single image.

    Args:
        image_path: Path to image file
        model_path: Path to trained model checkpoint
        confidence_threshold: Minimum confidence to keep a prediction (default 0.5).
                             Lowered from 0.7 to account for class imbalance weighting
        batch_size: How many windows to classify at once

    Returns:
        List of predictions: [{x, y, width, height, class_name, confidence}, ...]
    """
    print(f"\nRunning inference on {Path(image_path).name}...")

    model, processor, categories = load_model(model_path)
    image = Image.open(image_path).convert('L').convert('RGB')
    img_w, img_h = image.size

    window_sizes = WINDOW_SIZES

    windows = sliding_window(img_w, img_h, window_sizes, stride_ratio=0.7)
    print(f"Scanning {len(windows)} windows...")

    # Classify windows in batches
    predictions = []

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i + batch_size]

        # Crop all windows
        crops = []
        for x, y, w, h in batch_windows:
            crop = image.crop((x, y, x + w, y + h))
            crops.append(crop)

        # Process batch through CLIP
        inputs = processor(images=crops, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        with torch.no_grad():
            logits = model(pixel_values)
            class_indices = logits.argmax(dim=1)
            confidences = torch.sigmoid(logits.max(dim=1).values)

        # Keep high-confidence predictions (skip background class 0)
        for j, (x, y, w, h) in enumerate(batch_windows):
            conf = confidences[j].item()
            cls_idx = class_indices[j].item()

            # Skip background class (index 0)
            if cls_idx == 0:
                continue

            if conf >= confidence_threshold:
                predictions.append({
                    'x': x, 'y': y,
                    'width': w, 'height': h,
                    'class_name': categories[cls_idx]['name'],
                    'confidence': conf
                })

    # Remove overlapping boxes
    predictions = nms(predictions, iou_threshold=0.1)

    print(f"Found {len(predictions)} predictions")
    return predictions
