"""Generate background training crops from unselected image regions."""

# Window sizes — larger sizes only to reduce false positives
WINDOW_SIZES = [
    (190, 180),  # medium — ~P50
    (250, 230),  # large — ~P75
    (320, 300),  # extra-large — ~P90
]


def calc_iou(crop_x: int, crop_y: int, crop_w: int, crop_h: int,
             ann_bbox: list[float]) -> float:
    """Calculate intersection over union between crop and annotation.

    Args:
        crop_x, crop_y, crop_w, crop_h: Crop bounding box
        ann_bbox: [x, y, width, height] annotation from COCO

    Returns:
        IoU value between 0 and 1
    """
    ann_x, ann_y, ann_w, ann_h = ann_bbox

    ix1, iy1 = max(crop_x, ann_x), max(crop_y, ann_y)
    ix2, iy2 = min(crop_x + crop_w, ann_x + ann_w), min(crop_y + crop_h, ann_y + ann_h)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = crop_w * crop_h + ann_w * ann_h - intersection
    return intersection / union if union > 0 else 0.0


def is_background_crop(crop_x: int, crop_y: int, crop_w: int, crop_h: int,
                       annotations: list[list[float]], iou_threshold: float = 0.1) -> bool:
    """Check if crop is valid background (doesn't overlap annotations significantly).

    Args:
        crop_x, crop_y, crop_w, crop_h: Crop coordinates
        annotations: List of [x, y, width, height] bounding boxes
        iou_threshold: Maximum IoU allowed with any annotation

    Returns:
        True if crop qualifies as background
    """
    for ann_bbox in annotations:
        if calc_iou(crop_x, crop_y, crop_w, crop_h, ann_bbox) > iou_threshold:
            return False
    return True


def generate_background_crops(image_width: int, image_height: int,
                              annotations: list[list[float]],
                              crop_sizes: list[tuple[int, int]] = None,
                              stride_ratio: float = 0.7,
                              max_bg_per_image: int = 200) -> list[dict]:
    """Generate background crops by systematic grid covering entire image.

    Covers all unselected regions by sliding a grid across the image,
    keeping only crops that don't significantly overlap annotations.
    Capped at max_bg_per_image to prevent extreme class imbalance.

    Args:
        image_width, image_height: Image dimensions
        annotations: List of [x, y, width, height] annotated regions
        crop_sizes: Window sizes to generate, defaults to common sizes
        stride_ratio: Grid step size as fraction of window (0.3 = 30% of window)
        max_bg_per_image: Maximum background crops per image (prevents data imbalance)

    Returns:
        List of background crop dicts with keys: bbox, label
    """
    if crop_sizes is None:
        crop_sizes = WINDOW_SIZES

    bg_crops = []

    for crop_w, crop_h in crop_sizes:
        # Skip if crop size is invalid or if already at max
        if crop_w <= 0 or crop_h <= 0 or len(bg_crops) >= max_bg_per_image:
            continue

        stride_x = max(1, int(crop_w * stride_ratio))
        stride_y = max(1, int(crop_h * stride_ratio))

        y = 0
        while y + crop_h <= image_height and len(bg_crops) < max_bg_per_image:
            x = 0
            while x + crop_w <= image_width and len(bg_crops) < max_bg_per_image:
                if is_background_crop(x, y, crop_w, crop_h, annotations):
                    bg_crops.append({
                        'bbox': [x, y, crop_w, crop_h],
                        'label': 0
                    })
                x += stride_x
            y += stride_y

    return bg_crops
