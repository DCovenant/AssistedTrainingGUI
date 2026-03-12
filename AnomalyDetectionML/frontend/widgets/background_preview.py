"""Visualize background regions that will be extracted during training."""

from PyQt6.QtCore import QRect


def get_annotation_regions(annotations: list[dict]) -> list[QRect]:
    """Convert annotations to QRect regions to exclude from background overlay.

    Args:
        annotations: List of annotation dicts with x, y, width, height keys

    Returns:
        List of QRect objects representing annotated regions
    """
    regions = []
    for ann in annotations:
        rect = QRect(int(ann['x']), int(ann['y']),
                     int(ann['width']), int(ann['height']))
        regions.append(rect)
    return regions
