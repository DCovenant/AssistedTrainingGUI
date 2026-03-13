"""Validate COCO JSON files for invalid annotations."""

import json
from pathlib import Path

def validate_coco(coco_json_path: str) -> dict:
    """Check COCO JSON for invalid bounding boxes.

    Args:
        coco_json_path: Path to COCO JSON file

    Returns:
        Dict with validation results
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    results = {
        "total_images": len(coco_data['images']),
        "total_annotations": len(coco_data['annotations']),
        "invalid_bbox": [],
        "out_of_bounds": []
    }

    # Build image lookup
    id_to_image = {img['id']: img for img in coco_data['images']}

    for ann_idx, ann in enumerate(coco_data['annotations']):
        x, y, w, h = ann['bbox']
        img_id = ann['image_id']
        img = id_to_image.get(img_id)

        # Check for invalid dimensions
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            results["invalid_bbox"].append({
                "annotation_id": ann['id'],
                "image_id": img_id,
                "bbox": [x, y, w, h],
                "issue": "Invalid width/height or negative coords"
            })

        # Check bounds
        if img:
            if x + w > img['width'] or y + h > img['height']:
                results["out_of_bounds"].append({
                    "annotation_id": ann['id'],
                    "image_id": img_id,
                    "image": img['file_name'],
                    "bbox": [x, y, w, h],
                    "image_size": [img['width'], img['height']]
                })

    return results

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    for split in ["train", "dev", "test"]:
        json_path = project_root / "ml" / "data" / "coco" / f"coco_{split}.json"
        if json_path.exists():
            print(f"\n{'='*50}")
            print(f"Validating {split.upper()}")
            print(f"{'='*50}")

            results = validate_coco(str(json_path))
            print(f"Total images: {results['total_images']}")
            print(f"Total annotations: {results['total_annotations']}")
            print(f"Invalid bbox: {len(results['invalid_bbox'])}")
            print(f"Out of bounds: {len(results['out_of_bounds'])}")

            if results["invalid_bbox"]:
                print(f"\n❌ Invalid bounding boxes:")
                for item in results["invalid_bbox"][:5]:
                    print(f"  Ann {item['annotation_id']}: {item['bbox']} - {item['issue']}")

            if results["out_of_bounds"]:
                print(f"\n⚠️ Out of bounds annotations:")
                for item in results["out_of_bounds"][:5]:
                    print(f"  Ann {item['annotation_id']} ({item['image']}): bbox {item['bbox']} vs image {item['image_size']}")
