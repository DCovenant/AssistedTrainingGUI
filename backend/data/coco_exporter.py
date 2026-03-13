import json
from pathlib import Path
from typing import TypedDict

from .annotation_database import AnnotationDatabase
from .image_metadata import get_image_dimensions  # Used in build_coco_images()


class COCOImage(TypedDict):
    """COCO format image metadata."""
    id: int
    file_name: str
    width: int
    height: int


class COCOAnnotation(TypedDict):
    """COCO format bounding box annotation."""
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: float
    iscrowd: int


class COCOCategory(TypedDict):
    """COCO format class category."""
    id: int
    name: str


class COCODataset(TypedDict):
    """Complete COCO format dataset."""
    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    categories: list[COCOCategory]


def build_coco_images(
    image_ids: list[str],
    raw_images_directory: Path
) -> tuple[list[COCOImage], dict[str, int]]:
    """Create COCO image metadata from image files in raw_images.

    Args:
        image_ids: List of image filenames in split
        raw_images_directory: Path to raw_images folder (all images)

    Returns:
        (coco_images list, mapping of filename to image_id)
    """
    coco_images: list[COCOImage] = []
    filename_to_id: dict[str, int] = {}

    for image_idx, image_name in enumerate(image_ids, start=1):
        image_path = raw_images_directory / image_name
        width, height = get_image_dimensions(image_path)

        coco_images.append({
            "id": image_idx,
            "file_name": image_name,
            "width": width,
            "height": height
        })
        filename_to_id[image_name] = image_idx

    return coco_images, filename_to_id


def build_coco_annotations(
    database_annotations: list[dict],
    filename_to_image_id: dict[str, int]
) -> list[COCOAnnotation]:
    """Create COCO annotation objects from database records.

    Args:
        database_annotations: Annotations from SQLite database (pixel coordinates)
        filename_to_image_id: Mapping of filenames to COCO image IDs

    Returns:
        List of COCO annotations
    """
    coco_annotations: list[COCOAnnotation] = []

    for ann_idx, annotation in enumerate(database_annotations, start=1):
        image_name = annotation["image_id"]
        image_id = filename_to_image_id[image_name]

        area = annotation["width"] * annotation["height"]

        coco_annotations.append({
            "id": ann_idx,
            "image_id": image_id,
            "category_id": annotation["class_id"],
            "bbox": [annotation["x"], annotation["y"], annotation["width"], annotation["height"]],
            "area": area,
            "iscrowd": 0
        })

    return coco_annotations


def build_coco_categories(database_classes: list[dict]) -> list[COCOCategory]:
    """Create COCO category objects from database classes.

    Args:
        database_classes: Classes from SQLite database

    Returns:
        List of COCO categories
    """
    return [
        {"id": cls["id"], "name": cls["name"]}
        for cls in database_classes
    ]


def export_split_to_coco(
    split: str,
    output_path: Path,
    database: AnnotationDatabase,
    raw_images_directory: Path
) -> None:
    """Export single dataset split to COCO JSON format.

    Args:
        split: Dataset split name ("train", "dev", or "test")
        output_path: Where to save COCO JSON file
        database: AnnotationDatabase instance
        raw_images_directory: Path to raw_images folder (contains all images)
    """
    image_ids = database.get_images_by_split(split)
    coco_images, filename_to_image_id = build_coco_images(image_ids, raw_images_directory)

    database_annotations = database.get_annotations_by_split(split)
    coco_annotations = build_coco_annotations(database_annotations, filename_to_image_id)

    database_classes = database.get_all_classes()
    coco_categories = build_coco_categories(database_classes)

    coco_dataset: COCODataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }

    with open(output_path, "w") as f:
        json.dump(coco_dataset, f, indent=2)


def export_all_splits_to_coco(
    output_directory: Path,
    database: AnnotationDatabase,
    raw_images_directory: Path
) -> dict[str, Path]:
    """Export all dataset splits to COCO JSON format.

    Args:
        output_directory: Where to save COCO JSON files
        database: AnnotationDatabase instance
        raw_images_directory: Path to raw_images folder

    Returns:
        Mapping of split names to output file paths
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    for split in ["train", "dev", "test"]:
        output_path = output_directory / f"coco_{split}.json"
        export_split_to_coco(split, output_path, database, raw_images_directory)
        output_paths[split] = output_path

    return output_paths
