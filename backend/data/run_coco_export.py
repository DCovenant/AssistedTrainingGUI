"""Export annotated dataset to COCO JSON format for Florence-2 training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation_database import AnnotationDatabase
from coco_exporter import export_all_splits_to_coco


def main() -> None:
    """Export all splits to COCO format."""
    database = AnnotationDatabase()
    project_root = Path(__file__).parent.parent.parent
    raw_images_directory = project_root / "ml" / "data" / "raw_images"
    output_directory = project_root / "ml" / "data" / "coco"

    print("Exporting dataset to COCO format...")
    output_paths = export_all_splits_to_coco(output_directory, database, raw_images_directory)

    for split, path in output_paths.items():
        print(f"✓ {split}: {path}")

    print("\nExport complete!")


if __name__ == "__main__":
    main()
