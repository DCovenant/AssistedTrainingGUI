"""Validate all PNG images for corruption and proper format."""

from pathlib import Path
from PIL import Image

def validate_images(images_dir: str | None = None) -> dict:
    """Check all images for issues.

    Args:
        images_dir: Path to images folder. If None, uses default ml/data/raw_images.

    Returns:
        Dict with validation results
    """
    if images_dir is None:
        project_root = Path(__file__).parent.parent.parent
        images_dir = str(project_root / "ml" / "data" / "raw_images")

    images_path = Path(images_dir)
    results = {
        "total": 0,
        "valid": 0,
        "corrupted": [],
        "grayscale": [],
        "issues": []
    }

    # Validate all image types (PNG, JPG, JPEG)
    image_files = sorted(
        f for ext in ("*.png", "*.jpg", "*.jpeg")
        for f in images_path.glob(ext)
    )
    for image_file in image_files:
        results["total"] += 1
        try:
            img = Image.open(image_file)

            # Check if can be opened
            img.verify()

            # Re-open after verify (verify closes the file)
            img = Image.open(image_file)

            # Check dimensions
            if img.size[0] == 0 or img.size[1] == 0:
                results["issues"].append(f"{image_file.name}: Invalid dimensions {img.size}")
                continue

            # Check color mode
            if img.mode != 'RGB':
                if img.mode == 'L':
                    results["grayscale"].append(image_file.name)
                else:
                    results["issues"].append(f"{image_file.name}: Mode {img.mode} (not RGB)")
                continue

            results["valid"] += 1

        except Exception as e:
            results["corrupted"].append(f"{image_file.name}: {str(e)}")

    return results

if __name__ == "__main__":
    results = validate_images()

    print(f"Total images: {results['total']}")
    print(f"Valid: {results['valid']}")
    print(f"Grayscale: {len(results['grayscale'])}")
    print(f"Corrupted: {len(results['corrupted'])}")
    print(f"Other issues: {len(results['issues'])}")

    if results["grayscale"]:
        print("\n❌ Grayscale images (need to convert to RGB):")
        for img in results["grayscale"][:10]:
            print(f"  - {img}")

    if results["corrupted"]:
        print("\n❌ Corrupted images:")
        for img in results["corrupted"][:10]:
            print(f"  - {img}")

    if results["issues"]:
        print("\n⚠️ Other issues:")
        for issue in results["issues"][:10]:
            print(f"  - {issue}")
