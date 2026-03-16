from backend.data.annotation_database import AnnotationDatabase
import random


def create_initial_split(
    all_image_ids: list[str],
    train_percentage: int,
    dev_percentage: int,
    test_percentage: int
) -> dict[str, int]:
    """Create train/dev/test split for labeled images only.

    Splits annotated (labeled) images into train/dev/test by percentages.
    Unannotated (unlabeled) images left unassigned for active learning.
    Call once at start of active learning loop.

    Args:
        all_image_ids: All image filenames available
        train_percentage: Percentage of labeled images for training
        dev_percentage: Percentage of labeled images for validation
        test_percentage: Percentage of labeled images for testing

    Returns:
        Dict with counts: {train_count, dev_count, test_count, unlabeled_count}
    """
    db = AnnotationDatabase()

    labeled_images = db.get_annotated_images()
    random.seed(42)
    random.shuffle(labeled_images)
    unlabeled_images = db.get_unannotated_images(all_image_ids)

    total_labeled = len(labeled_images)
    train_count = int(total_labeled * train_percentage / 100)
    dev_count = int(total_labeled * dev_percentage / 100)

    train_images = labeled_images[:train_count]
    dev_images = labeled_images[train_count : train_count + dev_count]
    test_images = labeled_images[train_count + dev_count :]

    assign_images_to_split(train_images, "train", db)
    assign_images_to_split(dev_images, "dev", db)
    assign_images_to_split(test_images, "test", db)

    return {
        "train_count": len(train_images),
        "dev_count": len(dev_images),
        "test_count": len(test_images),
        "unlabeled_count": len(unlabeled_images)
    }


def add_labeled_to_train(image_ids: list[str]) -> int:
    """Add newly labeled images to training set.

    Used in active learning loop: add corrected predictions to training set.

    Args:
        image_ids: List of image filenames to add

    Returns:
        Count of images added
    """
    db = AnnotationDatabase()
    assign_images_to_split(image_ids, "train", db)
    return len(image_ids)


def assign_images_to_split(image_ids: list[str], split: str, db: AnnotationDatabase) -> None:
    """Assign images to a split in database."""
    for image_id in image_ids:
        db.add_image_split(image_id, split)
