from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataStatus:
    """Data availability check result"""
    has_raw_images: bool
    has_raw_pdfs: bool
    raw_images_path: Path
    raw_pdfs_path: Path

    @property
    def status_message(self) -> str:
        """Human-readable status for UI"""
        if self.has_raw_images:
            return "Raw images available for annotation"
        elif self.has_raw_pdfs:
            return "Only PDFs found (no images)"
        else:
            return "No data found"


def check_data_availability(base_path: str | None = None) -> DataStatus:
    """Check if raw images or PDFs are available.

    Args:
        base_path: Root data directory. If None, uses default ml/data relative to this module.

    Returns:
        DataStatus with availability of raw_images and raw_pdfs
    """
    if base_path is None:
        base_path = str(Path(__file__).parent.parent.parent / "ml" / "data")
    base = Path(base_path)
    raw_images_path = base / "raw_images"
    raw_pdfs_path = base / "raw_pdfs"

    has_raw_images = (
        _folder_has_files(raw_images_path, "*.png")
        or _folder_has_files(raw_images_path, "*.jpg")
        or _folder_has_files(raw_images_path, "*.jpeg")
    )
    has_raw_pdfs = _folder_has_files(raw_pdfs_path, "*.pdf")

    return DataStatus(
        has_raw_images=has_raw_images,
        has_raw_pdfs=has_raw_pdfs,
        raw_images_path=raw_images_path,
        raw_pdfs_path=raw_pdfs_path
    )


def _folder_has_files(folder_path: Path, pattern: str) -> bool:
    """Check if folder exists and contains files matching pattern."""
    if not folder_path.exists():
        return False
    return bool(list(folder_path.glob(pattern)))
