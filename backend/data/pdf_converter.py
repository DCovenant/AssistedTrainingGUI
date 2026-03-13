from pathlib import Path
from typing import Callable
import fitz


def get_converted_pdfs(raw_images_path: str) -> set[str]:
    """Get set of PDF basenames already converted to PNG.

    Args:
        raw_images_path: Folder containing converted PNG images

    Returns:
        Set of PDF stem names (e.g., {'doc1', 'doc2'})
    """
    images_path = Path(raw_images_path)
    if not images_path.exists():
        return set()

    converted = set()
    for png_file in images_path.glob("*.png"):
        # Extract PDF name from PNG name (format: pdf_stem_page_N.png)
        parts = png_file.stem.split("_page_")
        if len(parts) == 2:
            pdf_stem = parts[0]
            converted.add(pdf_stem)

    return converted


def convert_pdfs_to_png(
    raw_pdfs_path: str,
    raw_images_path: str,
    progress_callback: Callable[[int, int], None] | None = None
) -> dict[str, int]:
    """
    Convert all PDF pages to PNG images.

    Args:
        raw_pdfs_path: Folder containing PDF files
        raw_images_path: Folder to save PNG images
        progress_callback: Function called with (current, total) to report progress

    Returns:
        Dict with conversion stats: {total_pdfs, total_pages}
    """
    pdfs_path = Path(raw_pdfs_path)
    images_path = Path(raw_images_path)

    images_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdfs_path.glob("*.pdf"))
    total_pdfs = len(pdf_files)
    total_pages = 0

    for pdf_idx, pdf_file in enumerate(pdf_files):
        try:
            doc = fitz.open(pdf_file)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                image_name = f"{pdf_file.stem}_page_{page_num + 1}.png"
                image_path = images_path / image_name

                pix.save(image_path)
                total_pages += 1

                if progress_callback:
                    progress_callback(pdf_idx + 1, total_pdfs)

            doc.close()
        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}")

    return {"total_pdfs": total_pdfs, "total_pages": total_pages}


def rescan_and_convert_new_pdfs(
    raw_pdfs_path: str,
    raw_images_path: str,
    progress_callback: Callable[[int, int], None] | None = None
) -> dict[str, int]:
    """
    Scan for new PDFs and convert only those not yet converted.

    Args:
        raw_pdfs_path: Folder containing PDF files
        raw_images_path: Folder to save PNG images
        progress_callback: Function called with (current, total) to report progress

    Returns:
        Dict with conversion stats: {total_pdfs, total_pages, new_pdfs, skipped_pdfs}
    """
    pdfs_path = Path(raw_pdfs_path)
    images_path = Path(raw_images_path)

    images_path.mkdir(parents=True, exist_ok=True)

    # Get all PDFs and already-converted PDFs
    pdf_files = sorted(pdfs_path.glob("*.pdf"))
    converted_stems = get_converted_pdfs(raw_images_path)

    # Filter to only new PDFs
    new_pdfs = [pdf for pdf in pdf_files if pdf.stem not in converted_stems]

    total_new = len(new_pdfs)
    total_skipped = len(pdf_files) - total_new
    total_pages = 0

    for pdf_idx, pdf_file in enumerate(new_pdfs):
        try:
            doc = fitz.open(pdf_file)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

                image_name = f"{pdf_file.stem}_page_{page_num + 1}.png"
                image_path = images_path / image_name

                pix.save(image_path)
                total_pages += 1

                if progress_callback:
                    progress_callback(pdf_idx + 1, total_new)

            doc.close()
        except Exception as e:
            print(f"Error converting {pdf_file.name}: {e}")

    return {
        "total_pdfs": len(pdf_files),
        "new_pdfs": total_new,
        "skipped_pdfs": total_skipped,
        "total_pages": total_pages
    }
