import sqlite3
from pathlib import Path
from typing import Optional


class AnnotationDatabase:
    """Manage annotation data in SQLite database."""

    DB_PATH = Path(__file__).parent.parent.parent / "ml" / "data" / "data.db"

    def __init__(self) -> None:
        """Initialize database connection."""
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.connection: sqlite3.Connection | None = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create database and tables if they don't exist."""
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                color TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                dataset TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (class_id) REFERENCES classes(id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute("PRAGMA table_info(annotations)")
        columns = [col[1] for col in cursor.fetchall()]
        if "text" not in columns:
            cursor.execute("ALTER TABLE annotations ADD COLUMN text TEXT")
            conn.commit()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_splits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT UNIQUE NOT NULL,
                split TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign key constraints enabled."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.DB_PATH)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
        return self.connection

    def add_class(self, name: str, color: str) -> int:
        """Add class to database. Returns class id."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO classes (name, color) VALUES (?, ?)",
            (name, color)
        )
        conn.commit()
        return cursor.lastrowid

    def get_class_by_name(self, name: str) -> Optional[dict]:
        """Get class by name."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, color FROM classes WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all_classes(self) -> list[dict]:
        """Get all classes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, color FROM classes ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]

    def add_annotation(
        self, image_id: str, class_id: int, dataset: str, x: int, y: int, width: int, height: int, text: str = ""
    ) -> int:
        """Add annotation to database. Returns annotation id.

        Args:
            image_id: Image filename
            class_id: Class id from classes table
            dataset: Dataset name ("train", "dev", or "test")
            x, y, width, height: Pixel coordinates (integers)
            text: Text content found in annotation region
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO annotations (image_id, class_id, dataset, x, y, width, height, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, class_id, dataset, x, y, width, height, text)
        )
        conn.commit()
        return cursor.lastrowid

    def get_annotations_by_image(self, image_id: str, dataset: str) -> list[dict]:
        """Get all annotations for image in specific dataset.

        Args:
            image_id: Image filename
            dataset: Dataset name ("train" or "dev")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT a.id, a.image_id, a.class_id, a.x, a.y, a.width, a.height, a.text,
                   c.name, c.color
            FROM annotations a
            JOIN classes c ON a.class_id = c.id
            WHERE a.image_id = ? AND a.dataset = ?
            ORDER BY a.id
            """,
            (image_id, dataset)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_annotations_by_class(self, class_id: int) -> list[dict]:
        """Get all annotations for class."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT a.id, a.image_id, a.class_id, a.x, a.y, a.width, a.height
            FROM annotations a
            WHERE a.class_id = ?
            ORDER BY a.image_id
            """,
            (class_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete annotation by id. Returns success."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        conn.commit()
        return cursor.rowcount > 0

    def delete_annotations_by_image(self, image_id: str) -> int:
        """Delete all annotations for an image. Returns count deleted."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        conn.commit()
        return cursor.rowcount

    def delete_class(self, class_id: int) -> bool:
        """Delete class by id (cascades to annotations). Returns success."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM classes WHERE id = ?", (class_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_annotated_images(self) -> list[str]:
        """Get list of unique image IDs that have annotations (labeled images)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT image_id FROM annotations ORDER BY image_id"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_unannotated_images(self, all_image_ids: list[str]) -> list[str]:
        """Get images without annotations (unlabeled images).

        Args:
            all_image_ids: All image filenames available

        Returns:
            List of image IDs with no annotations
        """
        annotated = set(self.get_annotated_images())
        return [img_id for img_id in all_image_ids if img_id not in annotated]

    def add_image_split(self, image_id: str, split: str) -> int:
        """Add or update image split assignment. Returns split id."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR REPLACE INTO image_splits (image_id, split) VALUES (?, ?)",
            (image_id, split)
        )
        conn.commit()
        return cursor.lastrowid

    def get_image_split(self, image_id: str) -> Optional[str]:
        """Get split assignment for image."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT split FROM image_splits WHERE image_id = ?", (image_id,))
        row = cursor.fetchone()
        return row[0] if row else None

    def clear_image_splits(self) -> None:
        """Clear all image split assignments."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM image_splits")
        conn.commit()

    def get_images_by_split(self, split: str) -> list[str]:
        """Get all images assigned to a specific split."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT image_id FROM image_splits WHERE split = ? ORDER BY image_id",
            (split,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_annotations_by_split(self, split: str) -> list[dict]:
        """Get all annotations for images in a specific split.

        Args:
            split: Split name ("train", "dev", or "test")

        Returns:
            List of annotations with class data
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT a.id, a.image_id, a.class_id, a.x, a.y, a.width, a.height, a.text,
                   c.name, c.color
            FROM annotations a
            JOIN classes c ON a.class_id = c.id
            JOIN image_splits s ON a.image_id = s.image_id
            WHERE s.split = ?
            ORDER BY a.image_id, a.id
            """,
            (split,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def load_data(self) -> dict:
        """Load all annotations organized by train/dev/test splits.

        Returns:
            {
                'classes': [...],
                'train': [...],
                'dev': [...],
                'test': [...]
            }
            Each item in splits contains:
            {
                'image_path': Path,
                'image_id': str,
                'annotation_id': int,
                'class_id': int,
                'class_name': str,
                'bbox': {'x': float, 'y': float, 'width': float, 'height': float},
                'text': str
            }
        """
        base_path = Path(__file__).parent.parent.parent / "ml" / "data" / "raw_images"
        data = {
            'classes': self.get_all_classes(),
            'train': [],
            'dev': [],
            'test': []
        }

        # Load annotations for each split
        for split in ['train', 'dev', 'test']:
            annotations = self.get_annotations_by_split(split)

            for annotation in annotations:
                data[split].append({
                    'image_path': base_path / annotation['image_id'],
                    'image_id': annotation['image_id'],
                    'annotation_id': annotation['id'],
                    'class_id': annotation['class_id'],
                    'class_name': annotation['name'],
                    'bbox': {
                        'x': annotation['x'],
                        'y': annotation['y'],
                        'width': annotation['width'],
                        'height': annotation['height']
                    },
                    'text': annotation['text']
                })

        return data
