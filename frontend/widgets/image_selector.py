import shutil
from pathlib import Path
from random import shuffle
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QSplitter, QLineEdit, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSignal, Qt


class ImageSelector(QMainWindow):
    """Select which images from raw_images folder to use."""

    selection_changed = pyqtSignal(str, bool)
    done_selecting = pyqtSignal()

    def __init__(self, images_folder: str) -> None:
        super().__init__()
        self.setWindowTitle("Select Images to Use")
        self.setGeometry(100, 100, 1400, 800)

        self.images_folder = Path(images_folder)
        # Collect all image files (PNG, JPG, JPEG)
        self.image_files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in self.images_folder.glob(ext)
        )
        self.current_index = 0
        self.selected_images: set[str] = set()

        self._setup_ui()
        if self.image_files:
            self._load_image(0)

    def _setup_ui(self) -> None:
        """Create UI layout with image viewer and selection lists."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")
        splitter.addWidget(self.image_label)

        right_panel = self._create_selection_lists_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 70)
        splitter.setStretchFactor(1, 30)

        main_layout.addWidget(splitter, 1)

        self.counter_label = QLabel()
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.counter_label)

        button_layout = self._create_button_controls()
        main_layout.addLayout(button_layout)

    def _create_selection_lists_panel(self) -> QWidget:
        """Create panel with dataset percentages and selection lists."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(self._create_dataset_percentages_panel())

        considered_label = QLabel("Considered")
        considered_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(considered_label)

        self.considered_list = QListWidget()
        self.considered_list.itemClicked.connect(self._on_considered_item_clicked)
        layout.addWidget(self.considered_list)

        not_considered_label = QLabel("Not Considered")
        not_considered_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(not_considered_label)

        self.not_considered_list = QListWidget()
        self.not_considered_list.itemClicked.connect(self._on_not_considered_item_clicked)
        layout.addWidget(self.not_considered_list)

        return container

    def _create_dataset_percentages_panel(self) -> QWidget:
        """Create panel with dataset split percentages."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("Datasets percentages:")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        percentages_layout = QHBoxLayout()

        train_label = QLabel("Train:")
        self.train_input = QLineEdit()
        self.train_input.setText("70")
        self.train_input.setMaximumWidth(50)
        percentages_layout.addWidget(train_label)
        percentages_layout.addWidget(self.train_input)

        dev_label = QLabel("Dev:")
        self.dev_input = QLineEdit()
        self.dev_input.setText("15")
        self.dev_input.setMaximumWidth(50)
        percentages_layout.addWidget(dev_label)
        percentages_layout.addWidget(self.dev_input)

        test_label = QLabel("Test:")
        self.test_input = QLineEdit()
        self.test_input.setText("15")
        self.test_input.setMaximumWidth(50)
        percentages_layout.addWidget(test_label)
        percentages_layout.addWidget(self.test_input)

        percentages_layout.addStretch()
        layout.addLayout(percentages_layout)

        return container

    def _create_button_controls(self) -> QHBoxLayout:
        """Create navigation and selection buttons."""
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("<")
        self.prev_button.clicked.connect(self._go_previous)
        button_layout.addWidget(self.prev_button)

        self.reject_button = QPushButton("Dont consider")
        self.reject_button.clicked.connect(self._reject_image)
        button_layout.addWidget(self.reject_button)

        self.accept_button = QPushButton("Consider")
        self.accept_button.clicked.connect(self._accept_image)
        button_layout.addWidget(self.accept_button)

        self.next_button = QPushButton(">")
        self.next_button.clicked.connect(self._go_next)
        button_layout.addWidget(self.next_button)

        self.done_button = QPushButton("Done")
        self.done_button.clicked.connect(self._on_done_clicked)
        button_layout.addWidget(self.done_button)

        return button_layout

    def _load_image(self, index: int) -> None:
        """Load and display image at index."""
        if 0 <= index < len(self.image_files):
            self.current_index = index
            image_path = self.image_files[index]

            pixmap = QPixmap(str(image_path))
            scaled = pixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)

            self._update_counter()
            self._update_button_states()

    def _update_counter(self) -> None:
        """Show current position."""
        total = len(self.image_files)
        current = self.current_index + 1
        self.counter_label.setText(f"{current} / {total}")

    def _update_button_states(self) -> None:
        """Enable/disable navigation buttons."""
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.image_files) - 1)

    def _go_previous(self) -> None:
        """Go to previous image."""
        if self.current_index > 0:
            self._load_image(self.current_index - 1)

    def _go_next(self) -> None:
        """Go to next image."""
        if self.current_index < len(self.image_files) - 1:
            self._load_image(self.current_index + 1)

    def _accept_image(self) -> None:
        """Mark image as considered and update lists."""
        current_path = str(self.image_files[self.current_index])
        self.selected_images.add(current_path)
        self.selection_changed.emit(current_path, True)

        self._update_selection_lists()

        if self.current_index < len(self.image_files) - 1:
            self._go_next()

    def _reject_image(self) -> None:
        """Mark image as not considered and update lists."""
        current_path = str(self.image_files[self.current_index])
        self.selected_images.discard(current_path)
        self.selection_changed.emit(current_path, False)

        self._update_selection_lists()

        if self.current_index < len(self.image_files) - 1:
            self._go_next()

    def _update_selection_lists(self) -> None:
        """Refresh considered and not considered lists."""
        self.considered_list.clear()
        self.not_considered_list.clear()

        for image_path in self.image_files:
            image_name = Path(image_path).name
            if str(image_path) in self.selected_images:
                self.considered_list.addItem(image_name)
            else:
                self.not_considered_list.addItem(image_name)

    def _on_considered_item_clicked(self, item) -> None:
        """Load image when considered list item is clicked."""
        image_name = item.text()
        self._load_image_by_name(image_name)

    def _on_not_considered_item_clicked(self, item) -> None:
        """Load image when not considered list item is clicked."""
        image_name = item.text()
        self._load_image_by_name(image_name)

    def _load_image_by_name(self, image_name: str) -> None:
        """Load image by filename."""
        for idx, image_path in enumerate(self.image_files):
            if Path(image_path).name == image_name:
                self._load_image(idx)
                break

    def _on_done_clicked(self) -> None:
        """Validate percentages and split images if confirmed."""
        try:
            train = int(self.train_input.text())
            dev = int(self.dev_input.text())
            test = int(self.test_input.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter valid numbers for Train, Dev, and Test percentages."
            )
            return

        total = train + dev + test

        if total != 100:
            QMessageBox.warning(
                self,
                "Invalid Distribution",
                f"The sum of percentages must equal 100.\nCurrent sum: {total}"
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Split",
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._split_images(train, dev, test)

    def _split_images(self, train_pct: int, dev_pct: int, test_pct: int) -> None:
        """Split selected images into train/dev/test folders."""
        project_root = Path(__file__).parent.parent.parent
        splits_path = project_root / "ml" / "data" / "splits"
        train_path = splits_path / "train"
        dev_path = splits_path / "dev"
        test_path = splits_path / "test"

        self._clear_split_folders(train_path, dev_path, test_path)

        train_path.mkdir(parents=True, exist_ok=True)
        dev_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

        selected_paths = list(self.selected_images)
        shuffle(selected_paths)

        total_images = len(selected_paths)
        train_count = int(total_images * train_pct / 100)
        dev_count = int(total_images * dev_pct / 100)

        train_images = selected_paths[:train_count]
        dev_images = selected_paths[train_count : train_count + dev_count]
        test_images = selected_paths[train_count + dev_count :]

        for image_path in train_images:
            shutil.copy(image_path, train_path / Path(image_path).name)

        for image_path in dev_images:
            shutil.copy(image_path, dev_path / Path(image_path).name)

        for image_path in test_images:
            shutil.copy(image_path, test_path / Path(image_path).name)

        QMessageBox.information(
            self,
            "Split Complete",
            f"Images split successfully!\n\nTrain: {len(train_images)}\nDev: {len(dev_images)}\nTest: {len(test_images)}"
        )

        self.done_selecting.emit()
        self.close()

    def _clear_split_folders(
        self, train_path: Path, dev_path: Path, test_path: Path
    ) -> None:
        """Remove existing files from split folders."""
        for folder in [train_path, dev_path, test_path]:
            if folder.exists():
                for ext in ("*.png", "*.jpg", "*.jpeg"):
                    for file in folder.glob(ext):
                        file.unlink()

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        key = event.text().lower()

        if key == "a":
            self._go_previous()
        elif key == "d":
            self._go_next()
        elif key == "w":
            self._accept_image()
        elif key == "s":
            self._reject_image()
        else:
            super().keyPressEvent(event)

    def get_selected_images(self) -> set[str]:
        """Return paths of selected images."""
        return self.selected_images
