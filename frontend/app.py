import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QListWidget, QLabel, QMessageBox, QProgressDialog, QPushButton, QDialog,
    QScrollArea, QFileDialog, QComboBox
)
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRect
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.data.data_checker import check_data_availability
from backend.data.pdf_converter import convert_pdfs_to_png, rescan_and_convert_new_pdfs
from backend.data.annotation_database import AnnotationDatabase
from backend.data.dataset_splitter import create_initial_split
from backend.data.coco_exporter import export_all_splits_to_coco
from frontend.widgets.image_viewer import ImageViewer
from frontend.widgets.background_preview import get_annotation_regions
from frontend.widgets.class_config_dialog import ClassConfigDialog
from frontend.widgets.class_removal_dialog import ClassRemovalDialog
from frontend.widgets.dataset_division_dialog import DatasetDivisionDialog
from frontend.widgets.training_progress_dialog import TrainingProgressDialog

class PDFConversionWorker(QThread):
    """Background thread for PDF to PNG conversion."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict)

    def __init__(self, raw_pdfs_path: str, raw_images_path: str, rescan: bool = False) -> None:
        super().__init__()
        self.raw_pdfs_path = raw_pdfs_path
        self.raw_images_path = raw_images_path
        self.rescan = rescan

    def run(self) -> None:
        """Execute PDF to PNG conversion with progress updates."""
        if self.rescan:
            from backend.data.pdf_converter import rescan_and_convert_new_pdfs
            stats = rescan_and_convert_new_pdfs(
                self.raw_pdfs_path,
                self.raw_images_path,
                progress_callback=lambda c, t: self.progress.emit(c, t)
            )
        else:
            stats = convert_pdfs_to_png(
                self.raw_pdfs_path,
                self.raw_images_path,
                progress_callback=lambda c, t: self.progress.emit(c, t)
            )
        self.finished.emit(stats)


class TerminalDetectorApp(QMainWindow):
    """Main application window for terminal detection."""

    PROJECT_ROOT = Path(__file__).parent.parent

    def __init__(self) -> None:
        super().__init__()
        self.current_selection_class: str | None = None
        self.current_selection_class_id: int | None = None
        self.current_class_button: QPushButton | None = None
        self.current_image_name: str | None = None
        self.class_id_to_button: dict[int, QPushButton] = {}
        self.class_buttons_list: list[QPushButton] = []
        self.last_annotation_id: int | None = None
        self.delete_mode: bool = False
        self.current_annotations: list[dict] = []
        self.current_predictions: list[dict] = []
        self.db = AnnotationDatabase()
        self.last_removed_image_name: str | None = None
        self.last_removed_image_annotations: list[dict] = []
        self.pdf_worker: PDFConversionWorker | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Initialize user interface layout."""
        self.setWindowTitle("BT-7274")
        self.setGeometry(100, 100, 1400, 850)

        # Setup menu bar
        self._setup_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        toolbar_layout = QHBoxLayout()

        self.train_model_button = QPushButton("Train Model")
        self.train_model_button.clicked.connect(self._on_train_model_clicked)
        toolbar_layout.addWidget(self.train_model_button)

        self.test_button = QPushButton("Test")
        self.test_button.clicked.connect(self._on_test_clicked)
        toolbar_layout.addWidget(self.test_button)

        self.burn_predictions_button = QPushButton("Burn Predictions")
        self.burn_predictions_button.clicked.connect(self._on_burn_predictions_clicked)
        self.burn_predictions_button.setEnabled(False)
        toolbar_layout.addWidget(self.burn_predictions_button)

        # Model selector
        toolbar_layout.addWidget(QLabel("Model:"))
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(150)
        toolbar_layout.addWidget(self.model_selector)
        self._load_available_models()

        info_button = QPushButton("i")
        info_button.setFixedWidth(28)
        info_button.setToolTip("Keyboard shortcuts")
        info_button.clicked.connect(self._show_shortcuts_info)
        toolbar_layout.addWidget(info_button)

        toolbar_layout.addStretch()

        self.undo_predictions_button = QPushButton("Undo Predictions")
        self.undo_predictions_button.clicked.connect(self._on_undo_predictions_clicked)
        self.undo_predictions_button.setEnabled(False)
        toolbar_layout.addWidget(self.undo_predictions_button)

        self.background_button = QPushButton("Show Background")
        self.background_button.clicked.connect(self._on_background_toggled)
        self.background_button.setCheckable(True)
        toolbar_layout.addWidget(self.background_button)

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.undo_button.setEnabled(False)
        toolbar_layout.addWidget(self.undo_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete_mode_clicked)
        self.delete_button.setEnabled(False)
        toolbar_layout.addWidget(self.delete_button)

        main_layout.addLayout(toolbar_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        self.image_viewer = ImageViewer()
        splitter.addWidget(self.image_viewer)

        splitter.setStretchFactor(0, 10)
        splitter.setStretchFactor(1, 90)
        main_layout.addWidget(splitter)

        self._load_images()

        # Connect viewer signals
        self.image_viewer.selection_clicked.connect(self._on_selection_clicked_for_delete)

        # Set up keyboard shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(self._navigate_prev_image)
        QShortcut(QKeySequence("D"), self).activated.connect(self._navigate_next_image)
        QShortcut(QKeySequence("S"), self).activated.connect(self._on_undo_clicked)
        QShortcut(QKeySequence("W"), self).activated.connect(self._on_delete_mode_clicked)
        QShortcut(QKeySequence("X"), self).activated.connect(self._remove_current_image)
        QShortcut(QKeySequence("Z"), self).activated.connect(self._undo_remove_image)
        self._setup_class_shortcuts()

    def _load_available_models(self) -> None:
        """Load available model versions from ml/models/versions directory."""
        models_path = self.PROJECT_ROOT / "ml" / "models" / "versions"

        # Clear existing items
        self.model_selector.clear()

        if not models_path.exists():
            self.model_selector.addItem("No models found")
            self.model_selector.setEnabled(False)
            return

        # Get all subdirectories in versions folder
        model_dirs = sorted([d.name for d in models_path.iterdir() if d.is_dir()])

        if not model_dirs:
            self.model_selector.addItem("No models found")
            self.model_selector.setEnabled(False)
            return

        # Add model directories to combobox
        for model_name in model_dirs:
            self.model_selector.addItem(model_name)

        # Set best_model as default if it exists
        if "best_model" in model_dirs:
            self.model_selector.setCurrentText("best_model")

        self.model_selector.setEnabled(True)

    def _get_selected_model_path(self) -> Path:
        """Get the full path to the selected model."""
        model_name = self.model_selector.currentText()
        if model_name == "No models found":
            return None
        return self.PROJECT_ROOT / "ml" / "models" / "versions" / model_name / "model.pt"

    def _setup_menu_bar(self) -> None:
        """Create menu bar with File menu."""
        menubar = self.menuBar()

        # Files menu
        files_menu = menubar.addMenu("Files")

        # Add Files action
        add_files_action = files_menu.addAction("Add Files")
        add_files_action.triggered.connect(self._on_add_files_clicked)

        # Rescan PDFs action
        rescan_pdfs_action = files_menu.addAction("Rescan PDFs")
        rescan_pdfs_action.triggered.connect(self._on_rescan_pdfs_menu_clicked)

    def _on_add_files_clicked(self) -> None:
        """Handle Add Files menu action - opens file dialog to select PDFs and images."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilters([
            "PDF and Image Files (*.pdf *.jpg *.jpeg)",
            "PDF Files (*.pdf)",
            "Image Files (*.jpg *.jpeg)",
            "All Files (*)"
        ])

        if file_dialog.exec() == QDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                # Split files into PDFs and images
                pdf_files = [f for f in selected_files if f.lower().endswith('.pdf')]
                jpg_files = [f for f in selected_files if f.lower().endswith(('.jpg', '.jpeg'))]

                if jpg_files:
                    self._copy_images_to_raw_images(jpg_files)

                if pdf_files:
                    self._copy_pdfs_to_raw_folder(pdf_files)
                    self._rescan_and_convert_pdfs()

    def _copy_pdfs_to_raw_folder(self, file_paths: list[str]) -> None:
        """Copy selected PDF files to raw_pdfs folder."""
        raw_pdfs_path = self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"
        raw_pdfs_path.mkdir(parents=True, exist_ok=True)

        # Show progress dialog
        progress_dialog = QProgressDialog(
            "Copying PDF files...", None, 0, len(file_paths), self
        )
        progress_dialog.setWindowTitle("Adding Files")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)

        copied_count = 0
        for idx, file_path in enumerate(file_paths):
            progress_dialog.setValue(idx + 1)
            try:
                file_name = Path(file_path).name
                dest_path = raw_pdfs_path / file_name
                shutil.copy2(file_path, dest_path)
                copied_count += 1
            except Exception as e:
                QMessageBox.warning(
                    self, "Copy Error",
                    f"Failed to copy {Path(file_path).name}: {str(e)}"
                )

        progress_dialog.close()

        if copied_count > 0:
            QMessageBox.information(
                self, "Files Copied",
                f"Successfully copied {copied_count} PDF file(s) to raw_pdfs folder."
            )

    def _copy_images_to_raw_images(self, file_paths: list[str]) -> None:
        """Copy selected JPG/JPEG files directly to raw_images folder."""
        raw_images_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images"
        raw_images_path.mkdir(parents=True, exist_ok=True)

        # Show progress dialog
        progress_dialog = QProgressDialog(
            "Copying image files...", None, 0, len(file_paths), self
        )
        progress_dialog.setWindowTitle("Adding Files")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)

        copied_count = 0
        for idx, file_path in enumerate(file_paths):
            progress_dialog.setValue(idx + 1)
            try:
                file_name = Path(file_path).name
                dest_path = raw_images_path / file_name
                shutil.copy2(file_path, dest_path)
                copied_count += 1
            except Exception as e:
                QMessageBox.warning(
                    self, "Copy Error",
                    f"Failed to copy {Path(file_path).name}: {str(e)}"
                )

        progress_dialog.close()

        if copied_count > 0:
            QMessageBox.information(
                self, "Files Copied",
                f"Successfully copied {copied_count} image file(s) to raw_images folder."
            )
            self._load_images()

    def _on_rescan_pdfs_menu_clicked(self) -> None:
        """Handle Rescan PDFs menu action - reconvert all PDFs without duplicates."""
        pdfs_path = self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"
        if not pdfs_path.exists() or not list(pdfs_path.glob("*.pdf")):
            QMessageBox.information(
                self, "No PDFs",
                "No PDF files found in ml/data/raw_pdfs"
            )
            return

        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            "Rescan PDFs",
            "This will delete all existing PNG images and reconvert all PDFs.\n"
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._clear_and_reconvert_all_pdfs()

    def _clear_and_reconvert_all_pdfs(self) -> None:
        """Delete all PNG images and reconvert all PDFs without duplicates."""
        raw_images_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images"

        # Delete all existing images (PNG and JPG/JPEG from PDF conversions, keep native JPGs)
        if raw_images_path.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                for image_file in raw_images_path.glob(ext):
                    try:
                        image_file.unlink()
                    except Exception as e:
                        print(f"Failed to delete {image_file.name}: {e}")

        # Now convert all PDFs fresh
        self.pdf_worker = PDFConversionWorker(
            raw_pdfs_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"),
            raw_images_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_images"),
            rescan=False  # Use full conversion, not just new PDFs
        )

        progress_dialog = QProgressDialog(
            "Rescanning and converting all PDFs...", None, 0, 0, self
        )
        progress_dialog.setWindowTitle("Rescan PDFs")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)

        self.pdf_worker.progress.connect(
            lambda c, t: progress_dialog.setMaximum(t) or progress_dialog.setValue(c)
        )
        self.pdf_worker.finished.connect(
            lambda stats: self._on_reconvert_complete(progress_dialog, stats)
        )
        self.pdf_worker.finished.connect(self.pdf_worker.deleteLater)
        self.pdf_worker.start()

    def _on_reconvert_complete(self, progress_dialog: QProgressDialog, stats: dict) -> None:
        """Handle full PDF reconversion completion."""
        progress_dialog.close()

        message = (
            f"Rescan complete!\n"
            f"Converted: {stats.get('total_pdfs', 0)} PDFs to {stats.get('total_pages', 0)} PNG images"
        )
        QMessageBox.information(self, "Rescan Complete", message)
        self.statusBar().showMessage(message)
        self._load_images()

    def _create_left_panel(self) -> QWidget:
        """Create left panel with image list and classes controls."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        image_list_label = QLabel("Images")
        image_list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(image_list_label)

        self.page_list = QListWidget()
        self.page_list.itemClicked.connect(self._on_image_selected)
        layout.addWidget(self.page_list, 1)

        classes_panel = self._create_classes_panel()
        layout.addWidget(classes_panel, 1)

        self._load_classes_from_db()

        return container

    def _create_classes_panel(self) -> QWidget:
        """Create panel with classes label and action buttons."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        classes_label = QLabel("Classes")
        classes_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(classes_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        self.add_class_button = QPushButton("Add")
        self.add_class_button.clicked.connect(self._on_add_class_clicked)
        buttons_layout.addWidget(self.add_class_button)

        self.remove_class_button = QPushButton("Remove")
        self.remove_class_button.clicked.connect(self._on_remove_class_clicked)
        buttons_layout.addWidget(self.remove_class_button)

        layout.addLayout(buttons_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        self.classes_container = QWidget()
        self.classes_layout = QVBoxLayout(self.classes_container)
        self.classes_layout.setContentsMargins(0, 0, 0, 0)
        self.classes_layout.addStretch()

        scroll_area.setWidget(self.classes_container)
        layout.addWidget(scroll_area)

        return container

    def _on_add_class_clicked(self) -> None:
        """Handle Add class button click."""
        dialog = ClassConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            class_data = dialog.get_class_data()
            self.db.add_class(class_data["name"], class_data["color"])
            self._add_class_button(class_data["name"], class_data["color"])

    def _add_class_button(self, name: str, color: str) -> None:
        """Add class button to classes panel."""
        class_button = QPushButton(name)
        class_button.setProperty("color", color)
        self._update_button_style(class_button, is_active=False)
        class_button.clicked.connect(
            lambda: self._on_class_button_clicked(name, color, class_button)
        )
        self.classes_layout.insertWidget(
            self.classes_layout.count() - 1, class_button
        )

        class_data = self.db.get_class_by_name(name)
        if class_data:
            self.class_id_to_button[class_data["id"]] = class_button

        self.class_buttons_list.append(class_button)

    def _update_button_style(self, button: QPushButton, is_active: bool) -> None:
        """Update button style based on active state."""
        color = button.property("color")
        bg_color = "#d0d0d0" if is_active else "white"

        button.setStyleSheet(
            f"QPushButton {{"
            f"border: 2px solid {color}; "
            f"border-radius: 4px; "
            f"padding: 6px 12px; "
            f"color: {color}; "
            f"background-color: {bg_color}; "
            f"font-weight: bold;"
            f"}}"
            f"QPushButton:hover {{"
            f"background-color: {'#c0c0c0' if is_active else '#f0f0f0'};"
            f"}}"
            f"QPushButton:pressed {{"
            f"background-color: #b0b0b0;"
            f"}}"
        )

    def _on_class_button_clicked(
        self, class_name: str, class_color: str, button: QPushButton
    ) -> None:
        """Toggle selection mode for class."""
        if self.current_selection_class == class_name:
            self._deselect_class(button)
        else:
            self._select_class(class_name, class_color, button)

    def _deselect_class(self, button: QPushButton) -> None:
        """Deselect currently active class."""
        self.image_viewer.set_selection_mode(False)
        self.current_selection_class = None
        self.current_selection_class_id = None
        self._update_button_style(button, is_active=False)
        self.current_class_button = None
        self.last_annotation_id = None
        self.undo_button.setEnabled(False)
        try:
            self.image_viewer.selection_made.disconnect()
        except TypeError:
            pass

    def _select_class(self, class_name: str, class_color: str, button: QPushButton) -> None:
        """Select a class for annotation."""
        if self.current_class_button:
            self._update_button_style(self.current_class_button, is_active=False)

        class_data = self.db.get_class_by_name(class_name)
        self.current_selection_class = class_name
        self.current_selection_class_id = class_data["id"] if class_data else None
        self.current_class_button = button
        self._update_button_style(button, is_active=True)
        self.delete_button.setEnabled(True)
        self.image_viewer.set_selection_mode(True)
        try:
            self.image_viewer.selection_made.disconnect()
        except TypeError:
            pass
        self.image_viewer.selection_made.connect(
            lambda rect: self._on_selection_made(class_name, class_color, rect)
        )

    def _on_selection_made(self, class_name: str, class_color: str, rect: QRect) -> None:
        """Handle selection made on image."""
        if self.current_image_name and self.current_selection_class_id:
            img_dims = self.image_viewer.get_image_dimensions()
            if img_dims:
                img_w, img_h = img_dims
                rect = rect.normalized()
                x = max(0, rect.x())
                y = max(0, rect.y())
                w = min(rect.width(), img_w - x)
                h = min(rect.height(), img_h - y)
            else:
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            annotation_id = self.db.add_annotation(
                image_id=self.current_image_name,
                class_id=self.current_selection_class_id,
                dataset="default",
                x=x,
                y=y,
                width=w,
                height=h,
                text=""
            )
            self.last_annotation_id = annotation_id
            self.undo_button.setEnabled(True)
            self.image_viewer.add_selection(rect, class_color, "")

    def _on_undo_clicked(self) -> None:
        """Undo last selection made."""
        if not self.last_annotation_id or not self.current_image_name:
            return

        success = self.db.delete_annotation(self.last_annotation_id)
        if success:
            self.last_annotation_id = None
            self._load_annotations_for_image(self.current_image_name)
            self.undo_button.setEnabled(False)

    def _on_delete_mode_clicked(self) -> None:
        """Toggle delete mode."""
        if not self.delete_mode:
            self.delete_mode = True
            self._set_ui_enabled(False)
            self.image_viewer.set_delete_mode(True)
            self.delete_button.setText("Cancel")
        else:
            self._exit_delete_mode()

    def _exit_delete_mode(self) -> None:
        """Exit delete mode and re-enable UI."""
        self.delete_mode = False
        self._set_ui_enabled(True)
        self.image_viewer.set_delete_mode(False)
        self.delete_button.setText("Delete")

    def _on_selection_clicked_for_delete(self, x: int, y: int, width: int, height: int) -> None:
        """Handle selection click in delete mode."""
        for annotation in self.current_annotations:
            if (int(annotation["x"]) == x and
                int(annotation["y"]) == y and
                int(annotation["width"]) == width and
                int(annotation["height"]) == height):
                reply = QMessageBox.question(
                    self,
                    "Delete Selection",
                    "Delete this selection?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.db.delete_annotation(annotation["id"])
                    self._load_annotations_for_image(self.current_image_name)
                break

    def _set_ui_enabled(self, enabled: bool) -> None:
        """Enable or disable all UI elements except image viewer."""
        self.page_list.setEnabled(enabled)
        self.add_class_button.setEnabled(enabled)
        self.remove_class_button.setEnabled(enabled)
        self.undo_button.setEnabled(enabled and self.last_annotation_id is not None)
        if enabled:
            has_selections = len(self.current_annotations) > 0
            self.delete_button.setEnabled(has_selections)
        else:
            self.delete_button.setEnabled(True)
        for button in self.class_id_to_button.values():
            button.setEnabled(enabled)

    def _on_remove_class_clicked(self) -> None:
        """Handle Remove class button click."""
        classes = self.db.get_all_classes()
        if not classes:
            QMessageBox.information(
                self, "No Classes", "No classes to remove."
            )
            return

        dialog = ClassRemovalDialog(classes, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_class = dialog.get_selected_class()
            if selected_class:
                self._remove_class(selected_class["id"], selected_class["name"])

    def _remove_class(self, class_id: int, class_name: str) -> None:
        """Remove class from database and UI."""
        self.db.delete_class(class_id)

        if class_id in self.class_id_to_button:
            button = self.class_id_to_button[class_id]
            button.deleteLater()
            del self.class_id_to_button[class_id]

        if self.current_selection_class_id == class_id:
            self.current_selection_class = None
            self.current_selection_class_id = None
            self.current_class_button = None
            self.image_viewer.set_selection_mode(False)

        if not self.class_id_to_button:
            self.current_selection_class = None
            self.current_selection_class_id = None
            self.current_class_button = None
            self.image_viewer.set_selection_mode(False)

        if self.current_image_name:
            self._load_annotations_for_image(self.current_image_name)

        QMessageBox.information(
            self, "Class Removed", f"Class '{class_name}' and its selections have been removed."
        )

    def _load_classes_from_db(self) -> None:
        """Load all classes from database and create buttons."""
        classes = self.db.get_all_classes()
        for class_data in classes:
            self._add_class_button(class_data["name"], class_data["color"])

    def _load_images(self) -> None:
        """Load images from raw_images folder and select first."""
        images_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images"
        if not images_path.exists():
            return

        self.page_list.clear()
        # Collect all image files (PNG, JPG, JPEG)
        image_files = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg")
            for f in images_path.glob(ext)
        )
        for image_file in image_files:
            self.page_list.addItem(image_file.name)

        if self.page_list.count() > 0:
            self.page_list.setCurrentRow(0)
            self._on_image_selected(self.page_list.item(0))

    def _on_image_selected(self, item) -> None:
        """Display selected image and load its annotations."""
        image_name = item.text()
        self.current_image_name = image_name
        image_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images" / image_name
        self.image_viewer.load_image(str(image_path))
        self._load_annotations_for_image(image_name)

    def _load_annotations_for_image(self, image_name: str) -> None:
        """Load and display annotations for image from database."""
        self.image_viewer.clear_selections()
        self.current_annotations = self.db.get_annotations_by_image(image_name, "default")

        for annotation in self.current_annotations:
            rect = QRect(
                int(annotation["x"]),
                int(annotation["y"]),
                int(annotation["width"]),
                int(annotation["height"])
            )
            annotation_text = annotation.get("text", "")
            self.image_viewer.add_selection(rect, annotation["color"], annotation_text)

        self._generate_background_preview()

        if not self.delete_mode:
            has_selections = len(self.current_annotations) > 0
            self.delete_button.setEnabled(has_selections)

    def _generate_background_preview(self) -> None:
        """Set excluded regions (annotations) for background overlay."""
        regions = get_annotation_regions(self.current_annotations)
        self.image_viewer.set_excluded_regions(regions)

    def _on_background_toggled(self) -> None:
        """Handle background preview toggle button."""
        self.image_viewer.toggle_background_display()

    def _on_undo_predictions_clicked(self) -> None:
        """Clear all model predictions from current image."""
        self.image_viewer.clear_predictions()
        self.current_predictions = []
        self.undo_predictions_button.setEnabled(False)
        self.burn_predictions_button.setEnabled(False)

    def _on_burn_predictions_clicked(self) -> None:
        """Save predictions to database as annotations."""
        if not self.current_predictions or not self.current_image_name:
            QMessageBox.warning(
                self, "No Predictions",
                "No predictions to save. Run Test first."
            )
            return

        # Confirm before burning
        reply = QMessageBox.question(
            self, "Confirm",
            f"Save {len(self.current_predictions)} predictions as annotations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.burn_predictions_button.setEnabled(False)
        predictions_to_save = self.current_predictions[:]
        self.current_predictions = []

        saved_count = 0
        for pred in predictions_to_save:
            class_data = self.db.get_class_by_name(pred['class_name'])
            if not class_data:
                print(f"Warning: Class '{pred['class_name']}' not found in database")
                continue

            class_id = class_data['id']
            class_color = class_data['color']

            # Save annotation
            annotation_id = self.db.add_annotation(
                image_id=self.current_image_name,
                class_id=class_id,
                dataset="default",
                x=int(pred['x']),
                y=int(pred['y']),
                width=int(pred['width']),
                height=int(pred['height']),
                text=f"auto:{pred['confidence']:.0%}"
            )

            if annotation_id:
                saved_count += 1

        # Reload annotations to show with proper colors
        self._load_annotations_for_image(self.current_image_name)
        self.image_viewer.clear_predictions()
        self.current_predictions = []
        self.undo_predictions_button.setEnabled(False)
        self.burn_predictions_button.setEnabled(False)

        QMessageBox.information(
            self, "Saved",
            f"Saved {saved_count} predictions as annotations."
        )

    def _show_pdf_conversion_progress(self) -> None:
        """Show progress dialog and convert PDFs to PNG."""
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            QMessageBox.warning(self, "Busy", "PDF conversion already in progress.")
            return

        self.pdf_worker = PDFConversionWorker(
            raw_pdfs_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"),
            raw_images_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_images")
        )

        progress_dialog = QProgressDialog(
            "Converting PDFs to PNG...", None, 0, 0, self
        )
        progress_dialog.setWindowTitle("PDF Conversion")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)

        self.pdf_worker.progress.connect(
            lambda c, t: progress_dialog.setMaximum(t) or progress_dialog.setValue(c)
        )
        self.pdf_worker.finished.connect(
            lambda stats: self._on_pdf_conversion_complete(progress_dialog, stats)
        )
        self.pdf_worker.finished.connect(self.pdf_worker.deleteLater)
        self.pdf_worker.start()

    def _on_pdf_conversion_complete(
        self, progress_dialog: QProgressDialog, stats: dict
    ) -> None:
        """Handle PDF conversion completion."""
        progress_dialog.close()
        message = (
            f"Converted {stats['total_pdfs']} PDFs to {stats['total_pages']} PNG images"
        )
        self.statusBar().showMessage(message)
        QMessageBox.information(self, "Conversion Complete", message)
        self._load_images()

    def _on_train_model_clicked(self) -> None:
        """Handle Train Model button — shows dataset division dialog, then starts training."""
        annotated_images = self.db.get_annotated_images()

        if not annotated_images:
            QMessageBox.warning(
                self,
                "No Annotations",
                "No annotated images found. Please annotate images first."
            )
            return

        # Show dataset division dialog
        all_images = [item.text() for item in self._get_all_images()]
        dialog = DatasetDivisionDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        percentages = dialog.get_percentages()
        self._split_dataset(all_images, percentages)

        # Check if COCO files exist, create them if missing
        coco_dir = self.PROJECT_ROOT / "ml" / "data" / "coco"
        train_json = coco_dir / "coco_train.json"
        dev_json = coco_dir / "coco_dev.json"

        if not train_json.exists() or not dev_json.exists():
            reply = QMessageBox.question(
                self,
                "Create COCO Files?",
                "COCO JSON files are required for training.\n\n"
                "Create them now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._create_coco_json()
            else:
                return

        # Start training
        training_dialog = TrainingProgressDialog(self)
        training_dialog.start_training()
        training_dialog.exec()

    def _on_test_clicked(self) -> None:
        """Run model predictions on the currently selected image."""
        if not self.current_image_name:
            QMessageBox.warning(self, "No Image", "Please select an image first.")
            return

        model_path = self._get_selected_model_path()
        if not model_path or not model_path.exists():
            QMessageBox.warning(
                self, "No Model",
                "Selected model not found. Please select a valid model."
            )
            return

        from backend.fine_tuning.inference import run_inference
        image_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images" / self.current_image_name
        predictions = run_inference(str(image_path), str(model_path))

        has_predictions = len(predictions) > 0
        self.current_predictions = predictions
        self.undo_predictions_button.setEnabled(has_predictions)
        self.burn_predictions_button.setEnabled(has_predictions)

        # Display predictions on image viewer
        for pred in predictions:
            rect = QRect(
                int(pred['x']), int(pred['y']),
                int(pred['width']), int(pred['height'])
            )
            self.image_viewer.add_selection(
                rect, "#ff0000",
                f"{pred['class_name']} {pred['confidence']:.0%}",
                is_prediction=True
            )

    def _on_rescan_pdfs_clicked(self) -> None:
        """Handle Rescan PDFs button click."""
        pdfs_path = self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"
        if not pdfs_path.exists() or not list(pdfs_path.glob("*.pdf")):
            QMessageBox.information(
                self, "No PDFs",
                "No PDF files found in ml/data/raw_pdfs"
            )
            return

        self._rescan_and_convert_pdfs()

    def _rescan_and_convert_pdfs(self) -> None:
        """Scan for new PDFs and convert them."""
        if self.pdf_worker is not None and self.pdf_worker.isRunning():
            QMessageBox.warning(self, "Busy", "PDF conversion already in progress.")
            return

        self.pdf_worker = PDFConversionWorker(
            raw_pdfs_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_pdfs"),
            raw_images_path=str(self.PROJECT_ROOT / "ml" / "data" / "raw_images"),
            rescan=True
        )

        progress_dialog = QProgressDialog(
            "Scanning for new PDFs...", None, 0, 0, self
        )
        progress_dialog.setWindowTitle("Rescan PDFs")
        progress_dialog.setCancelButton(None)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)

        self.pdf_worker.progress.connect(
            lambda c, t: progress_dialog.setMaximum(t) or progress_dialog.setValue(c)
        )
        self.pdf_worker.finished.connect(
            lambda stats: self._on_rescan_complete(progress_dialog, stats)
        )
        self.pdf_worker.finished.connect(self.pdf_worker.deleteLater)
        self.pdf_worker.start()

    def _on_rescan_complete(
        self, progress_dialog: QProgressDialog, stats: dict
    ) -> None:
        """Handle rescan completion."""
        progress_dialog.close()

        if stats["new_pdfs"] == 0:
            message = f"No new PDFs found. ({stats['total_pdfs']} total PDFs, {stats['skipped_pdfs']} already converted)"
            QMessageBox.information(self, "Rescan Complete", message)
            self.statusBar().showMessage(message)
        else:
            message = (
                f"Rescan complete!\n"
                f"New PDFs: {stats['new_pdfs']}\n"
                f"Converted to: {stats['total_pages']} PNG images\n"
                f"Skipped: {stats['skipped_pdfs']} (already converted)"
            )
            QMessageBox.information(self, "Rescan Complete", message)
            self.statusBar().showMessage(message)
            self._load_images()

    def _get_all_images(self) -> list:
        """Get all image items from list widget."""
        return [self.page_list.item(i) for i in range(self.page_list.count())]

    def _split_dataset(self, all_image_ids: list[str], percentages: dict) -> None:
        """Create train/dev/test split for labeled images only."""
        try:
            stats = create_initial_split(
                all_image_ids=all_image_ids,
                train_percentage=percentages["train"],
                dev_percentage=percentages["dev"],
                test_percentage=percentages["test"]
            )

            message = (
                f"Dataset split successful!\n"
                f"Train: {stats['train_count']}\n"
                f"Dev: {stats['dev_count']}\n"
                f"Test: {stats['test_count']}\n"
                f"Unlabeled (for active learning): {stats['unlabeled_count']}"
            )
            QMessageBox.information(self, "Split Complete", message)
            self.statusBar().showMessage(message)

            # Ask if user wants to create COCO JSONs
            self._prompt_create_coco_json()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Split Error",
                f"Failed to split dataset: {str(e)}"
            )

    def _navigate_prev_image(self) -> None:
        """Navigate to previous image in list."""
        self._navigate_image(-1)

    def _navigate_next_image(self) -> None:
        """Navigate to next image in list."""
        self._navigate_image(1)

    def _navigate_image(self, delta: int) -> None:
        """Move selection by delta in image list and load image."""
        current_row = self.page_list.currentRow()
        new_row = current_row + delta
        if 0 <= new_row < self.page_list.count():
            self.page_list.setCurrentRow(new_row)
            self._on_image_selected(self.page_list.item(new_row))

    def _remove_current_image(self) -> None:
        """Remove current image and save for undo."""
        if not self.current_image_name:
            return

        self.last_removed_image_name = self.current_image_name
        self.last_removed_image_annotations = self.db.get_annotations_by_image(
            self.current_image_name, "default"
        )

        self._move_image_to_trash(self.current_image_name)
        self.db.delete_annotations_by_image(self.current_image_name)
        self._update_image_list_after_removal()

    def _move_image_to_trash(self, image_name: str) -> None:
        """Move image file to trash folder."""
        images_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images" / image_name
        trash_path = self.PROJECT_ROOT / "ml" / "data" / ".trash_images"
        trash_path.mkdir(parents=True, exist_ok=True)

        if images_path.exists():
            images_path.rename(trash_path / image_name)

    def _update_image_list_after_removal(self) -> None:
        """Update UI after removing image from dataset."""
        current_row = self.page_list.currentRow()
        self.page_list.takeItem(current_row)

        new_row = min(current_row, self.page_list.count() - 1)
        if new_row >= 0:
            self.page_list.setCurrentRow(new_row)
            self._on_image_selected(self.page_list.item(new_row))
        else:
            self.current_image_name = None
            self.image_viewer.clear_selections()

    def _undo_remove_image(self) -> None:
        """Undo last image removal."""
        if not self.last_removed_image_name:
            return

        self._restore_image_from_trash(self.last_removed_image_name)
        self._restore_annotations(self.last_removed_image_annotations)
        self._add_restored_image_to_list(self.last_removed_image_name)
        self._clear_undo_state()

    def _restore_image_from_trash(self, image_name: str) -> None:
        """Restore image file from trash."""
        trash_path = self.PROJECT_ROOT / "ml" / "data" / ".trash_images" / image_name
        images_path = self.PROJECT_ROOT / "ml" / "data" / "raw_images" / image_name

        if trash_path.exists():
            trash_path.rename(images_path)

    def _restore_annotations(self, annotations: list[dict]) -> None:
        """Restore annotations to database."""
        for ann in annotations:
            self.db.add_annotation(
                image_id=ann["image_id"],
                class_id=ann["class_id"],
                dataset=ann["dataset"],
                x=ann["x"],
                y=ann["y"],
                width=ann["width"],
                height=ann["height"],
                text=ann.get("text", "")
            )

    def _add_restored_image_to_list(self, image_name: str) -> None:
        """Add restored image to list and select it."""
        self.page_list.addItem(image_name)
        self._sort_image_list()
        self._select_restored_image(image_name)

    def _sort_image_list(self) -> None:
        """Sort images in list alphabetically."""
        items = [self.page_list.item(i).text() for i in range(self.page_list.count())]
        items.sort()
        self.page_list.clear()
        for item_text in items:
            self.page_list.addItem(item_text)

    def _select_restored_image(self, image_name: str) -> None:
        """Select restored image in list."""
        for i in range(self.page_list.count()):
            if self.page_list.item(i).text() == image_name:
                self.page_list.setCurrentRow(i)
                self._on_image_selected(self.page_list.item(i))
                break

    def _clear_undo_state(self) -> None:
        """Clear undo state after restore."""
        self.last_removed_image_name = None
        self.last_removed_image_annotations = []

    def _show_shortcuts_info(self) -> None:
        """Show available keyboard shortcuts."""
        shortcuts_text = (
            "A — Previous image\n"
            "D — Next image\n"
            "S — Undo last annotation\n"
            "W — Delete mode\n"
            "X — Remove image from dataset\n"
            "Z — Undo last removal\n"
            "\n"
            "1-9 — Select class (in order)"
        )
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            shortcuts_text
        )

    def _setup_class_shortcuts(self) -> None:
        """Set up keyboard shortcuts for class buttons (1-9)."""
        for i, button in enumerate(self.class_buttons_list[:9], start=1):
            key = str(i)
            QShortcut(QKeySequence(key), self).activated.connect(button.click)

    def _prompt_create_coco_json(self) -> None:
        """Ask user if they want to create COCO JSON files for training."""
        reply = QMessageBox.question(
            self,
            "Create COCO JSON?",
            "Export dataset to COCO format?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._create_coco_json()

    def _create_coco_json(self) -> None:
        """Export all splits to COCO JSON format."""
        try:
            raw_images_directory = self.PROJECT_ROOT / "ml" / "data" / "raw_images"
            output_directory = self.PROJECT_ROOT / "ml" / "data" / "coco"

            export_all_splits_to_coco(output_directory, self.db, raw_images_directory)

            message = (
                "COCO JSON files created successfully!\n\n"
                f"Location: {output_directory}\n"
                "Files:\n"
                "  - coco_train.json\n"
                "  - coco_dev.json\n"
                "  - coco_test.json"
            )
            QMessageBox.information(self, "COCO Export Complete", message)
            self.statusBar().showMessage("COCO JSON files created")
        except Exception as e:
            QMessageBox.critical(
                self,
                "COCO Export Error",
                f"Failed to create COCO JSON files: {str(e)}"
            )

    def _cleanup_trash(self) -> None:
        """Clean up trash folder on app close."""
        trash_path = self.PROJECT_ROOT / "ml" / "data" / ".trash_images"
        if trash_path.exists():
            for item in trash_path.iterdir():
                if item.is_file():
                    item.unlink()
            trash_path.rmdir()

    def closeEvent(self, event) -> None:
        """Handle application close event."""
        self._cleanup_trash()
        self._cleanup_threads()
        self.db.close()
        super().closeEvent(event)

    def _cleanup_threads(self) -> None:
        """Wait for background threads to finish before closing."""
        if self.pdf_worker and self.pdf_worker.isRunning():
            self.pdf_worker.wait(5000)


def main() -> None:
    """Launch application. Handle PDF conversion if needed."""
    app = QApplication(sys.argv)
    data_status = check_data_availability()

    main_window = TerminalDetectorApp()

    if data_status.has_raw_pdfs and not data_status.has_raw_images:
        reply = QMessageBox.question(
            main_window,
            "Convert PDFs to Images",
            "PDF files found. Convert to PNG images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            main_window.show()
            main_window._show_pdf_conversion_progress()
        else:
            main_window.show()
    else:
        main_window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
