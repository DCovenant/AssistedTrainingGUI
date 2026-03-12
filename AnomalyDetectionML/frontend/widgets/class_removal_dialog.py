from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt


class ClassRemovalDialog(QDialog):
    """Dialog for selecting and removing a class."""

    def __init__(self, classes: list[dict], parent=None) -> None:
        """
        Initialize dialog with list of classes.

        Args:
            classes: List of class dicts with 'id', 'name', 'color' keys
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Remove Class")
        self.setGeometry(200, 200, 400, 300)
        self.classes = classes
        self.selected_class: dict | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create dialog layout."""
        layout = QVBoxLayout()

        label = QLabel("Select class to remove:")
        layout.addWidget(label)

        self.classes_list = QListWidget()
        for class_data in self.classes:
            item = QListWidgetItem(class_data["name"])
            item.setData(Qt.ItemDataRole.UserRole, class_data)
            self.classes_list.addItem(item)

        self.classes_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.classes_list)

        button_layout = QHBoxLayout()
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.remove_button.setEnabled(False)
        button_layout.addWidget(self.remove_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _on_selection_changed(self) -> None:
        """Enable remove button when item is selected."""
        has_selection = len(self.classes_list.selectedItems()) > 0
        self.remove_button.setEnabled(has_selection)

    def _on_remove_clicked(self) -> None:
        """Handle remove button click."""
        selected_items = self.classes_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            self.selected_class = item.data(Qt.ItemDataRole.UserRole)

            reply = QMessageBox.question(
                self,
                "Confirm Removal",
                f"Are you sure you want to remove class '{self.selected_class['name']}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.accept()

    def get_selected_class(self) -> dict | None:
        """Return selected class data."""
        return self.selected_class
