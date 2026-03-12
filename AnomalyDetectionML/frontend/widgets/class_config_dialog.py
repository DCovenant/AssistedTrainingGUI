from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt


class ClassConfigDialog(QDialog):
    """Dialog for configuring class name and color."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Class config")
        self.setGeometry(200, 200, 400, 200)

        self.selected_color = QColor(Qt.GlobalColor.black)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create dialog layout."""
        layout = QVBoxLayout()

        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)

        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_button = QPushButton("Choose Color")
        color_button.clicked.connect(self._open_color_picker)
        self.color_display = QPushButton()
        self.color_display.setMaximumWidth(50)
        self._update_color_display()
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_button)
        color_layout.addWidget(self.color_display)
        color_layout.addStretch()
        layout.addLayout(color_layout)

        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self._on_ok_clicked)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _on_ok_clicked(self) -> None:
        """Validate class name and accept dialog."""
        class_name = self.name_input.text().strip()

        if not class_name:
            QMessageBox.warning(
                self,
                "Empty Name",
                "Please enter a class name."
            )
            return

        self.accept()

    def _open_color_picker(self) -> None:
        """Open color picker dialog."""
        from PyQt6.QtWidgets import QColorDialog

        color = QColorDialog.getColor(
            self.selected_color, self, "Choose Class Color"
        )
        if color.isValid():
            self.selected_color = color
            self._update_color_display()

    def _update_color_display(self) -> None:
        """Update color display button."""
        stylesheet = f"background-color: {self.selected_color.name()};"
        self.color_display.setStyleSheet(stylesheet)

    def get_class_data(self) -> dict:
        """Return class name and color."""
        return {
            "name": self.name_input.text(),
            "color": self.selected_color.name()
        }
