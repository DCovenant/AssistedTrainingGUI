from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
)


class AnnotationTextDialog(QDialog):
    """Dialog for entering text content of annotation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Enter Text")
        self.setGeometry(300, 300, 400, 150)
        self.text_content: str = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create dialog layout with text input."""
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Text on this annotation:"))

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("e.g., -X1:1")
        layout.addWidget(self.text_input)

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
        """Accept dialog and store text."""
        self.text_content = self.text_input.text()
        self.accept()

    def get_text(self) -> str:
        """Return entered text."""
        return self.text_content
