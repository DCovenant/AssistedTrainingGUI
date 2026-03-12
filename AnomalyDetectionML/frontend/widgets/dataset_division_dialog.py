from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from PyQt6.QtCore import Qt


class DatasetDivisionDialog(QDialog):
    """Dialog for setting train/dev/test split percentages for labeled images."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Dataset Division")
        self.setGeometry(200, 200, 400, 250)
        self.train_percentage: int = 0
        self.dev_percentage: int = 0
        self.test_percentage: int = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create dialog layout with train/dev/test inputs."""
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Split labeled images into train/dev/test:"))
        layout.addWidget(QLabel("(Unlabeled images stay for active learning)"))

        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train:"))
        self.train_input = QLineEdit()
        self.train_input.setText("70")
        self.train_input.setMaximumWidth(60)
        train_layout.addWidget(self.train_input)
        train_layout.addWidget(QLabel("%"))
        train_layout.addStretch()
        layout.addLayout(train_layout)

        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Dev:"))
        self.dev_input = QLineEdit()
        self.dev_input.setText("15")
        self.dev_input.setMaximumWidth(60)
        dev_layout.addWidget(self.dev_input)
        dev_layout.addWidget(QLabel("%"))
        dev_layout.addStretch()
        layout.addLayout(dev_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test:"))
        self.test_input = QLineEdit()
        self.test_input.setText("15")
        self.test_input.setMaximumWidth(60)
        test_layout.addWidget(self.test_input)
        test_layout.addWidget(QLabel("%"))
        test_layout.addStretch()
        layout.addLayout(test_layout)

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
        """Validate percentages and accept dialog."""
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
                "Invalid Sum",
                f"Train + Dev + Test must sum to 100.\nCurrent sum: {total}"
            )
            return

        self.train_percentage = train
        self.dev_percentage = dev
        self.test_percentage = test
        self.accept()

    def get_percentages(self) -> dict:
        """Return train/dev/test percentages."""
        return {
            "train": self.train_percentage,
            "dev": self.dev_percentage,
            "test": self.test_percentage
        }
