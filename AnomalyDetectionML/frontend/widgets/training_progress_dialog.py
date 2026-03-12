"""Training progress dialog with real-time updates and loss/accuracy graph."""

import sys
import time
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel, QTextEdit, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.fine_tuning.training import TrainingLauncher


class TrainingWorker(QThread):
    """Runs training in a background thread, emitting progress signals.

    Signals:
        batch_progress: (current_batch, total_batches) within each epoch
        epoch_complete: (metrics_dict) after each epoch's train+validation
        training_complete: (best_dev_accuracy) when all epochs finish
        training_error: (error_message) if training fails
    """
    batch_progress = pyqtSignal(int, int)
    epoch_complete = pyqtSignal(dict)
    training_complete = pyqtSignal(float)
    training_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_requested = False

    def stop(self) -> None:
        """Signal the training to stop after current epoch."""
        self.stop_requested = True

    def run(self) -> None:
        try:
            launcher = TrainingLauncher()
            best_acc = launcher.train(
                on_batch=lambda cur, total: self.batch_progress.emit(cur, total),
                on_epoch=lambda metrics: self.epoch_complete.emit(metrics),
                should_stop=lambda: self.stop_requested
            )
            self.training_complete.emit(best_acc)
        except Exception as e:
            self.training_error.emit(str(e))


class TrainingProgressDialog(QDialog):
    """Dialog showing real-time training progress with live loss/accuracy graph.

    Layout:
        - Epoch counter label
        - Batch progress bar
        - Time elapsed / ETA
        - [Left] Loss & accuracy graph  |  [Right] Epoch log text
        - Close button
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Progress")
        self.setMinimumSize(900, 500)
        self.worker: TrainingWorker | None = None
        self.start_time: float = 0.0
        self.epoch_times: list[float] = []

        # Metric history for plotting
        self._epochs: list[int] = []
        self._train_loss: list[float] = []
        self._dev_loss: list[float] = []
        self._train_acc: list[float] = []
        self._dev_acc: list[float] = []

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.epoch_label = QLabel("Starting training...")
        self.epoch_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.epoch_label)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setTextVisible(True)
        self.batch_progress_bar.setFormat("Batch %v / %m")
        layout.addWidget(self.batch_progress_bar)

        self.time_label = QLabel("")
        layout.addWidget(self.time_label)

        # Graph + log side by side
        mid_layout = QHBoxLayout()

        # Matplotlib canvas with two subplots
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.set_facecolor("#2b2b2b")
        self.ax_loss = self.figure.add_subplot(211)
        self.ax_acc = self.figure.add_subplot(212)
        self.figure.subplots_adjust(hspace=0.4, left=0.15, right=0.95, top=0.92, bottom=0.1)
        self._style_axes()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumWidth(400)
        mid_layout.addWidget(self.canvas, stretch=3)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        mid_layout.addWidget(self.log_text, stretch=2)

        layout.addLayout(mid_layout)

        button_layout = QHBoxLayout()

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        button_layout.addWidget(self.stop_button)

        button_layout.addStretch()

        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def _style_axes(self) -> None:
        """Apply dark theme to both axes."""
        for ax, title, ylabel in [
            (self.ax_loss, "Loss", "Loss"),
            (self.ax_acc, "Accuracy", "Acc %"),
        ]:
            ax.set_facecolor("#1e1e1e")
            ax.set_title(title, color="white", fontsize=10)
            ax.set_ylabel(ylabel, color="white", fontsize=8)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.grid(True, alpha=0.2, color="white")

    def _update_graph(self) -> None:
        """Redraw both subplots with current metric history."""
        # Loss plot
        self.ax_loss.clear()
        self.ax_loss.plot(self._epochs, self._train_loss, "o-", color="#4fc3f7", markersize=4, label="Train")
        self.ax_loss.plot(self._epochs, self._dev_loss, "o-", color="#ff8a65", markersize=4, label="Dev")
        self.ax_loss.legend(fontsize=7, loc="upper right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="white")

        # Accuracy plot
        self.ax_acc.clear()
        self.ax_acc.plot(self._epochs, self._train_acc, "o-", color="#4fc3f7", markersize=4, label="Train")
        self.ax_acc.plot(self._epochs, self._dev_acc, "o-", color="#ff8a65", markersize=4, label="Dev")
        self.ax_acc.legend(fontsize=7, loc="lower right", facecolor="#2b2b2b", edgecolor="#555555", labelcolor="white")

        # Re-apply styling after clear
        self._style_axes()

        # Set x-axis to integer epochs
        if self._epochs:
            self.ax_acc.set_xlabel("Epoch", color="white", fontsize=8)
            for ax in (self.ax_loss, self.ax_acc):
                ax.set_xlim(0.5, max(self._epochs) + 0.5)
                ax.set_xticks(self._epochs)

        self.canvas.draw_idle()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string like '2m 30s' or '1h 5m'
        """
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {secs}s"
        hours, mins = divmod(minutes, 60)
        return f"{hours}h {mins}m"

    def start_training(self) -> None:
        """Launch training in background thread."""
        self.start_time = time.time()
        self.worker = TrainingWorker()
        self.worker.batch_progress.connect(self._on_batch_progress)
        self.worker.epoch_complete.connect(self._on_epoch_complete)
        self.worker.training_complete.connect(self._on_training_complete)
        self.worker.training_error.connect(self._on_training_error)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_batch_progress(self, current: int, total: int) -> None:
        """Update batch progress bar.

        Args:
            current: Current batch number
            total: Total batches in epoch (train + dev)
        """
        self.batch_progress_bar.setMaximum(total)
        self.batch_progress_bar.setValue(current)

    def _on_epoch_complete(self, metrics: dict) -> None:
        """Append epoch results to log and update graph.

        Args:
            metrics: Dict with epoch, total_epochs, train_loss, train_acc,
                     dev_loss, dev_acc, is_best
        """
        epoch = metrics['epoch']
        total = metrics['total_epochs']
        remaining = total - epoch

        now = time.time()
        self.epoch_times.append(now)

        # Calculate time per epoch from actual measurements
        if len(self.epoch_times) == 1:
            avg_epoch_time = now - self.start_time
        else:
            avg_epoch_time = (now - self.start_time) / epoch

        elapsed = self._format_duration(now - self.start_time)
        eta = self._format_duration(avg_epoch_time * remaining)

        self.epoch_label.setText(f"Epoch {epoch} / {total}")
        self.time_label.setText(f"Elapsed: {elapsed}  |  ETA: {eta}")

        best_marker = "  *best*" if metrics['is_best'] else ""
        line = (
            f"Epoch {epoch}/{total}  |  "
            f"Train Loss: {metrics['train_loss']:.4f}  Acc: {metrics['train_acc']:.1f}%  |  "
            f"Dev Loss: {metrics['dev_loss']:.4f}  Acc: {metrics['dev_acc']:.1f}%"
            f"{best_marker}"
        )
        self.log_text.append(line)

        # Update graph data
        self._epochs.append(epoch)
        self._train_loss.append(metrics['train_loss'])
        self._dev_loss.append(metrics['dev_loss'])
        self._train_acc.append(metrics['train_acc'])
        self._dev_acc.append(metrics['dev_acc'])
        self._update_graph()

    def _on_stop_clicked(self) -> None:
        """Handle Stop button click."""
        if self.worker:
            self.log_text.append("\nStopping training...")
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def _on_training_complete(self, best_acc: float) -> None:
        """Show completion message.

        Args:
            best_acc: Best dev accuracy achieved
        """
        total_time = self._format_duration(time.time() - self.start_time)
        self.epoch_label.setText("Training complete")
        self.time_label.setText(f"Total time: {total_time}")
        self.log_text.append(f"\nBest Dev Accuracy: {best_acc:.1f}%  |  Total: {total_time}")
        self.batch_progress_bar.setValue(self.batch_progress_bar.maximum())
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)

    def _on_training_error(self, error: str) -> None:
        """Show error message.

        Args:
            error: Error description
        """
        self.epoch_label.setText("Training failed")
        self.log_text.append(f"\nERROR: {error}")
        self.stop_button.setEnabled(False)
        self.close_button.setEnabled(True)

    def closeEvent(self, event) -> None:
        """Wait for worker thread before closing."""
        if self.worker and self.worker.isRunning():
            self.worker.wait(5000)
        super().closeEvent(event)
