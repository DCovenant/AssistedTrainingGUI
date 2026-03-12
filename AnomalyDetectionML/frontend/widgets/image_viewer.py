from pathlib import Path
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QPoint, QRect, pyqtSignal


class ImageViewer(QWidget):
    """Image viewer with zoom, pan, and selection capabilities."""

    ZOOM_STEP: float = 0.1
    MIN_ZOOM: float = 0.1
    MAX_ZOOM: float = 3.0
    INITIAL_ZOOM: float = 0.7

    selection_made = pyqtSignal(QRect)
    selection_clicked = pyqtSignal(int, int, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.original_image: QPixmap | None = None
        self.zoom_level: float = self.INITIAL_ZOOM
        self.pan_offset: QPoint = QPoint(0, 0)
        self.is_dragging: bool = False
        self.drag_start_pos: QPoint = QPoint(0, 0)
        self.selection_mode: bool = False
        self.selection_start: QPoint = QPoint(0, 0)
        self.selection_rect: QRect = QRect()
        self.completed_selections: list[dict] = []
        self.delete_mode: bool = False
        self.hovered_selection_index: int | None = None
        self.excluded_regions: list[QRect] = []
        self.show_background: bool = False

        self.setStyleSheet("background-color: white;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def load_image(self, image_path: str) -> None:
        """Load image from file path and fit to viewport."""
        if Path(image_path).exists():
            self.original_image = QPixmap(image_path)
            self.zoom_level = self.INITIAL_ZOOM
            self._fit_image_to_viewport()

    def resizeEvent(self, event) -> None:
        """Recenter image when viewport is resized."""
        super().resizeEvent(event)
        if self.original_image is not None:
            self.pan_offset = self._calculate_centered_pan_offset()
            self.update()

    def _fit_image_to_viewport(self) -> None:
        """Calculate zoom level to fit entire image in viewport and center it."""
        if self.original_image is None or self.width() == 0 or self.height() == 0:
            self.zoom_level = self.INITIAL_ZOOM
            return

        width_ratio = self.width() / self.original_image.width()
        height_ratio = self.height() / self.original_image.height()

        fit_zoom = min(width_ratio, height_ratio) * 0.95
        self.zoom_level = max(self.MIN_ZOOM, min(fit_zoom, self.MAX_ZOOM))
        self.pan_offset = self._calculate_centered_pan_offset()
        self.update()

    def _calculate_centered_pan_offset(self) -> QPoint:
        """Calculate pan offset to center the scaled image in viewport."""
        if self.original_image is None:
            return QPoint(0, 0)

        scaled_width = int(self.original_image.width() * self.zoom_level)
        scaled_height = int(self.original_image.height() * self.zoom_level)

        center_x = (self.width() - scaled_width) // 2
        center_y = (self.height() - scaled_height) // 2

        return QPoint(max(0, center_x), max(0, center_y))

    def wheelEvent(self, event) -> None:
        """Handle mouse wheel for zoom centered on cursor."""
        if self.original_image is None:
            return

        mouse_pos = QPoint(int(event.position().x()), int(event.position().y()))
        delta = event.angleDelta().y()
        zoom_factor = self.ZOOM_STEP if delta > 0 else -self.ZOOM_STEP
        new_zoom = self.zoom_level + zoom_factor

        if self.MIN_ZOOM <= new_zoom <= self.MAX_ZOOM:
            image_point = (mouse_pos - self.pan_offset) / self.zoom_level
            self.zoom_level = new_zoom
            self.pan_offset = mouse_pos - image_point * self.zoom_level

            # Recenter if scaled image is smaller than viewport
            centered_offset = self._calculate_centered_pan_offset()
            if self.zoom_level * self.original_image.width() < self.width():
                self.pan_offset.setX(centered_offset.x())
            if self.zoom_level * self.original_image.height() < self.height():
                self.pan_offset.setY(centered_offset.y())

            self.update()

    def set_selection_mode(self, enabled: bool) -> None:
        """Enable or disable selection mode."""
        self.selection_mode = enabled
        self.selection_rect = QRect()
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_delete_mode(self, enabled: bool) -> None:
        """Enable or disable delete mode."""
        self.delete_mode = enabled
        if not enabled:
            self.hovered_selection_index = None
        self.setCursor(Qt.CursorShape.PointingHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def add_selection(self, rect: QRect, color: str, text: str = "", is_prediction: bool = False) -> None:
        """Add completed selection to display.

        Args:
            rect: Selection rectangle in image coordinates
            color: Hex color string for rectangle outline
            text: Text content to display on selection
            is_prediction: Mark as model prediction for selective clearing
        """
        self.completed_selections.append({
            "rect": rect,
            "color": color,
            "text": text,
            "is_prediction": is_prediction
        })
        self.update()

    def clear_selections(self) -> None:
        """Clear all completed selections."""
        self.completed_selections.clear()
        self.update()

    def clear_predictions(self) -> None:
        """Remove only model predictions, keep user annotations."""
        self.completed_selections = [
            s for s in self.completed_selections if not s.get("is_prediction", False)
        ]
        self.update()

    def set_excluded_regions(self, regions: list[QRect]) -> None:
        """Set annotated regions to exclude from background overlay.

        Args:
            regions: List of QRect objects to exclude from background paint
        """
        self.excluded_regions = regions
        self.update()

    def toggle_background_display(self) -> None:
        """Toggle background crops visibility."""
        self.show_background = not self.show_background
        self.update()

    def get_image_dimensions(self) -> tuple[int, int] | None:
        """Get original image width and height in pixels.

        Returns:
            Tuple of (width, height) or None if no image loaded
        """
        if self.original_image is None:
            return None
        return (self.original_image.width(), self.original_image.height())

    def mousePressEvent(self, event) -> None:
        """Handle mouse press for pan, selection, or delete."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.delete_mode:
                screen_point = event.pos()
                image_point = self._screen_to_image_point(screen_point)
                for selection in self.completed_selections:
                    if selection["rect"].contains(image_point):
                        rect = selection["rect"]
                        self.selection_clicked.emit(rect.x(), rect.y(), rect.width(), rect.height())
                        return
            elif self.selection_mode:
                self.selection_start = event.pos()
                self.selection_rect = QRect(self.selection_start, self.selection_start)
            else:
                self.is_dragging = True
                self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move for pan, selection, or delete hover."""
        # Always track which selection is under the mouse for hover effect
        screen_point = event.pos()
        image_point = self._screen_to_image_point(screen_point)
        hovered_index = None
        for idx, selection in enumerate(self.completed_selections):
            if selection["rect"].contains(image_point):
                hovered_index = idx
                break
        if hovered_index != self.hovered_selection_index:
            self.hovered_selection_index = hovered_index
            self.update()

        if self.delete_mode:
            pass  # Hover tracking handled above
        elif self.selection_mode:
            screen_rect = QRect(self.selection_start, event.pos()).normalized()
            self.selection_rect = self._screen_to_image_rect(screen_rect)
            self.update()
        elif self.is_dragging:
            delta = event.pos() - self.drag_start_pos
            self.pan_offset += delta
            self.drag_start_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release to end pan or selection."""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.selection_mode and self.selection_rect.width() > 0 and self.selection_rect.height() > 0:
                self.selection_made.emit(self.selection_rect)
                self.selection_rect = QRect()
                self.update()
            self.is_dragging = False

    def _screen_to_image_rect(self, screen_rect: QRect) -> QRect:
        """Convert screen coordinates to image coordinates."""
        top_left = self._screen_to_image_point(screen_rect.topLeft())
        bottom_right = self._screen_to_image_point(screen_rect.bottomRight())
        return QRect(top_left, bottom_right).normalized()

    def _screen_to_image_point(self, screen_point: QPoint) -> QPoint:
        """Convert screen point to image point."""
        x = int((screen_point.x() - self.pan_offset.x()) / self.zoom_level)
        y = int((screen_point.y() - self.pan_offset.y()) / self.zoom_level)
        return QPoint(x, y)

    def _image_to_screen_point(self, image_point: QPoint) -> QPoint:
        """Convert image point to screen point."""
        x = int(image_point.x() * self.zoom_level + self.pan_offset.x())
        y = int(image_point.y() * self.zoom_level + self.pan_offset.y())
        return QPoint(x, y)

    def _image_to_screen_rect(self, image_rect: QRect) -> QRect:
        """Convert image rectangle to screen rectangle."""
        top_left = self._image_to_screen_point(image_rect.topLeft())
        bottom_right = self._image_to_screen_point(image_rect.bottomRight())
        return QRect(top_left, bottom_right).normalized()

    def _draw_background_overlay(self, painter: QPainter, image_width: int, image_height: int) -> None:
        """Draw transparent overlay over background areas (unselected regions).

        Paints semi-transparent overlay over entire image, then excludes
        annotated regions to show which areas are background.

        Args:
            painter: QPainter instance for drawing
            image_width, image_height: Original image dimensions
        """
        overlay_color = QColor(135, 206, 235, 80)
        image_rect = QRect(0, 0, image_width, image_height)
        screen_image_rect = self._image_to_screen_rect(image_rect)

        painter.fillRect(screen_image_rect, overlay_color)

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        painter.fillColor = QColor(0, 0, 0, 255)

        for region in self.excluded_regions:
            screen_region = self._image_to_screen_rect(region)
            painter.fillRect(screen_region, QColor(0, 0, 0, 255))

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def _draw_selection_text(self, painter: QPainter, screen_rect: QRect, text: str, color: str) -> None:
        """Draw text label on top of selection rectangle.

        Args:
            painter: QPainter instance for drawing
            screen_rect: Rectangle in screen coordinates
            text: Text to display
            color: Hex color string for text
        """
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        text_x = screen_rect.x() + 5
        text_y = screen_rect.y() - 5

        painter.setPen(QPen(QColor("black"), 1))
        painter.drawText(text_x + 1, text_y + 1, text)

        painter.setPen(QPen(QColor(color), 1))
        painter.drawText(text_x, text_y, text)

    def paintEvent(self, event) -> None:
        """Draw image with zoom, pan, and selections."""
        if self.original_image is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        scaled_width = int(self.original_image.width() * self.zoom_level)
        scaled_height = int(self.original_image.height() * self.zoom_level)

        scaled_image = self.original_image.scaled(
            scaled_width,
            scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        painter.drawPixmap(self.pan_offset, scaled_image)

        if self.show_background:
            self._draw_background_overlay(painter, self.original_image.width(), self.original_image.height())

        for idx, selection in enumerate(self.completed_selections):
            image_rect = selection["rect"]
            color = selection["color"]
            text = selection["text"]
            screen_rect = self._image_to_screen_rect(image_rect)
            is_hovered = (idx == self.hovered_selection_index)

            pen_width = 4 if is_hovered else 2
            pen = QPen(QColor(color), pen_width, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(screen_rect)

            if is_hovered and self.delete_mode:
                painter.fillRect(screen_rect, QColor(255, 255, 255, 30))

            if text:
                self._draw_selection_text(painter, screen_rect, text, color)

        if self.selection_mode and not self.selection_rect.isNull():
            screen_rect = self._image_to_screen_rect(self.selection_rect)
            pen = QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(screen_rect)
