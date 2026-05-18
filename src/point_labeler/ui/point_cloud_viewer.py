"""Point cloud viewer with camera-aware screen-space annotation."""

from __future__ import annotations

import logging
from typing import Callable, Literal

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

ThemeMode = Literal["dark", "light"]
AnnotationMode = Literal["浏览", "框选", "刷子"]
PolygonCallback = Callable[[np.ndarray], None]
BrushCallback = Callable[[np.ndarray], None]
BrushRadiusCallback = Callable[[float], None]
BrushStrokeCallback = Callable[[], None]

logger = logging.getLogger("point_labeler.viewer")
logger.setLevel(logging.WARNING)


class _OverlayWidget(QWidget):
    """Draw annotation overlays above the 3D canvas."""

    def __init__(self, viewer: "PointCloudViewer", parent: QWidget) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        self._viewer._draw_overlay(self)


class PointCloudViewer(QWidget):
    """3D point cloud viewer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend_ready = False
        self._point_size = 2
        self._theme_mode: ThemeMode = "dark"
        self._point_color = "#ffffff"
        self._annotation_mode: AnnotationMode = "浏览"
        self._brush_radius = 30.0  # pixels
        self._annotation_active = False
        self._alt_was_pressed = False
        self._alt_down = False

        self._current_points: np.ndarray | None = None
        self._current_xyz: np.ndarray | None = None
        self._current_rgb_colors: np.ndarray | None = None
        self._filtered_to_original: np.ndarray | None = None

        self._polygon_callback: PolygonCallback | None = None
        self._brush_callback: BrushCallback | None = None
        self._brush_radius_callback: BrushRadiusCallback | None = None
        self._brush_stroke_begin_callback: BrushStrokeCallback | None = None
        self._brush_stroke_end_callback: BrushStrokeCallback | None = None

        self._brush_dragging = False
        self._brush_center_xy: tuple[float, float] | None = None
        self._origin_logged = False
        self._brush_cursor_cache_key: tuple[int, int] | None = None
        self._brush_cursor_cache: QCursor | None = None

        self._polygon_points_xy: list[tuple[float, float]] = []
        self._polygon_drag_vertex_idx: int | None = None

        self._plotter = None
        self._overlay: _OverlayWidget | None = None
        self._actor = None
        self._fallback_label: QLabel | None = None
        self._box_actor2d = None
        self._box_mapper2d = None
        self._box_polydata = None
        self._box_preview_visible = False
        self._original_to_filtered: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._root_layout = layout

        self._setup_backend()

    def _setup_backend(self) -> None:
        try:
            from pyvistaqt import QtInteractor  # type: ignore
        except Exception:
            self._backend_ready = False
            self._fallback_label = QLabel("3D后端不可用（需要 pyvista / pyvistaqt / vtk）。")
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._root_layout.addWidget(self._fallback_label)
            return

        self._backend_ready = True
        self._plotter = QtInteractor(self)
        self._plotter.setMouseTracking(True)
        self._plotter.setFocusPolicy(Qt.StrongFocus)
        self._plotter.enable_trackball_style()
        try:
            self._plotter.interactor.SetMotionFactor(1.0)
            self._plotter.interactor.SetMouseWheelMotionFactor(1.0)
        except Exception:
            pass

        self._plotter.installEventFilter(self)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        try:
            self._plotter.iren.add_observer("EndInteractionEvent", lambda *_: self._update_overlay())
        except Exception:
            pass

        self._apply_plotter_background()
        self._root_layout.addWidget(self._plotter)

        self._overlay = _OverlayWidget(self, self._plotter)
        self._sync_overlay_geometry()
        self._overlay.show()
        self._overlay.raise_()
        self._setup_box_preview_actor()
        logger.debug(
            "viewer backend ready: plotter=%s overlay=%s",
            self._plotter.__class__.__name__,
            self._overlay.__class__.__name__ if self._overlay else None,
        )

    @property
    def backend_ready(self) -> bool:
        return self._backend_ready

    def set_box_callback(self, callback: PolygonCallback | None) -> None:
        self._polygon_callback = callback

    def set_brush_callback(self, callback: BrushCallback | None) -> None:
        self._brush_callback = callback

    def set_brush_radius_callback(self, callback: BrushRadiusCallback | None) -> None:
        self._brush_radius_callback = callback

    def set_brush_stroke_begin_callback(self, callback: BrushStrokeCallback | None) -> None:
        self._brush_stroke_begin_callback = callback

    def set_brush_stroke_end_callback(self, callback: BrushStrokeCallback | None) -> None:
        self._brush_stroke_end_callback = callback

    def set_annotation_mode(self, mode: AnnotationMode) -> None:
        self._annotation_mode = mode
        self._annotation_active = False
        self._alt_down = False
        self._alt_was_pressed = False
        self._update_plotter_cursor()
        self._clear_annotation_primitives()
        logger.debug("set_annotation_mode mode=%s", mode)

    def set_brush_radius(self, radius: float) -> None:
        self._brush_radius = min(100.0, max(1.0, float(radius)))
        logger.debug("set_brush_radius radius=%.3f", self._brush_radius)
        self._brush_cursor_cache_key = None
        self._brush_cursor_cache = None
        if self._brush_radius_callback is not None:
            self._brush_radius_callback(self._brush_radius)
        self._update_plotter_cursor()
        self._update_overlay()

    def brush_radius(self) -> float:
        return self._brush_radius

    def set_theme(self, mode: ThemeMode) -> None:
        self._theme_mode = mode
        self._point_color = "#ffffff" if mode == "dark" else "#000000"
        self._apply_plotter_background()
        if self._current_points is not None:
            self.set_points(self._current_points, reset_camera=False, rgb_colors=self._current_rgb_colors)

    def set_point_size(self, size: int) -> None:
        self._point_size = max(1, int(size))
        if self._current_points is not None:
            self.set_points(self._current_points, reset_camera=False, rgb_colors=self._current_rgb_colors)

    def reset_camera(self) -> None:
        if self._backend_ready:
            self._plotter.reset_camera()
            self._plotter.render()

    def clear(self) -> None:
        self._current_points = None
        self._current_xyz = None
        self._current_rgb_colors = None
        self._filtered_to_original = None
        self._original_to_filtered = None
        self._remove_point_actor()
        if self._backend_ready:
            self._plotter.render()
        self._clear_annotation_primitives()

    def set_points(
        self,
        points: np.ndarray,
        reset_camera: bool = True,
        rgb_colors: np.ndarray | None = None,
    ) -> None:
        self._current_points = points
        self._current_rgb_colors = rgb_colors
        if not self._backend_ready:
            return
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"points must be Nx3/Nx4, got {points.shape}")

        xyz = np.asarray(points[:, :3], dtype=np.float32)
        finite_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite_mask]
        self._current_xyz = xyz
        self._filtered_to_original = np.where(finite_mask)[0].astype(np.int64, copy=False)
        self._original_to_filtered = np.full(points.shape[0], -1, dtype=np.int64)
        self._original_to_filtered[self._filtered_to_original] = np.arange(
            self._filtered_to_original.shape[0], dtype=np.int64
        )

        rgb_filtered: np.ndarray | None = None
        if rgb_colors is not None:
            if rgb_colors.ndim != 2 or rgb_colors.shape[1] != 3:
                raise ValueError(f"rgb_colors must be Nx3, got {rgb_colors.shape}")
            if rgb_colors.shape[0] != points.shape[0]:
                raise ValueError("rgb_colors length mismatch")
            rgb_filtered = np.asarray(rgb_colors, dtype=np.uint8)[finite_mask]

        camera_position = self._plotter.camera_position if not reset_camera else None
        self._remove_point_actor()
        if xyz.shape[0] == 0:
            self._plotter.render()
            return

        if rgb_filtered is None:
            self._actor = self._plotter.add_points(
                xyz,
                color=self._point_color,
                point_size=self._point_size,
                render_points_as_spheres=False,
                show_scalar_bar=False,
                pickable=True,
            )
        else:
            self._actor = self._plotter.add_points(
                xyz,
                scalars=rgb_filtered,
                rgb=True,
                point_size=self._point_size,
                render_points_as_spheres=False,
                show_scalar_bar=False,
                pickable=True,
            )

        if reset_camera:
            self._plotter.reset_camera()
        elif camera_position is not None:
            self._plotter.camera_position = camera_position
        self._plotter.render()

    def update_rgb_for_original_indices(
        self,
        original_indices: np.ndarray,
        rgb_color: np.ndarray,
    ) -> bool:
        """Update actor point colors in place for selected original indices."""
        if (
            self._plotter is None
            or self._actor is None
            or self._original_to_filtered is None
            or original_indices.size == 0
        ):
            return False

        try:
            mapper = self._actor.mapper
            dataset = mapper.dataset
            if dataset is None or "Data" not in dataset.point_data:
                return False
            scalar = np.asarray(dataset.point_data["Data"])
            if scalar.ndim != 2 or scalar.shape[1] != 3:
                return False

            idx = np.asarray(original_indices, dtype=np.int64)
            in_range = (idx >= 0) & (idx < self._original_to_filtered.shape[0])
            if not np.any(in_range):
                return False
            idx = idx[in_range]
            local = self._original_to_filtered[idx]
            local = local[local >= 0]
            if local.size == 0:
                return False
            local = np.unique(local)

            color = np.asarray(rgb_color, dtype=np.uint8).reshape(1, 3)
            scalar[local] = color
            dataset.point_data["Data"] = scalar
            dataset.Modified()
            mapper.Modified()

            # Keep cached RGB state consistent with in-place actor updates.
            if self._current_rgb_colors is not None:
                cache = np.asarray(self._current_rgb_colors)
                if cache.ndim == 2 and cache.shape[1] == 3:
                    cache_idx = idx
                    if cache_idx.size > 0:
                        cache[cache_idx] = color
                        self._current_rgb_colors = cache

            self._plotter.render()
            return True
        except Exception:
            logger.exception("update_rgb_for_original_indices failed")
            return False

    def pick_points_in_display_circle(self, center_xy: tuple[float, float], radius_px: float) -> np.ndarray:
        if self._current_xyz is None or self._current_xyz.shape[0] == 0:
            return np.array([], dtype=np.int64)

        projected = self._project_points_to_display(self._current_xyz)
        if projected is None:
            return np.array([], dtype=np.int64)

        disp_xy, valid = projected
        dx = disp_xy[:, 0] - float(center_xy[0])
        dy = disp_xy[:, 1] - float(center_xy[1])
        dist2 = dx * dx + dy * dy
        local = np.where(valid & (dist2 <= float(radius_px) * float(radius_px)))[0].astype(np.int64, copy=False)
        logger.debug(
            "pick_circle center=(%.1f,%.1f) radius=%.1f valid=%d selected=%d",
            center_xy[0],
            center_xy[1],
            radius_px,
            int(np.count_nonzero(valid)),
            int(local.size),
        )
        if valid.size > 0:
            logger.debug("pick_circle valid_ratio=%.4f", float(np.count_nonzero(valid)) / float(valid.size))

        if self._filtered_to_original is None:
            return local
        return self._filtered_to_original[local]

    def pick_points_in_display_polygon(self, polygon_xy: np.ndarray) -> np.ndarray:
        if (
            self._current_xyz is None
            or self._current_xyz.shape[0] == 0
            or polygon_xy.ndim != 2
            or polygon_xy.shape[0] < 3
            or polygon_xy.shape[1] != 2
        ):
            return np.array([], dtype=np.int64)

        projected = self._project_points_to_display(self._current_xyz)
        if projected is None:
            return np.array([], dtype=np.int64)

        disp_xy, valid_mask = projected
        if disp_xy.shape[0] == 0:
            return np.array([], dtype=np.int64)

        min_x = float(np.min(polygon_xy[:, 0]))
        max_x = float(np.max(polygon_xy[:, 0]))
        min_y = float(np.min(polygon_xy[:, 1]))
        max_y = float(np.max(polygon_xy[:, 1]))

        bbox_mask = (
            valid_mask
            & (disp_xy[:, 0] >= min_x)
            & (disp_xy[:, 0] <= max_x)
            & (disp_xy[:, 1] >= min_y)
            & (disp_xy[:, 1] <= max_y)
        )
        candidate_local = np.where(bbox_mask)[0]
        if candidate_local.size == 0:
            return np.array([], dtype=np.int64)

        inside = self._points_in_polygon(disp_xy[candidate_local], polygon_xy)
        selected_local = candidate_local[inside]
        logger.debug(
            "pick_polygon verts=%d candidate=%d selected=%d",
            int(polygon_xy.shape[0]),
            int(candidate_local.size),
            int(selected_local.size),
        )

        if self._filtered_to_original is None:
            return selected_local.astype(np.int64, copy=False)
        return self._filtered_to_original[selected_local]

    def eventFilter(self, obj, event):  # type: ignore[override]
        if not self._backend_ready or self._plotter is None:
            return False

        et = event.type()
        mouse_event_types = (
            QEvent.MouseButtonPress,
            QEvent.MouseButtonDblClick,
            QEvent.MouseMove,
            QEvent.MouseButtonRelease,
            QEvent.Wheel,
            QEvent.NativeGesture,
        )
        pan_keys = {
            Qt.Key_W,
            Qt.Key_A,
            Qt.Key_S,
            Qt.Key_D,
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Left,
            Qt.Key_Right,
        }

        related = obj is self._plotter
        if not related and isinstance(obj, QWidget):
            try:
                related = self._plotter.isAncestorOf(obj)
            except Exception:
                related = False

        # Border mouse events can be delivered to adjacent widgets. As long as
        # the cursor is inside plotter global rect, treat it as plotter-related.
        if not related and et in mouse_event_types:
            gp = self._extract_event_global(event)
            if gp is not None:
                try:
                    gp_q = QPoint(int(round(gp[0])), int(round(gp[1])))
                    top_left = self._plotter.mapToGlobal(QPoint(0, 0))
                    rect_g = self._plotter.rect().translated(top_left)
                    related = rect_g.contains(gp_q)
                except Exception:
                    related = False

        if not related:
            return False

        if et in (QEvent.Leave, QEvent.FocusOut):
            self._end_brush_stroke_if_needed()
            self._alt_down = False
            self._alt_was_pressed = False
            self._annotation_active = False
            self._clear_annotation_primitives()
            return False

        if et == QEvent.Enter:
            try:
                self._plotter.setFocus()
            except Exception:
                pass
            return False

        if et in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self._sync_overlay_geometry()
            return False

        if et == QEvent.KeyPress:
            try:
                if event.key() == Qt.Key_Alt:
                    self._alt_down = True
                    return False
                if self._handle_pan_key(event.key()):
                    return True
            except Exception:
                pass
            return False

        if et == QEvent.ShortcutOverride:
            try:
                if event.key() in pan_keys:
                    event.accept()
                    return True
            except Exception:
                pass
            return False

        if et == QEvent.KeyRelease:
            try:
                if event.key() in pan_keys:
                    return True
            except Exception:
                pass
            # fallthrough to Alt handling below
            try:
                if event.key() == Qt.Key_Alt:
                    self._end_brush_stroke_if_needed()
                    self._alt_down = False
                    self._alt_was_pressed = False
                    self._annotation_active = False
                    self._clear_annotation_primitives()
            except Exception:
                pass
            return False

        if et not in mouse_event_types:
            return False

        alt_pressed = self._is_alt_pressed(event)
        if self._alt_was_pressed and not alt_pressed:
            self._clear_annotation_primitives()
        self._alt_was_pressed = alt_pressed
        camera_active = self._annotation_mode == "浏览" or not alt_pressed
        self._annotation_active = (self._annotation_mode != "浏览") and alt_pressed
        self._update_plotter_cursor()

        if et != QEvent.MouseMove:
            xy_dbg = self._extract_event_xy(obj, event)
            logger.debug(
                "event type=%s obj=%s alt=%s mode=%s camera_active=%s xy=%s",
                int(et),
                type(obj).__name__,
                alt_pressed,
                self._annotation_mode,
                camera_active,
                xy_dbg,
            )

        if et == QEvent.NativeGesture:
            if self._annotation_mode == "刷子" and alt_pressed:
                try:
                    gtype = event.gestureType()
                except Exception:
                    gtype = None
                # Trackpad pinch-to-zoom fallback on platforms where wheel delta is zero.
                if gtype == Qt.ZoomNativeGesture:
                    try:
                        val = float(event.value())
                    except Exception:
                        val = 0.0
                    logger.debug("native_gesture zoom value=%.6f", val)
                    if abs(val) > 1e-6:
                        scale = 1.0 + val
                        if scale <= 0.0:
                            scale = 0.5
                        self.set_brush_radius(self._brush_radius * scale)
                        logger.debug("native_gesture new_radius=%.3f", self._brush_radius)
                    return True
            return False

        if et == QEvent.Wheel:
            if self._annotation_mode == "刷子" and alt_pressed:
                delta = 0
                ax = ay = px = py = 0
                try:
                    ad = event.angleDelta()
                    ax, ay = int(ad.x()), int(ad.y())
                except Exception:
                    ax, ay = 0, 0
                try:
                    pd = event.pixelDelta()
                    px, py = int(pd.x()), int(pd.y())
                except Exception:
                    px, py = 0, 0

                candidates = [ay, ax, py, px]
                for cand in candidates:
                    if cand != 0:
                        delta = cand
                        break

                logger.debug(
                    "wheel delta_components angle=(%d,%d) pixel=(%d,%d) chosen=%d",
                    ax,
                    ay,
                    px,
                    py,
                    delta,
                )
                if delta != 0:
                    logger.debug("wheel delta=%d old_radius=%.3f", int(delta), self._brush_radius)
                    self.set_brush_radius(self._brush_radius * (1.1 if delta > 0 else 0.9))
                    logger.debug("wheel new_radius=%.3f", self._brush_radius)
                return True
            return False

        if camera_active:
            self._end_brush_stroke_if_needed()
            self._brush_dragging = False
            self._polygon_drag_vertex_idx = None
            if self._annotation_mode == "刷子":
                self._brush_center_xy = None
            self._update_overlay()
            return False

        if self._annotation_mode == "刷子":
            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._brush_dragging = True
                if self._brush_stroke_begin_callback is not None:
                    self._brush_stroke_begin_callback()
                self._apply_brush_from_event(obj, event, apply=True)
                return True
            if et == QEvent.MouseMove:
                self._apply_brush_from_event(obj, event, apply=self._brush_dragging)
                return True
            if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._end_brush_stroke_if_needed()
                self._brush_dragging = False
                self._brush_center_xy = None
                self._update_overlay()
                return True
            return True

        if self._annotation_mode == "框选":
            if et == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
                self._apply_polygon_selection()
                return True

            if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                xy = self._extract_event_xy(obj, event)
                if xy is not None:
                    self._log_box_cursor(event, xy, action="press")
                    idx = self._find_polygon_vertex_near(xy, 10.0)
                    if idx is None:
                        self._polygon_points_xy.append(xy)
                        idx = len(self._polygon_points_xy) - 1
                    self._polygon_drag_vertex_idx = idx
                    self._update_overlay()
                return True

            if et == QEvent.MouseMove and self._polygon_drag_vertex_idx is not None:
                xy = self._extract_event_xy(obj, event)
                if xy is not None and 0 <= self._polygon_drag_vertex_idx < len(self._polygon_points_xy):
                    self._log_box_cursor(event, xy, action="drag")
                    self._polygon_points_xy[self._polygon_drag_vertex_idx] = xy
                    self._update_overlay()
                return True

            if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._polygon_drag_vertex_idx = None
                return True

            if et == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                self._apply_polygon_selection()
                return True

            if et == QEvent.MouseButtonPress and event.button() == Qt.MiddleButton:
                self._polygon_points_xy.clear()
                self._polygon_drag_vertex_idx = None
                self._update_overlay()
                return True

            return True

        return False

    def _apply_brush_from_event(self, obj, event, apply: bool) -> None:
        xy = self._extract_event_xy(obj, event)
        if xy is None:
            return
        self._brush_center_xy = xy
        mouse_global = self._extract_event_global(event)
        self._log_brush_cursor(mouse_global, xy, apply)
        self._update_overlay()

        if not apply or self._brush_callback is None:
            return

        indices = self.pick_points_in_display_circle(xy, self._brush_radius)
        if indices.size > 0:
            self._log_brush_selection_points(indices, max_points=8)
            self._brush_callback(indices)

    def _apply_polygon_selection(self) -> None:
        if self._polygon_callback is None or len(self._polygon_points_xy) < 3:
            logger.debug("polygon_apply skipped verts=%d", len(self._polygon_points_xy))
            return

        self._log_polygon_overlay_points()
        poly = np.array(self._polygon_points_xy, dtype=np.float64)
        indices = self.pick_points_in_display_polygon(poly)
        if indices.size > 0:
            logger.debug("polygon_apply selected=%d", int(indices.size))
            self._polygon_callback(indices)
        self._clear_annotation_primitives()

    def _find_polygon_vertex_near(self, xy: tuple[float, float], threshold_px: float) -> int | None:
        if not self._polygon_points_xy:
            return None
        x, y = float(xy[0]), float(xy[1])
        best_idx: int | None = None
        best_dist2 = float(threshold_px) * float(threshold_px)
        for i, p in enumerate(self._polygon_points_xy):
            dx = p[0] - x
            dy = p[1] - y
            d2 = dx * dx + dy * dy
            if d2 <= best_dist2:
                best_dist2 = d2
                best_idx = i
        return best_idx

    def _extract_event_xy(self, obj, event) -> tuple[float, float] | None:
        if self._plotter is None:
            return None

        # Always normalize through global position to avoid per-widget local
        # coordinate differences near edges of the VTK/Qt interactor.
        gp = self._extract_event_global(event)
        if gp is not None:
            p = self._plotter.mapFromGlobal(QPoint(int(round(gp[0])), int(round(gp[1]))))
            return float(p.x()), float(p.y())

        local_xy: tuple[float, float] | None = None
        try:
            p = event.position()
            local_xy = float(p.x()), float(p.y())
        except Exception:
            pass
        if local_xy is None:
            try:
                local_xy = float(event.x()), float(event.y())
            except Exception:
                return None
        if not isinstance(obj, QWidget):
            return None
        if obj is self._plotter:
            return local_xy
        mapped = obj.mapTo(self._plotter, QPointF(local_xy[0], local_xy[1]).toPoint())
        return float(mapped.x()), float(mapped.y())

    def _extract_event_global(self, event) -> tuple[float, float] | None:
        try:
            gp = event.globalPosition()
            return float(gp.x()), float(gp.y())
        except Exception:
            pass
        try:
            gp = event.globalPos()
            return float(gp.x()), float(gp.y())
        except Exception:
            return None

    def _plotter_xy_to_global(self, xy: tuple[float, float]) -> tuple[float, float] | None:
        if self._plotter is None:
            return None
        p = self._plotter.mapToGlobal(QPoint(int(round(xy[0])), int(round(xy[1]))))
        return float(p.x()), float(p.y())

    def _log_brush_cursor(
        self,
        mouse_global: tuple[float, float] | None,
        brush_xy: tuple[float, float],
        apply: bool,
    ) -> None:
        self._log_plotter_vtk_top_left_once()
        brush_global = self._plotter_xy_to_global(brush_xy)
        if mouse_global is None:
            logger.info(
                "mouse_global=(N/A,N/A) brush_global=(%s,%s) brush_local=(%.1f,%.1f) apply=%s radius=%.1f",
                f"{brush_global[0]:.1f}" if brush_global is not None else "N/A",
                f"{brush_global[1]:.1f}" if brush_global is not None else "N/A",
                brush_xy[0],
                brush_xy[1],
                apply,
                self._brush_radius,
            )
            return
        logger.debug(
            "mouse_global=(%.1f,%.1f) brush_global=(%s,%s) brush_local=(%.1f,%.1f) apply=%s radius=%.1f",
            mouse_global[0],
            mouse_global[1],
            f"{brush_global[0]:.1f}" if brush_global is not None else "N/A",
            f"{brush_global[1]:.1f}" if brush_global is not None else "N/A",
            brush_xy[0],
            brush_xy[1],
            apply,
            self._brush_radius,
        )

    def _log_plotter_vtk_top_left_once(self) -> None:
        if self._origin_logged or self._plotter is None:
            return
        self._origin_logged = True
        try:
            plotter_tl = self._plotter.mapToGlobal(QPoint(0, 0))
            plotter_tl_xy = (float(plotter_tl.x()), float(plotter_tl.y()))

            rw = self._plotter.GetRenderWindow()
            vtk_w, vtk_h = rw.GetSize() if rw is not None else (0, 0)
            dpr = self._pixel_ratio()

            # VTK display top-left corresponds to Qt local (0, 0):
            # y_qt = qt_h - (y_vtk / dpr) - 1, with y_vtk = vtk_h - 1.
            vtk_tl_local_qt = (0.0, (float(max(vtk_h, 1)) / dpr) - float(self._plotter.height()))
            vtk_tl_global = self._plotter.mapToGlobal(
                QPoint(int(round(vtk_tl_local_qt[0])), int(round(vtk_tl_local_qt[1])))
            )
            vtk_tl_global_xy = (float(vtk_tl_global.x()), float(vtk_tl_global.y()))

            dx = vtk_tl_global_xy[0] - plotter_tl_xy[0]
            dy = vtk_tl_global_xy[1] - plotter_tl_xy[1]
            logger.debug(
                "origin_compare plotter_tl_global=(%.1f,%.1f) vtk_tl_global=(%.1f,%.1f) "
                "delta=(%.1f,%.1f) qt_size=(%d,%d) vtk_size=(%d,%d) dpr=%.3f",
                plotter_tl_xy[0],
                plotter_tl_xy[1],
                vtk_tl_global_xy[0],
                vtk_tl_global_xy[1],
                dx,
                dy,
                int(self._plotter.width()),
                int(self._plotter.height()),
                int(vtk_w),
                int(vtk_h),
                float(dpr),
            )
        except Exception:
            logger.exception("origin_compare failed")

    def _log_polygon_overlay_points(self) -> None:
        if not self._polygon_points_xy:
            return
        points_text = ", ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in self._polygon_points_xy)
        logger.debug("polygon_screen_points=[%s]", points_text)

    def _log_box_cursor(self, event, box_xy: tuple[float, float], action: str) -> None:
        mouse_global = self._extract_event_global(event)
        box_global = self._plotter_xy_to_global(box_xy)
        if mouse_global is None:
            logger.debug(
                "mouse_global=(N/A,N/A) box_global=(%s,%s) box_local=(%.1f,%.1f) action=%s",
                f"{box_global[0]:.1f}" if box_global is not None else "N/A",
                f"{box_global[1]:.1f}" if box_global is not None else "N/A",
                box_xy[0],
                box_xy[1],
                action,
            )
            return
        logger.debug(
            "mouse_global=(%.1f,%.1f) box_global=(%s,%s) box_local=(%.1f,%.1f) action=%s",
            mouse_global[0],
            mouse_global[1],
            f"{box_global[0]:.1f}" if box_global is not None else "N/A",
            f"{box_global[1]:.1f}" if box_global is not None else "N/A",
            box_xy[0],
            box_xy[1],
            action,
        )

    def _log_brush_selection_points(self, indices: np.ndarray, max_points: int = 8) -> None:
        if self._current_xyz is None or self._filtered_to_original is None:
            return
        if indices.size == 0:
            logger.debug("brush_selected_points count=0")
            return

        max_idx = int(np.max(indices))
        if max_idx >= int(self._filtered_to_original.shape[0]):
            logger.debug("brush_selected_points count=%d sample=[index_mismatch]", int(indices.size))
            return
        local_map = np.full(self._filtered_to_original.shape[0], -1, dtype=np.int64)
        local_map[self._filtered_to_original] = np.arange(self._filtered_to_original.shape[0], dtype=np.int64)
        local_indices = local_map[indices]
        valid_local = local_indices >= 0
        if not np.any(valid_local):
            return
        valid_indices = indices[valid_local]
        local_indices = local_indices[valid_local]
        xyz = self._current_xyz[local_indices]
        projected = self._project_points_to_display(xyz)
        if projected is None:
            return
        disp_xy, valid_disp = projected

        count = int(indices.size)
        samples: list[str] = []
        limit = min(int(max_points), int(local_indices.size))
        for i in range(limit):
            world = xyz[i]
            if valid_disp[i]:
                sx, sy = float(disp_xy[i, 0]), float(disp_xy[i, 1])
                samples.append(
                    f"id={int(valid_indices[i])} world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f}) screen=({sx:.1f},{sy:.1f})"
                )
            else:
                samples.append(
                    f"id={int(valid_indices[i])} world=({world[0]:.3f},{world[1]:.3f},{world[2]:.3f}) screen=(N/A,N/A)"
                )
        logger.debug("brush_selected_points count=%d sample=[%s]", count, "; ".join(samples))

    def _project_points_to_display(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self._plotter is None:
            return None
        try:
            renderer = self._plotter.renderer
            dpr = self._pixel_ratio()
            qt_h = float(max(1, int(self._plotter.height())))
            n = xyz.shape[0]
            disp = np.zeros((n, 2), dtype=np.float64)
            valid = np.zeros(n, dtype=bool)

            for i in range(n):
                p = xyz[i]
                if not np.isfinite(p).all():
                    continue
                renderer.SetWorldPoint(float(p[0]), float(p[1]), float(p[2]), 1.0)
                renderer.WorldToDisplay()
                out = renderer.GetDisplayPoint()
                if out is None:
                    continue
                x_vtk, y_vtk, z_vtk = float(out[0]), float(out[1]), float(out[2])
                if not np.isfinite(x_vtk) or not np.isfinite(y_vtk) or not np.isfinite(z_vtk):
                    continue
                # Keep only points that are inside the camera depth range.
                if z_vtk < 0.0 or z_vtk > 1.0:
                    continue
                x_qt = x_vtk / dpr
                y_qt = qt_h - (y_vtk / dpr) - 1.0
                if not np.isfinite(x_qt) or not np.isfinite(y_qt):
                    continue
                disp[i, 0] = x_qt
                disp[i, 1] = y_qt
                valid[i] = True

            return disp, valid
        except Exception:
            logger.exception("project_points_to_display failed")
            return None

    @staticmethod
    def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
        x = points_xy[:, 0]
        y = points_xy[:, 1]
        poly_x = polygon_xy[:, 0]
        poly_y = polygon_xy[:, 1]

        inside = np.zeros(points_xy.shape[0], dtype=bool)
        j = polygon_xy.shape[0] - 1

        for i in range(polygon_xy.shape[0]):
            xi, yi = poly_x[i], poly_y[i]
            xj, yj = poly_x[j], poly_y[j]

            denom = (yj - yi)
            denom = denom if abs(denom) > 1e-12 else (1e-12 if denom >= 0 else -1e-12)
            intersects = ((yi > y) != (yj > y)) & (x < ((xj - xi) * (y - yi) / denom + xi))
            inside ^= intersects
            j = i

        return inside

    def _remove_point_actor(self) -> None:
        if self._plotter is None or self._actor is None:
            return
        try:
            self._plotter.remove_actor(self._actor, reset_camera=False)
        except Exception:
            self._plotter.clear()
        self._actor = None

    def _apply_plotter_background(self) -> None:
        if self._backend_ready:
            self._plotter.set_background("#ffffff" if self._theme_mode == "light" else "#000000")

    def _setup_box_preview_actor(self) -> None:
        if not self._backend_ready or self._plotter is None:
            return
        try:
            from vtkmodules.vtkCommonDataModel import vtkPolyData
            from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D
        except Exception:
            logger.exception("setup_box_preview_actor failed: vtk imports unavailable")
            return

        if self._box_actor2d is not None:
            return

        self._box_polydata = vtkPolyData()
        self._box_mapper2d = vtkPolyDataMapper2D()
        self._box_mapper2d.SetInputData(self._box_polydata)

        self._box_actor2d = vtkActor2D()
        self._box_actor2d.SetMapper(self._box_mapper2d)
        prop = self._box_actor2d.GetProperty()
        prop.SetColor(0.254, 0.784, 1.0)  # #41c8ff
        prop.SetLineWidth(2.0)
        prop.SetPointSize(7.0)
        try:
            prop.SetDisplayLocationToForeground()
        except Exception:
            pass
        self._box_actor2d.SetVisibility(False)

        self._plotter.renderer.AddActor2D(self._box_actor2d)

    def _qt_xy_to_vtk_display(self, xy: tuple[float, float]) -> tuple[float, float]:
        dpr = self._pixel_ratio()
        qt_h = float(max(1, int(self._plotter.height()))) if self._plotter is not None else 1.0
        x_vtk = float(xy[0]) * dpr
        y_vtk = (qt_h - float(xy[1]) - 1.0) * dpr
        return x_vtk, y_vtk

    def _update_box_preview_actor(self) -> None:
        if self._plotter is None or self._box_actor2d is None or self._box_polydata is None:
            return

        show = self._annotation_active and self._annotation_mode == "框选" and bool(self._polygon_points_xy)
        if not show:
            if self._box_preview_visible:
                self._box_actor2d.SetVisibility(False)
                self._box_preview_visible = False
                try:
                    self._plotter.render()
                except Exception:
                    pass
            return

        try:
            from vtkmodules.vtkCommonCore import vtkPoints
            from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData
        except Exception:
            logger.exception("update_box_preview_actor failed: vtk imports unavailable")
            return

        points = vtkPoints()
        for xy in self._polygon_points_xy:
            xv, yv = self._qt_xy_to_vtk_display(xy)
            points.InsertNextPoint(xv, yv, 0.0)

        lines = vtkCellArray()
        n = len(self._polygon_points_xy)
        for i in range(1, n):
            seg = vtkLine()
            seg.GetPointIds().SetId(0, i - 1)
            seg.GetPointIds().SetId(1, i)
            lines.InsertNextCell(seg)
        if n >= 3:
            seg = vtkLine()
            seg.GetPointIds().SetId(0, n - 1)
            seg.GetPointIds().SetId(1, 0)
            lines.InsertNextCell(seg)

        verts = vtkCellArray()
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)
        poly.SetVerts(verts)
        self._box_polydata.ShallowCopy(poly)

        self._box_actor2d.SetVisibility(True)
        self._box_preview_visible = True
        try:
            self._plotter.render()
        except Exception:
            pass

    def _draw_overlay(self, target: QWidget) -> None:
        painter = QPainter(target)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(target.rect(), QColor(0, 0, 0, 0))
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.Antialiasing)

        if not self._annotation_active:
            painter.end()
            return

        painter.end()

    def _update_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.update()
        self._update_box_preview_actor()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_overlay_geometry()

    def _sync_overlay_geometry(self) -> None:
        if self._overlay is None or self._plotter is None:
            return
        self._overlay.setGeometry(self._plotter.rect())
        self._overlay.raise_()
        logger.debug(
            "overlay_geometry plotter_rect=%s overlay_rect=%s",
            self._plotter.rect(),
            self._overlay.rect(),
        )

    def _clear_annotation_primitives(self) -> None:
        logger.debug(
            "clear_annotation brush_center=%s poly_verts=%d",
            self._brush_center_xy,
            len(self._polygon_points_xy),
        )
        self._brush_center_xy = None
        self._brush_dragging = False
        self._polygon_points_xy.clear()
        self._polygon_drag_vertex_idx = None
        self._update_overlay()

    def _end_brush_stroke_if_needed(self) -> None:
        if not self._brush_dragging:
            return
        if self._brush_stroke_end_callback is not None:
            self._brush_stroke_end_callback()

    def _update_plotter_cursor(self) -> None:
        if self._plotter is None:
            return
        try:
            if self._annotation_active and self._annotation_mode == "刷子":
                brush_cursor = self._build_brush_cursor()
                if brush_cursor is not None:
                    self._plotter.setCursor(brush_cursor)
                else:
                    self._plotter.setCursor(Qt.CrossCursor)
            elif self._annotation_active and self._annotation_mode == "框选":
                self._plotter.setCursor(Qt.CrossCursor)
            else:
                self._plotter.unsetCursor()
        except Exception:
            pass

    def _build_brush_cursor(self) -> QCursor | None:
        radius = max(1.0, float(self._brush_radius))
        dpr = self._pixel_ratio()
        key = (int(round(radius * 10.0)), int(round(dpr * 100.0)))
        if self._brush_cursor_cache_key == key and self._brush_cursor_cache is not None:
            return self._brush_cursor_cache

        logical_diameter = int(np.ceil(radius * 2.0)) + 6
        logical_size = max(16, min(256, logical_diameter))
        if logical_size != logical_diameter:
            # Keep cursor drawable if requested radius exceeds platform cursor size.
            radius = max(1.0, (float(logical_size) - 6.0) * 0.5)

        physical_size = max(1, int(np.ceil(float(logical_size) * dpr)))
        pix = QPixmap(physical_size, physical_size)
        pix.fill(Qt.transparent)
        pix.setDevicePixelRatio(dpr)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(Qt.NoBrush)

        ring_pen = QPen(QColor("#ffd400"), 2)
        painter.setPen(ring_pen)

        c = (float(logical_size) - 1.0) * 0.5
        half_w = max(0.0, float(ring_pen.widthF()) * 0.5)
        draw_r = max(1.0, min(radius - half_w, c - 1.0))
        rect = QRectF(c - draw_r, c - draw_r, 2.0 * draw_r, 2.0 * draw_r)
        painter.drawEllipse(rect)

        center_pen = QPen(QColor("#ffd400"), 1)
        painter.setPen(center_pen)
        painter.drawLine(QPointF(c - 4.0, c), QPointF(c + 4.0, c))
        painter.drawLine(QPointF(c, c - 4.0), QPointF(c, c + 4.0))
        painter.end()

        hot = int(round(c))
        cursor = QCursor(pix, hot, hot)
        self._brush_cursor_cache_key = key
        self._brush_cursor_cache = cursor
        return cursor

    def _pixel_ratio(self) -> float:
        if self._plotter is None:
            return 1.0
        try:
            ratio = float(self._plotter._getPixelRatio())  # type: ignore[attr-defined]
        except Exception:
            ratio = float(getattr(self._plotter, "devicePixelRatioF", lambda: 1.0)())
        if not np.isfinite(ratio) or ratio <= 0:
            return 1.0
        return ratio

    def _handle_pan_key(self, key: int) -> bool:
        key_map = {
            Qt.Key_W: (0.0, 1.0),
            Qt.Key_S: (0.0, -1.0),
            Qt.Key_A: (-1.0, 0.0),
            Qt.Key_D: (1.0, 0.0),
            Qt.Key_Up: (0.0, 1.0),
            Qt.Key_Down: (0.0, -1.0),
            Qt.Key_Left: (-1.0, 0.0),
            Qt.Key_Right: (1.0, 0.0),
        }
        delta = key_map.get(key)
        if delta is None:
            return False
        return self._pan_view(delta[0], delta[1])

    def _pan_view(self, dir_x: float, dir_y: float) -> bool:
        if self._plotter is None:
            return False
        try:
            camera = self._plotter.camera
            pos = np.asarray(camera.position, dtype=np.float64)
            focal = np.asarray(camera.focal_point, dtype=np.float64)
            up = np.asarray(camera.up, dtype=np.float64)

            view = focal - pos
            view_norm = np.linalg.norm(view)
            up_norm = np.linalg.norm(up)
            if view_norm <= 1e-12 or up_norm <= 1e-12:
                return False

            view_dir = view / view_norm
            up_dir = up / up_norm
            right = np.cross(view_dir, up_dir)
            right_norm = np.linalg.norm(right)
            if right_norm <= 1e-12:
                return False
            right_dir = right / right_norm

            # Pan step scales with camera distance to keep controls consistent.
            step = max(1e-3, float(view_norm) * 0.03)
            delta = right_dir * (float(dir_x) * step) + up_dir * (float(dir_y) * step)

            camera.position = tuple((pos + delta).tolist())
            camera.focal_point = tuple((focal + delta).tolist())
            self._plotter.render()
            return True
        except Exception:
            logger.exception("pan_view failed")
            return False

    def _is_alt_pressed(self, event=None) -> bool:
        mods = Qt.NoModifier
        if event is not None:
            try:
                mods = mods | event.modifiers()
            except Exception:
                pass
        try:
            mods = mods | QApplication.keyboardModifiers()
        except Exception:
            pass
        try:
            mods = mods | QApplication.queryKeyboardModifiers()
        except Exception:
            pass
        return self._alt_down or bool(mods & Qt.AltModifier)
