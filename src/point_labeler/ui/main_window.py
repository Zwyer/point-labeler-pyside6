"""主窗口：点云语义标注、类别管理、自动保存。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import logging

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from point_labeler.io import (
    SemanticKittiSequence,
    SequenceFrame,
    load_semantickitti_sequence,
    pack_semantickitti_label,
    read_semantickitti_bin,
    read_semantickitti_label,
    split_semantickitti_label,
    write_semantickitti_label,
)
from point_labeler.ui.point_cloud_viewer import PointCloudViewer

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LabelClass:
    """单个语义类别配置。"""

    class_id: int
    name: str
    color_hex: str
    locked: bool = False
    visible: bool = True


@dataclass
class LabelEditAction:
    """Single undoable semantic-label edit."""

    indices: np.ndarray
    old_labels: np.ndarray


class MainWindow(QMainWindow):
    """点云语义分割标注主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self._log = logging.getLogger("point_labeler.main_window")
        self.setWindowTitle("点云语义标注工具")
        self.resize(1450, 860)

        self._current_sequence: SemanticKittiSequence | None = None
        self._current_frame_index: int = -1
        self._current_points: np.ndarray | None = None
        self._current_semantic_labels: np.ndarray | None = None
        self._current_instance_labels: np.ndarray | None = None

        self._class_palette: dict[int, LabelClass] = {}
        self._selected_class_id: int = 0
        self._suppress_class_signals = False

        self._annotation_mode: str = "浏览"
        self._brush_radius: float = 30.0
        self._pan_speed_multiplier: float = 1.0
        self._undo_stack: list[LabelEditAction] = []
        self._max_undo_steps = 200
        self._brush_stroke_active = False
        self._brush_stroke_changed_mask: np.ndarray | None = None
        self._brush_stroke_old_labels: np.ndarray | None = None

        self._setup_default_classes()
        self._setup_ui()
        self._setup_shortcuts()
        self._refresh_class_list()
        self._apply_theme("dark")

    def _setup_default_classes(self) -> None:
        defaults = [
            LabelClass(0, "未标注", "#808080"),
            LabelClass(1, "汽车", "#e74c3c"),
            LabelClass(2, "自行车", "#f39c12"),
            LabelClass(3, "行人", "#2ecc71"),
            LabelClass(4, "道路", "#3498db"),
            LabelClass(5, "建筑", "#9b59b6"),
            LabelClass(6, "植被", "#27ae60"),
        ]
        self._class_palette = {c.class_id: c for c in defaults}
        self._selected_class_id = 0

    def _setup_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)

        topbar = QHBoxLayout()

        self.btn_open_seq = QPushButton("打开序列")
        self.btn_open_seq.clicked.connect(self._on_open_sequence)
        topbar.addWidget(self.btn_open_seq)

        self.btn_save_label = QPushButton("保存标签")
        self.btn_save_label.clicked.connect(self._on_save_label)
        topbar.addWidget(self.btn_save_label)

        self.btn_import_classes = QPushButton("导入类别")
        self.btn_import_classes.clicked.connect(self._on_import_classes)
        topbar.addWidget(self.btn_import_classes)

        self.btn_export_classes = QPushButton("导出类别")
        self.btn_export_classes.clicked.connect(self._on_export_classes)
        topbar.addWidget(self.btn_export_classes)

        self.lbl_sequence = QLabel("未加载序列")
        self.lbl_sequence.setTextInteractionFlags(Qt.TextSelectableByMouse)
        topbar.addWidget(self.lbl_sequence, 1)

        topbar.addWidget(QLabel("主题"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("黑夜", "dark")
        self.combo_theme.addItem("白天", "light")
        self.combo_theme.currentIndexChanged.connect(self._on_theme_changed)
        topbar.addWidget(self.combo_theme)

        root_layout.addLayout(topbar)

        body = QHBoxLayout()
        root_layout.addLayout(body, 1)

        left = QWidget()
        left.setMinimumWidth(320)
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("类别管理"))
        self.list_classes = QListWidget()
        self.list_classes.currentItemChanged.connect(self._on_class_selection_changed)
        left_layout.addWidget(self.list_classes, 1)

        class_btn_row = QHBoxLayout()

        self.btn_add_class = QPushButton("新增")
        self.btn_add_class.clicked.connect(self._on_add_class)
        class_btn_row.addWidget(self.btn_add_class)

        self.btn_remove_class = QPushButton("删除")
        self.btn_remove_class.clicked.connect(self._on_remove_class)
        class_btn_row.addWidget(self.btn_remove_class)

        left_layout.addLayout(class_btn_row)

        self.btn_color_class = QPushButton("改颜色")
        self.btn_color_class.clicked.connect(self._on_set_class_color)
        left_layout.addWidget(self.btn_color_class)

        self.lbl_current_class = QLabel("当前类别：-")
        left_layout.addWidget(self.lbl_current_class)

        left_layout.addWidget(QLabel("标注模式"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("浏览", "浏览")
        self.combo_mode.addItem("框选", "框选")
        self.combo_mode.addItem("刷子", "刷子")
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        left_layout.addWidget(self.combo_mode)

        left_layout.addWidget(QLabel("刷子半径（像素）"))
        self.spin_brush_radius = QSpinBox()
        self.spin_brush_radius.setRange(1, 100)
        self.spin_brush_radius.setValue(30)
        self.spin_brush_radius.valueChanged.connect(self._on_brush_radius_changed)
        left_layout.addWidget(self.spin_brush_radius)

        body.addWidget(left, 0)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        viewer_tools = QHBoxLayout()
        viewer_tools.addWidget(QLabel("点大小"))

        self.btn_size_down = QPushButton("-")
        self.btn_size_down.setFixedWidth(32)
        self.btn_size_down.clicked.connect(self._on_point_size_down)
        viewer_tools.addWidget(self.btn_size_down)

        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 12)
        self.spin_point_size.setValue(2)
        self.spin_point_size.setButtonSymbols(QSpinBox.NoButtons)
        self.spin_point_size.valueChanged.connect(self._on_point_size_changed)
        self.spin_point_size.setFixedWidth(56)
        viewer_tools.addWidget(self.spin_point_size)

        self.btn_size_up = QPushButton("+")
        self.btn_size_up.setFixedWidth(32)
        self.btn_size_up.clicked.connect(self._on_point_size_up)
        viewer_tools.addWidget(self.btn_size_up)

        self.btn_reset_camera = QPushButton("重置视角")
        self.btn_reset_camera.clicked.connect(self._on_reset_camera)
        viewer_tools.addWidget(self.btn_reset_camera)

        viewer_tools.addWidget(QLabel("移动速度"))
        self.slider_pan_speed = QSlider(Qt.Horizontal)
        self.slider_pan_speed.setRange(20, 300)
        self.slider_pan_speed.setValue(100)
        self.slider_pan_speed.setFixedWidth(120)
        self.slider_pan_speed.valueChanged.connect(self._on_pan_speed_changed)
        viewer_tools.addWidget(self.slider_pan_speed)

        self.lbl_pan_speed = QLabel("1.00x")
        self.lbl_pan_speed.setMinimumWidth(46)
        viewer_tools.addWidget(self.lbl_pan_speed)

        viewer_tools.addStretch(1)
        right_layout.addLayout(viewer_tools)

        self.viewer = PointCloudViewer()
        self.viewer.set_box_callback(self._on_box_selected)
        self.viewer.set_brush_callback(self._on_brush_selected)
        self.viewer.set_brush_radius_callback(self._on_viewer_brush_radius_changed)
        self.viewer.set_brush_stroke_begin_callback(self._on_brush_stroke_begin)
        self.viewer.set_brush_stroke_end_callback(self._on_brush_stroke_end)
        self.viewer.set_annotation_mode(self._annotation_mode)
        self.viewer.set_brush_radius(self._brush_radius)
        self.viewer.set_pan_speed_multiplier(self._pan_speed_multiplier)
        right_layout.addWidget(self.viewer, 1)

        body.addWidget(right, 1)

        bottom = QHBoxLayout()

        self.slider_frames = QSlider(Qt.Horizontal)
        self.slider_frames.setEnabled(False)
        self.slider_frames.setMinimum(0)
        self.slider_frames.setMaximum(0)
        self.slider_frames.valueChanged.connect(self._on_slider_changed)
        bottom.addWidget(self.slider_frames, 1)

        self.lbl_frame_pos = QLabel("0/0 | -")
        self.lbl_frame_pos.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_frame_pos.setMinimumWidth(320)
        self.lbl_frame_pos.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bottom.addWidget(self.lbl_frame_pos)

        root_layout.addLayout(bottom)

    def _setup_shortcuts(self) -> None:
        undo_shortcut = QShortcut(QKeySequence.Undo, self)
        undo_shortcut.activated.connect(self._on_undo_last_action)

    def _refresh_class_list(self) -> None:
        self._suppress_class_signals = True
        self.list_classes.clear()

        for class_id in sorted(self._class_palette.keys()):
            cls = self._class_palette[class_id]
            item = QListWidgetItem(f"{cls.class_id}: {cls.name}")
            item.setData(Qt.UserRole, cls.class_id)
            color = QColor(cls.color_hex)
            if not cls.visible:
                color = color.darker(170)
            item.setBackground(color)
            item.setForeground(QColor("#f2f2f2") if color.lightness() < 128 else QColor("#101010"))
            self.list_classes.addItem(item)
            row_widget = self._build_class_row_widget(class_id, cls, color)
            item.setSizeHint(row_widget.sizeHint())
            self.list_classes.setItemWidget(item, row_widget)

        self._suppress_class_signals = False

        ids = sorted(self._class_palette.keys())
        if self._selected_class_id in ids:
            self.list_classes.setCurrentRow(ids.index(self._selected_class_id))

        self._update_current_class_label()

    def _update_current_class_label(self) -> None:
        cls = self._class_palette.get(self._selected_class_id)
        if cls is None:
            self.lbl_current_class.setText("当前类别：-")
            return
        lock_text = "锁定" if cls.locked else "未锁定"
        vis_text = "显示" if cls.visible else "隐藏"
        self.lbl_current_class.setText(
            f"当前类别：{cls.class_id} - {cls.name} ({cls.color_hex}) | {lock_text} | {vis_text}"
        )

    def _build_class_row_widget(self, class_id: int, cls: LabelClass, color: QColor) -> QWidget:
        row = QWidget(self.list_classes)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 2, 6, 2)
        row_layout.setSpacing(6)

        text = QLabel(f"{cls.class_id}: {cls.name}")
        text.setStyleSheet(
            f"color: {'#f2f2f2' if color.lightness() < 128 else '#101010'}; background: transparent;"
        )
        row_layout.addWidget(text, 1)

        lock_btn = QToolButton(row)
        lock_btn.setText("🔒" if cls.locked else "🔓")
        lock_btn.setToolTip("解锁该类别" if cls.locked else "锁定该类别")
        lock_btn.setAutoRaise(True)
        lock_btn.clicked.connect(lambda _=False, cid=class_id: self._on_toggle_class_lock_by_id(cid))
        row_layout.addWidget(lock_btn, 0)

        vis_btn = QToolButton(row)
        vis_btn.setText("👁" if cls.visible else "🚫")
        vis_btn.setToolTip("隐藏该类别" if cls.visible else "显示该类别")
        vis_btn.setAutoRaise(True)
        vis_btn.clicked.connect(lambda _=False, cid=class_id: self._on_toggle_class_visibility_by_id(cid))
        row_layout.addWidget(vis_btn, 0)

        row.setStyleSheet(f"background-color: {color.name(QColor.HexRgb)}; border: none;")
        return row

    def _on_open_sequence(self) -> None:
        self._auto_save_current_frame()
        selected_dir = QFileDialog.getExistingDirectory(self, "选择序列目录", str(Path.cwd()))
        if not selected_dir:
            return

        try:
            sequence = load_semantickitti_sequence(selected_dir)
        except Exception as exc:
            QMessageBox.critical(self, "打开失败", str(exc))
            return

        self._current_sequence = sequence
        self._current_frame_index = -1
        self._clear_frame_state()

        self.lbl_sequence.setText(
            f"序列目录：{sequence.sequence_dir} | 帧数：{len(sequence.frames)} | 标签目录：{sequence.sequence_dir / 'labels'}"
        )

        self._refresh_sequence_slider(sequence)
        self._load_frame_by_index(0)

    def _refresh_sequence_slider(self, sequence: SemanticKittiSequence) -> None:
        count = len(sequence.frames)
        enabled = count > 0

        self.slider_frames.blockSignals(True)
        self.slider_frames.setEnabled(enabled)
        if enabled:
            self.slider_frames.setMinimum(0)
            self.slider_frames.setMaximum(count - 1)
            self.slider_frames.setValue(0)
        else:
            self.slider_frames.setMinimum(0)
            self.slider_frames.setMaximum(0)
            self.slider_frames.setValue(0)
        self.slider_frames.blockSignals(False)

        self._update_frame_pos_label()

    def _clear_frame_state(self) -> None:
        self._current_points = None
        self._current_semantic_labels = None
        self._current_instance_labels = None
        self._undo_stack.clear()
        self._reset_brush_stroke_state()
        self.viewer.clear()
        self._update_frame_pos_label()

    def _update_frame_pos_label(self) -> None:
        if self._current_sequence is None or not self._current_sequence.frames:
            self.lbl_frame_pos.setText("0/0 | -")
            return

        total = len(self._current_sequence.frames)
        idx = self._current_frame_index + 1 if self._current_frame_index >= 0 else 0

        name = "-"
        if 0 <= self._current_frame_index < total:
            frame = self._current_sequence.frames[self._current_frame_index]
            name = frame.bin_path.name

        self.lbl_frame_pos.setText(f"{idx}/{total} | {name}")

    def _on_slider_changed(self, index: int) -> None:
        self._load_frame_by_index(index)

    def _load_frame_by_index(self, index: int) -> None:
        if self._current_sequence is None:
            return
        if index < 0 or index >= len(self._current_sequence.frames):
            return
        if index == self._current_frame_index:
            return

        self._auto_save_current_frame()

        frame = self._current_sequence.frames[index]
        try:
            points = read_semantickitti_bin(frame.bin_path, mode="xyzi")
        except Exception as exc:
            QMessageBox.critical(self, "读取点云失败", f"{frame.bin_path}\n{exc}")
            return

        sem_labels, ins_labels = self._load_frame_labels(frame, points.shape[0])

        self._current_points = points
        self._current_semantic_labels = sem_labels
        self._current_instance_labels = ins_labels
        self._current_frame_index = index
        self._undo_stack.clear()
        self._reset_brush_stroke_state()

        self._ensure_label_file_exists(frame, sem_labels, ins_labels)
        self._render_current_frame(reset_camera=True)

        self.slider_frames.blockSignals(True)
        self.slider_frames.setValue(index)
        self.slider_frames.blockSignals(False)
        self._update_frame_pos_label()

    def _load_frame_labels(
        self,
        frame: SequenceFrame,
        point_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if frame.label_path is None or not frame.label_path.exists():
            return np.zeros(point_count, dtype=np.uint16), np.zeros(point_count, dtype=np.uint16)

        raw = read_semantickitti_label(frame.label_path)
        if raw.shape[0] != point_count:
            return np.zeros(point_count, dtype=np.uint16), np.zeros(point_count, dtype=np.uint16)

        return split_semantickitti_label(raw)

    def _ensure_label_file_exists(
        self,
        frame: SequenceFrame,
        semantic_labels: np.ndarray,
        instance_labels: np.ndarray,
    ) -> None:
        if self._current_sequence is None or self._current_frame_index < 0:
            return

        labels_dir = self._current_sequence.sequence_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        out_path = labels_dir / f"{frame.frame_id}.label"

        if not out_path.exists():
            self._write_label_file(out_path, semantic_labels, instance_labels)

        self._current_sequence.frames[self._current_frame_index] = replace(frame, label_path=out_path)

    def _write_label_file(
        self,
        out_path: Path,
        semantic_labels: np.ndarray,
        instance_labels: np.ndarray,
    ) -> None:
        packed = pack_semantickitti_label(semantic_labels, instance_labels)
        write_semantickitti_label(out_path, packed)

    def _auto_save_current_frame(self) -> None:
        if (
            self._current_sequence is None
            or self._current_frame_index < 0
            or self._current_semantic_labels is None
            or self._current_instance_labels is None
        ):
            return

        frame = self._current_sequence.frames[self._current_frame_index]
        labels_dir = self._current_sequence.sequence_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        out_path = labels_dir / f"{frame.frame_id}.label"

        try:
            self._write_label_file(out_path, self._current_semantic_labels, self._current_instance_labels)
            self._current_sequence.frames[self._current_frame_index] = replace(frame, label_path=out_path)
        except Exception:
            pass

    def _render_current_frame(self, reset_camera: bool) -> None:
        if self._current_points is None or self._current_semantic_labels is None:
            return
        rgb = self._build_rgb_colors(self._current_semantic_labels)
        points_for_render = self._build_points_for_render(self._current_points, self._current_semantic_labels)
        self.viewer.set_points(points_for_render, reset_camera=reset_camera, rgb_colors=rgb)

    def _build_points_for_render(self, points: np.ndarray, semantic_labels: np.ndarray) -> np.ndarray:
        hidden_ids = [cid for cid, cls in self._class_palette.items() if not cls.visible]
        if not hidden_ids:
            return points
        out = np.array(points, copy=True)
        hidden_mask = np.isin(semantic_labels, np.asarray(hidden_ids, dtype=semantic_labels.dtype))
        out[hidden_mask, :3] = np.nan
        return out

    def _build_rgb_colors(self, semantic_labels: np.ndarray) -> np.ndarray:
        if semantic_labels.size == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        max_label = int(np.max(semantic_labels))
        lut = np.full((max_label + 1, 3), 255, dtype=np.uint8)
        for class_id, cls in self._class_palette.items():
            if 0 <= class_id <= max_label:
                lut[class_id] = self._hex_to_rgb(cls.color_hex)
        return lut[semantic_labels]

    @staticmethod
    def _hex_to_rgb(color_hex: str) -> np.ndarray:
        text = color_hex.strip().lstrip("#")
        if len(text) != 6:
            return np.array([255, 255, 255], dtype=np.uint8)
        return np.array(
            [int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)],
            dtype=np.uint8,
        )

    def _on_save_label(self) -> None:
        self._auto_save_current_frame()

        if self._current_sequence is None or self._current_frame_index < 0:
            QMessageBox.information(self, "提示", "当前没有可保存的帧。")
            return

        frame = self._current_sequence.frames[self._current_frame_index]
        QMessageBox.information(self, "已保存", f"已保存：{frame.label_path}")

    def _on_class_selection_changed(
        self,
        current: QListWidgetItem | None,
        _: QListWidgetItem | None,
    ) -> None:
        if self._suppress_class_signals or current is None:
            return

        class_id = current.data(Qt.UserRole)
        if isinstance(class_id, int):
            self._selected_class_id = class_id

        self._update_current_class_label()

    def _on_add_class(self) -> None:
        class_id, ok = QInputDialog.getInt(
            self,
            "新增类别",
            "类别 ID：",
            0,
            0,
            65535,
            1,
        )
        if not ok:
            return

        if class_id in self._class_palette:
            QMessageBox.warning(self, "重复 ID", f"类别 ID {class_id} 已存在。")
            return

        name, ok = QInputDialog.getText(self, "新增类别", "类别名称：")
        if not ok or not name.strip():
            return

        self._class_palette[class_id] = LabelClass(class_id, name.strip(), "#ffffff")
        self._selected_class_id = class_id
        self._refresh_class_list()
        self._render_current_frame(reset_camera=False)

    def _on_remove_class(self) -> None:
        if self._selected_class_id == 0:
            QMessageBox.warning(self, "禁止操作", "未标注类别不能删除。")
            return

        if self._selected_class_id in self._class_palette:
            del self._class_palette[self._selected_class_id]

        self._selected_class_id = 0
        self._refresh_class_list()
        self._render_current_frame(reset_camera=False)

    def _on_set_class_color(self) -> None:
        cls = self._class_palette.get(self._selected_class_id)
        if cls is None:
            return

        color = QColorDialog.getColor(QColor(cls.color_hex), self, "选择类别颜色")
        if not color.isValid():
            return

        cls.color_hex = color.name(QColor.HexRgb)
        self._class_palette[cls.class_id] = cls
        self._refresh_class_list()
        self._render_current_frame(reset_camera=False)

    def _on_toggle_class_lock_by_id(self, class_id: int) -> None:
        self._selected_class_id = class_id
        self._select_class_row(class_id)
        cls = self._class_palette.get(class_id)
        if cls is None:
            return
        cls.locked = not cls.locked
        self._class_palette[cls.class_id] = cls
        self._refresh_class_list()

    def _on_toggle_class_visibility_by_id(self, class_id: int) -> None:
        self._selected_class_id = class_id
        self._select_class_row(class_id)
        cls = self._class_palette.get(class_id)
        if cls is None:
            return
        cls.visible = not cls.visible
        self._class_palette[cls.class_id] = cls
        self._refresh_class_list()
        self._render_current_frame(reset_camera=False)

    def _select_class_row(self, class_id: int) -> None:
        ids = sorted(self._class_palette.keys())
        if class_id not in ids:
            return
        self.list_classes.blockSignals(True)
        self.list_classes.setCurrentRow(ids.index(class_id))
        self.list_classes.blockSignals(False)

    def _on_mode_changed(self) -> None:
        mode = self.combo_mode.currentData()
        if not isinstance(mode, str):
            mode = "浏览"
        self._annotation_mode = mode
        self._log.debug("mode_changed mode=%s", mode)
        self.viewer.set_annotation_mode(mode)

    def _on_brush_radius_changed(self) -> None:
        self._brush_radius = float(self.spin_brush_radius.value())
        self._log.debug("ui_brush_radius_changed radius=%.3f", self._brush_radius)
        self.viewer.set_brush_radius(self._brush_radius)

    def _on_viewer_brush_radius_changed(self, radius: float) -> None:
        self._brush_radius = radius
        self._log.debug("viewer_brush_radius_changed radius=%.3f", radius)

        value = int(round(radius))
        value = max(self.spin_brush_radius.minimum(), min(self.spin_brush_radius.maximum(), value))

        self.spin_brush_radius.blockSignals(True)
        self.spin_brush_radius.setValue(value)
        self.spin_brush_radius.blockSignals(False)

    def _on_box_selected(self, indices: np.ndarray) -> None:
        if self._annotation_mode != "框选" or self._current_semantic_labels is None:
            return

        if indices.size == 0:
            return
        self._log.debug("box_selected count=%d", int(indices.size))

        self._apply_label_to_indices(indices)

    def _on_brush_selected(self, indices: np.ndarray) -> None:
        if self._annotation_mode != "刷子" or self._current_semantic_labels is None:
            return
        if indices.size == 0:
            return
        self._log.debug("brush_selected count=%d", int(indices.size))

        self._apply_label_to_indices(indices)

    def _apply_label_to_indices(self, indices: np.ndarray) -> None:
        if self._current_semantic_labels is None:
            return

        candidate = np.asarray(indices, dtype=np.int64)
        if candidate.size == 0:
            return
        total = int(self._current_semantic_labels.shape[0])
        in_range = (candidate >= 0) & (candidate < total)
        if not np.any(in_range):
            self._log.debug("apply_label skipped: all target points are out of range")
            return
        candidate = candidate[in_range]

        hidden_ids = np.array(
            [cid for cid, cls in self._class_palette.items() if not cls.visible],
            dtype=np.uint16,
        )
        if hidden_ids.size > 0:
            hidden_mask = np.isin(self._current_semantic_labels[candidate], hidden_ids)
            candidate = candidate[~hidden_mask]
        if candidate.size == 0:
            self._log.debug("apply_label skipped: all target points are hidden")
            return

        locked_ids = np.array(
            [cid for cid, cls in self._class_palette.items() if cls.locked and cid != self._selected_class_id],
            dtype=np.uint16,
        )
        if locked_ids.size > 0:
            protected = np.isin(self._current_semantic_labels[candidate], locked_ids)
            candidate = candidate[~protected]
        if candidate.size == 0:
            self._log.debug("apply_label skipped: all target points are locked")
            return

        unique_indices = np.unique(candidate)
        unique_count = int(unique_indices.size)
        old_labels = self._current_semantic_labels[unique_indices].copy()
        new_label = np.uint16(self._selected_class_id)
        change_mask = old_labels != new_label
        if not np.any(change_mask):
            self._log.debug("apply_label skipped: labels are already target class")
            return

        changed_indices = unique_indices[change_mask]
        changed_old_labels = old_labels[change_mask]
        if self._annotation_mode == "刷子" and self._brush_stroke_active:
            self._accumulate_brush_stroke_change(changed_indices, changed_old_labels)
        else:
            self._push_undo_action(changed_indices, changed_old_labels)

        self._log.debug(
            "apply_label class_id=%d raw_count=%d unique_count=%d",
            int(self._selected_class_id),
            int(indices.size),
            int(changed_indices.size),
        )
        self._current_semantic_labels[changed_indices] = new_label
        cls = self._class_palette.get(self._selected_class_id)
        incremental_ok = False
        if cls is not None:
            color = self._hex_to_rgb(cls.color_hex)
            incremental_ok = self.viewer.update_rgb_for_original_indices(changed_indices, color)
        if not incremental_ok:
            self._render_current_frame(reset_camera=False)
        if self._annotation_mode != "刷子" or not self._brush_stroke_active:
            self._auto_save_current_frame()

    def _on_undo_last_action(self) -> None:
        if self._current_semantic_labels is None:
            return
        if not self._undo_stack:
            return

        action = self._undo_stack.pop()
        if action.indices.size == 0:
            return

        self._current_semantic_labels[action.indices] = action.old_labels
        self._render_current_frame(reset_camera=False)
        self._auto_save_current_frame()
        self._log.debug("undo_applied restored_points=%d", int(action.indices.size))

    def _push_undo_action(self, indices: np.ndarray, old_labels: np.ndarray) -> None:
        if indices.size == 0:
            return
        self._undo_stack.append(LabelEditAction(indices=indices.copy(), old_labels=old_labels.copy()))
        if len(self._undo_stack) > self._max_undo_steps:
            self._undo_stack.pop(0)

    def _reset_brush_stroke_state(self) -> None:
        self._brush_stroke_active = False
        self._brush_stroke_changed_mask = None
        self._brush_stroke_old_labels = None

    def _on_brush_stroke_begin(self) -> None:
        if self._current_semantic_labels is None:
            self._reset_brush_stroke_state()
            return
        self._brush_stroke_active = True
        n = int(self._current_semantic_labels.shape[0])
        self._brush_stroke_changed_mask = np.zeros(n, dtype=bool)
        self._brush_stroke_old_labels = np.zeros(n, dtype=np.uint16)

    def _accumulate_brush_stroke_change(self, changed_indices: np.ndarray, changed_old_labels: np.ndarray) -> None:
        if (
            not self._brush_stroke_active
            or self._brush_stroke_changed_mask is None
            or self._brush_stroke_old_labels is None
            or changed_indices.size == 0
        ):
            return
        seen_mask = self._brush_stroke_changed_mask[changed_indices]
        first_touch = ~seen_mask
        if np.any(first_touch):
            first_idx = changed_indices[first_touch]
            self._brush_stroke_old_labels[first_idx] = changed_old_labels[first_touch]
        self._brush_stroke_changed_mask[changed_indices] = True

    def _on_brush_stroke_end(self) -> None:
        if (
            not self._brush_stroke_active
            or self._brush_stroke_changed_mask is None
            or self._brush_stroke_old_labels is None
        ):
            self._reset_brush_stroke_state()
            return
        changed_indices = np.flatnonzero(self._brush_stroke_changed_mask).astype(np.int64, copy=False)
        if changed_indices.size > 0:
            old_labels = self._brush_stroke_old_labels[changed_indices]
            self._push_undo_action(changed_indices, old_labels)
            self._log.debug("brush_stroke_end undo_points=%d", int(changed_indices.size))
            self._auto_save_current_frame()
        self._reset_brush_stroke_state()

    def _on_point_size_changed(self, value: int) -> None:
        self.viewer.set_point_size(value)

    def _on_point_size_down(self) -> None:
        self.spin_point_size.setValue(max(1, self.spin_point_size.value() - 1))

    def _on_point_size_up(self) -> None:
        self.spin_point_size.setValue(min(12, self.spin_point_size.value() + 1))

    def _on_reset_camera(self) -> None:
        self.viewer.reset_camera()

    def _on_pan_speed_changed(self, value: int) -> None:
        multiplier = max(0.2, float(value) / 100.0)
        self._pan_speed_multiplier = multiplier
        self.lbl_pan_speed.setText(f"{multiplier:.2f}x")
        self.viewer.set_pan_speed_multiplier(multiplier)

    def _on_theme_changed(self) -> None:
        mode = self.combo_theme.currentData()
        if not isinstance(mode, str):
            mode = "dark"
        self._apply_theme(mode)

    def _apply_theme(self, mode: str) -> None:
        if mode == "light":
            stylesheet = """
QMainWindow, QWidget { background-color: #f7f7f7; color: #1f1f1f; }
QLabel { color: #1f1f1f; }
QPushButton {
    background-color: #e6e6e6; color: #1f1f1f; border: 1px solid #bdbdbd;
    border-radius: 4px; padding: 4px 8px;
}
QPushButton:hover { background-color: #dcdcdc; }
QComboBox, QSpinBox, QListWidget {
    background-color: #ffffff; color: #1f1f1f; border: 1px solid #bdbdbd;
}
QSlider::groove:horizontal { border: 1px solid #bdbdbd; height: 6px; background: #d9d9d9; }
QSlider::handle:horizontal { background: #8a8a8a; width: 14px; margin: -5px 0; border-radius: 7px; }
            """
        else:
            stylesheet = """
QMainWindow, QWidget { background-color: #1b1b1b; color: #d8d8d8; }
QLabel { color: #d8d8d8; }
QPushButton {
    background-color: #2c2c2c; color: #d8d8d8; border: 1px solid #4d4d4d;
    border-radius: 4px; padding: 4px 8px;
}
QPushButton:hover { background-color: #383838; }
QComboBox, QSpinBox, QListWidget {
    background-color: #232323; color: #d8d8d8; border: 1px solid #4d4d4d;
}
QSlider::groove:horizontal { border: 1px solid #4d4d4d; height: 6px; background: #2b2b2b; }
QSlider::handle:horizontal { background: #9b9b9b; width: 14px; margin: -5px 0; border-radius: 7px; }
            """

        self.setStyleSheet(stylesheet)
        self.viewer.set_theme(mode)

    def _on_import_classes(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入类别配置",
            str(Path.cwd()),
            "配置文件 (*.json *.yaml *.yml)",
        )
        if not path:
            return

        try:
            self._class_palette = self._load_class_config(Path(path))
            if 0 not in self._class_palette:
                self._class_palette[0] = LabelClass(0, "未标注", "#808080")
            self._selected_class_id = 0
            self._refresh_class_list()
            self._render_current_frame(reset_camera=False)
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def _on_export_classes(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出类别配置",
            str(Path.cwd() / "classes.json"),
            "配置文件 (*.json *.yaml *.yml)",
        )
        if not path:
            return

        try:
            self._save_class_config(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _save_class_config(self, path: Path) -> None:
        data = [
            {
                "id": c.class_id,
                "name": c.name,
                "color": c.color_hex,
                "locked": c.locked,
                "visible": c.visible,
            }
            for c in sorted(self._class_palette.values(), key=lambda x: x.class_id)
        ]

        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("未安装 PyYAML，无法导出 YAML。")
            path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            return

        raise RuntimeError("仅支持 .json / .yaml / .yml")

    def _load_class_config(self, path: Path) -> dict[int, LabelClass]:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("未安装 PyYAML，无法导入 YAML。")
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raise RuntimeError("仅支持 .json / .yaml / .yml")

        if not isinstance(data, list):
            raise RuntimeError("类别配置格式错误：应为列表。")

        result: dict[int, LabelClass] = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            class_id = int(item.get("id", 0))
            name = str(item.get("name", f"class_{class_id}"))
            color = str(item.get("color", "#ffffff"))
            locked = bool(item.get("locked", False))
            visible = bool(item.get("visible", True))
            result[class_id] = LabelClass(class_id, name, color, locked=locked, visible=visible)

        return result

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._auto_save_current_frame()
        super().closeEvent(event)
