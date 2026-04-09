# combined_app.py — стеклянный UI с оверлеями, тулбаром, мини-картой и тостами
# 3D-мир + Mind Trainer (эволюция/метрики/мозг агентов)
# ПКМ в 3D -> задать goal агенту в локальном мире тренера.
# Выбор агента синхронен между 3D и инспектором мозга.

import sys
import math
import random
from typing import Dict, Any, Optional, List, Tuple
from collections.abc import Mapping, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QStyle, QFileDialog

from OpenGL.GLU import gluUnProject
from OpenGL.GL import (
    glGetDoublev, glGetIntegerv,
    GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT
)

# --- 3D: движок, окружение
from engine3d import MiniMatrixEngine
from env_cinematic import build_cinematic_environment

# --- Бой
from combat_system import CombatSystem

# --- Конфиг/мир
import config
from world import WorldObject

# --- Mind Trainer панели
from mind_trainer_gui import MindTrainerInteractive, TrainerStatsWidget, AgentBrainWidget
from training_room import LAB_AGENT_TAG, TRAINING_ROOM_TAG


# =========================
# 0) Токены темы/стилей
# =========================
COL_BG = "#0a0c12"
COL_BG_GRAD_A = "#0b0d14"
COL_BG_GRAD_B = "#07090f"

COL_PANEL_BG = "rgba(18,22,30,0.78)"
COL_BORDER = "#2a2f3a"
COL_ACCENT = "#7aa2ff"
COL_ACCENT_2 = "#9bd1ff"
COL_TEXT = "#e8eaf6"
COL_TEXT_DIM = "#b9bed3"
COL_SUCCESS = "#5bd1a5"
COL_WARN = "#ffbf69"

RADIUS = 16
SHADOW_ALPHA = 150
APP_FONT = "Inter, Segoe UI, Helvetica Neue, Arial"

APP_QSS_BASE = f"""
QMainWindow {{
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {COL_BG_GRAD_A}, stop:1 {COL_BG_GRAD_B});
  color:{COL_TEXT};
}}
QStatusBar {{ color:#c7cbe4; background-color: rgba(10,11,16,0.65); border-top:1px solid {COL_BORDER}; }}
QToolBar {{
  background: rgba(14,16,22,0.6);
  border-bottom: 1px solid {COL_BORDER};
  padding: 6px 8px;
}}
QToolButton {{
  color:{COL_TEXT}; background: rgba(24,28,36,0.6);
  border:1px solid {COL_BORDER}; border-radius:10px; padding:6px 10px;
}}
QToolButton:hover {{ background: rgba(32,38,48,0.7); }}
QToolButton:pressed {{ background: rgba(18,22,30,0.7); }}
QComboBox, QLineEdit {{
  background: rgba(20,24,32,0.65); border:1px solid {COL_BORDER};
  border-radius:10px; padding:6px 8px; color:{COL_TEXT};
}}
QComboBox QAbstractItemView {{ background:#141827; color:{COL_TEXT}; selection-background-color:#2a3147; }}
QScrollBar:vertical {{ background:transparent; width:10px; margin:6px 2px 6px 2px; }}
QScrollBar::handle:vertical {{ background: rgba(122,162,255,0.35); border-radius:6px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(122,162,255,0.55); }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QSplitter::handle {{ background:#171c28; width:8px; border-radius:3px; }}
QSplitter::handle:hover {{ background:#22283a; }}
QFrame[card="true"] {{
  background:{COL_PANEL_BG}; border:1px solid {COL_BORDER}; border-radius:{RADIUS}px;
}}
QLabel[cardTitle="true"] {{
  color:{COL_TEXT}; font-size:13px; font-weight:680;
}}
QProgressBar {{
  background: rgba(18,22,30,0.65); border:1px solid {COL_BORDER}; border-radius:10px; height:12px;
  qproperty-textVisible: false;  /* фикс: вместо text-visible */
}}
QProgressBar::chunk {{
  border-radius:9px; margin:1px;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {{ACC1}}, stop:1 {{ACC2}});
}}
QCheckBox::indicator {{
  width:18px; height:18px; border-radius:5px; border:1px solid {COL_BORDER}; background: rgba(24,28,36,0.6);
}}
QCheckBox::indicator:checked {{
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {{ACC1}}, stop:1 {{ACC2}});
}}
QSlider::groove:horizontal {{
  height:6px; border-radius:3px; background: rgba(255,255,255,0.08);
}}
QSlider::handle:horizontal {{
  width:14px; height:14px; margin:-5px 0; border-radius:7px;
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {{ACC1}}, stop:1 {{ACC2}});
  border:1px solid #31405d;
}}
"""

APP_QSS = APP_QSS_BASE.replace("{{ACC1}}", COL_ACCENT).replace("{{ACC2}}", COL_ACCENT_2)
APP_QSS_EMERALD = APP_QSS_BASE.replace("{{ACC1}}", "#41d6b0").replace("{{ACC2}}", "#6ef7c9")

HELP_PILL_STYLE = """
QLabel {
  color:#cdd5ff; background: rgba(28,32,48,0.72);
  border:1px solid #374058; border-radius:10px; padding:8px 10px; font-size:12px;
}
"""

FRAME3D_STYLE = f"""
QFrame {{
  background:#000; border:1px solid {COL_BORDER}; border-radius:{RADIUS}px;
}}
"""

STATUSBAR_STYLE = "QStatusBar { color:#c7cbe4; background-color:#11121a; }"


# =====================================================
# Константы showcase-карты
# =====================================================
SHOWCASE_WORLD_WIDTH = 180.0
SHOWCASE_WORLD_HEIGHT = 180.0
SHOWCASE_SAFE_HAVENS = 8
SHOWCASE_ENV_SEED = 2026
VIEW_FRAME_INTERVAL_MS = 16
TRAINER_TICK_INTERVAL_MS = 24
SNAPSHOT_PUSH_INTERVAL_MS = 24
OVERLAY_RELAYOUT_INTERVAL_MS = 33
ENGINE_CHAT_TAIL = 120
ENGINE_EVENT_TAIL = 120
ENGINE_GLOBAL_EVENT_TAIL = 60


# =====================================================
# 1) Утилиты и базовые виджеты
# =====================================================
def _iter_vals(maybe_collection):
    if maybe_collection is None:
        return []
    if isinstance(maybe_collection, Mapping):
        return list(maybe_collection.values())
    if isinstance(maybe_collection, Sequence) and not isinstance(maybe_collection, (str, bytes, bytearray)):
        return list(maybe_collection)
    return [maybe_collection]


def _public_tags(entity: Any) -> List[str]:
    state = getattr(entity, "public_state", {}) or {}
    tags = state.get("tags", [])
    if not isinstance(tags, list):
        return []
    return [str(t) for t in tags]


_CONTROL_TEXT_KEY_ALIASES: Dict[str, int] = {
    "w": int(Qt.Key_W),
    "ц": int(Qt.Key_W),
    "a": int(Qt.Key_A),
    "ф": int(Qt.Key_A),
    "s": int(Qt.Key_S),
    "ы": int(Qt.Key_S),
    "d": int(Qt.Key_D),
    "в": int(Qt.Key_D),
    "v": int(Qt.Key_V),
    "м": int(Qt.Key_V),
    "q": int(Qt.Key_Q),
    "й": int(Qt.Key_Q),
    "e": int(Qt.Key_E),
    "у": int(Qt.Key_E),
    "g": int(Qt.Key_G),
    "п": int(Qt.Key_G),
    "m": int(Qt.Key_M),
    "ь": int(Qt.Key_M),
    "r": int(Qt.Key_R),
    "к": int(Qt.Key_R),
    "f": int(Qt.Key_F),
    "а": int(Qt.Key_F),
}


def _normalize_control_key_event(e: QtGui.QKeyEvent) -> int:
    key = int(e.key())
    text = str(e.text() or "").strip().casefold()
    if text:
        mapped = _CONTROL_TEXT_KEY_ALIASES.get(text)
        if mapped is not None:
            return int(mapped)
    return key


class GlassCard(QtWidgets.QFrame):
    """Полупрозрачная «акриловая» карточка с мягкой тенью и верхним бликом."""
    def __init__(self, parent=None, radius:int=RADIUS, shadow_blur:int=28, shadow_offset:QtCore.QPoint=QtCore.QPoint(0,6)):
        super().__init__(parent)
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(shadow_blur)
        shadow.setOffset(shadow_offset)
        shadow.setColor(QtGui.QColor(0, 0, 0, SHADOW_ALPHA))
        self.setGraphicsEffect(shadow)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect().adjusted(1,1,-1,-1)
        grad = QtGui.QLinearGradient(r.topLeft(), r.bottomLeft())
        grad.setColorAt(0.0, QtGui.QColor(26,30,40,210))
        grad.setColorAt(1.0, QtGui.QColor(16,18,26,210))
        p.setBrush(grad)
        p.setPen(QtGui.QPen(QtGui.QColor(COL_BORDER), 1))
        p.drawRoundedRect(r, self.radius, self.radius)
        hl = QtGui.QLinearGradient(r.topLeft(), r.center())
        hl.setColorAt(0.0, QtGui.QColor(255,255,255,12))
        hl.setColorAt(1.0, QtGui.QColor(255,255,255,0))
        p.setBrush(hl); p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(QtCore.QRectF(r.x()+1, r.y()+1, r.width()-2, r.height()*0.45), self.radius-2, self.radius-2)
        p.end()
        super().paintEvent(ev)


class OverlayLabel(QtWidgets.QLabel):
    """Плавающая подсказка поверх 3D."""
    def __init__(self, parent, text: str):
        super().__init__(parent)
        self.setText(text)
        self.setStyleSheet(HELP_PILL_STYLE)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QtCore.QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0); self._anim.setEndValue(1.0); self._anim.start()

    def fade(self, show: bool):
        self._anim.stop()
        self._anim.setDirection(QtCore.QAbstractAnimation.Forward if show else QtCore.QAbstractAnimation.Backward)
        self._anim.start()


def make_card(title: str, inner: QtWidgets.QWidget) -> GlassCard:
    card = GlassCard()
    v = QtWidgets.QVBoxLayout(card)
    v.setContentsMargins(12, 12, 12, 12)
    v.setSpacing(10)
    title_lbl = QtWidgets.QLabel(title)
    title_lbl.setProperty("cardTitle", True)
    v.addWidget(title_lbl)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setStyleSheet("QScrollArea { background: transparent; }")
    scroll.setWidget(inner)
    v.addWidget(scroll, 1)
    return card


def apply_expand_policy(w: QtWidgets.QWidget, *, w_stretch=False):
    sp = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Expanding if w_stretch else QtWidgets.QSizePolicy.Preferred,
        QtWidgets.QSizePolicy.Expanding
    )
    w.setSizePolicy(sp)


class SnackBar(QtWidgets.QFrame):
    """Тост-уведомление внизу 3D."""
    def __init__(self, parent: QtWidgets.QWidget, text: str, msec:int=1800):
        super().__init__(parent)
        self.setStyleSheet("""
        QFrame { background: rgba(25,30,42,0.90); border: 1px solid #2a2f3a; border-radius: 12px; }
        QLabel { color:#e8eaf6; padding:8px 12px; }
        """)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(QtWidgets.QLabel(text))
        self._opacity = QtWidgets.QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QtCore.QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(240)
        self._anim.setStartValue(0.0); self._anim.setEndValue(1.0)
        self._anim.start()
        QtCore.QTimer.singleShot(msec, self.dismiss)

    def place(self, parent_rect: QtCore.QRect):
        self.adjustSize()
        w = self.sizeHint().width()
        h = self.sizeHint().height()
        self.setGeometry(parent_rect.center().x()-w//2, parent_rect.bottom()-h-18, w, h)

    def dismiss(self):
        self._anim.setDirection(QtCore.QAbstractAnimation.Backward)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()


class MiniMapWidget(QtWidgets.QWidget):
    """Мини-карта мира в экранных координатах, кликом — перелёт камеры."""
    clickedWorld = QtCore.Signal(float, float)  # x, y in world

    def __init__(self, shared, parent=None):
        super().__init__(parent)
        self.shared = shared
        self._player_provider = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, True)
        self.setFixedSize(180, 180)
        self.setToolTip("Mini-map: click to center camera")
        self._bg = QtGui.QColor(20, 24, 32, 180)
        self._pen_grid = QtGui.QPen(QtGui.QColor(255,255,255,20), 1)
        self._pen_border = QtGui.QPen(QtGui.QColor("#2a2f3a"), 1)
        self._brush_agent = QtGui.QBrush(QtGui.QColor(122,162,255,220))
        self._brush_lab = QtGui.QBrush(QtGui.QColor(88, 235, 176, 232))
        self._brush_sel = QtGui.QBrush(QtGui.QColor(255,255,255,240))
        self._brush_animal = QtGui.QBrush(QtGui.QColor(255,191,105,220))
        self._brush_player = QtGui.QBrush(QtGui.QColor(86, 236, 182, 232))
        self._pen_player = QtGui.QPen(QtGui.QColor(210, 255, 238, 240), 1.4)

    def set_player_provider(self, provider):
        self._player_provider = provider
        self.update()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect().adjusted(1,1,-1,-1)
        # card back
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, 14, 14)
        p.fillPath(path, self._bg)
        p.setPen(self._pen_border); p.drawPath(path)

        # grid
        p.setPen(self._pen_grid)
        for i in range(1,4):
            p.drawLine(r.left()+i*r.width()/4, r.top()+6, r.left()+i*r.width()/4, r.bottom()-6)
            p.drawLine(r.left()+6, r.top()+i*r.height()/4, r.right()-6, r.top()+i*r.height()/4)

        # entities
        ww, wh = max(1.0, self.shared.world_w), max(1.0, self.shared.world_h)
        def map_xy(x, y):
            px = r.left()+6 + (x/ww)*(r.width()-12)
            py = r.top()+6 + (y/wh)*(r.height()-12)
            return px, py

        # agents
        sel = self.shared.get_selected_agent_id()
        try:
            for aid, ent in self.shared.engine.agents.items():
                x = getattr(ent.transform.pos, "x", 0.0)
                z = getattr(ent.transform.pos, "z", 0.0)
                px, py = map_xy(x, z)
                tags = set(_public_tags(ent))
                if aid == sel:
                    brush = self._brush_sel
                elif LAB_AGENT_TAG in tags:
                    brush = self._brush_lab
                else:
                    brush = self._brush_agent
                p.setBrush(brush)
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawEllipse(QtCore.QPointF(px, py), 4.5 if aid==sel else 3.0, 4.5 if aid==sel else 3.0)
        except Exception:
            pass

        # animals (если есть)
        try:
            for ani in _iter_vals(getattr(self.shared.engine, "animals", [])):
                x = getattr(ani.transform.pos, "x", 0.0)
                z = getattr(ani.transform.pos, "z", 0.0)
                px, py = map_xy(x, z)
                p.setBrush(self._brush_animal); p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawRect(QtCore.QRectF(px-2, py-2, 4, 4))
        except Exception:
            pass

        marker = None
        if callable(self._player_provider):
            try:
                marker = self._player_provider()
            except Exception:
                marker = None
        if isinstance(marker, dict) and marker.get("active"):
            px, py = map_xy(float(marker.get("x", 0.0)), float(marker.get("z", 0.0)))
            yaw = float(marker.get("yaw_rad", 0.0))
            dx = math.cos(yaw) * 9.0
            dy = math.sin(yaw) * 9.0
            p.setPen(self._pen_player)
            p.setBrush(self._brush_player)
            p.drawEllipse(QtCore.QPointF(px, py), 4.2, 4.2)
            p.drawLine(QtCore.QPointF(px, py), QtCore.QPointF(px + dx, py + dy))
            p.setPen(QtCore.Qt.PenStyle.NoPen)

        p.end()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        r = self.rect().adjusted(6,6,-6,-6)
        ww, wh = self.shared.world_w, self.shared.world_h
        t = max(0.0, min(1.0, (e.position().x()-r.left())/max(1,r.width())))
        u = max(0.0, min(1.0, (e.position().y()-r.top())/max(1,r.height())))
        self.clickedWorld.emit(t*ww, u*wh)


class WorldMapOverlay(QtWidgets.QWidget):
    """Полноэкранная карта мира поверх интерфейса."""
    clickedWorld = QtCore.Signal(float, float)
    closed = QtCore.Signal()

    def __init__(self, shared, parent=None):
        super().__init__(parent)
        self.shared = shared
        self._player_provider = None
        self._background_cache_key = None
        self._background_cache = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.hide()

    def set_player_provider(self, provider):
        self._player_provider = provider
        self.update()

    def _zone_cache_signature(self) -> Tuple[Any, ...]:
        zones = list(getattr(getattr(self.shared.engine, "world", None), "zones", []) or [])
        signature = []
        for zone in zones:
            signature.append((
                str(getattr(zone, "obj_id", "") or ""),
                str(getattr(zone, "kind", "") or ""),
                round(float(getattr(zone, "x", 0.0)), 2),
                round(float(getattr(zone, "z", 0.0)), 2),
                round(float(getattr(zone, "radius", 0.0)), 2),
            ))
        return tuple(signature)

    def _invalidate_background_cache(self):
        self._background_cache_key = None
        self._background_cache = None

    def resizeEvent(self, event: QtGui.QResizeEvent):
        self._invalidate_background_cache()
        return super().resizeEvent(event)

    def show_map(self):
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self.update()

    def hide_map(self):
        if not self.isVisible():
            return
        self.hide()
        self.closed.emit()

    def _map_rect(self) -> QtCore.QRectF:
        outer = self.rect().adjusted(28, 28, -28, -28)
        panel = outer.adjusted(0, 56, 0, -42)
        ww = max(1.0, float(self.shared.world_w))
        wh = max(1.0, float(self.shared.world_h))
        aspect = ww / wh
        pw = panel.width()
        ph = panel.height()
        if pw / max(1.0, ph) > aspect:
            map_h = ph
            map_w = map_h * aspect
        else:
            map_w = pw
            map_h = map_w / max(1e-6, aspect)
        x = panel.center().x() - map_w * 0.5
        y = panel.center().y() - map_h * 0.5
        return QtCore.QRectF(x, y, map_w, map_h)

    def _world_to_screen(self, rect: QtCore.QRectF, x: float, y: float) -> QtCore.QPointF:
        ww = max(1.0, float(self.shared.world_w))
        wh = max(1.0, float(self.shared.world_h))
        px = rect.left() + (float(x) / ww) * rect.width()
        py = rect.top() + (float(y) / wh) * rect.height()
        return QtCore.QPointF(px, py)

    def _screen_to_world(self, rect: QtCore.QRectF, pos: QtCore.QPointF) -> Tuple[float, float]:
        ww = max(1.0, float(self.shared.world_w))
        wh = max(1.0, float(self.shared.world_h))
        tx = max(0.0, min(1.0, (pos.x() - rect.left()) / max(1.0, rect.width())))
        ty = max(0.0, min(1.0, (pos.y() - rect.top()) / max(1.0, rect.height())))
        return tx * ww, ty * wh

    def _paint_cached_background(self, painter: QtGui.QPainter) -> QtCore.QRectF:
        key = (
            int(self.width()),
            int(self.height()),
            int(round(float(self.shared.world_w))),
            int(round(float(self.shared.world_h))),
            self._zone_cache_signature(),
        )
        if self._background_cache_key != key or self._background_cache is None:
            pixmap = QtGui.QPixmap(self.size())
            pixmap.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(pixmap)
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            p.fillRect(self.rect(), QtGui.QColor(6, 10, 16, 228))

            outer = self.rect().adjusted(18, 18, -18, -18)
            panel_path = QtGui.QPainterPath()
            panel_path.addRoundedRect(QtCore.QRectF(outer), 26.0, 26.0)
            panel_grad = QtGui.QLinearGradient(outer.topLeft(), outer.bottomRight())
            panel_grad.setColorAt(0.0, QtGui.QColor(12, 18, 28, 242))
            panel_grad.setColorAt(1.0, QtGui.QColor(7, 11, 18, 242))
            p.fillPath(panel_path, panel_grad)
            p.setPen(QtGui.QPen(QtGui.QColor(82, 112, 158, 180), 1.2))
            p.drawPath(panel_path)

            title_font = p.font()
            title_font.setPointSize(15)
            title_font.setWeight(QtGui.QFont.DemiBold)
            p.setFont(title_font)
            p.setPen(QtGui.QColor(236, 244, 255))
            title_rect = QtCore.QRectF(outer.left() + 20, outer.top() + 14, outer.width() - 40, 26)
            p.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, "Карта мира")

            meta_font = p.font()
            meta_font.setPointSize(10)
            meta_font.setWeight(QtGui.QFont.Medium)
            p.setFont(meta_font)
            p.setPen(QtGui.QColor(170, 190, 214))
            world_meta = f"{self.shared.world_w:.0f} x {self.shared.world_h:.0f}  |  ЛКМ — перейти  |  Esc / M — закрыть"
            p.drawText(QtCore.QRectF(outer.left() + 20, outer.top() + 42, outer.width() - 40, 20), Qt.AlignLeft | Qt.AlignVCenter, world_meta)

            map_rect = self._map_rect()
            map_path = QtGui.QPainterPath()
            map_path.addRoundedRect(map_rect, 18.0, 18.0)
            p.fillPath(map_path, QtGui.QColor(16, 24, 34, 242))
            p.setPen(QtGui.QPen(QtGui.QColor(62, 88, 118, 210), 1.1))
            p.drawPath(map_path)

            clip_path = QtGui.QPainterPath()
            clip_path.addRoundedRect(map_rect.adjusted(1, 1, -1, -1), 17.0, 17.0)
            p.save()
            p.setClipPath(clip_path)
            for i in range(1, 8):
                x = map_rect.left() + map_rect.width() * (i / 8.0)
                y = map_rect.top() + map_rect.height() * (i / 8.0)
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 22), 1.0))
                p.drawLine(QtCore.QPointF(x, map_rect.top()), QtCore.QPointF(x, map_rect.bottom()))
                p.drawLine(QtCore.QPointF(map_rect.left(), y), QtCore.QPointF(map_rect.right(), y))

            for zone in list(getattr(getattr(self.shared.engine, "world", None), "zones", []) or []):
                kind = str(getattr(zone, "kind", "") or "")
                obj_id = str(getattr(zone, "obj_id", "") or "")
                if "training_room" in obj_id:
                    fill = QtGui.QColor(92, 18, 18, 78)
                    stroke = QtGui.QColor(138, 248, 196, 168)
                elif kind == "safe":
                    fill = QtGui.QColor(74, 210, 184, 58)
                    stroke = QtGui.QColor(108, 245, 210, 132)
                elif kind == "hazard":
                    fill = QtGui.QColor(255, 106, 106, 54)
                    stroke = QtGui.QColor(255, 148, 120, 138)
                else:
                    fill = QtGui.QColor(120, 140, 190, 26)
                    stroke = QtGui.QColor(146, 166, 214, 84)
                center = self._world_to_screen(map_rect, float(getattr(zone, "x", 0.0)), float(getattr(zone, "z", 0.0)))
                radius = (float(getattr(zone, "radius", 1.0)) / max(1.0, float(self.shared.world_w))) * map_rect.width()
                radius = max(4.0, radius)
                p.setPen(QtGui.QPen(stroke, 1.0))
                p.setBrush(fill)
                p.drawEllipse(center, radius, radius)
            p.restore()

            footer = QtCore.QRectF(outer.left() + 20, outer.bottom() - 28, outer.width() - 40, 18)
            p.setPen(QtGui.QColor(162, 180, 201))
            p.drawText(footer, Qt.AlignLeft | Qt.AlignVCenter, "Белые точки — выбранные, зелёно-мятные — лабораторный агент, зелёный маркер — игрок, оранжевые — животные")
            p.end()
            self._background_cache = pixmap
            self._background_cache_key = key
        painter.drawPixmap(0, 0, self._background_cache)
        return self._map_rect()

    def paintEvent(self, _event: QtGui.QPaintEvent):
        if not self.isVisible():
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        map_rect = self._paint_cached_background(p)

        p.save()
        clip_path = QtGui.QPainterPath()
        clip_path.addRoundedRect(map_rect.adjusted(1, 1, -1, -1), 17.0, 17.0)
        p.setClipPath(clip_path)
        selected_id = self.shared.get_selected_agent_id()
        for aid, ent in list(getattr(self.shared.engine, "agents", {}).items()):
            center = self._world_to_screen(map_rect, float(getattr(ent.transform.pos, "x", 0.0)), float(getattr(ent.transform.pos, "z", 0.0)))
            sel = aid == selected_id
            tags = set(_public_tags(ent))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            if sel:
                brush = QtGui.QColor(244, 248, 255, 244)
            elif LAB_AGENT_TAG in tags:
                brush = QtGui.QColor(88, 235, 176, 234)
            else:
                brush = QtGui.QColor(122, 162, 255, 224)
            p.setBrush(brush)
            r = 6.6 if sel else 4.2
            p.drawEllipse(center, r, r)
            if sel:
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 1.2))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(center, 10.0, 10.0)

        for ani in _iter_vals(getattr(self.shared.engine, "animals", [])):
            center = self._world_to_screen(map_rect, float(getattr(ani.transform.pos, "x", 0.0)), float(getattr(ani.transform.pos, "z", 0.0)))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(255, 191, 105, 218))
            p.drawRect(QtCore.QRectF(center.x() - 3.1, center.y() - 3.1, 6.2, 6.2))

        marker = None
        if callable(self._player_provider):
            try:
                marker = self._player_provider()
            except Exception:
                marker = None
        if isinstance(marker, dict) and marker.get("active"):
            center = self._world_to_screen(map_rect, float(marker.get("x", 0.0)), float(marker.get("z", 0.0)))
            yaw = float(marker.get("yaw_rad", 0.0))
            dx = math.cos(yaw) * 16.0
            dy = math.sin(yaw) * 16.0
            p.setPen(QtGui.QPen(QtGui.QColor(214, 255, 238, 248), 2.0))
            p.setBrush(QtGui.QColor(86, 236, 182, 240))
            p.drawEllipse(center, 6.8, 6.8)
            p.drawLine(center, QtCore.QPointF(center.x() + dx, center.y() + dy))

        p.restore()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() not in (Qt.LeftButton, Qt.RightButton):
            e.accept()
            return
        map_rect = self._map_rect()
        if e.button() == Qt.LeftButton and map_rect.contains(e.position()):
            wx, wy = self._screen_to_world(map_rect, e.position())
            self.clickedWorld.emit(float(wx), float(wy))
        self.hide_map()
        e.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        key = _normalize_control_key_event(e)
        if key in (Qt.Key_Escape, Qt.Key_M, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.hide_map()
            e.accept()
            return
        super().keyPressEvent(e)


# ======================================================
# 2) SharedState (лайт; мост между тренером и 3D)
# ======================================================
class SharedState(QtCore.QObject):
    updated = QtCore.Signal()

    def __init__(self, engine: MiniMatrixEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.tick: int = 0
        self.world_w: float = 100.0
        self.world_h: float = 100.0
        self.chat_tail: List[str] = []
        self.event_tail: List[Dict[str, Any]] = []
        self.selected_agent_id: Optional[str] = None
        self._connected = True  # локальный мост

    @Slot(dict)
    def update_from_snapshot(self, snap: Dict[str, Any]):
        self.tick = snap.get("tick", self.tick)
        w = snap.get("world", {})
        self.world_w = float(w.get("width", self.world_w))
        self.world_h = float(w.get("height", self.world_h))
        self.chat_tail = snap.get("chat", self.chat_tail)
        self.event_tail = snap.get("events", self.event_tail)
        self.engine.sync_from_world(snap)
        self._apply_selection_to_engine()
        self.updated.emit()

    def set_connected(self, ok: bool):
        self._connected = ok
        self.updated.emit()

    def is_connected(self) -> bool:
        return self._connected

    def get_tick(self) -> int:
        return self.tick

    def get_chat_lines(self) -> List[str]:
        return list(self.chat_tail)

    def get_world_events_lines(self) -> List[str]:
        lines: List[str] = []
        for ev in self.event_tail[-50:]:
            if isinstance(ev, dict):
                et = ev.get("type", "?")
                tk = ev.get("tick", "?")
                short = {k: v for k, v in ev.items() if k != "tick"}
                lines.append(f"[t={tk}] {et}: {short}")
            else:
                lines.append(str(ev))
        return lines

    def set_selected_agent(self, agent_id: Optional[str]):
        self.selected_agent_id = agent_id
        self._apply_selection_to_engine()
        self.updated.emit()

    def cycle_next_agent(self):
        ids = list(self.engine.agents.keys())
        if not ids:
            return
        ids.sort()
        if self.selected_agent_id in ids:
            i = (ids.index(self.selected_agent_id) + 1) % len(ids)
        else:
            i = 0
        self.selected_agent_id = ids[i]
        self._apply_selection_to_engine()
        self.updated.emit()

    def get_selected_agent_id(self) -> Optional[str]:
        return self.selected_agent_id

    def _apply_selection_to_engine(self):
        for aid, ent in self.engine.agents.items():
            try:
                ent.selected = (aid == self.selected_agent_id)
            except Exception:
                pass

    def get_selected_agent_debug(self) -> Dict[str, Any]:
        aid = self.selected_agent_id
        if not aid:
            return {}
        ent = self.engine.agents.get(aid)
        if not ent:
            return {}
        st = getattr(ent, "public_state", {}) or {}

        def _f(d: dict, k: str, default=0.0) -> float:
            try:
                return float(d.get(k, default))
            except Exception:
                return default

        pos = st.get("pos", {})
        goal = st.get("goal", {})
        vel = st.get("vel", {})
        mind = st.get("mind", {}) or {}
        return {
            "id": aid,
            "name": st.get("name", aid),
            "tags": list(st.get("tags", []) or []),
            "pos": {"x": _f(pos, "x"), "y": _f(pos, "y")},
            "goal": {"x": _f(goal, "x"), "y": _f(goal, "y")},
            "vel": {"x": _f(vel, "x"), "y": _f(vel, "y")},
            "fear": _f(st, "fear"),
            "health": _f(st, "health", 100.0),
            "energy": _f(st, "energy", 100.0),
            "hunger": _f(st, "hunger", 0.0),
            "age_ticks": int(st.get("age_ticks", 0)),
            "alive": bool(st.get("alive", True)),
            "cause_of_death": st.get("cause_of_death"),
            "mind_drive": mind.get("current_drive"),
            "mind_survival_score": mind.get("survival_score"),
            "mind_behavior_rules": mind.get("behavior_rules", {}),
            "mind_beliefs": mind.get("beliefs", []),
            "mind_memory_tail": mind.get("memory_tail", []),
        }


# ======================================
# 3) 3D-вью + HUD оверлей + FPS
# ======================================
class World3DView(QOpenGLWidget):
    requestSetGoal = QtCore.Signal(str, float, float)  # agent_id, x, z
    fpsUpdated = QtCore.Signal(float)
    firstPersonChanged = QtCore.Signal(bool)
    firstPersonMessage = QtCore.Signal(str)
    worldMapRequested = QtCore.Signal()
    gameFullscreenRequested = QtCore.Signal()

    def __init__(self, shared: SharedState, parent=None, *, player_drive_callback=None):
        super().__init__(parent)
        self.shared = shared
        self.engine = shared.engine
        self._player_drive_callback = player_drive_callback
        self.shared.updated.connect(self._on_shared_updated)

        self.center_x = 50.0
        self.center_z = 50.0
        self.distance = 140.0
        self.yaw_deg = -135.0
        self.pitch_deg = 40.0
        self.fov_deg = 45.0

        self._mv = None
        self._proj = None
        self._viewport = None
        self._last_mouse_pos: Optional[QtCore.QPointF] = None
        self._btns = Qt.NoButton

        self._camera_mode = "orbit"
        self._pressed_keys: set[int] = set()
        self._fp_x = self.center_x
        self._fp_z = self.center_z
        self._fp_yaw_deg = 45.0
        self._fp_pitch_deg = -6.0
        self._fp_target_yaw_deg = self._fp_yaw_deg
        self._fp_target_pitch_deg = self._fp_pitch_deg
        self._fp_eye_height = 1.72
        self._fp_radius = 0.92
        self._fp_walk_speed = 8.5
        self._fp_run_speed = 13.5
        self._fp_precision_speed = 4.8
        self._fp_vel_x = 0.0
        self._fp_vel_z = 0.0
        self._fp_accel_response = 12.0
        self._fp_decel_response = 9.0
        self._fp_look_response = 18.0
        self._fp_headbob_phase = 0.0
        self._fp_headbob_amount = 0.0
        self._fp_last_speed_mode = "walk"
        self._fp_captured = False
        self._fp_mouse_grabbed = False
        self._fp_keyboard_grabbed = False
        self._ignore_mouse_warp = False
        self._mouse_sensitivity = 0.10
        self._mouse_edge_margin = 48.0
        self._fp_collision_padding = 0.22
        self._fp_focus_info: Dict[str, Any] = {}
        self._fp_center_ground: Optional[Tuple[float, float]] = None
        self._orbit_fov_deg = self.fov_deg
        self._fp_fov_deg = 72.0
        self._fp_fov_run_deg = 78.0
        self._fp_fov_current = self._fp_fov_deg

        self._timer = QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.setInterval(VIEW_FRAME_INTERVAL_MS)
        self._timer.timeout.connect(self._frame_tick)
        self._timer.start()
        self._last_frame_time = QtCore.QElapsedTimer()
        self._last_frame_time.start()

        self._fps_smooth = 0.0
        self._scene_dirty = True
        self._idle_repaint_accum = 0.0
        platform_name = str(QtGui.QGuiApplication.platformName() or "").strip().lower()
        self._qt_platform_name = platform_name
        self._supports_native_pointer_grab = platform_name in {"xcb", "windows", "cocoa"}
        self._supports_keyboard_grab = platform_name in {"xcb", "windows", "cocoa"}

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    @Slot()
    def _on_shared_updated(self):
        self._scene_dirty = True

    def is_first_person_mode(self) -> bool:
        return self._camera_mode == "first_person"

    def get_first_person_hud_state(self) -> Dict[str, Any]:
        selected_name = str(self.shared.get_selected_agent_debug().get("name") or "—")
        driving = bool(self._player_control_agent_id())
        return {
            "active": self.is_first_person_mode(),
            "captured": self._fp_captured,
            "x": float(self._fp_x),
            "z": float(self._fp_z),
            "yaw_deg": float(self._fp_yaw_deg),
            "pitch_deg": float(self._fp_pitch_deg),
            "speed_mode": self._fp_last_speed_mode,
            "focus_name": self._fp_focus_info.get("name"),
            "focus_kind": self._fp_focus_info.get("kind"),
            "focus_distance": self._fp_focus_info.get("distance"),
            "has_ground_target": bool(self._fp_center_ground),
            "selected_name": selected_name,
            "driving": driving,
        }

    def get_minimap_player_marker(self) -> Dict[str, Any]:
        return {
            "active": self.is_first_person_mode(),
            "x": float(self._fp_x),
            "z": float(self._fp_z),
            "yaw_rad": math.radians(self._fp_yaw_deg),
        }

    def camera_status_text(self) -> str:
        if self.is_first_person_mode():
            if self._player_control_agent_id():
                who = str(self.shared.get_selected_agent_debug().get("name") or self.shared.get_selected_agent_id() or "agent")
                return f"Cam: FPS / drive {who} x={self._fp_x:0.1f}, z={self._fp_z:0.1f}, yaw={self._fp_yaw_deg:0.0f}°, pitch={self._fp_pitch_deg:0.0f}°"
            return f"Cam: FPS x={self._fp_x:0.1f}, z={self._fp_z:0.1f}, yaw={self._fp_yaw_deg:0.0f}°, pitch={self._fp_pitch_deg:0.0f}°"
        return f"Cam: ORBIT x={self.center_x:0.1f}, z={self.center_z:0.1f}, dist={self.distance:0.0f}"

    @staticmethod
    def _angle_diff_deg(target: float, current: float) -> float:
        return (target - current + 180.0) % 360.0 - 180.0

    def _camera_forward_vector(self) -> Tuple[float, float, float]:
        yaw = math.radians(self._fp_yaw_deg)
        pitch = math.radians(self._fp_pitch_deg)
        cos_p = math.cos(pitch)
        return cos_p * math.cos(yaw), math.sin(pitch), cos_p * math.sin(yaw)

    def _current_fov_deg(self) -> float:
        return self._fp_fov_current if self.is_first_person_mode() else self._orbit_fov_deg

    def _orbit_camera_position(self) -> Tuple[float, float, float]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        r = max(10.0, self.distance)
        cx, cz = self.center_x, self.center_z
        cos_p = math.cos(pitch); sin_p = math.sin(pitch)
        cam_x = cx + r * cos_p * math.cos(yaw)
        cam_y = max(5.0, r * sin_p)
        cam_z = cz + r * cos_p * math.sin(yaw)
        return cam_x, cam_y, cam_z

    def _first_person_eye_y(self) -> float:
        return self._fp_eye_height + math.sin(self._fp_headbob_phase) * self._fp_headbob_amount

    def _camera_position(self) -> Tuple[float, float, float]:
        if self.is_first_person_mode():
            return self._fp_x, self._first_person_eye_y(), self._fp_z
        return self._orbit_camera_position()

    def _camera_look_target(self) -> Tuple[float, float, float]:
        if self.is_first_person_mode():
            fx, fy, fz = self._camera_forward_vector()
            cam_x, cam_y, cam_z = self._camera_position()
            return (
                cam_x + fx * 8.0,
                cam_y + fy * 8.0,
                cam_z + fz * 8.0,
            )
        return self.center_x, 0.0, self.center_z

    def _clamp_center(self):
        self.center_x = max(0.0, min(self.shared.world_w, self.center_x))
        self.center_z = max(0.0, min(self.shared.world_h, self.center_z))

    def _clamp_first_person(self):
        margin = self._fp_radius + 0.4
        self._fp_x = max(margin, min(self.shared.world_w - margin, self._fp_x))
        self._fp_z = max(margin, min(self.shared.world_h - margin, self._fp_z))
        self.center_x = self._fp_x
        self.center_z = self._fp_z

    def _selected_agent_entity(self):
        sel = self.shared.get_selected_agent_id()
        if sel and sel in self.engine.agents:
            return self.engine.agents[sel]
        return None

    def _sync_first_person_agent_visibility(self) -> None:
        hidden: set[str] = set()
        if self.is_first_person_mode():
            control_id = self._player_control_agent_id()
            if control_id:
                hidden.add(str(control_id))
        try:
            self.engine.hidden_agent_ids = hidden
        except Exception:
            pass

    def _player_control_agent_id(self) -> Optional[str]:
        sel = self.shared.get_selected_agent_id()
        if not sel or not callable(self._player_drive_callback):
            return None
        return str(sel)

    def _ensure_control_target_selected(self) -> Optional[str]:
        sel = self.shared.get_selected_agent_id()
        if sel and sel in self.engine.agents:
            return str(sel)
        if not getattr(self.engine, "agents", None):
            return None
        best_id = None
        best_d2 = float("inf")
        ref_x = float(self.center_x)
        ref_z = float(self.center_z)
        for aid, ent in list(self.engine.agents.items()):
            dx = float(ent.transform.pos.x) - ref_x
            dz = float(ent.transform.pos.z) - ref_z
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_id = aid
        if best_id:
            self.shared.set_selected_agent(best_id)
            return str(best_id)
        return None

    def _seed_first_person_from_context(self):
        self._sync_first_person_agent_visibility()
        if callable(self._player_drive_callback):
            self._ensure_control_target_selected()
        ent = self._selected_agent_entity()
        if ent is not None:
            base_yaw = math.degrees(float(ent.transform.yaw))
            self._fp_x = float(ent.transform.pos.x)
            self._fp_z = float(ent.transform.pos.z)
            self._fp_yaw_deg = base_yaw
            self._fp_pitch_deg = -4.0
        else:
            cam_x, cam_y, cam_z = self._orbit_camera_position()
            dir_x = self.center_x - cam_x
            dir_z = self.center_z - cam_z
            horiz = max(1e-6, math.hypot(dir_x, dir_z))
            self._fp_x = float(self.center_x)
            self._fp_z = float(self.center_z)
            self._fp_yaw_deg = math.degrees(math.atan2(dir_z, dir_x))
            self._fp_pitch_deg = max(-22.0, min(18.0, math.degrees(math.atan2(-cam_y, horiz))))
        self._fp_x, self._fp_z = self._resolve_first_person_collisions(self._fp_x, self._fp_z)
        self._clamp_first_person()
        self._fp_target_yaw_deg = self._fp_yaw_deg
        self._fp_target_pitch_deg = self._fp_pitch_deg
        self._fp_vel_x = 0.0
        self._fp_vel_z = 0.0
        self._fp_headbob_amount = 0.0
        self._fp_fov_current = self._fp_fov_deg
        self._update_first_person_focus()

    def _drive_selected_agent(self, x: float, z: float, dt: float, *, facing_x: float, facing_z: float) -> Tuple[float, float]:
        agent_id = self._player_control_agent_id()
        if agent_id is None:
            return float(x), float(z)
        try:
            result = self._player_drive_callback(
                agent_id,
                float(x),
                float(z),
                max(float(dt), 1e-6),
                float(facing_x),
                float(facing_z),
                str(self._fp_last_speed_mode),
            )
        except Exception:
            return float(x), float(z)
        nx = float(x)
        nz = float(z)
        ok = True
        if isinstance(result, Mapping):
            ok = bool(result.get("ok", True))
            nx = float(result.get("x", nx))
            nz = float(result.get("z", nz))
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)) and len(result) >= 2:
            nx = float(result[0])
            nz = float(result[1])
        if not ok:
            return float(self._fp_x), float(self._fp_z)
        return nx, nz

    def _mesh_collision_radius(self, inst) -> float:
        kind = str(getattr(inst, "kind", "") or "")
        sx = abs(float(getattr(getattr(inst, "scale", None), "x", 1.0)))
        sz = abs(float(getattr(getattr(inst, "scale", None), "z", 1.0)))
        base = 0.5 * (sx + sz)
        mul = {
            "house": 0.52,
            "tree": 0.24,
            "tower": 0.36,
            "well": 0.38,
            "shrine": 0.40,
            "wall": 0.58,
            "rock": 0.42,
            "log": 0.34,
            "lantern": 0.16,
            "lake": 0.62,
            "fire": 0.22,
        }.get(kind)
        if mul is None:
            return 0.0
        return max(0.55, base * mul)

    def _iter_first_person_colliders(self):
        sel = self.shared.get_selected_agent_id()
        for inst in list(getattr(self.engine, "static_meshes", []) or []):
            radius = self._mesh_collision_radius(inst)
            if radius <= 0.0:
                continue
            pos = getattr(inst, "pos", None)
            yield float(getattr(pos, "x", 0.0)), float(getattr(pos, "z", 0.0)), radius
        for aid, ent in list(getattr(self.engine, "agents", {}).items()):
            if aid == sel:
                continue
            yield float(ent.transform.pos.x), float(ent.transform.pos.z), 0.78
        for ani in _iter_vals(getattr(self.engine, "animals", [])):
            transform = getattr(ani, "transform", None)
            pos = getattr(transform, "pos", None)
            if pos is None:
                continue
            yield float(getattr(pos, "x", 0.0)), float(getattr(pos, "z", 0.0)), 0.64

    def _screen_center_world_hit(self) -> Optional[Tuple[float, float]]:
        if self.width() <= 0 or self.height() <= 0:
            return None
        return self._screen_to_world_plane(self.width() * 0.5, self.height() * 0.5)

    def _update_first_person_focus(self):
        self._fp_center_ground = self._screen_center_world_hit()
        cam_x, cam_y, cam_z = self._camera_position()
        fx, fy, fz = self._camera_forward_vector()
        best: Optional[Dict[str, Any]] = None
        max_dist = 26.0
        hidden_agent_id = self._player_control_agent_id()

        def _try(kind: str, name: str, entity_id: Optional[str], ex: float, ey: float, ez: float, radius: float):
            nonlocal best
            vx = ex - cam_x
            vy = ey - cam_y
            vz = ez - cam_z
            t = vx * fx + vy * fy + vz * fz
            if t < 0.35 or t > max_dist:
                return
            perp2 = max(0.0, vx * vx + vy * vy + vz * vz - t * t)
            lock_r = radius + 0.05 * t
            if perp2 > lock_r * lock_r:
                return
            candidate = {
                "kind": kind,
                "name": name,
                "entity_id": entity_id,
                "distance": t,
            }
            if best is None or t < float(best["distance"]):
                best = candidate

        for aid, ent in list(getattr(self.engine, "agents", {}).items()):
            if hidden_agent_id and aid == hidden_agent_id:
                continue
            _try(
                "agent",
                str(getattr(ent, "name", aid)),
                aid,
                float(ent.transform.pos.x),
                1.05,
                float(ent.transform.pos.z),
                0.72,
            )
        for ani in _iter_vals(getattr(self.engine, "animals", [])):
            transform = getattr(ani, "transform", None)
            pos = getattr(transform, "pos", None)
            if pos is None:
                continue
            _try(
                "animal",
                str(getattr(ani, "name", getattr(ani, "species", "animal"))),
                None,
                float(getattr(pos, "x", 0.0)),
                0.85,
                float(getattr(pos, "z", 0.0)),
                0.62,
            )
        self._fp_focus_info = best or {}

    def _resolve_first_person_collisions(self, x: float, z: float) -> Tuple[float, float]:
        margin = self._fp_radius + 0.4
        x = max(margin, min(self.shared.world_w - margin, x))
        z = max(margin, min(self.shared.world_h - margin, z))
        for i in range(3):
            moved = False
            for cx, cz, radius in self._iter_first_person_colliders():
                dx = x - cx
                dz = z - cz
                min_dist = self._fp_radius + radius + self._fp_collision_padding
                d2 = dx * dx + dz * dz
                if d2 >= min_dist * min_dist:
                    continue
                if d2 < 1e-8:
                    ang = math.radians(self._fp_yaw_deg + 90.0 * (i + 1))
                    dx = math.cos(ang)
                    dz = math.sin(ang)
                    dist = 1.0
                else:
                    dist = math.sqrt(d2)
                nx = dx / max(1e-6, dist)
                nz = dz / max(1e-6, dist)
                x = cx + nx * min_dist
                z = cz + nz * min_dist
                x = max(margin, min(self.shared.world_w - margin, x))
                z = max(margin, min(self.shared.world_h - margin, z))
                moved = True
            if not moved:
                break
        return x, z

    def _engage_mouse_capture(self):
        if self._fp_captured or not self.isVisible():
            return
        self._fp_mouse_grabbed = False
        self.setCursor(Qt.BlankCursor)
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if self._supports_native_pointer_grab:
            try:
                self.grabMouse(QtGui.QCursor(Qt.BlankCursor))
                self._fp_mouse_grabbed = True
            except TypeError:
                try:
                    self.grabMouse()
                    self._fp_mouse_grabbed = True
                except Exception:
                    self._fp_mouse_grabbed = False
            except Exception:
                self._fp_mouse_grabbed = False
        self._fp_captured = True
        self._engage_keyboard_capture()
        self._warp_mouse_to_center()

    def _release_mouse_capture(self):
        if not self._fp_captured:
            return
        if self._fp_mouse_grabbed:
            try:
                self.releaseMouse()
            except Exception:
                pass
        self.unsetCursor()
        self._fp_mouse_grabbed = False
        self._fp_captured = False
        self._ignore_mouse_warp = False

    def _engage_keyboard_capture(self):
        if self._fp_keyboard_grabbed:
            return
        self._fp_keyboard_grabbed = False
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        if self._supports_keyboard_grab:
            try:
                self.grabKeyboard()
                self._fp_keyboard_grabbed = True
            except Exception:
                self._fp_keyboard_grabbed = False

    def _release_keyboard_capture(self):
        if not self._fp_keyboard_grabbed:
            return
        try:
            self.releaseKeyboard()
        except Exception:
            pass
        self._fp_keyboard_grabbed = False

    def _warp_mouse_to_center(self):
        if not self.isVisible() or not self._fp_mouse_grabbed:
            return
        center = QtCore.QPoint(self.width() // 2, self.height() // 2)
        self._ignore_mouse_warp = True
        QtGui.QCursor.setPos(self.mapToGlobal(center))
        self._last_mouse_pos = QtCore.QPointF(center)

    def _toggle_first_person_capture(self):
        if not self.is_first_person_mode():
            return
        if self._fp_captured:
            self._release_mouse_capture()
            self.firstPersonMessage.emit("First person: cursor released")
        else:
            self._engage_mouse_capture()
            self.firstPersonMessage.emit("First person: cursor captured")

    def suspend_first_person_capture(self):
        if not self.is_first_person_mode():
            return
        self._pressed_keys.clear()
        self._release_mouse_capture()
        self._release_keyboard_capture()
        self._last_mouse_pos = None

    def resume_first_person_capture(self):
        if not self.is_first_person_mode():
            return
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._engage_mouse_capture()
        self._engage_keyboard_capture()

    def _first_person_primary_action(self):
        focus = dict(self._fp_focus_info or {})
        if focus.get("kind") == "agent" and focus.get("entity_id"):
            self.shared.set_selected_agent(str(focus["entity_id"]))
            self.firstPersonMessage.emit(f"Selected {focus.get('name')}")
            return
        if focus.get("kind") == "animal":
            self.firstPersonMessage.emit(f"Observed {focus.get('name')}")
            return
        if self._fp_center_ground is not None:
            x, z = self._fp_center_ground
            self.firstPersonMessage.emit(f"Ground target: ({x:.1f}, {z:.1f})")

    def _first_person_secondary_action(self):
        sel = self.shared.get_selected_agent_id()
        if not sel:
            focus = dict(self._fp_focus_info or {})
            if focus.get("kind") == "agent" and focus.get("entity_id"):
                self.shared.set_selected_agent(str(focus["entity_id"]))
                self.firstPersonMessage.emit(f"Selected {focus.get('name')}")
            else:
                self.firstPersonMessage.emit("No selected agent for command")
            return
        if self._fp_center_ground is None:
            self.firstPersonMessage.emit("No ground point under crosshair")
            return
        x, z = self._fp_center_ground
        self.requestSetGoal.emit(str(sel), float(x), float(z))
        self.firstPersonMessage.emit(f"Goal for {sel} -> ({x:.1f}, {z:.1f})")

    def set_first_person_mode(self, enabled: bool):
        enabled = bool(enabled)
        if enabled == self.is_first_person_mode():
            if enabled:
                self._engage_mouse_capture()
                self._sync_first_person_agent_visibility()
            return
        if enabled:
            self._camera_mode = "first_person"
            self._seed_first_person_from_context()
            self._engage_mouse_capture()
            self._engage_keyboard_capture()
            self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
            self.firstPersonChanged.emit(True)
            control_target = self.shared.get_selected_agent_debug().get("name") or self.shared.get_selected_agent_id()
            if control_target:
                self.firstPersonMessage.emit(f"Controlling {control_target}")
            self.update()
            return
        self._release_mouse_capture()
        self._release_keyboard_capture()
        self._pressed_keys.clear()
        self._camera_mode = "orbit"
        self._sync_first_person_agent_visibility()
        self.center_x = self._fp_x
        self.center_z = self._fp_z
        self.yaw_deg = self._fp_yaw_deg + 180.0
        self.pitch_deg = 28.0
        self.distance = max(44.0, min(120.0, self.distance))
        self._clamp_center()
        self._fp_focus_info = {}
        self._fp_center_ground = None
        self.firstPersonChanged.emit(False)
        self.update()

    def toggle_first_person_mode(self):
        self.set_first_person_mode(not self.is_first_person_mode())

    def reset_view(self):
        if self.is_first_person_mode():
            self._seed_first_person_from_context()
            self._engage_mouse_capture()
            self.update()
            return
        self.center_x = self.shared.world_w * 0.5
        self.center_z = self.shared.world_h * 0.5
        self.distance = max(140.0, max(self.shared.world_w, self.shared.world_h) * 1.25)
        self.yaw_deg = -135.0
        self.pitch_deg = 40.0
        self.update()

    def focus_selected_target(self):
        ent = self._selected_agent_entity()
        if ent is None:
            return
        if self.is_first_person_mode():
            self._fp_x = float(ent.transform.pos.x)
            self._fp_z = float(ent.transform.pos.z)
            self._fp_yaw_deg = math.degrees(float(ent.transform.yaw))
            self._fp_x, self._fp_z = self._resolve_first_person_collisions(self._fp_x, self._fp_z)
            self._clamp_first_person()
            self._engage_mouse_capture()
            self.update()
            return
        self.center_x = ent.transform.pos.x
        self.center_z = ent.transform.pos.z
        self._clamp_center()
        self.update()

    def cycle_next_target(self):
        self.shared.cycle_next_agent()
        self._sync_first_person_agent_visibility()
        self.focus_selected_target()

    def move_first_person_to(self, x: float, z: float):
        self._fp_x = float(x)
        self._fp_z = float(z)
        self._fp_x, self._fp_z = self._resolve_first_person_collisions(self._fp_x, self._fp_z)
        self._fp_vel_x = 0.0
        self._fp_vel_z = 0.0
        facing_x = math.cos(math.radians(self._fp_yaw_deg))
        facing_z = math.sin(math.radians(self._fp_yaw_deg))
        self._fp_x, self._fp_z = self._drive_selected_agent(
            self._fp_x,
            self._fp_z,
            1.0 / 60.0,
            facing_x=facing_x,
            facing_z=facing_z,
        )
        self._clamp_first_person()
        self.update()

    def _update_first_person(self, dt: float):
        if not self.is_first_person_mode():
            return
        self._sync_first_person_agent_visibility()
        dt = max(0.0, min(dt, 0.05))
        yaw_diff = self._angle_diff_deg(self._fp_target_yaw_deg, self._fp_yaw_deg)
        self._fp_yaw_deg += yaw_diff * min(1.0, dt * self._fp_look_response)
        pitch_alpha = min(1.0, dt * self._fp_look_response)
        self._fp_pitch_deg += (self._fp_target_pitch_deg - self._fp_pitch_deg) * pitch_alpha

        move_local_x = 0.0
        move_local_z = 0.0
        yaw = math.radians(self._fp_yaw_deg)
        fwd_x = math.cos(yaw)
        fwd_z = math.sin(yaw)
        right_x = -math.sin(yaw)
        right_z = math.cos(yaw)

        if Qt.Key_W in self._pressed_keys or Qt.Key_Up in self._pressed_keys:
            move_local_z += 1.0
        if Qt.Key_S in self._pressed_keys or Qt.Key_Down in self._pressed_keys:
            move_local_z -= 0.78
        if Qt.Key_A in self._pressed_keys:
            move_local_x -= 0.92
        if Qt.Key_D in self._pressed_keys:
            move_local_x += 0.92

        move_x = right_x * move_local_x + fwd_x * move_local_z
        move_z = right_z * move_local_x + fwd_z * move_local_z
        length = math.hypot(move_x, move_z)
        running = Qt.Key_Shift in self._pressed_keys
        precision = Qt.Key_Control in self._pressed_keys
        target_bob = 0.0
        if precision:
            self._fp_last_speed_mode = "precision"
        else:
            self._fp_last_speed_mode = "sprint" if running else "walk"

        target_vx = 0.0
        target_vz = 0.0
        if length > 1e-6:
            move_x /= length
            move_z /= length
            if precision:
                speed = self._fp_precision_speed
            else:
                speed = self._fp_run_speed if running else self._fp_walk_speed
            target_vx = move_x * speed
            target_vz = move_z * speed
            target_bob = 0.058 if running else 0.034
            self._fp_headbob_phase += dt * (11.5 if running else 7.2)
        vel_alpha = min(1.0, dt * (self._fp_accel_response if length > 1e-6 else self._fp_decel_response))
        self._fp_vel_x += (target_vx - self._fp_vel_x) * vel_alpha
        self._fp_vel_z += (target_vz - self._fp_vel_z) * vel_alpha
        next_x = self._fp_x + self._fp_vel_x * dt
        next_z = self._fp_z + self._fp_vel_z * dt
        self._fp_x, self._fp_z = self._resolve_first_person_collisions(next_x, next_z)
        if abs(next_x - self._fp_x) > 1e-4:
            self._fp_vel_x = 0.0
        if abs(next_z - self._fp_z) > 1e-4:
            self._fp_vel_z = 0.0
        self._clamp_first_person()
        if abs(self._fp_vel_x) + abs(self._fp_vel_z) > 0.06:
            target_bob = max(target_bob, 0.028)
        self._fp_headbob_amount += (target_bob - self._fp_headbob_amount) * min(1.0, dt * 7.5)
        fov_target = self._fp_fov_run_deg if running and not precision else self._fp_fov_deg
        self._fp_fov_current += (fov_target - self._fp_fov_current) * min(1.0, dt * 6.0)
        self._fp_x, self._fp_z = self._drive_selected_agent(
            self._fp_x,
            self._fp_z,
            dt,
            facing_x=fwd_x,
            facing_z=fwd_z,
        )
        self.center_x = self._fp_x
        self.center_z = self._fp_z
        self._update_first_person_focus()

    def paintGL(self):
        cam_x, cam_y, cam_z = self._camera_position()
        self.engine.setup_viewport_and_camera(
            w=self.width(), h=self.height(),
            cam_pos=(cam_x, cam_y, cam_z),
            cam_look=self._camera_look_target(),
            fov_deg=self._current_fov_deg(),
        )
        self.engine.render_opengl()
        self._mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport = glGetIntegerv(GL_VIEWPORT)

    def _frame_tick(self):
        dt = self._last_frame_time.elapsed() / 1000.0
        self._last_frame_time.restart()
        try:
            self.engine.update(dt)
            self._update_first_person(dt)
        finally:
            repaint_now = self._scene_dirty or self.is_first_person_mode()
            if not repaint_now:
                self._idle_repaint_accum += max(0.0, dt)
                repaint_now = self._idle_repaint_accum >= 0.05
            if repaint_now:
                self._scene_dirty = False
                self._idle_repaint_accum = 0.0
                self.update()
            if dt > 0:
                fps = 1.0/dt
                self._fps_smooth = 0.9*self._fps_smooth + 0.1*fps if self._fps_smooth else fps
                self.fpsUpdated.emit(self._fps_smooth)

    def _screen_to_world_plane(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self._mv is None or self._proj is None or self._viewport is None:
            return None
        gl_y = self._viewport[3] - y
        p0 = gluUnProject(x, gl_y, 0.0, self._mv, self._proj, self._viewport)
        p1 = gluUnProject(x, gl_y, 1.0, self._mv, self._proj, self._viewport)
        if not p0 or not p1:
            return None
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        dy = (y1 - y0)
        if abs(dy) < 1e-6:
            return None
        t = -y0 / dy
        wx = x0 + (x1 - x0) * t
        wz = z0 + (z1 - z0) * t
        return float(wx), float(wz)

    def _pick_agent_near(self, wx: float, wz: float, radius: float = 2.0) -> Optional[str]:
        best_id = None
        best_d2 = radius * radius
        for aid, ent in self.engine.agents.items():
            dx = ent.transform.pos.x - wx
            dz = ent.transform.pos.z - wz
            d2 = dx * dx + dz * dz
            if d2 <= best_d2:
                best_d2 = d2
                best_id = aid
        return best_id

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self.is_first_person_mode():
            if not self._fp_captured:
                self._engage_mouse_capture()
                e.accept()
                return
            if e.button() == Qt.LeftButton:
                self._first_person_primary_action()
                e.accept()
                return
            if e.button() == Qt.RightButton:
                self._first_person_secondary_action()
                e.accept()
                return
            e.accept()
            return
        self._btns = e.buttons()
        self._last_mouse_pos = e.position()
        if e.button() == Qt.LeftButton:
            hit = self._screen_to_world_plane(e.position().x(), e.position().y())
            if hit:
                wx, wz = hit
                aid = self._pick_agent_near(wx, wz, radius=2.0)
                if aid:
                    self.shared.set_selected_agent(aid)
        if e.button() == Qt.RightButton:
            sel = self.shared.get_selected_agent_id()
            if sel:
                hit = self._screen_to_world_plane(e.position().x(), e.position().y())
                if hit:
                    wx, wz = hit
                    # Ctrl+ПКМ — контекст без приказа; обычный ПКМ — приказ как раньше
                    if e.modifiers() & Qt.ControlModifier:
                        menu = QtWidgets.QMenu(self)
                        act_goal = menu.addAction("Set goal here")
                        act_focus = menu.addAction("Focus camera here")
                        act_copy = menu.addAction("Copy coords")
                        chosen = menu.exec(self.mapToGlobal(QtCore.QPoint(int(e.position().x()), int(e.position().y()))))
                        if chosen == act_goal:
                            self.requestSetGoal.emit(sel, wx, wz)
                        elif chosen == act_focus:
                            self.center_x, self.center_z = wx, wz; self._clamp_center(); self.update()
                        elif chosen == act_copy:
                            QtGui.QGuiApplication.clipboard().setText(f"{wx:.2f},{wz:.2f}")
                    else:
                        self.requestSetGoal.emit(sel, wx, wz)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self.is_first_person_mode():
            e.accept()
            return
        self._btns = e.buttons()
        self._last_mouse_pos = None
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.is_first_person_mode():
            if not self._fp_captured:
                self._last_mouse_pos = e.position()
                e.accept()
                return
            if self._ignore_mouse_warp:
                self._ignore_mouse_warp = False
                self._last_mouse_pos = e.position()
                e.accept()
                return
            if self._last_mouse_pos is None:
                self._last_mouse_pos = e.position()
                e.accept()
                return
            delta = e.position() - self._last_mouse_pos
            self._last_mouse_pos = e.position()
            dx = max(-42.0, min(42.0, float(delta.x())))
            dy = max(-42.0, min(42.0, float(delta.y())))
            self._fp_target_yaw_deg += dx * self._mouse_sensitivity
            self._fp_target_pitch_deg = max(-78.0, min(68.0, self._fp_target_pitch_deg - dy * self._mouse_sensitivity * 0.88))
            if self._fp_mouse_grabbed and (
                e.position().x() < self._mouse_edge_margin
                or e.position().x() > self.width() - self._mouse_edge_margin
                or e.position().y() < self._mouse_edge_margin
                or e.position().y() > self.height() - self._mouse_edge_margin
            ):
                self._warp_mouse_to_center()
            e.accept()
            return
        if self._last_mouse_pos is None:
            self._last_mouse_pos = e.position()
        delta = e.position() - self._last_mouse_pos
        self._last_mouse_pos = e.position()

        if self._btns & Qt.RightButton:  # орбита
            self.yaw_deg += delta.x() * 0.25
            self.pitch_deg = max(15.0, min(80.0, self.pitch_deg - delta.y() * 0.25))
            self.update(); return

        if self._btns & Qt.MiddleButton:  # пан
            pan_speed = max(0.1, self.distance * 0.01)
            yaw = math.radians(self.yaw_deg)
            right_x = math.cos(yaw); right_z = math.sin(yaw)
            fwd_x = -math.sin(yaw);  fwd_z = math.cos(yaw)
            self.center_x -= (right_x * delta.x() + fwd_x * delta.y()) * pan_speed * 0.02
            self.center_z -= (right_z * delta.x() + fwd_z * delta.y()) * pan_speed * 0.02
            self._clamp_center(); self.update(); return

        super().mouseMoveEvent(e)

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if self.is_first_person_mode():
            e.accept()
            return
        delta = e.angleDelta().y() / 120.0
        self.distance *= math.pow(0.9, delta)
        self.distance = max(20.0, min(600.0, self.distance))
        self.update()
        super().wheelEvent(e)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        key = _normalize_control_key_event(e)
        move_keys = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_Up, Qt.Key_Down, Qt.Key_Shift, Qt.Key_Control}
        if key in move_keys:
            self._pressed_keys.add(key)
            e.accept()
            return
        if e.isAutoRepeat():
            e.accept()
            return
        if key == Qt.Key_V:
            self.toggle_first_person_mode()
            e.accept()
        elif key == Qt.Key_Escape and self.is_first_person_mode():
            self.set_first_person_mode(False)
            e.accept()
        elif key == Qt.Key_Q and self.is_first_person_mode():
            self._toggle_first_person_capture()
            e.accept()
        elif key == Qt.Key_E and self.is_first_person_mode():
            self._first_person_primary_action()
            e.accept()
        elif key == Qt.Key_G and self.is_first_person_mode():
            self._first_person_secondary_action()
            e.accept()
        elif key == Qt.Key_M and self.is_first_person_mode():
            self.worldMapRequested.emit()
            e.accept()
        elif key == Qt.Key_F11:
            self.gameFullscreenRequested.emit()
            e.accept()
        elif key == Qt.Key_R:
            self.reset_view()
            e.accept()
        elif key == Qt.Key_F:
            self.focus_selected_target()
            e.accept()
        elif key == Qt.Key_Tab:
            self.cycle_next_target()
            e.accept()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.isAutoRepeat():
            e.accept()
            return
        self._pressed_keys.discard(_normalize_control_key_event(e))
        super().keyReleaseEvent(e)

    def focusOutEvent(self, e: QtGui.QFocusEvent):
        self._pressed_keys.clear()
        if self.is_first_person_mode():
            self._release_mouse_capture()
            self._release_keyboard_capture()
        super().focusOutEvent(e)


# ====================================================
# 4) Мост: trainer.world -> engine (для 3D синхронизации)
# ====================================================
class TrainerToEngineBridge(QtCore.QObject):
    def __init__(self, trainer: MindTrainerInteractive, shared: SharedState, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.shared = shared
        self._snapshot_push_pending = False
        self._snapshot_timer = QtCore.QTimer(self)
        self._snapshot_timer.setSingleShot(True)
        self._snapshot_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._snapshot_timer.setInterval(SNAPSHOT_PUSH_INTERVAL_MS)
        self._snapshot_timer.timeout.connect(self._flush_snapshot_push)
        self.trainer.world_changed.connect(self._schedule_snapshot_push)

    @Slot()
    def _schedule_snapshot_push(self):
        self._snapshot_push_pending = True
        if not self._snapshot_timer.isActive():
            self._snapshot_timer.start()

    @Slot()
    def _flush_snapshot_push(self):
        if not self._snapshot_push_pending:
            return
        self._snapshot_push_pending = False
        self._push_snapshot()
        if self._snapshot_push_pending:
            self._snapshot_timer.start()

    @Slot()
    def _push_snapshot(self):
        world = self.trainer.world
        if world is None:
            return
        snap = _build_engine_snapshot(
            world,
            tick=self.trainer.monitor.tick,
            selected_agent_id=self.shared.get_selected_agent_id(),
        )
        self.shared.update_from_snapshot(snap)


# ====================================================
# 5) HUD поверх 3D (имя агента и бары)
# ====================================================
class AgentHud(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget, shared: SharedState):
        super().__init__(parent)
        self.shared = shared
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("""
        QFrame { background: rgba(18,22,30,0.72); border:1px solid #2a2f3a; border-radius:12px; }
        QLabel { color: #e8eaf6; }
        """)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self.title = QtWidgets.QLabel("—")
        f = self.title.font(); f.setBold(True); self.title.setFont(f)
        self.drive = QtWidgets.QLabel("")
        self.pb_health = QtWidgets.QProgressBar(); self.pb_energy = QtWidgets.QProgressBar(); self.pb_fear = QtWidgets.QProgressBar()
        for pb in (self.pb_health, self.pb_energy, self.pb_fear):
            pb.setTextVisible(False); pb.setMinimum(0); pb.setMaximum(100)

        lay.addWidget(self.title)
        lay.addWidget(self.drive)
        lay.addWidget(QtWidgets.QLabel("Health"))
        lay.addWidget(self.pb_health)
        lay.addWidget(QtWidgets.QLabel("Energy"))
        lay.addWidget(self.pb_energy)
        lay.addWidget(QtWidgets.QLabel("Fear"))
        lay.addWidget(self.pb_fear)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(24); self.shadow.setOffset(0,6)
        self.shadow.setColor(QtGui.QColor(0,0,0,140))
        self.setGraphicsEffect(self.shadow)

        self.shared.updated.connect(self.refresh)
        self.refresh()

    @Slot()
    def refresh(self):
        info = self.shared.get_selected_agent_debug()
        name = info.get("name") or info.get("id") or "—"
        tags = set(str(t) for t in list(info.get("tags", []) or []))
        suffix = " [LAB]" if LAB_AGENT_TAG in tags else ""
        self.title.setText(f"{name}{suffix}")
        drive = info.get("mind_drive") or "—"
        score = info.get("mind_survival_score")
        room_state = " | Room: confined" if TRAINING_ROOM_TAG in tags else ""
        self.drive.setText(f"Drive: {drive}   |   Survival: {score if score is not None else '—'}{room_state}")
        self.pb_health.setValue(int(info.get("health") or 0))
        self.pb_energy.setValue(int(info.get("energy") or 0))
        fear = info.get("fear") or 0.0
        self.pb_fear.setValue(max(0, min(100, int(fear*100))) if fear <= 1.0 else int(fear))

    def place(self, parent_rect: QtCore.QRect):
        # левый верхний угол с отступом
        self.setGeometry(QtCore.QRect(parent_rect.x()+14, parent_rect.y()+14, 260, 168))


class LiveTickPill(QtWidgets.QLabel):
    def __init__(self, parent: QtWidgets.QWidget, shared: SharedState):
        super().__init__(parent)
        self.shared = shared
        self.setStyleSheet("""
        QLabel { color:#dfe6ff; background: rgba(28,32,48,0.75);
                 border:1px solid #374058; border-radius: 10px; padding:6px 10px; }
        """)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._dot = True
        self._blink = QTimer(self); self._blink.setInterval(700); self._blink.timeout.connect(self._toggle)
        self._blink.start()
        self.shared.updated.connect(self.refresh)
        self.refresh()

    def _toggle(self):
        self._dot = not self._dot
        self.refresh()

    @Slot()
    def refresh(self):
        dot = "●" if self._dot else "○"
        self.setText(f"{dot} LIVE  t={self.shared.get_tick()}")

    def place(self, parent_rect: QtCore.QRect):
        self.adjustSize()
        w = self.sizeHint().width()
        self.setGeometry(parent_rect.right()-w-14, parent_rect.y()+14, w, 28)


class FirstPersonOverlay(QtWidgets.QWidget):
    """Прозрачный HUD для режима прогулки от первого лица."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._state: Dict[str, Any] = {"active": False}

    def set_state(self, state: Dict[str, Any]) -> None:
        self._state = dict(state or {})
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent):
        state = self._state
        if not state.get("active"):
            return

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        r = self.rect()
        cx = r.center().x()
        cy = r.center().y()

        # Crosshair
        glow_pen = QtGui.QPen(QtGui.QColor(110, 220, 255, 75), 4)
        core_pen = QtGui.QPen(QtGui.QColor(236, 246, 255, 220), 1.4)
        for pen in (glow_pen, core_pen):
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(cx - 12, cy), QtCore.QPointF(cx - 4, cy))
            p.drawLine(QtCore.QPointF(cx + 4, cy), QtCore.QPointF(cx + 12, cy))
            p.drawLine(QtCore.QPointF(cx, cy - 12), QtCore.QPointF(cx, cy - 4))
            p.drawLine(QtCore.QPointF(cx, cy + 4), QtCore.QPointF(cx, cy + 12))

        # Mode pill
        pill = QtCore.QRectF(r.center().x() - 108, r.top() + 14, 216, 30)
        grad = QtGui.QLinearGradient(pill.topLeft(), pill.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(18, 30, 48, 214))
        grad.setColorAt(1.0, QtGui.QColor(12, 18, 30, 214))
        p.setPen(QtGui.QPen(QtGui.QColor(84, 150, 206, 180), 1.0))
        p.setBrush(grad)
        p.drawRoundedRect(pill, 14.0, 14.0)
        pill_font = p.font()
        pill_font.setPointSize(10)
        pill_font.setWeight(QtGui.QFont.DemiBold)
        p.setFont(pill_font)
        p.setPen(QtGui.QColor(222, 240, 255))
        pill_text = "FIRST PERSON / CHARACTER" if state.get("driving") else "FIRST PERSON / FREE WALK"
        p.drawText(pill, Qt.AlignCenter, pill_text)

        # Stats card
        panel = QtCore.QRectF(r.left() + 16, r.bottom() - 126, 388, 110)
        p.setPen(QtGui.QPen(QtGui.QColor(64, 92, 120, 180), 1.0))
        p.setBrush(QtGui.QColor(10, 14, 22, 184))
        p.drawRoundedRect(panel, 14.0, 14.0)

        font = p.font()
        font.setPointSize(9)
        font.setWeight(QtGui.QFont.Medium)
        p.setFont(font)
        p.setPen(QtGui.QColor(216, 229, 246))
        capture = "мышь захвачена" if state.get("captured") else "кликни в окно для захвата мыши"
        speed = state.get("speed_mode", "walk")
        if speed == "sprint":
            speed_label = "SPRINT"
        elif speed == "precision":
            speed_label = "PRECISION"
        else:
            speed_label = "WALK"
        focus_name = state.get("focus_name")
        focus_kind = state.get("focus_kind")
        focus_distance = state.get("focus_distance")
        if focus_name:
            dist_txt = f"{focus_distance:.1f}m" if isinstance(focus_distance, (int, float)) else "?"
            focus_line = f"Прицел: {focus_name} [{focus_kind}] {dist_txt}  |  E выбрать"
        elif state.get("has_ground_target"):
            focus_line = "Прицел: земля  |  RMB/G задать goal выбранному агенту"
        else:
            focus_line = "Прицел: нет цели  |  Q освободить/вернуть курсор"
        lines = [
            f"M/Ь карта мира  |  WASD/ЦФЫВ вести агента  |  Shift {speed_label}  |  Ctrl точный шаг  |  V/М или Esc выход",
            f"Позиция: x={state.get('x', 0.0):.1f}  z={state.get('z', 0.0):.1f}  yaw={state.get('yaw_deg', 0.0):.0f}°  pitch={state.get('pitch_deg', 0.0):.0f}°",
            f"Управляемый агент: {state.get('selected_name') or '—'}" if state.get("driving") else f"Выбранный агент: {state.get('selected_name') or '—'}",
            focus_line,
            f"Состояние: {capture}",
        ]
        y = panel.top() + 14
        for line in lines:
            p.drawText(QtCore.QRectF(panel.left() + 12, y, panel.width() - 24, 20), Qt.AlignLeft | Qt.AlignVCenter, line)
            y += 18


# ====================================================
# 6) Главное окно: 3 колонки (Stats | 3D | Brain) + Toolbar
# ====================================================
class CombinedMainWindow(QtWidgets.QMainWindow):
    def _initial_agent_lineup(self) -> list[dict[str, str]]:
        return [
            {"id": "a1", "name": "Echo", "persona":
             "Ты Echo. Осторожный выживальщик. Бережёшь Nova, шутишь, но всегда смотришь по сторонам."},
            {"id": "a2", "name": "Nova", "persona":
             "Ты Nova. Смелая исследовательница. Действуешь решительно, но не рискуешь зря."},
            {"id": "agent_0", "name": "A0", "persona": "scout/explorer"},
        ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-Matrix Lab — 3D + Mind Trainer")
        self.resize(1600, 900)
        self.settings = QtCore.QSettings("MiniMatrixLab", "CombinedApp")
        self._game_fullscreen = False
        self._game_fullscreen_saved_ui: Dict[str, Any] = {}
        self._pending_restore_game_fullscreen = bool(self.settings.value("game_fullscreen", False, type=bool))
        self._main_toolbar: Optional[QtWidgets.QToolBar] = None
        self._outer_layout: Optional[QtWidgets.QVBoxLayout] = None
        self._frame3d: Optional[QtWidgets.QFrame] = None
        self._frame3d_layout: Optional[QtWidgets.QVBoxLayout] = None
        self._overlay_relayout_pending = False
        self._overlay_relayout_timer: Optional[QtCore.QTimer] = None
        self._trainer_tick_error: Optional[str] = None

        # Глобальный шрифт
        font = QtGui.QFont(APP_FONT, 10)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.setFont(font)

        # Showcase-конфигурация мира (крупнее дефолтного).
        self._showcase_world_w = float(SHOWCASE_WORLD_WIDTH)
        self._showcase_world_h = float(SHOWCASE_WORLD_HEIGHT)
        self._showcase_safe_havens = int(SHOWCASE_SAFE_HAVENS)
        self._showcase_env_seed = int(SHOWCASE_ENV_SEED)
        config.WORLD_WIDTH = self._showcase_world_w
        config.WORLD_HEIGHT = self._showcase_world_h

        # 1) 3D движок
        self.engine = MiniMatrixEngine()
        self.engine.world.width = self._showcase_world_w
        self.engine.world.height = self._showcase_world_h

        # 2) тренер + бой
        agent_lineup = list(self._initial_agent_lineup() or [])
        self.trainer = MindTrainerInteractive(
            num_agents=max(1, len(agent_lineup) or 3), max_ticks_per_epoch=2000, seed=1234,
            disaster_interval_ticks=400, relief_after_disaster=80,
            fresh_start=False,
            agent_lineup=agent_lineup,
        )
        # При смене эпохи у трейнера пересобираем карту и перепривязываем боёвку.
        self.trainer.epoch_changed.connect(self._on_trainer_epoch_changed)
        self._prepare_showcase_world(announce=False, rebuild_environment=True)

        self.combat = CombatSystem(self.trainer.world)
        self._combat_timer = QtCore.QTimer(self)
        self._combat_timer.setInterval(50)  # ~20 Гц
        self._combat_timer.timeout.connect(self._on_combat_tick)
        self._combat_timer.start()
        self._combat_paused = False

        # 3) Shared + 3D виджет + оверлеи
        self.shared = SharedState(self.engine)
        self.view3d = World3DView(self.shared, player_drive_callback=self._drive_player_agent)
        self.view3d.center_x = self._showcase_world_w * 0.5
        self.view3d.center_z = self._showcase_world_h * 0.5
        self.view3d.distance = max(160.0, max(self._showcase_world_w, self._showcase_world_h) * 1.25)
        apply_expand_policy(self.view3d, w_stretch=True)
        self.view3d.setMinimumSize(980, 720)
        self.view3d.requestSetGoal.connect(self._on_set_goal_from_3d)
        self.view3d.fpsUpdated.connect(self._update_fps)
        self.view3d.firstPersonChanged.connect(self._on_first_person_changed)
        self.view3d.firstPersonMessage.connect(lambda text: self.statusBar().showMessage(text, 1800))
        self.view3d.worldMapRequested.connect(self._toggle_world_map_overlay)
        self.view3d.gameFullscreenRequested.connect(self._toggle_game_fullscreen_shortcut)

        frame3d = QtWidgets.QFrame()
        frame3d.setStyleSheet(FRAME3D_STYLE)
        frame3d.setProperty("card", True)
        lay3d = QtWidgets.QVBoxLayout(frame3d)
        lay3d.setContentsMargins(10, 10, 10, 10)
        lay3d.setSpacing(8)
        lay3d.addWidget(self.view3d, 1)
        self._frame3d = frame3d
        self._frame3d_layout = lay3d

        # Оверлеи
        self.help_overlay = OverlayLabel(frame3d,
            self._help_text_for_current_mode())
        self.hud_overlay = AgentHud(frame3d, self.shared)
        self.live_pill = LiveTickPill(frame3d, self.shared)
        self.minimap = MiniMapWidget(self.shared, frame3d)
        self.minimap.set_player_provider(self.view3d.get_minimap_player_marker)
        self.minimap.clickedWorld.connect(self._center_to)
        self.fp_overlay = FirstPersonOverlay(frame3d)
        self.fp_overlay.set_state(self.view3d.get_first_person_hud_state())

        def place_overlays():
            r = frame3d.rect()
            if self.hud_overlay.isVisible():
                self.hud_overlay.place(r)
            self.live_pill.place(r)
            self.fp_overlay.setGeometry(self.view3d.geometry())
            if hasattr(self, "world_map_overlay"):
                self.world_map_overlay.setGeometry(central.rect())
            # help — под HUD или в верхнем левом углу
            help_size = self.help_overlay.sizeHint()
            self.help_overlay.resize(help_size)
            help_y = self.hud_overlay.geometry().bottom() + 8 if self.hud_overlay.isVisible() else r.y() + 14
            self.help_overlay.move(r.x()+14, help_y)
            # minimap — правый нижний
            self.minimap.move(r.right()-self.minimap.width()-14, r.bottom()-self.minimap.height()-14)
            self.fp_overlay.raise_()
            self.live_pill.raise_()
            self.help_overlay.raise_()
            self.minimap.raise_()
            if self.hud_overlay.isVisible():
                self.hud_overlay.raise_()
            if hasattr(self, "world_map_overlay") and self.world_map_overlay.isVisible():
                self.world_map_overlay.raise_()

        frame3d.installEventFilter(self)
        self.view3d.installEventFilter(self)
        self._place_overlays = place_overlays

        # 4) панели тренера
        self.statsWidget = TrainerStatsWidget(self.trainer)
        self.statsWidget.setMinimumWidth(360)
        self.statsWidget.setMaximumWidth(480)
        self.stats_card = make_card("Панель эволюции / метрик", self.statsWidget)

        self.brainWidget = AgentBrainWidget(self.trainer)
        self.brainWidget.setMinimumWidth(360)
        self.brainWidget.setMaximumWidth(520)
        self.brain_card = make_card("Инспектор мозга агента", self.brainWidget)

        # синхронизация выбора
        self.shared.updated.connect(self._sync_selection_into_brain_panel)
        self.brainWidget.comboAgent.currentIndexChanged.connect(self._sync_selection_from_brain_panel)

        # 5) сплиттер: Stats | 3D | Brain
        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.setOpaqueResize(False)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.stats_card)
        self.splitter.addWidget(frame3d)
        self.splitter.addWidget(self.brain_card)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([420, 900, 420])

        # 6) центральный виджет
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)
        outer.addWidget(self.splitter, 1)
        self.setCentralWidget(central)
        self._outer_layout = outer
        self.world_map_overlay = WorldMapOverlay(self.shared, central)
        self.world_map_overlay.set_player_provider(self.view3d.get_minimap_player_marker)
        self.world_map_overlay.clickedWorld.connect(self._center_to)
        self.world_map_overlay.closed.connect(self._on_world_map_closed)
        self._overlay_relayout_timer = QtCore.QTimer(self)
        self._overlay_relayout_timer.setSingleShot(True)
        self._overlay_relayout_timer.setTimerType(QtCore.Qt.CoarseTimer)
        self._overlay_relayout_timer.setInterval(OVERLAY_RELAYOUT_INTERVAL_MS)
        self._overlay_relayout_timer.timeout.connect(self._flush_overlay_relayout)
        central.installEventFilter(self)
        QtWidgets.QApplication.instance().installEventFilter(self)

        # 7) статусбар
        self.statusBar().setStyleSheet(STATUSBAR_STYLE)
        self._lbl_fps = QtWidgets.QLabel("FPS: —")
        self._lbl_cam = QtWidgets.QLabel("Cam: —")
        self._lbl_fps.setStyleSheet("QLabel{color:#a9c1ff}")
        self._lbl_cam.setStyleSheet("QLabel{color:#a9c1ff}")
        self.statusBar().addPermanentWidget(self._lbl_fps)
        self.statusBar().addPermanentWidget(self._lbl_cam)

        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(250)
        self._ui_timer.timeout.connect(self._tick_ui)
        self._ui_timer.start()

        # 8) мост мира в 3D
        self.bridge = TrainerToEngineBridge(self.trainer, self.shared, self)

        # 8.1) Основной тик симуляции мира и мозга агента.
        self._trainer_timer = QtCore.QTimer(self)
        self._trainer_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._trainer_timer.setInterval(TRAINER_TICK_INTERVAL_MS)
        self._trainer_timer.timeout.connect(self._on_trainer_tick)
        self._trainer_timer.start()

        # 9) Toolbar & actions
        self._make_toolbar()

        # 10) тема/палитра и состояние
        self._apply_palette()
        self._apply_theme(self.settings.value("theme", "blue"))
        self._restore_geometry()

        # 11) начальный пуш снапшота
        self.bridge._push_snapshot()
        self._ensure_training_room_selection()
        self._schedule_overlay_relayout()
        self._on_first_person_changed(self.view3d.is_first_person_mode())

    # --- eventFilter для перекладки оверлеев
    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.KeyRelease):
            if self._forward_key_event_to_view3d(ev, source=obj):
                return True
        if ev.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            watched = {
                self,
                self.centralWidget(),
                getattr(self, "_frame3d", None),
                getattr(self, "view3d", None),
                getattr(self, "splitter", None),
            }
            if obj in watched:
                self._schedule_overlay_relayout()
        return super().eventFilter(obj, ev)

    def _schedule_overlay_relayout(self):
        if not hasattr(self, "_place_overlays"):
            return
        self._overlay_relayout_pending = True
        timer = getattr(self, "_overlay_relayout_timer", None)
        if timer is None:
            QtCore.QTimer.singleShot(OVERLAY_RELAYOUT_INTERVAL_MS, self._flush_overlay_relayout)
            return
        timer.start()

    @Slot()
    def _flush_overlay_relayout(self):
        self._overlay_relayout_pending = False
        if hasattr(self, "_place_overlays"):
            self._place_overlays()

    def _resize_world_if_needed(self, world, target_w: float, target_h: float) -> bool:
        """
        Масштабирует координаты мира к новому размеру.
        Нужен как safety-net, если мир создан не с теми размерами.
        """
        old_w = float(getattr(world, "width", target_w))
        old_h = float(getattr(world, "height", target_h))
        if old_w <= 1e-6 or old_h <= 1e-6:
            return False
        if abs(old_w - target_w) < 1e-6 and abs(old_h - target_h) < 1e-6:
            return False

        sx = target_w / old_w
        sy = target_h / old_h
        sr = math.sqrt(max(0.01, sx * sy))

        for ag in _iter_vals(getattr(world, "agents", {})):
            for attr, mul in (("x", sx), ("goal_x", sx), ("y", sy), ("goal_y", sy)):
                if hasattr(ag, attr):
                    try:
                        setattr(ag, attr, float(getattr(ag, attr, 0.0)) * mul)
                    except Exception:
                        pass

        for ani in _iter_vals(getattr(world, "animals", {})):
            for attr, mul in (("x", sx), ("y", sy)):
                if hasattr(ani, attr):
                    try:
                        setattr(ani, attr, float(getattr(ani, attr, 0.0)) * mul)
                    except Exception:
                        pass

        for obj in _iter_vals(getattr(world, "objects", [])):
            for attr, mul in (("x", sx), ("y", sy), ("radius", sr)):
                if hasattr(obj, attr):
                    try:
                        setattr(obj, attr, float(getattr(obj, attr, 0.0)) * mul)
                    except Exception:
                        pass

        acts = dict(getattr(world, "activities", {}) or {})
        for rec in acts.values():
            if not isinstance(rec, dict):
                continue
            area = rec.get("area")
            if not isinstance(area, dict):
                continue
            try:
                area["x"] = float(area.get("x", 0.0)) * sx
                area["y"] = float(area.get("y", 0.0)) * sy
                area["radius"] = float(area.get("radius", 0.0)) * sr
            except Exception:
                pass
        if hasattr(world, "set_activity_registry"):
            try:
                world.set_activity_registry(acts)
            except Exception:
                pass

        world.width = float(target_w)
        world.height = float(target_h)

        try:
            spx, spy = getattr(config, "SAFE_POINT", (old_w * 0.5, old_h * 0.5))
            config.SAFE_POINT = (float(spx) * sx, float(spy) * sy)
        except Exception:
            config.SAFE_POINT = (target_w * 0.5, target_h * 0.5)
        return True

    def _add_showcase_safe_havens(self, world) -> int:
        """
        Добавляет дополнительные safe-зоны для отхила и восстановления.
        """
        if world is None:
            return 0

        existing_ids = {
            str(getattr(obj, "obj_id", ""))
            for obj in _iter_vals(getattr(world, "objects", []))
        }
        registry = dict(getattr(world, "activities", {}) or {})
        anchors = [
            (0.12, 0.14), (0.30, 0.11), (0.50, 0.14), (0.70, 0.12),
            (0.86, 0.28), (0.88, 0.52), (0.80, 0.76), (0.56, 0.86),
            (0.31, 0.82), (0.14, 0.66), (0.13, 0.42), (0.24, 0.25),
        ]
        rnd = random.Random(1000 + int(getattr(self.trainer, "current_epoch", 0)))
        target_count = max(1, min(self._showcase_safe_havens, len(anchors)))

        hazards = [obj for obj in _iter_vals(getattr(world, "objects", [])) if str(getattr(obj, "kind", "")) == "hazard"]

        def _far_from_hazard(x: float, y: float, r: float) -> bool:
            for hz in hazards:
                hx = float(getattr(hz, "x", 0.0))
                hy = float(getattr(hz, "y", 0.0))
                hr = float(getattr(hz, "radius", 0.0))
                dx = x - hx
                dy = y - hy
                if dx * dx + dy * dy < (r + hr + 5.0) * (r + hr + 5.0):
                    return False
            return True

        added = 0
        for i in range(target_count):
            obj_id = f"showcase_safe_{i}"
            if obj_id in existing_ids:
                continue

            fx, fy = anchors[i]
            radius = rnd.uniform(7.0, 10.5)
            px = float(world.width) * fx + rnd.uniform(-2.6, 2.6)
            py = float(world.height) * fy + rnd.uniform(-2.6, 2.6)
            px = max(4.0, min(float(world.width) - 4.0, px))
            py = max(4.0, min(float(world.height) - 4.0, py))

            # Сдвигаем позицию, если опасность слишком близко.
            tries = 0
            while tries < 8 and not _far_from_hazard(px, py, radius):
                px = max(4.0, min(float(world.width) - 4.0, px + rnd.uniform(-8.0, 8.0)))
                py = max(4.0, min(float(world.height) - 4.0, py + rnd.uniform(-8.0, 8.0)))
                tries += 1

            haven = WorldObject(
                obj_id=obj_id,
                name=f"Оазис_{i + 1}",
                kind="safe",
                x=px,
                y=py,
                radius=radius,
                danger_level=0.0,
                comfort_level=1.0,
            )
            world.add_object(haven)
            existing_ids.add(obj_id)
            added += 1

            registry[obj_id] = {
                "name": haven.name,
                "activity_tags": ["heal", "rest", "eat", "calm", "sleep", "repair_self", "restock_food"],
                "comfort_level": 1.0,
                "danger_level": 0.0,
                "area": {"x": px, "y": py, "radius": radius},
            }

        if hasattr(world, "set_activity_registry"):
            try:
                world.set_activity_registry(registry)
            except Exception:
                pass

        # Основную точку убежища переносим ближе к центру карты.
        if target_count > 0:
            aid = "showcase_safe_0"
            rec = registry.get(aid, {})
            area = rec.get("area", {})
            if isinstance(area, dict):
                try:
                    config.SAFE_POINT = (float(area.get("x", world.width * 0.5)), float(area.get("y", world.height * 0.5)))
                except Exception:
                    config.SAFE_POINT = (world.width * 0.5, world.height * 0.5)
        return added

    def _rebuild_environment_for_world(self, world):
        if world is None:
            return
        self.engine.world.width = float(getattr(world, "width", self._showcase_world_w))
        self.engine.world.height = float(getattr(world, "height", self._showcase_world_h))
        try:
            meshes = build_cinematic_environment(
                world_w=self.engine.world.width,
                world_h=self.engine.world.height,
                seed=self._showcase_env_seed + int(getattr(self.trainer, "current_epoch", 0)),
            )
        except TypeError:
            meshes = build_cinematic_environment(self.engine.world.width, self.engine.world.height)

        if hasattr(self.engine, "load_static_environment"):
            try:
                self.engine.load_static_environment(meshes)
            except Exception as e:
                print("[env] load_static_environment failed:", e)

    def _prepare_showcase_world(self, *, announce: bool, rebuild_environment: bool):
        world = getattr(self.trainer, "world", None)
        if world is None:
            return
        self._resize_world_if_needed(world, self._showcase_world_w, self._showcase_world_h)
        added = self._add_showcase_safe_havens(world)
        training_room = getattr(self.trainer, "training_room", None)
        if training_room is not None:
            preferred_id = getattr(training_room, "agent_id", None)
            training_room.attach_world(world, preferred_agent_id=preferred_id, announce=announce)
        if rebuild_environment:
            self._rebuild_environment_for_world(world)
        if announce and hasattr(world, "add_chat_line"):
            try:
                world.add_chat_line(
                    f"[world] showcase-map ready: {world.width:.0f}x{world.height:.0f}, +{added} safe havens"
                )
            except Exception:
                pass

    def _training_room(self):
        return getattr(self.trainer, "training_room", None)

    def _training_room_agent_id(self) -> Optional[str]:
        room = self._training_room()
        aid = getattr(room, "agent_id", None) if room is not None else None
        return str(aid) if aid else None

    def _ensure_training_room_selection(self) -> None:
        aid = self._training_room_agent_id()
        if not aid:
            return
        current = self.shared.get_selected_agent_id()
        if current and current in getattr(self.engine, "agents", {}):
            return
        self.shared.set_selected_agent(aid)

    @Slot()
    def _send_selected_agent_to_training_room(self):
        room = self._training_room()
        world = getattr(self.trainer, "world", None)
        if room is None or world is None:
            return
        agent_id = self.shared.get_selected_agent_id() or self._training_room_agent_id()
        agent_id = self.trainer.assign_agent_to_training_room(agent_id, announce=True)
        if not agent_id:
            self._toast("No agent available for training room")
            return
        self.shared.set_selected_agent(agent_id)
        self.bridge._push_snapshot()
        self._toast(f"{agent_id} moved to training room")

    @Slot()
    def _release_training_room_agent(self):
        room = self._training_room()
        world = getattr(self.trainer, "world", None)
        if room is None or world is None:
            return
        agent_id = self.trainer.release_training_room_agent(announce=True)
        if not agent_id:
            self._toast("Training room is empty")
            return
        self.shared.set_selected_agent(agent_id)
        self.bridge._push_snapshot()
        self._toast(f"{agent_id} released into world")

    @Slot()
    def _on_trainer_epoch_changed(self):
        self._prepare_showcase_world(announce=True, rebuild_environment=True)
        if hasattr(self, "combat") and self.combat is not None:
            # У trainer новый world на эпоху — боевую систему тоже перепривязываем.
            self.combat.world = self.trainer.world
        if hasattr(self, "bridge"):
            self.bridge._push_snapshot()
        self._ensure_training_room_selection()

    @Slot()
    def _on_trainer_tick(self):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        try:
            trainer.step_tick()
            self._trainer_tick_error = None
        except Exception as exc:
            self._trainer_tick_error = str(exc)
            if hasattr(self, "_trainer_timer") and self._trainer_timer.isActive():
                self._trainer_timer.stop()
            try:
                self.statusBar().showMessage(f"Trainer tick error: {exc}", 5000)
            except Exception:
                pass
            print(f"[combined_app] trainer tick failed: {exc}")

    # --- toolbar
    def _make_toolbar(self):
        tb = QtWidgets.QToolBar("Controls", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self._main_toolbar = tb

        # Play/Pause боя
        self.act_play = QtGui.QAction(self.style().standardIcon(QStyle.SP_MediaPause), "Play/Pause бой (Space)", self)
        self.act_play.setShortcut("Space")
        self.act_play.triggered.connect(self._toggle_combat)
        tb.addAction(self.act_play)

        # Сброс камеры
        act_reset_cam = QtGui.QAction(self.style().standardIcon(QStyle.SP_BrowserReload), "Сброс камеры (R)", self)
        act_reset_cam.setShortcut("R")
        act_reset_cam.triggered.connect(lambda: self.view3d.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_R, Qt.NoModifier)))
        tb.addAction(act_reset_cam)

        # Фокус на агенте
        act_focus = QtGui.QAction(self.style().standardIcon(QStyle.SP_ArrowRight), "Фокус на агенте (F)", self)
        act_focus.setShortcut("F")
        act_focus.triggered.connect(lambda: self.view3d.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_F, Qt.NoModifier)))
        tb.addAction(act_focus)

        # Следующий агент
        act_next = QtGui.QAction("Next (Tab)", self)
        act_next.setShortcut("Tab")
        act_next.triggered.connect(lambda: self.view3d.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)))
        tb.addAction(act_next)

        self.act_first_person = QtGui.QAction("1st Person (V)", self, checkable=True)
        self.act_first_person.setShortcut("V")
        self.act_first_person.toggled.connect(self._toggle_first_person)
        tb.addAction(self.act_first_person)

        self.act_game_fullscreen = QtGui.QAction("Game Fullscreen (F11)", self, checkable=True)
        self.act_game_fullscreen.setShortcut("F11")
        self.act_game_fullscreen.toggled.connect(self._toggle_game_fullscreen)
        self.addAction(self.act_game_fullscreen)
        tb.addAction(self.act_game_fullscreen)

        tb.addSeparator()

        # Волки
        act_spawn_wolves = QtGui.QAction("Spawn wolves (Ctrl+W)", self)
        act_spawn_wolves.setShortcut("Ctrl+W")
        act_spawn_wolves.triggered.connect(self._spawn_wolves)
        self.addAction(act_spawn_wolves)
        tb.addAction(act_spawn_wolves)

        act_room_assign = QtGui.QAction("To Room (Ctrl+Shift+T)", self)
        act_room_assign.setShortcut("Ctrl+Shift+T")
        act_room_assign.triggered.connect(self._send_selected_agent_to_training_room)
        self.addAction(act_room_assign)
        tb.addAction(act_room_assign)

        act_room_release = QtGui.QAction("Release Room Agent (Ctrl+Shift+G)", self)
        act_room_release.setShortcut("Ctrl+Shift+G")
        act_room_release.triggered.connect(self._release_training_room_agent)
        self.addAction(act_room_release)
        tb.addAction(act_room_release)

        # Скриншот
        act_shot = QtGui.QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Screenshot 3D (Ctrl+Shift+S)", self)
        act_shot.setShortcut("Ctrl+Shift+S")
        act_shot.triggered.connect(self._screenshot_3d)
        tb.addAction(act_shot)

        # Zen-режим
        self.act_zen = QtGui.QAction("Zen (Ctrl+J)", self, checkable=True)
        self.act_zen.setShortcut("Ctrl+J")
        self.act_zen.triggered.connect(self._toggle_zen)
        tb.addAction(self.act_zen)

        # Тема
        self.act_theme = QtGui.QAction("Theme: Emerald", self)
        self.act_theme.setShortcut("Ctrl+E")
        self.act_theme.triggered.connect(self._toggle_theme)
        tb.addAction(self.act_theme)

        # Индикатор соединения
        conn = QtWidgets.QLabel("Link: Local")
        conn.setStyleSheet("QLabel { color:#a9c1ff; }")
        tb.addSeparator()
        tb.addWidget(conn)

        # Help overlay toggle
        self.act_help = QtGui.QAction("Help (H)", self, checkable=True)
        self.act_help.setShortcut("H")
        self.act_help.setChecked(True)
        self.act_help.toggled.connect(lambda on: self.help_overlay.fade(on))
        tb.addAction(self.act_help)

    # --- play/pause боя
    def _toggle_combat(self):
        self._combat_paused = not self._combat_paused
        if self._combat_paused:
            self._combat_timer.stop()
            self.statusBar().showMessage("Combat: paused", 2000)
            self.act_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self._combat_timer.start()
            self.statusBar().showMessage("Combat: running", 2000)
            self.act_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    # --- боевой тик
    def _on_combat_tick(self):
        if getattr(self, 'combat', None) and getattr(self.trainer, 'world', None):
            room = self._training_room()
            if room is not None:
                room.maintain_world(self.trainer.world)
            try:
                self.combat.step(0.05)
            finally:
                if room is not None:
                    room.maintain_world(self.trainer.world)
                self.bridge._push_snapshot()

    # --- спавн волков возле выбранного агента (или центра)
    def _spawn_wolves(self):
        w = self.trainer.world
        if not w or not getattr(self, 'combat', None):
            return
        sel = self.shared.get_selected_agent_id()
        room = self._training_room()
        if sel and hasattr(w, "get_agent_by_id"):
            ag = w.get_agent_by_id(sel)
            cx = getattr(ag, "x", w.width * 0.5)
            cy = getattr(ag, "y", w.height * 0.5)
        else:
            cx, cy = w.width * 0.5, w.height * 0.5
        if room is not None and room.is_confined(sel):
            cx, cy = room.release_point(w)
            self._toast("Training room is protected: wolves spawned outside the room")
        try:
            self.combat.spawn_wave("wolf", n=3, around=(cx, cy))
            if hasattr(w, "add_chat_line"):
                w.add_chat_line(f"[system] Spawned 3 wolves near ({cx:.1f},{cy:.1f})")
            self._toast(f"Spawned 3 wolves near ({cx:.1f},{cy:.1f})")
        finally:
            self.bridge._push_snapshot()

    # --- goal из 3D в локальный мир тренера
    @Slot(str, float, float)
    def _on_set_goal_from_3d(self, agent_id: str, x: float, z: float):
        w = self.trainer.world
        if not w:
            return
        room = self._training_room()
        if room is not None:
            clamped_x, clamped_z = room.clamp_point_for_agent(w, agent_id, x, z)
            if abs(clamped_x - x) > 1e-6 or abs(clamped_z - z) > 1e-6:
                self._toast("Goal clamped to training room")
            x, z = clamped_x, clamped_z
        x = max(0.0, min(w.width, x))
        z = max(0.0, min(w.height, z))
        if hasattr(w, "set_agent_goal"):
            try:
                ok = bool(w.set_agent_goal(agent_id, x, z))
            except Exception:
                ok = False
            if not ok:
                return
        else:
            ag = None
            if hasattr(w, "get_agent_by_id"):
                ag = w.get_agent_by_id(agent_id)
            else:
                for a in _iter_vals(getattr(w, "agents", [])):
                    if getattr(a, "id", None) == agent_id:
                        ag = a
                        break
            if ag is None:
                return
            if hasattr(ag, "set_goal"):
                try:
                    ag.set_goal(x, z, reason="external_command", tick=getattr(w, "tick_count", 0))
                except Exception:
                    setattr(ag, "goal_x", x)
                    setattr(ag, "goal_y", z)
            else:
                setattr(ag, "goal_x", x)
                setattr(ag, "goal_y", z)
            if hasattr(w, "add_chat_line"):
                try:
                    w.add_chat_line(f"[observer] {getattr(ag, 'name', agent_id)} → goal=({x:.1f},{z:.1f})")
                except Exception:
                    pass

        self.bridge._push_snapshot()
        self._toast(f"Goal for {agent_id} → ({x:.1f}, {z:.1f})")

    def _get_world_agent_by_id(self, world, agent_id: Optional[str]):
        if world is None or not agent_id:
            return None
        if hasattr(world, "get_agent_by_id"):
            try:
                ag = world.get_agent_by_id(agent_id)
                if ag is not None:
                    return ag
            except Exception:
                pass
        for ag in _iter_vals(getattr(world, "agents", {})):
            if getattr(ag, "id", None) == agent_id or getattr(ag, "agent_id", None) == agent_id:
                return ag
        return None

    def _sync_engine_agent_from_world(self, ag, *, facing_x: float, facing_z: float) -> None:
        aid = str(getattr(ag, "id", getattr(ag, "agent_id", "")) or "")
        if not aid:
            return
        ent = self.engine.agents.get(aid)
        if ent is None:
            return
        px = float(getattr(ag, "x", ent.transform.pos.x))
        pz = float(getattr(ag, "y", ent.transform.pos.z))
        vx = float(getattr(ag, "vx", 0.0))
        vz = float(getattr(ag, "vy", 0.0))
        ent.name = str(getattr(ag, "name", ent.name))
        ent.transform.pos.x = px
        ent.transform.pos.z = pz
        ent.target_pos.x = px
        ent.target_pos.z = pz
        ent.net.server_pos.x = px
        ent.net.server_pos.z = pz
        ent.net.server_vel.x = vx
        ent.net.server_vel.z = vz
        ent.net.since_snap = 0.0
        ent.body.vel.x = vx
        ent.body.vel.z = vz
        dir_x = float(facing_x)
        dir_z = float(facing_z)
        vel_len = math.hypot(vx, vz)
        if vel_len > 1e-6:
            dir_x = vx / vel_len
            dir_z = vz / vel_len
        else:
            face_len = math.hypot(dir_x, dir_z)
            if face_len > 1e-6:
                dir_x /= face_len
                dir_z /= face_len
            else:
                dir_x = float(getattr(ent.brain.desired_dir, "x", 1.0))
                dir_z = float(getattr(ent.brain.desired_dir, "z", 0.0))
        ent.brain.desired_dir.x = dir_x
        ent.brain.desired_dir.z = dir_z
        try:
            alive = bool(getattr(ag, "is_alive", lambda: True)())
        except Exception:
            alive = bool(getattr(ag, "alive", True))
        ps = dict(getattr(ent, "public_state", {}) or {})
        energy = float(getattr(ag, "energy", ps.get("energy", 100.0)))
        hunger = float(getattr(ag, "hunger", ps.get("hunger", 0.0)))
        if energy <= 1.5:
            energy *= 100.0
        if hunger <= 1.5:
            hunger *= 100.0
        ent.anim.alive = alive
        ps["id"] = aid
        ps["name"] = ent.name
        ps["alive"] = alive
        ps["pos"] = {"x": px, "y": pz}
        ps["vel"] = {"x": vx, "y": vz}
        ps["goal"] = {
            "x": float(getattr(ag, "goal_x", px)),
            "y": float(getattr(ag, "goal_y", pz)),
        }
        ps["facing"] = {"x": dir_x, "y": dir_z}
        ps["health"] = float(getattr(ag, "health", ps.get("health", 100.0)))
        ps["energy"] = energy
        ps["hunger"] = hunger
        ps["fear"] = float(getattr(ag, "fear", ps.get("fear", 0.0)))
        ent.public_state = ps

    def _drive_player_agent(
        self,
        agent_id: str,
        x: float,
        z: float,
        dt: float,
        facing_x: float,
        facing_z: float,
        speed_mode: str,
    ) -> Dict[str, Any]:
        world = self.trainer.world
        if world is None:
            return {"ok": False, "x": float(x), "z": float(z)}
        ag = self._get_world_agent_by_id(world, agent_id)
        if ag is None:
            return {"ok": False, "x": float(x), "z": float(z)}
        try:
            alive = bool(getattr(ag, "is_alive", lambda: True)())
        except Exception:
            alive = bool(getattr(ag, "alive", True))
        if not alive:
            return {
                "ok": False,
                "x": float(getattr(ag, "x", x)),
                "z": float(getattr(ag, "y", z)),
            }

        nx = max(0.0, min(float(getattr(world, "width", x)), float(x)))
        nz = max(0.0, min(float(getattr(world, "height", z)), float(z)))
        room = self._training_room()
        if room is not None:
            nx, nz = room.clamp_point_for_agent(world, agent_id, nx, nz)
        ok = False
        if hasattr(world, "drive_agent"):
            try:
                ok = bool(world.drive_agent(
                    agent_id,
                    nx,
                    nz,
                    max(float(dt), 1e-6),
                    facing=(float(facing_x), float(facing_z)),
                    source=f"player:{speed_mode}",
                    hold_ticks=4,
                ))
            except Exception:
                ok = False
        if not ok:
            old_x = float(getattr(ag, "x", nx))
            old_y = float(getattr(ag, "y", nz))
            setattr(ag, "x", nx)
            setattr(ag, "y", nz)
            setattr(ag, "goal_x", nx)
            setattr(ag, "goal_y", nz)
            denom = max(float(dt), 1e-6)
            try:
                setattr(ag, "vx", (nx - old_x) / denom)
                setattr(ag, "vy", (nz - old_y) / denom)
            except Exception:
                pass
            if hasattr(ag, "mark_manual_control"):
                try:
                    ag.mark_manual_control(
                        tick=int(getattr(world, "tick_count", 0)),
                        hold_ticks=4,
                        source=f"player:{speed_mode}",
                    )
                except Exception:
                    pass
            face_len = math.hypot(float(facing_x), float(facing_z))
            if face_len > 1e-6:
                try:
                    setattr(ag, "_manual_facing_x", float(facing_x) / face_len)
                    setattr(ag, "_manual_facing_y", float(facing_z) / face_len)
                except Exception:
                    pass
            ok = True

        self._sync_engine_agent_from_world(ag, facing_x=float(facing_x), facing_z=float(facing_z))
        return {
            "ok": ok,
            "x": float(getattr(ag, "x", nx)),
            "z": float(getattr(ag, "y", nz)),
            "name": str(getattr(ag, "name", agent_id)),
        }

    # --- синхронизация выделения (3D -> инспектор)
    @Slot()
    def _sync_selection_into_brain_panel(self):
        sel = self.shared.get_selected_agent_id()
        if not sel:
            return
        box = self.brainWidget.comboAgent
        for i in range(box.count()):
            if box.itemData(i) == sel:
                if box.currentIndex() != i:
                    box.blockSignals(True)
                    box.setCurrentIndex(i)
                    box.blockSignals(False)
                break

    # --- синхронизация выделения (инспектор -> 3D)
    @Slot(int)
    def _sync_selection_from_brain_panel(self, _idx: int):
        ag = self.brainWidget.get_selected_agent()
        if not ag:
            return
        self.shared.set_selected_agent(getattr(ag, "id", None))

    # --- центрирование камеры (из мини-карты)
    def _center_to(self, x: float, y: float):
        if self.view3d.is_first_person_mode():
            self.view3d.move_first_person_to(x, y)
            self._toast(f"Player → ({x:.1f}, {y:.1f})")
            return
        self.view3d.center_x, self.view3d.center_z = x, y
        self.view3d._clamp_center()
        self.view3d.update()
        self._toast(f"Camera → ({x:.1f}, {y:.1f})")

    # --- FPS/status
    def _update_fps(self, fps: float):
        self._lbl_fps.setText(f"FPS: {fps:0.1f}")

    def _tick_ui(self):
        self._lbl_cam.setText(self.view3d.camera_status_text())
        self.fp_overlay.set_state(self.view3d.get_first_person_hud_state())
        self.minimap.update()

    # --- тост
    def _toast(self, text: str):
        sb = SnackBar(self.view3d.parentWidget(), text)
        sb.place(self.view3d.parentWidget().rect())

    # --- скриншот
    def _screenshot_3d(self):
        try:
            img = self.view3d.grabFramebuffer()
            if img.isNull():
                self._toast("Cannot grab framebuffer")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", "minimatrix_3d.png", "PNG Images (*.png)")
            if path:
                img.save(path, "PNG")
                self._toast(f"Saved: {path}")
        except Exception as e:
            self._toast(f"Shot error: {e}")

    # --- Zen
    def _toggle_zen(self, on: bool):
        if on:
            self._prev_sizes = self.splitter.sizes()
            self.splitter.setSizes([0, 1, 0])
            self.stats_card.hide()
            self.brain_card.hide()
            self._toast("Zen on")
        else:
            self.stats_card.show()
            self.brain_card.show()
            if hasattr(self, "_prev_sizes") and self._prev_sizes:
                self.splitter.setSizes(self._prev_sizes)
            else:
                self.splitter.setSizes([420, 900, 420])
            self._toast("Zen off")

    def _set_game_fullscreen(self, on: bool):
        on = bool(on)
        if on == self._game_fullscreen:
            if hasattr(self, "act_game_fullscreen"):
                self.act_game_fullscreen.blockSignals(True)
                self.act_game_fullscreen.setChecked(on)
                self.act_game_fullscreen.blockSignals(False)
            return

        if on:
            margins = self._outer_layout.contentsMargins() if self._outer_layout is not None else QtCore.QMargins(12, 12, 12, 12)
            frame_margins = self._frame3d_layout.contentsMargins() if self._frame3d_layout is not None else QtCore.QMargins(10, 10, 10, 10)
            self._game_fullscreen_saved_ui = {
                "splitter_sizes": self.splitter.sizes(),
                "stats_visible": self.stats_card.isVisible(),
                "brain_visible": self.brain_card.isVisible(),
                "toolbar_visible": bool(self._main_toolbar and self._main_toolbar.isVisible()),
                "status_visible": self.statusBar().isVisible(),
                "help_visible": self.help_overlay.isVisible(),
                "minimap_visible": self.minimap.isVisible(),
                "live_visible": self.live_pill.isVisible(),
                "hud_visible": self.hud_overlay.isVisible(),
                "outer_margins": (margins.left(), margins.top(), margins.right(), margins.bottom()),
                "outer_spacing": self._outer_layout.spacing() if self._outer_layout is not None else 12,
                "frame_margins": (frame_margins.left(), frame_margins.top(), frame_margins.right(), frame_margins.bottom()),
                "frame_spacing": self._frame3d_layout.spacing() if self._frame3d_layout is not None else 8,
            }

            self.stats_card.hide()
            self.brain_card.hide()
            self.splitter.setSizes([0, 1, 0])
            if self._main_toolbar is not None:
                self._main_toolbar.hide()
            self.statusBar().hide()
            self.help_overlay.hide()
            self.minimap.hide()
            self.live_pill.hide()
            self.hud_overlay.hide()
            if self._outer_layout is not None:
                self._outer_layout.setContentsMargins(0, 0, 0, 0)
                self._outer_layout.setSpacing(0)
            if self._frame3d_layout is not None:
                self._frame3d_layout.setContentsMargins(0, 0, 0, 0)
                self._frame3d_layout.setSpacing(0)
            self.showFullScreen()
            self._game_fullscreen = True
            self._schedule_overlay_relayout()
            self.statusBar().showMessage("Game fullscreen: on", 1200)
        else:
            saved = dict(self._game_fullscreen_saved_ui or {})
            self.showNormal()
            self._game_fullscreen = False
            if self._outer_layout is not None:
                l, t, r, b = saved.get("outer_margins", (12, 12, 12, 12))
                self._outer_layout.setContentsMargins(int(l), int(t), int(r), int(b))
                self._outer_layout.setSpacing(int(saved.get("outer_spacing", 12)))
            if self._frame3d_layout is not None:
                l, t, r, b = saved.get("frame_margins", (10, 10, 10, 10))
                self._frame3d_layout.setContentsMargins(int(l), int(t), int(r), int(b))
                self._frame3d_layout.setSpacing(int(saved.get("frame_spacing", 8)))
            if saved.get("stats_visible", True):
                self.stats_card.show()
            if saved.get("brain_visible", True):
                self.brain_card.show()
            if self._main_toolbar is not None and saved.get("toolbar_visible", True):
                self._main_toolbar.show()
            if saved.get("status_visible", True):
                self.statusBar().show()
            if saved.get("help_visible", True) and getattr(self, "act_help", None) and self.act_help.isChecked():
                self.help_overlay.show()
            if saved.get("minimap_visible", True):
                self.minimap.show()
            if saved.get("live_visible", True):
                self.live_pill.show()
            self.hud_overlay.setVisible(bool(saved.get("hud_visible", True)) and not self.view3d.is_first_person_mode())
            sizes = saved.get("splitter_sizes")
            if isinstance(sizes, list):
                try:
                    self.splitter.setSizes([int(x) for x in sizes])
                except Exception:
                    pass
            self._schedule_overlay_relayout()
            if self.statusBar().isVisible():
                self.statusBar().showMessage("Game fullscreen: off", 1200)

        if hasattr(self, "act_game_fullscreen"):
            self.act_game_fullscreen.blockSignals(True)
            self.act_game_fullscreen.setChecked(bool(self._game_fullscreen))
            self.act_game_fullscreen.blockSignals(False)

    @Slot(bool)
    def _toggle_game_fullscreen(self, on: bool):
        self._set_game_fullscreen(bool(on))

    @Slot()
    def _toggle_game_fullscreen_shortcut(self):
        self._set_game_fullscreen(not self._game_fullscreen)

    def _help_text_for_current_mode(self) -> str:
        if self.view3d.is_first_person_mode():
            return "FPS: при входе открывается карта мира • M/Ь — карта • F11 — игровой fullscreen • мышь — обзор • WASD/ЦФЫВ — вести выбранного агента • Shift — бег • Ctrl — точный шаг • Q/Й — курсор • E/У или LMB — выбрать цель • RMB/G/П — goal • Ctrl+Shift+T — в комнату • Ctrl+Shift+G — выпуск • F/А — к агенту • Tab — след. • V/М или Esc — выход"
        return "ЛКМ — выбрать • ПКМ — приказать • Ctrl+ПКМ — меню • RMB — орбита • MMB — пан • колесо — зум • F11 — игровой fullscreen • V/М — first person • Tab — след. • F/А — фокус • R/К — сброс • Ctrl+W — волки • Ctrl+Shift+T — в комнату • Ctrl+Shift+G — выпуск"

    def _forward_key_event_to_view3d(self, e: QtGui.QKeyEvent, *, source=None) -> bool:
        if not hasattr(self, "view3d"):
            return False
        if source is self.world_map_overlay:
            return False
        if isinstance(source, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox)):
            return False
        key = _normalize_control_key_event(e)
        always_keys = {Qt.Key_V, Qt.Key_R, Qt.Key_F, Qt.Key_Tab, Qt.Key_F11}
        if hasattr(self, "world_map_overlay") and self.world_map_overlay.isVisible():
            return False
        if not self.view3d.is_first_person_mode() and key not in always_keys:
            return False
        forward_keys = {
            Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D,
            Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right,
            Qt.Key_Shift, Qt.Key_Control,
            Qt.Key_Q, Qt.Key_E, Qt.Key_G, Qt.Key_M,
            Qt.Key_R, Qt.Key_F, Qt.Key_Tab,
            Qt.Key_V, Qt.Key_Escape, Qt.Key_F11,
        }
        if key not in forward_keys:
            return False
        self.view3d.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        cloned = QtGui.QKeyEvent(
            e.type(),
            key,
            e.modifiers(),
            e.text(),
            e.isAutoRepeat(),
            e.count(),
        )
        if e.type() == QtCore.QEvent.Type.KeyPress:
            self.view3d.keyPressEvent(cloned)
        else:
            self.view3d.keyReleaseEvent(cloned)
        e.accept()
        return True

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if self._forward_key_event_to_view3d(e):
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if self._forward_key_event_to_view3d(e):
            return
        super().keyReleaseEvent(e)

    def _show_world_map_overlay(self):
        if not hasattr(self, "world_map_overlay") or not self.view3d.is_first_person_mode():
            return
        self.view3d.suspend_first_person_capture()
        self.world_map_overlay.setGeometry(self.centralWidget().rect())
        self.world_map_overlay.show_map()
        self.statusBar().showMessage("Карта мира открыта: ЛКМ — перейти, Esc/M — закрыть", 2400)

    def _toggle_world_map_overlay(self):
        if not hasattr(self, "world_map_overlay") or not self.view3d.is_first_person_mode():
            return
        if self.world_map_overlay.isVisible():
            self.world_map_overlay.hide_map()
            return
        self._show_world_map_overlay()

    @Slot()
    def _on_world_map_closed(self):
        if self.view3d.is_first_person_mode():
            self.view3d.resume_first_person_capture()
        self.statusBar().showMessage("Карта мира закрыта", 1200)

    @Slot(bool)
    def _toggle_first_person(self, on: bool):
        self.view3d.set_first_person_mode(bool(on))

    @Slot(bool)
    def _on_first_person_changed(self, on: bool):
        if hasattr(self, "act_first_person"):
            self.act_first_person.blockSignals(True)
            self.act_first_person.setChecked(bool(on))
            self.act_first_person.blockSignals(False)
        if hasattr(self, "hud_overlay"):
            self.hud_overlay.setVisible(not on)
        if hasattr(self, "help_overlay"):
            self.help_overlay.setText(self._help_text_for_current_mode())
        self._schedule_overlay_relayout()
        if hasattr(self, "fp_overlay"):
            self.fp_overlay.set_state(self.view3d.get_first_person_hud_state())
        if hasattr(self, "world_map_overlay"):
            if on:
                QtCore.QTimer.singleShot(0, self._show_world_map_overlay)
            elif self.world_map_overlay.isVisible():
                self.world_map_overlay.hide_map()
        self.statusBar().showMessage("First person: on" if on else "First person: off", 1800)

    # --- Theme
    def _toggle_theme(self):
        cur = self.settings.value("theme", "blue")
        new = "emerald" if cur == "blue" else "blue"
        self._apply_theme(new)
        self.settings.setValue("theme", new)
        self._toast(f"Theme: {new}")

    def _apply_theme(self, name: str):
        if name == "emerald":
            self._set_app_stylesheet(APP_QSS_EMERALD)
            self.act_theme.setText("Theme: Blue")
        else:
            self._set_app_stylesheet(APP_QSS)
            self.act_theme.setText("Theme: Emerald")

    def _set_app_stylesheet(self, qss: str):
        QtWidgets.QApplication.instance().setStyleSheet(qss)

    def _apply_palette(self):
        pal = QtWidgets.QApplication.instance().palette()
        CR = QtGui.QPalette.ColorRole
        pal.setColor(CR.Window, QtGui.QColor(COL_BG))
        pal.setColor(CR.WindowText, QtGui.QColor(COL_TEXT))
        pal.setColor(CR.Base, QtGui.QColor(12, 14, 20))
        pal.setColor(CR.AlternateBase, QtGui.QColor(18, 20, 28))
        pal.setColor(CR.ToolTipBase, QtGui.QColor(22, 24, 32))
        pal.setColor(CR.ToolTipText, QtGui.QColor(COL_TEXT))
        pal.setColor(CR.Text, QtGui.QColor(COL_TEXT))
        pal.setColor(CR.Button, QtGui.QColor(18, 20, 28))
        pal.setColor(CR.ButtonText, QtGui.QColor(COL_TEXT))
        pal.setColor(CR.Highlight, QtGui.QColor(90, 130, 255))
        pal.setColor(CR.HighlightedText, QtGui.QColor("#ffffff"))
        QtWidgets.QApplication.instance().setPalette(pal)

    # --- persist
    def _restore_geometry(self):
        geo = self.settings.value("geo")
        if isinstance(geo, QtCore.QByteArray):
            self.restoreGeometry(geo)
        state = self.settings.value("state")
        if isinstance(state, QtCore.QByteArray):
            self.restoreState(state)
        split = self.settings.value("splitter")
        if isinstance(split, list):
            try:
                self.splitter.setSizes([int(x) for x in split])
            except Exception:
                pass
        if self._pending_restore_game_fullscreen:
            QtCore.QTimer.singleShot(0, lambda: self._set_game_fullscreen(True))

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.settings.setValue("geo", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.settings.setValue("splitter", self.splitter.sizes())
        self.settings.setValue("game_fullscreen", bool(self._game_fullscreen))
        if hasattr(self, "_trainer_timer"):
            self._trainer_timer.stop()
        if hasattr(self, "_combat_timer"):
            self._combat_timer.stop()
        if hasattr(self, "_ui_timer"):
            self._ui_timer.stop()
        super().closeEvent(e)


# ====================================================
# 7) Сборка снапшота для движка 3D
# ====================================================
def _belief_to_dict(b) -> Dict[str, Any]:
    try:
        if isinstance(b, dict):
            return {
                "if": str(b.get("if") or b.get("condition") or ""),
                "then": str(b.get("then") or b.get("conclusion") or ""),
                "strength": float(b.get("strength", 0.0) or 0.0),
            }
        return {
            "if": getattr(b, "condition", ""),
            "then": getattr(b, "conclusion", ""),
            "strength": float(getattr(b, "strength", 0.0) or 0.0),
        }
    except Exception:
        return {}

def _memory_to_dict(ev) -> Dict[str, Any]:
    try:
        if isinstance(ev, dict):
            data = ev.get("data")
            if not isinstance(data, dict):
                data = {
                    k: v for k, v in ev.items()
                    if k not in ("type", "etype", "kind", "tick", "level", "actor", "pos", "private")
                }
            pos = ev.get("pos")
            if isinstance(pos, list):
                pos = tuple(pos)
            return {
                "type": str(ev.get("type") or ev.get("etype") or ev.get("kind") or "event"),
                "tick": ev.get("tick"),
                "level": str(ev.get("level", "info")),
                "actor": ev.get("actor"),
                "pos": pos if isinstance(pos, tuple) and len(pos) == 2 else None,
                "data": dict(data),
            }
        pos = getattr(ev, "pos", None)
        if isinstance(pos, list):
            pos = tuple(pos)
        return {
            "type": str(getattr(ev, "etype", getattr(ev, "type", "event"))),
            "tick": getattr(ev, "tick", None),
            "level": str(getattr(ev, "level", "info")),
            "actor": getattr(ev, "actor", None),
            "pos": pos if isinstance(pos, tuple) and len(pos) == 2 else None,
            "data": dict(getattr(ev, "data", {}) or {}),
        }
    except Exception:
        return {}

def _brain_to_dict(brain, *, detailed: bool = True) -> Dict[str, Any]:
    if brain is None:
        return {}
    if isinstance(brain, dict):
        data = dict(brain)
        rules_obj = data.get("behavior_rules", {})
    else:
        try:
            data = dict(brain.export_public_state_for_ui() or {})
        except Exception:
            data = {}
        rules_obj = data.get("behavior_rules", getattr(brain, "behavior_rules", None))
    rules: Dict[str, Any] = {}
    if isinstance(rules_obj, dict):
        rules = dict(rules_obj)
    elif rules_obj is not None:
        for k in dir(rules_obj):
            if k.startswith("_"):
                continue
            try:
                v = getattr(rules_obj, k)
            except Exception:
                continue
            if isinstance(v, (int, float, str, bool)):
                rules[k] = v
    beliefs = []
    mem_tail = []
    if detailed:
        try:
            beliefs_src = data.get("beliefs", getattr(brain, "beliefs", [])) if not isinstance(brain, dict) else data.get("beliefs", [])
            for b in list(beliefs_src or [])[-24:]:
                beliefs.append(_belief_to_dict(b))
        except Exception:
            pass
        try:
            mem_src = data.get("memory_tail", getattr(brain, "memory_tail", [])) if not isinstance(brain, dict) else data.get("memory_tail", [])
            mem_tail = [_memory_to_dict(ev) for ev in list(mem_src or [])[-16:]]
        except Exception:
            pass
    return {
        "current_drive": data.get("current_drive", getattr(brain, "current_drive", None) if not isinstance(brain, dict) else None),
        "survival_score": data.get("survival_score", getattr(brain, "survival_score", None) if not isinstance(brain, dict) else None),
        "behavior_rules": rules,
        "beliefs": beliefs,
        "memory_tail": mem_tail,
    }


def _trim_agent_payload_for_engine(row: Dict[str, Any], *, detailed: bool) -> Dict[str, Any]:
    packed = dict(row)
    packed["mind"] = _brain_to_dict(row.get("mind") or row.get("consciousness"), detailed=detailed)
    if detailed and "memory_tail" in packed:
        packed["memory_tail"] = [_memory_to_dict(ev) for ev in list(packed.get("memory_tail", []) or [])[-16:]]
    else:
        packed.pop("memory_tail", None)
    packed.pop("consciousness", None)
    return packed

def _build_engine_snapshot(world, *, tick: int, selected_agent_id: Optional[str] = None) -> Dict[str, Any]:
    if hasattr(world, "snapshot") and callable(getattr(world, "snapshot")):
        try:
            snap = dict(world.snapshot() or {})
            snap["tick"] = int(snap.get("tick", tick))
            snap["world"] = {
                "width": float((snap.get("world", {}) or {}).get("width", getattr(world, "width", 100.0))),
                "height": float((snap.get("world", {}) or {}).get("height", getattr(world, "height", 100.0))),
            }

            chat_src = snap.get("chat")
            if not isinstance(chat_src, list):
                if hasattr(world, "chat_log") and isinstance(world.chat_log, list):
                    chat_src = world.chat_log
                elif hasattr(world, "chat") and isinstance(world.chat, list):
                    chat_src = world.chat
                else:
                    chat_src = []
            snap["chat"] = [str(x) for x in list(chat_src)[-ENGINE_CHAT_TAIL:]]

            events_src = snap.get("events")
            if not isinstance(events_src, list):
                events_src = list(getattr(world, "event_log", []) or getattr(world, "events", []) or [])
            snap["events"] = list(events_src)[-ENGINE_EVENT_TAIL:]

            if not snap.get("global_events"):
                snap["global_events"] = [
                    f"[t={ev.get('tick', snap['tick'])}] {ev.get('type', 'event')}: "
                    f"{ev.get('name') or ev.get('who') or ev.get('victim_name') or ev.get('target') or ''}".rstrip(": ")
                    for ev in snap["events"][-ENGINE_GLOBAL_EVENT_TAIL:]
                ]

            agents_norm: List[Dict[str, Any]] = []
            for ag in list(snap.get("agents", []) or []):
                if not isinstance(ag, dict):
                    continue
                row = dict(ag)
                is_selected = bool(selected_agent_id) and str(row.get("id", "")) == str(selected_agent_id)
                row = _trim_agent_payload_for_engine(row, detailed=is_selected)
                try:
                    energy = float(row.get("energy", 100.0))
                    if energy <= 1.5:
                        row["energy"] = energy * 100.0
                except Exception:
                    pass
                try:
                    hunger = float(row.get("hunger", 0.0))
                    if hunger <= 1.5:
                        row["hunger"] = hunger * 100.0
                except Exception:
                    pass
                agents_norm.append(row)
            snap["agents"] = agents_norm
            return snap
        except Exception:
            pass

    agents_pack: List[Dict[str, Any]] = []
    for ag in _iter_vals(getattr(world, "agents", [])):
        brain = getattr(ag, "brain", None)
        agents_pack.append({
            "id": getattr(ag, "id", ""),
            "name": getattr(ag, "name", ""),
            "pos": {"x": float(getattr(ag, "x", 0.0)), "y": float(getattr(ag, "y", 0.0))},
            "goal": {"x": float(getattr(ag, "goal_x", getattr(ag, "x", 0.0))),
                     "y": float(getattr(ag, "goal_y", getattr(ag, "y", 0.0)))},
            "vel": {"x": float(getattr(ag, "vx", 0.0)), "y": float(getattr(ag, "vy", 0.0))},
            "fear": float(getattr(ag, "fear", 0.0)),
            "health": float(getattr(ag, "health", 100.0)),
            "energy": float(getattr(ag, "energy", 100.0)) * 100.0 if float(getattr(ag, "energy", 100.0)) <= 1.5 else float(getattr(ag, "energy", 100.0)),
            "hunger": float(getattr(ag, "hunger", 0.0)) * 100.0 if float(getattr(ag, "hunger", 0.0)) <= 1.5 else float(getattr(ag, "hunger", 0.0)),
            "age_ticks": int(getattr(ag, "age_ticks", 0)),
            "alive": bool(getattr(ag, "is_alive", lambda: True)()),
            "cause_of_death": getattr(ag, "cause_of_death", None),
            "tags": list(getattr(ag, "tags", []) or []),
            "mind": _brain_to_dict(
                brain,
                detailed=(bool(selected_agent_id) and str(getattr(ag, "id", "")) == str(selected_agent_id)),
            ),
        })

    objects_pack: List[Dict[str, Any]] = []
    for obj in _iter_vals(getattr(world, "objects", [])):
        objects_pack.append({
            "id": getattr(obj, "obj_id", ""),
            "name": getattr(obj, "name", ""),
            "kind": getattr(obj, "kind", ""),
            "pos": {"x": float(getattr(obj, "x", 0.0)), "y": float(getattr(obj, "y", 0.0))},
            "radius": float(getattr(obj, "radius", 0.0)),
            "danger_level": float(getattr(obj, "danger_level", 0.0)),
            "comfort_level": float(getattr(obj, "comfort_level", 0.0)),
        })

    animals_pack: List[Dict[str, Any]] = []
    for ani in _iter_vals(getattr(world, "animals", [])):
        sp = getattr(ani, "species", None)
        animals_pack.append({
            "id": getattr(ani, "uid", ""),
            "species": getattr(sp, "species_id", getattr(sp, "name", "beast")) if sp else "beast",
            "pos": {"x": float(getattr(ani, "x", 0.0)), "y": float(getattr(ani, "y", 0.0))},
            "hp": float(getattr(ani, "hp", getattr(sp, "base_hp", 50.0) if sp else 50.0)),
            "aggressive": bool(getattr(sp, "aggressive", False)) if sp else False,
            "tamable": bool(getattr(sp, "tamable", False)) if sp else False,
            "tamed_by": getattr(ani, "tamed_by", None),
        })

    chat_lines: List[str] = []
    if hasattr(world, "chat") and isinstance(world.chat, list):
        chat_lines = [str(x) for x in world.chat[-ENGINE_CHAT_TAIL:]]
    elif hasattr(world, "chat_log") and isinstance(world.chat_log, list):
        chat_lines = [str(x) for x in world.chat_log[-ENGINE_CHAT_TAIL:]]

    snap: Dict[str, Any] = {
        "tick": int(tick),
        "world": {"width": float(getattr(world, "width", 100.0)),
                  "height": float(getattr(world, "height", 100.0))},
        "agents": agents_pack,
        "objects": objects_pack,
        "animals": animals_pack,
        "chat": chat_lines,
        "events": list(getattr(world, "event_log", []) or [])[-ENGINE_EVENT_TAIL:],
    }
    snap["global_events"] = [
        f"[t={ev.get('tick', tick)}] {ev.get('type', 'event')}: "
        f"{ev.get('name') or ev.get('who') or ev.get('victim_name') or ev.get('target') or ''}".rstrip(": ")
        for ev in snap["events"][-ENGINE_GLOBAL_EVENT_TAIL:]
    ]
    return snap


# =========================
# 8) main
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont(APP_FONT, 10))

    # Глобальная стеклянная тема (стартовая — blue)
    app.setStyleSheet(APP_QSS)

    # Лёгкая палитра (контраст текста)
    pal = app.palette()
    CR = QtGui.QPalette.ColorRole
    pal.setColor(CR.Window, QtGui.QColor(COL_BG))
    pal.setColor(CR.WindowText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Base, QtGui.QColor(12, 14, 20))
    pal.setColor(CR.AlternateBase, QtGui.QColor(18, 20, 28))
    pal.setColor(CR.ToolTipBase, QtGui.QColor(22, 24, 32))
    pal.setColor(CR.ToolTipText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Text, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Button, QtGui.QColor(18, 20, 28))
    pal.setColor(CR.ButtonText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Highlight, QtGui.QColor(90, 130, 255))
    pal.setColor(CR.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)

    win = CombinedMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
