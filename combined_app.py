# combined_app.py — стеклянный UI с оверлеями, тулбаром, мини-картой и тостами
# 3D-мир + Mind Trainer (эволюция/метрики/мозг агентов)
# ПКМ в 3D -> задать goal агенту в локальном мире тренера.
# Выбор агента синхронен между 3D и инспектором мозга.

import sys
import math
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
from env_lowpoly import build_lowpoly_village

# --- Бой
from combat_system import CombatSystem

# --- Mind Trainer панели
from mind_trainer_gui import MindTrainerInteractive, TrainerStatsWidget, AgentBrainWidget


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
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFixedSize(180, 180)
        self.setToolTip("Mini-map: click to center camera")
        self._bg = QtGui.QColor(20, 24, 32, 180)
        self._pen_grid = QtGui.QPen(QtGui.QColor(255,255,255,20), 1)
        self._pen_border = QtGui.QPen(QtGui.QColor("#2a2f3a"), 1)
        self._brush_agent = QtGui.QBrush(QtGui.QColor(122,162,255,220))
        self._brush_sel = QtGui.QBrush(QtGui.QColor(255,255,255,240))
        self._brush_animal = QtGui.QBrush(QtGui.QColor(255,191,105,220))

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
                p.setBrush(self._brush_sel if aid == sel else self._brush_agent)
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

        p.end()

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        r = self.rect().adjusted(6,6,-6,-6)
        ww, wh = self.shared.world_w, self.shared.world_h
        t = max(0.0, min(1.0, (e.position().x()-r.left())/max(1,r.width())))
        u = max(0.0, min(1.0, (e.position().y()-r.top())/max(1,r.height())))
        self.clickedWorld.emit(t*ww, u*wh)


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

    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared
        self.engine = shared.engine
        self.shared.updated.connect(self.update)

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

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._frame_tick)
        self._timer.start()
        self._last_frame_time = QtCore.QElapsedTimer()
        self._last_frame_time.start()

        self._fps_smooth = 0.0

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

    def _camera_position(self) -> Tuple[float, float, float]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        r = max(10.0, self.distance)
        cx, cz = self.center_x, self.center_z
        cos_p = math.cos(pitch); sin_p = math.sin(pitch)
        cam_x = cx + r * cos_p * math.cos(yaw)
        cam_y = max(5.0, r * sin_p)
        cam_z = cz + r * cos_p * math.sin(yaw)
        return cam_x, cam_y, cam_z

    def _clamp_center(self):
        self.center_x = max(0.0, min(self.shared.world_w, self.center_x))
        self.center_z = max(0.0, min(self.shared.world_h, self.center_z))

    def paintGL(self):
        cam_x, cam_y, cam_z = self._camera_position()
        self.engine.setup_viewport_and_camera(
            w=self.width(), h=self.height(),
            cam_pos=(cam_x, cam_y, cam_z),
            cam_look=(self.center_x, 0.0, self.center_z),
            fov_deg=self.fov_deg,
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
        finally:
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
        self._btns = e.buttons()
        self._last_mouse_pos = None
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
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
        delta = e.angleDelta().y() / 120.0
        self.distance *= math.pow(0.9, delta)
        self.distance = max(20.0, min(600.0, self.distance))
        self.update()
        super().wheelEvent(e)

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.key() == Qt.Key_R:
            self.center_x = self.shared.world_w * 0.5
            self.center_z = self.shared.world_h * 0.5
            self.distance = 140.0
            self.yaw_deg = -135.0
            self.pitch_deg = 40.0
            self.update()
        elif e.key() == Qt.Key_F:
            sel = self.shared.get_selected_agent_id()
            if sel and sel in self.engine.agents:
                ent = self.engine.agents[sel]
                self.center_x = ent.transform.pos.x
                self.center_z = ent.transform.pos.z
                self._clamp_center(); self.update()
        elif e.key() == Qt.Key_Tab:
            self.shared.cycle_next_agent()
            sel = self.shared.get_selected_agent_id()
            if sel and sel in self.engine.agents:
                ent = self.engine.agents[sel]
                self.center_x = ent.transform.pos.x
                self.center_z = ent.transform.pos.z
                self._clamp_center(); self.update()
        else:
            super().keyPressEvent(e)


# ====================================================
# 4) Мост: trainer.world -> engine (для 3D синхронизации)
# ====================================================
class TrainerToEngineBridge(QtCore.QObject):
    def __init__(self, trainer: MindTrainerInteractive, shared: SharedState, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.shared = shared
        self.trainer.world_changed.connect(self._push_snapshot)

    @Slot()
    def _push_snapshot(self):
        world = self.trainer.world
        if world is None:
            return
        snap = _build_engine_snapshot(world, tick=self.trainer.monitor.tick)
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
        self.title.setText(str(name))
        drive = info.get("mind_drive") or "—"
        score = info.get("mind_survival_score")
        self.drive.setText(f"Drive: {drive}   |   Survival: {score if score is not None else '—'}")
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


# ====================================================
# 6) Главное окно: 3 колонки (Stats | 3D | Brain) + Toolbar
# ====================================================
class CombinedMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-Matrix Lab — 3D + Mind Trainer")
        self.resize(1600, 900)
        self.settings = QtCore.QSettings("MiniMatrixLab", "CombinedApp")

        # Глобальный шрифт
        font = QtGui.QFont(APP_FONT, 10)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.setFont(font)

        # 1) 3D движок
        self.engine = MiniMatrixEngine()
        try:
            village_meshes = build_lowpoly_village(
                world_w=self.engine.world.width,
                world_h=self.engine.world.height,
            )
        except TypeError:
            village_meshes = build_lowpoly_village(self.engine.world.width, self.engine.world.height)
        if hasattr(self.engine, "load_static_environment"):
            try:
                self.engine.load_static_environment(village_meshes)
            except Exception as e:
                print("[env] load_static_environment failed:", e)

        # 2) тренер + бой
        self.trainer = MindTrainerInteractive(
            num_agents=3, max_ticks_per_epoch=2000, seed=1234,
            disaster_interval_ticks=400, relief_after_disaster=80,
            fresh_start=False,
            agent_lineup=[
                {"id": "a1", "name": "Echo", "persona":
                 "Ты Echo. Осторожный выживальщик. Бережёшь Nova, шутишь, но всегда смотришь по сторонам."},
                {"id": "a2", "name": "Nova", "persona":
                 "Ты Nova. Смелая исследовательница. Действуешь решительно, но не рискуешь зря."},
                {"id": "agent_0", "name": "A0", "persona": "scout/explorer"},
            ],
        )
        self.combat = CombatSystem(self.trainer.world)
        self._combat_timer = QtCore.QTimer(self)
        self._combat_timer.setInterval(50)  # ~20 Гц
        self._combat_timer.timeout.connect(self._on_combat_tick)
        self._combat_timer.start()
        self._combat_paused = False

        # 3) Shared + 3D виджет + оверлеи
        self.shared = SharedState(self.engine)
        self.view3d = World3DView(self.shared)
        apply_expand_policy(self.view3d, w_stretch=True)
        self.view3d.setMinimumSize(980, 720)
        self.view3d.requestSetGoal.connect(self._on_set_goal_from_3d)
        self.view3d.fpsUpdated.connect(self._update_fps)

        frame3d = QtWidgets.QFrame()
        frame3d.setStyleSheet(FRAME3D_STYLE)
        frame3d.setProperty("card", True)
        lay3d = QtWidgets.QVBoxLayout(frame3d)
        lay3d.setContentsMargins(10, 10, 10, 10)
        lay3d.setSpacing(8)
        lay3d.addWidget(self.view3d, 1)

        # Оверлеи
        self.help_overlay = OverlayLabel(frame3d,
            "ЛКМ — выбрать • ПКМ — приказать • Ctrl+ПКМ — меню • RMB — орбита • MMB — пан • колесо — зум • Tab — след. • F — фокус • R — сброс • Ctrl+W — волки")
        self.hud_overlay = AgentHud(frame3d, self.shared)
        self.live_pill = LiveTickPill(frame3d, self.shared)
        self.minimap = MiniMapWidget(self.shared, frame3d)
        self.minimap.clickedWorld.connect(self._center_to)

        def place_overlays():
            r = frame3d.rect()
            self.hud_overlay.place(r)
            self.live_pill.place(r)
            # help — под HUD
            self.help_overlay.adjustSize()
            self.help_overlay.move(r.x()+14, self.hud_overlay.geometry().bottom()+8)
            # minimap — правый нижний
            self.minimap.move(r.right()-self.minimap.width()-14, r.bottom()-self.minimap.height()-14)

        frame3d.installEventFilter(self)
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

        # 9) Toolbar & actions
        self._make_toolbar()

        # 10) тема/палитра и состояние
        self._apply_palette()
        self._apply_theme(self.settings.value("theme", "blue"))
        self._restore_geometry()

        # 11) начальный пуш снапшота
        self.bridge._push_snapshot()
        self._place_overlays()

    # --- eventFilter для перекладки оверлеев
    def eventFilter(self, obj, ev):
        if ev.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Show):
            if hasattr(self, "_place_overlays"):
                QtCore.QTimer.singleShot(0, self._place_overlays)
        return super().eventFilter(obj, ev)

    # --- toolbar
    def _make_toolbar(self):
        tb = QtWidgets.QToolBar("Controls", self)
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

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

        tb.addSeparator()

        # Волки
        act_spawn_wolves = QtGui.QAction("Spawn wolves (Ctrl+W)", self)
        act_spawn_wolves.setShortcut("Ctrl+W")
        act_spawn_wolves.triggered.connect(self._spawn_wolves)
        self.addAction(act_spawn_wolves)
        tb.addAction(act_spawn_wolves)

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
            try:
                self.combat.step(0.05)
            finally:
                self.bridge._push_snapshot()

    # --- спавн волков возле выбранного агента (или центра)
    def _spawn_wolves(self):
        w = self.trainer.world
        if not w or not getattr(self, 'combat', None):
            return
        sel = self.shared.get_selected_agent_id()
        if sel and hasattr(w, "get_agent_by_id"):
            ag = w.get_agent_by_id(sel)
            cx = getattr(ag, "x", w.width * 0.5)
            cy = getattr(ag, "y", w.height * 0.5)
        else:
            cx, cy = w.width * 0.5, w.height * 0.5
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
        ag = None
        if hasattr(w, "get_agent_by_id"):
            ag = w.get_agent_by_id(agent_id)
        else:
            for a in _iter_vals(getattr(w, "agents", [])):
                if getattr(a, "id", None) == agent_id:
                    ag = a; break
        if ag is None:
            return
        x = max(0.0, min(w.width, x))
        z = max(0.0, min(w.height, z))
        if hasattr(ag, "set_goal"):
            try: ag.set_goal(x, z)
            except Exception:
                setattr(ag, "goal_x", x); setattr(ag, "goal_y", z)
        else:
            setattr(ag, "goal_x", x); setattr(ag, "goal_y", z)

        if hasattr(w, "add_chat_line"):
            try: w.add_chat_line(f"[observer] {getattr(ag,'name',agent_id)} → goal=({x:.1f},{z:.1f})")
            except Exception: pass

        self.bridge._push_snapshot()
        self._toast(f"Goal for {agent_id} → ({x:.1f}, {z:.1f})")

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
        self.view3d.center_x, self.view3d.center_z = x, y
        self.view3d._clamp_center()
        self.view3d.update()
        self._toast(f"Camera → ({x:.1f}, {y:.1f})")

    # --- FPS/status
    def _update_fps(self, fps: float):
        self._lbl_fps.setText(f"FPS: {fps:0.1f}")

    def _tick_ui(self):
        self._lbl_cam.setText(f"Cam: x={self.view3d.center_x:0.1f}, z={self.view3d.center_z:0.1f}, dist={self.view3d.distance:0.0f}")

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

    def closeEvent(self, e: QtGui.QCloseEvent):
        self.settings.setValue("geo", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.settings.setValue("splitter", self.splitter.sizes())
        super().closeEvent(e)


# ====================================================
# 7) Сборка снапшота для движка 3D
# ====================================================
def _belief_to_dict(b) -> Dict[str, Any]:
    try:
        return {
            "if": getattr(b, "condition", ""),
            "then": getattr(b, "conclusion", ""),
            "strength": float(getattr(b, "strength", 0.0) or 0.0),
        }
    except Exception:
        return {}

def _brain_to_dict(brain) -> Dict[str, Any]:
    if brain is None:
        return {}
    rules_obj = getattr(brain, "behavior_rules", None)
    rules: Dict[str, Any] = {}
    if rules_obj is not None:
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
    try:
        for b in list(getattr(brain, "beliefs", []) or [])[-60:]:
            beliefs.append(_belief_to_dict(b))
    except Exception:
        pass
    mem_tail = []
    try:
        mem_tail = list(getattr(brain, "memory_tail", []) or [])[-40:]
    except Exception:
        pass
    return {
        "current_drive": getattr(brain, "current_drive", None),
        "survival_score": getattr(brain, "survival_score", None),
        "behavior_rules": rules,
        "beliefs": beliefs,
        "memory_tail": mem_tail,
    }

def _build_engine_snapshot(world, *, tick: int) -> Dict[str, Any]:
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
            "energy": float(getattr(ag, "energy", 100.0)),
            "hunger": float(getattr(ag, "hunger", 0.0)),
            "age_ticks": int(getattr(ag, "age_ticks", 0)),
            "alive": bool(getattr(ag, "is_alive", lambda: True)()),
            "cause_of_death": getattr(ag, "cause_of_death", None),
            "mind": _brain_to_dict(brain),
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
        chat_lines = [str(x) for x in world.chat[-200:]]

    snap: Dict[str, Any] = {
        "tick": int(tick),
        "world": {"width": float(getattr(world, "width", 100.0)),
                  "height": float(getattr(world, "height", 100.0))},
        "agents": agents_pack,
        "objects": objects_pack,
        "animals": animals_pack,
        "chat": chat_lines,
        "events": [],
    }
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
