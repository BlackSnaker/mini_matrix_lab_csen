# combined_app.py — обновлённый, «стеклянный» UI
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
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QStyle

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
COL_BG = "#0b0d12"        # фон окна
COL_PANEL_BG = "rgba(16,18,26,0.85)"
COL_PANEL_BG_DARK = "rgba(10,12,18,0.85)"
COL_BORDER = "#2a2f3a"
COL_ACCENT = "#7aa2ff"
COL_TEXT = "#e8eaf6"
COL_TEXT_DIM = "#b9bed3"

APP_FONT = "Inter, Segoe UI, Helvetica Neue, Arial"

CARD_STYLE = f"""
QFrame[card="true"] {{
    background: {COL_PANEL_BG};
    border: 1px solid {COL_BORDER};
    border-radius: 14px;
}}
QLabel[cardTitle="true"] {{
    color: {COL_TEXT};
    font-size: 12.5px;
    font-weight: 650;
    letter-spacing: .3px;
}}
"""

TABS_STYLE = f"""
QTabWidget::pane {{
    border: 1px solid {COL_BORDER};
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #14161f, stop:1 #0f1119);
    border-radius: 12px;
}}
QTabBar::tab {{
    background-color: #232634;
    color:{COL_TEXT};
    padding:7px 12px;
    margin-right:4px;
    border-top-left-radius:8px;
    border-top-right-radius:8px;
    font-size:11px;
}}
QTabBar::tab:selected {{ background-color:#34384a; color:#fff; }}
"""

HELP_PILL_STYLE = """
QLabel {
    color:#cdd5ff; background: rgba(28,32,48,0.7);
    border:1px solid #374058;
    border-radius: 10px; padding:6px 10px;
}
"""

SPLITTER_STYLE = """
QSplitter::handle {
    background: #1a1f2a;
    width: 7px;
    border-radius: 3px;
}
"""

FRAME3D_STYLE = f"""
QFrame {{
    background: #000;
    border: 1px solid {COL_BORDER};
    border-radius: 14px;
}}
"""

STATUSBAR_STYLE = "QStatusBar { color:#c7cbe4; background-color:#11121a; }"


# =====================================================
# 1) Утилиты: итерация, карточки, тени, размер политики
# =====================================================
def _iter_vals(maybe_collection):
    """Возвращает элементы коллекции как список: поддерживает dict, list/tuple, None и одиночные объекты."""
    if maybe_collection is None:
        return []
    if isinstance(maybe_collection, Mapping):
        return list(maybe_collection.values())
    if isinstance(maybe_collection, Sequence) and not isinstance(maybe_collection, (str, bytes, bytearray)):
        return list(maybe_collection)
    return [maybe_collection]


def make_card(title: str, inner: QtWidgets.QWidget) -> QtWidgets.QFrame:
    """Оборачивает виджет в «карточку» с заголовком и тенью."""
    card = QtWidgets.QFrame()
    card.setProperty("card", True)
    card.setStyleSheet(CARD_STYLE)

    v = QtWidgets.QVBoxLayout(card)
    v.setContentsMargins(12, 12, 12, 12)
    v.setSpacing(10)

    title_lbl = QtWidgets.QLabel(title)
    title_lbl.setProperty("cardTitle", True)
    v.addWidget(title_lbl)

    # если контента потенциально много — оборачиваем в ScrollArea
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setStyleSheet(f"QScrollArea {{ background: transparent; }}")
    scroll.setWidget(inner)
    v.addWidget(scroll, 1)

    shadow = QGraphicsDropShadowEffect(card)
    shadow.setBlurRadius(28)
    shadow.setOffset(0, 6)
    shadow.setColor(QtGui.QColor(0, 0, 0, 140))
    card.setGraphicsEffect(shadow)

    return card


def apply_expand_policy(w: QtWidgets.QWidget, *, w_stretch=False):
    sp = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Expanding if w_stretch else QtWidgets.QSizePolicy.Preferred,
        QtWidgets.QSizePolicy.Expanding
    )
    w.setSizePolicy(sp)


# ======================================================
# 2) SharedState (лайт; совместим с World3DView/engine3d)
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
# 3) Упрощённый 3D-вью на QOpenGLWidget
# ======================================
class World3DView(QOpenGLWidget):
    requestSetGoal = QtCore.Signal(str, float, float)  # agent_id, x, z

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

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)  # чуть плавнее вращение/пан

    def _camera_position(self) -> Tuple[float, float, float]:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        r = max(10.0, self.distance)
        cx, cz = self.center_x, self.center_z
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)
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
        # сохранить матрицы для пикинга
        self._mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport = glGetIntegerv(GL_VIEWPORT)

    def _frame_tick(self):
        dt = self._last_frame_time.elapsed() / 1000.0
        self._last_frame_time.restart()
        self.engine.update(dt)
        self.update()

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
        wz = z0 + (z1 - z0) * t   # корректный расчёт Z-проекции на плоскость y=0
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

        # RMB — орбита
        if self._btns & Qt.RightButton:
            self.yaw_deg += delta.x() * 0.25
            self.pitch_deg = max(15.0, min(80.0, self.pitch_deg - delta.y() * 0.25))
            self.update()
            return

        # MMB — пан
        if self._btns & Qt.MiddleButton:
            pan_speed = max(0.1, self.distance * 0.01)
            yaw = math.radians(self.yaw_deg)
            right_x = math.cos(yaw)
            right_z = math.sin(yaw)
            fwd_x = -math.sin(yaw)
            fwd_z = math.cos(yaw)
            self.center_x -= (right_x * delta.x() + fwd_x * delta.y()) * pan_speed * 0.02
            self.center_z -= (right_z * delta.x() + fwd_z * delta.y()) * pan_speed * 0.02
            self._clamp_center()
            self.update()
            return

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
                self._clamp_center()
                self.update()
        elif e.key() == Qt.Key_Tab:
            self.shared.cycle_next_agent()
            sel = self.shared.get_selected_agent_id()
            if sel and sel in self.engine.agents:
                ent = self.engine.agents[sel]
                self.center_x = ent.transform.pos.x
                self.center_z = ent.transform.pos.z
                self._clamp_center()
                self.update()
        else:
            super().keyPressEvent(e)


# ====================================================
# 4) Мост: trainer.world -> engine (для 3D синхронизации)
# ====================================================
class TrainerToEngineBridge(QtCore.QObject):
    """Слушает trainer.world_changed и толкает снапшот в SharedState/Engine."""
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
# 5) Главное окно: 3 колонки (Stats | 3D | Brain)
# ====================================================
class CombinedMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-Matrix Lab — 3D + Mind Trainer")
        self.resize(1600, 900)

        # Глобальный фон и шрифт
        self.setStyleSheet(f"QMainWindow {{ background-color:{COL_BG}; color:{COL_TEXT}; }} QLabel {{ color:{COL_TEXT}; }}")
        font = QtGui.QFont(APP_FONT, 10)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.setFont(font)

        # 1) 3D движок
        self.engine = MiniMatrixEngine()
        # окружение (low poly)
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

        # 3) Shared + 3D виджет с рамкой-карточкой
        self.shared = SharedState(self.engine)
        self.view3d = World3DView(self.shared)
        apply_expand_policy(self.view3d, w_stretch=True)
        self.view3d.setMinimumSize(980, 720)
        self.view3d.requestSetGoal.connect(self._on_set_goal_from_3d)

        frame3d = QtWidgets.QFrame()
        frame3d.setStyleSheet(FRAME3D_STYLE)
        frame3d.setProperty("card", True)
        lay3d = QtWidgets.QVBoxLayout(frame3d)
        lay3d.setContentsMargins(10, 10, 10, 10)
        lay3d.setSpacing(8)

        help_lbl = QtWidgets.QLabel(
            "ЛКМ — выбрать • ПКМ — приказать • RMB — орбита • MMB — пан • колесо — зум • Tab — след. • F — фокус • R — сброс • Ctrl+W — волки"
        )
        help_lbl.setStyleSheet(HELP_PILL_STYLE)
        help_lbl.setWordWrap(True)
        lay3d.addWidget(help_lbl, 0)
        lay3d.addWidget(self.view3d, 1)

        # Тень под 3D
        shadow3d = QGraphicsDropShadowEffect(frame3d)
        shadow3d.setBlurRadius(30)
        shadow3d.setOffset(0, 6)
        shadow3d.setColor(QtGui.QColor(0, 0, 0, 140))
        frame3d.setGraphicsEffect(shadow3d)

        # 4) панели тренера в карточках
        self.statsWidget = TrainerStatsWidget(self.trainer)
        self.statsWidget.setMinimumWidth(360)
        self.statsWidget.setMaximumWidth(480)
        stats_card = make_card("Панель эволюции / метрик", self.statsWidget)

        self.brainWidget = AgentBrainWidget(self.trainer)
        self.brainWidget.setMinimumWidth(360)
        self.brainWidget.setMaximumWidth(520)
        brain_card = make_card("Инспектор мозга агента", self.brainWidget)

        # синхронизация выбора
        self.shared.updated.connect(self._sync_selection_into_brain_panel)
        self.brainWidget.comboAgent.currentIndexChanged.connect(self._sync_selection_from_brain_panel)

        # 5) сплиттер: Stats | 3D | Brain
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(SPLITTER_STYLE)
        splitter.addWidget(stats_card)
        splitter.addWidget(frame3d)
        splitter.addWidget(brain_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([420, 900, 420])

        # 6) центральный виджет
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)
        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # 7) статусбар
        self.statusBar().setStyleSheet(STATUSBAR_STYLE)
        self.statusBar().showMessage("Ready")

        # 8) мост мира в 3D
        self.bridge = TrainerToEngineBridge(self.trainer, self.shared, self)

        # 9) хоткей волков
        act_spawn_wolves = QtGui.QAction("Spawn wolves", self)
        act_spawn_wolves.setShortcut("Ctrl+W")
        act_spawn_wolves.triggered.connect(self._spawn_wolves)
        self.addAction(act_spawn_wolves)

        # 10) начальный пуш снапшота
        self.bridge._push_snapshot()

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
            self.statusBar().showMessage(f"Spawned 3 wolves near ({cx:.1f},{cy:.1f})", 3000)
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
                    ag = a
                    break
        if ag is None:
            return
        x = max(0.0, min(w.width, x))
        z = max(0.0, min(w.height, z))
        if hasattr(ag, "set_goal"):
            try:
                ag.set_goal(x, z)
            except Exception:
                setattr(ag, "goal_x", x)
                setattr(ag, "goal_y", z)
        else:
            setattr(ag, "goal_x", x)
            setattr(ag, "goal_y", z)

        if hasattr(w, "add_chat_line"):
            try:
                w.add_chat_line(f"[observer] {getattr(ag,'name',agent_id)} → goal=({x:.1f},{z:.1f})")
            except Exception:
                pass

        self.bridge._push_snapshot()
        self.statusBar().showMessage(f"goal for {agent_id} → ({x:.1f}, {z:.1f})", 3000)

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


# ====================================================
# 6) Сборка снапшота для движка 3D
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
# 7) main
# =========================
def main():
    app = QtWidgets.QApplication(sys.argv)
    # (Qt6 автоматически включает HiDPI; AA_UseHighDpiPixmaps — deprecated, убираем)
    app.setFont(QtGui.QFont(APP_FONT, 10))

    # Лёгкая системная палитра (чуть светлее видимость контролов)
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
