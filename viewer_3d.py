import sys
import math
import json
from typing import Dict, Any, Optional, Tuple, List

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import (
    Qt,
    Signal,
    Slot,
    QUrl,
    QTimer,
    QPointF,
)
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGraphicsDropShadowEffect

# OpenGL utils
from OpenGL.GL import (
    glGetDoublev,
    glGetIntegerv,
    GL_MODELVIEW_MATRIX,
    GL_PROJECTION_MATRIX,
    GL_VIEWPORT,
)
from OpenGL.GLU import gluUnProject

# наш движок
from brain_visualizer import OrganicBrainView
from engine3d import MiniMatrixEngine
from env_lowpoly import build_lowpoly_village  # низкополигональная деревня


# =============================================================================
# 0) Маленькие утилиты UI (шрифты, элайдинг, политики размеров)
# =============================================================================

def mono_font(size: int = 11):
    f = QtGui.QFont()
    f.setFamilies(["JetBrains Mono", "Consolas", "SF Mono", "monospace"])
    f.setPointSize(size)
    return f

def ui_font(size: int = 11, weight: int = QtGui.QFont.Normal):
    f = QtGui.QFont()
    f.setFamilies(["Inter", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
    f.setPointSize(size)
    f.setWeight(weight)
    return f

class ElidedLabel(QtWidgets.QLabel):
    """Лейбл, который автоматически укорачивает строку и показывает тултип с полным текстом."""
    def __init__(self, text:str="", parent=None, mode=Qt.ElideRight):
        super().__init__(text, parent)
        self._full_text = text
        self._mode = mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setToolTip(text)

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        super().setText(text)
        self._update_elide()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        self._update_elide()

    def _update_elide(self):
        fm = self.fontMetrics()
        elided = fm.elidedText(self._full_text, self._mode, max(20, self.width() - 8))
        super().setText(elided)


def make_pill_label(text: str, ok: bool) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lbl.setFont(ui_font(10, QtGui.QFont.DemiBold))
    if ok:
        style = """
            QLabel {
                color:#c9ffe3; background: rgba(20,40,30,0.55);
                border:1px solid #2b6345; padding:6px 10px; border-radius: 12px;
            }"""
    else:
        style = """
            QLabel {
                color:#ffb8b8; background: rgba(50,18,18,0.55);
                border:1px solid #6a2b2b; padding:6px 10px; border-radius: 12px;
            }"""
    lbl.setStyleSheet(style)
    return lbl


# =============================================================================
# 1) WebSocket-клиент с авто-reconnect, ping/pong и логированием ack/error
# =============================================================================

WS_URL = "ws://localhost:8000/ws"


class WebSocketClient(QtCore.QObject):
    world_state_received = Signal(dict)
    connection_status = Signal(bool)
    message_log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.ws = QWebSocket()
        self.ws.errorOccurred.connect(self.on_error)
        self.ws.connected.connect(self.on_connected)
        self.ws.disconnected.connect(self.on_disconnected)
        self.ws.textMessageReceived.connect(self.on_text_message)

        self._url = WS_URL
        self._reconnect_delay_ms = 500

        self._reconnect_timer = QTimer(self)
        self._reconnect_timer.setSingleShot(True)
        self._reconnect_timer.timeout.connect(self._try_reconnect)

        self._ping_timer = QTimer(self)
        self._ping_timer.setInterval(10_000)  # каждые 10 секунд
        self._ping_timer.timeout.connect(self._send_ping)

    def connect(self, url: str = WS_URL):
        self._url = url
        self.ws.open(QUrl(url))

    def _try_reconnect(self):
        self.message_log.emit("[WS] reconnecting…")
        self.ws.open(QUrl(self._url))

    def _schedule_reconnect(self):
        # экспоненциальный backoff до 5с
        self._reconnect_delay_ms = min(self._reconnect_delay_ms * 2, 5000)
        self._reconnect_timer.start(self._reconnect_delay_ms)

    def _reset_backoff(self):
        self._reconnect_delay_ms = 500

    # --- события сокета

    @Slot()
    def on_connected(self):
        self.connection_status.emit(True)
        self.message_log.emit("[WS] connected")
        self._reset_backoff()
        self._ping_timer.start()
        self.ws.sendTextMessage(json.dumps({"type": "subscribe"}))

    @Slot()
    def on_disconnected(self):
        self.connection_status.emit(False)
        self.message_log.emit("[WS] disconnected")
        self._ping_timer.stop()
        self._schedule_reconnect()

    @Slot("QAbstractSocket::SocketError")
    def on_error(self, err):
        self.message_log.emit(f"[WS ERROR] {err}")

    @Slot(str)
    def on_text_message(self, msg: str):
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            self.message_log.emit("[WS WARN] non-JSON message")
            return

        mtype = payload.get("type")
        data = payload.get("data", {})

        if mtype == "world_state":
            self.world_state_received.emit(data)
        elif mtype == "pong":
            pass
        elif mtype == "ack":
            self.message_log.emit(f"[WS ACK] {data}")
        elif mtype == "error":
            self.message_log.emit(f"[WS ERROR msg] {data}")
        else:
            self.message_log.emit(f"[WS {mtype}] {data}")

    # --- команды наружу

    def _send_ping(self):
        try:
            self.ws.sendTextMessage(json.dumps({"type": "ping"}))
        except Exception:
            pass

    def send_set_goal(self, agent_id: str, x: float, y: float):
        try:
            msg = {
                "type": "set_goal",
                "agent_id": agent_id,
                "goal": {"x": x, "y": y},
            }
            self.ws.sendTextMessage(json.dumps(msg))
        except Exception as e:
            self.message_log.emit(f"[WS WARN] can't send set_goal: {e}")


# =============================================================================
# 2) SharedState — единый источник правды для UI
# =============================================================================

class SharedState(QtCore.QObject):
    updated = Signal()

    def __init__(self, engine: MiniMatrixEngine, parent=None):
        super().__init__(parent)
        self.engine = engine

        self.tick: int = 0
        self.world_w: float = 100.0
        self.world_h: float = 100.0

        self.chat_tail: List[str] = []
        self.event_tail: List[Dict[str, Any]] = []

        self.selected_agent_id: Optional[str] = None
        self._connected: bool = False

    # ---- снапшот от сервера

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

    # ---- соединение

    def set_connected(self, ok: bool):
        self._connected = ok
        self.updated.emit()

    def is_connected(self) -> bool:
        return self._connected

    # ---- простые геттеры

    def get_tick(self) -> int:
        return self.tick

    def get_chat_lines(self) -> List[str]:
        return list(self.chat_tail)

    def get_world_events_lines(self) -> List[str]:
        lines: List[str] = []
        for ev in self.event_tail[-200:]:
            if not isinstance(ev, dict):
                lines.append(str(ev))
                continue

            etype = ev.get("type", "?")
            tick = ev.get("tick", "?")

            if etype == "death":
                name = ev.get("name", ev.get("who", "<?>"))
                reason = ev.get("reason", "?")
                lines.append(f"[t={tick}] DEATH: {name} ({reason})")
                continue

            if etype == "command_goal":
                nm = ev.get("name", ev.get("who", "<?>"))
                goal = ev.get("goal", "?")
                lines.append(f"[t={tick}] CMD→{nm}: {goal}")
                continue

            if etype == "animal_death":
                nm = ev.get("name", ev.get("who", "<?>"))
                lines.append(f"[t={tick}] ANIMAL DOWN: {nm}")
                continue

            short = {k: v for k, v in ev.items() if k not in ("tick",)}
            lines.append(f"[t={tick}] {etype}: {short}")

        if not lines and getattr(self.engine, "global_events", None):
            return [str(s) for s in self.engine.global_events[-50:]]

        return lines

    # ---- выбор агента

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
            idx = ids.index(self.selected_agent_id)
            idx = (idx + 1) % len(ids)
        else:
            idx = 0

        self.selected_agent_id = ids[idx]
        self._apply_selection_to_engine()
        self.updated.emit()

    def get_selected_agent_id(self) -> Optional[str]:
        return self.selected_agent_id

    def _apply_selection_to_engine(self):
        for aid, ent in self.engine.agents.items():
            ent.selected = (aid == self.selected_agent_id)

    # ---- сводка по агенту

    def get_selected_agent_debug(self) -> Dict[str, Any]:
        aid = self.selected_agent_id
        if aid is None:
            return {}

        ent = self.engine.agents.get(aid)
        if ent is None:
            return {}

        st = getattr(ent, "public_state", {}) or {}

        name = st.get("name", aid)
        pos = st.get("pos", {})
        goal = st.get("goal", {})
        vel = st.get("vel", {})

        fear = float(st.get("fear", 0.0))
        hp = float(st.get("health", 100.0))
        energy = float(st.get("energy", 100.0))
        hunger = float(st.get("hunger", 0.0))

        age = st.get("age_ticks", 0)
        dz_count = st.get("danger_zones_count", 0)
        hz_known = st.get("hazards_known", 0)
        memory_tail = st.get("memory_tail", [])

        alive_flag = st.get("alive", None)
        if alive_flag is None:
            alive_flag = (hp > 0)

        cause_of_death = (
            st.get("cause_of_death")
            or st.get("death_reason")
            or st.get("death_cause")
            or None
        )

        mind_block = st.get("mind", None)
        if mind_block is None:
            mind_block = st.get("consciousness", {}) or {}
        if mind_block is None:
            mind_block = {}

        mind_survival_score = mind_block.get("survival_score", None)
        mind_drive = mind_block.get("current_drive", None)
        mind_behavior_rules = mind_block.get("behavior_rules", {})
        mind_beliefs = mind_block.get("beliefs", [])
        mind_memory_tail = mind_block.get("memory_tail", [])
        mind_trauma_map = mind_block.get("trauma_map", [])
        mind_curiosity_charge = mind_block.get("curiosity_charge", None)
        mind_last_thought = mind_block.get("last_thought", None)

        cmb = st.get("combat", {}) or {}
        combat_state = cmb.get("state")
        combat_skill = cmb.get("skill")
        combat_enemy_id = cmb.get("enemy_id")
        combat_just_hit = bool(cmb.get("just_hit", False))

        return {
            "id": aid,
            "name": name,
            "pos": {"x": float(pos.get("x", 0.0)), "y": float(pos.get("y", 0.0))},
            "vel": {"x": float(vel.get("x", 0.0)), "y": float(vel.get("y", 0.0))},
            "goal": {"x": float(goal.get("x", 0.0)), "y": float(goal.get("y", 0.0))},
            "fear": fear,
            "health": hp,
            "energy": energy,
            "hunger": hunger,
            "tick": int(self.tick),
            "age_ticks": age,
            "danger_zones_count": dz_count,
            "hazards_known": hz_known,
            "memory_tail": memory_tail,
            "alive": bool(alive_flag),
            "cause_of_death": cause_of_death,
            "mind_block": mind_block,
            "mind_survival_score": mind_survival_score,
            "mind_drive": mind_drive,
            "mind_behavior_rules": mind_behavior_rules,
            "mind_beliefs": mind_beliefs,
            "mind_memory_tail": mind_memory_tail,
            "mind_trauma_map": mind_trauma_map,
            "mind_curiosity_charge": mind_curiosity_charge,
            "mind_last_thought": mind_last_thought,
            "combat_state": combat_state,
            "combat_skill": combat_skill,
            "combat_enemy_id": combat_enemy_id,
            "combat_just_hit": combat_just_hit,
            "generation": int(st.get("generation", 0) or 0),
            "parents": list(st.get("parents", []) or []),
            "lineage_role": str(st.get("lineage_role", "balanced") or "balanced"),
            "tags": list(st.get("tags", []) or []),
        }


# =============================================================================
# 3) 3D-вьювер на QOpenGLWidget + HUD
# =============================================================================

class World3DView(QOpenGLWidget):
    """
    Окно с миром.
    Управляет орбитальной камерой и кликами по земле:
      - ЛКМ = выбрать агента
      - ПКМ = приказать выбранному куда идти
      - RMB drag = крутить камеру
      - MMB drag = панорамить
      - колесо = зум
      - F = сфокусироваться на выбранном агенте
      - R = сброс камеры в центр карты
      - Tab = выбрать следующего агента
    """
    requestSetGoal = Signal(str, float, float)  # agent_id, x, z

    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared
        self.engine = shared.engine

        self.shared.updated.connect(self.update)

        # орбитальная камера
        self.center_x = 50.0
        self.center_z = 50.0
        self.distance = 140.0
        self.yaw_deg = -135.0
        self.pitch_deg = 40.0
        self.fov_deg = 45.0

        # HUD
        self._fps = 0.0
        self._show_help = True
        self._hud_font = mono_font(10)
        self._help_font = ui_font(10)

        # follow selected
        self._follow_selected = False
        self._follow_lerp = 0.12  # плавность слежения

        # picking
        self._mv = None
        self._proj = None
        self._viewport = None

        # мышь / ввод
        self._last_mouse_pos: Optional[QPointF] = None
        self._btns = Qt.NoButton

        # таймер для апдейта физики/анимации движка (60 FPS)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._frame_tick)
        self._timer.start()

        self._last_frame_time = QtCore.QElapsedTimer()
        self._last_frame_time.start()

        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # API: включить/выключить автоследование
    def set_follow_selected(self, enabled: bool):
        self._follow_selected = bool(enabled)

    # ---- камера

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

    def _clamp_camera_center(self):
        self.center_x = max(0.0, min(self.shared.world_w, self.center_x))
        self.center_z = max(0.0, min(self.shared.world_h, self.center_z))

    # ---- OpenGL lifecycle

    def initializeGL(self):
        pass

    def resizeGL(self, w, h):
        pass

    def paintGL(self):
        cam_x, cam_y, cam_z = self._camera_position()

        self.engine.setup_viewport_and_camera(
            w=self.width(),
            h=self.height(),
            cam_pos=(cam_x, cam_y, cam_z),
            cam_look=(self.center_x, 0.0, self.center_z),
            fov_deg=self.fov_deg,
        )
        self.engine.render_opengl()

        # сохраняем матрицы/viewport для raycast по полу
        self._mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport = glGetIntegerv(GL_VIEWPORT)

        # --- HUD поверх OpenGL
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        def pill(x, y, w, h, alpha=140):
            rect = QtCore.QRectF(x, y, w, h)
            bg = QtGui.QColor(16, 16, 24, alpha)
            pen = QtGui.QPen(QtGui.QColor(80, 80, 90, 160))
            painter.setPen(pen)
            painter.setBrush(bg)
            painter.drawRoundedRect(rect, 10, 10)

        # верхний левый блок (FPS / tick / conn)
        hud_text = [
            f"FPS: {self._fps:4.1f}",
            f"Tick: {self.shared.get_tick():d}",
            "WS: " + ("● Connected" if self.shared.is_connected() else "○ Disconnected"),
        ]

        painter.setFont(self._hud_font)
        metrics = QtGui.QFontMetrics(painter.font())
        width_px = max(metrics.horizontalAdvance(t) for t in hud_text) + 16
        height_px = metrics.height() * len(hud_text) + 14

        pill(10, 10, width_px, height_px, alpha=130)
        painter.setPen(QtGui.QColor("#d7d7e0"))
        y = 10 + metrics.ascent() + 6
        for i, t in enumerate(hud_text):
            if i == 2:
                dot_color = QtGui.QColor(80, 200, 120) if self.shared.is_connected() else QtGui.QColor(180, 80, 80)
                painter.setPen(QtGui.QPen(dot_color))
                painter.drawText(18, y, "●")
                painter.setPen(QtGui.QPen(QtGui.QColor("#d7d7e0")))
                painter.drawText(32, y, t[3:])
            else:
                painter.drawText(18, y, t)
            y += metrics.height()

        # нижняя подсказка
        if self._show_help:
            painter.setFont(ui_font(10))
            help_str = "ЛКМ=выбрать • ПКМ=приказать • RMB drag=орбита • MMB drag=пан • колесо=зум • Tab=след. • F=фокус • R=сброс • H=hide help"
            m2 = QtGui.QFontMetrics(painter.font())
            w2 = m2.horizontalAdvance(help_str) + 24
            h2 = m2.height() + 12
            pill(10, self.height() - h2 - 10, min(w2, self.width() - 20), h2, alpha=120)
            painter.setPen(QtGui.QPen(QtGui.QColor("#cbd1ff")))
            painter.drawText(18, self.height() - 10 - m2.descent(), help_str)

        # вспышка «HIT!»
        sel = self.shared.get_selected_agent_id()
        if sel:
            info = self.shared.get_selected_agent_debug()
            if info.get("combat_just_hit"):
                flash_text = "HIT!"
                painter.setFont(ui_font(13, QtGui.QFont.Black))
                fm = QtGui.QFontMetrics(painter.font())
                tw = fm.horizontalAdvance(flash_text) + 20
                th = fm.height() + 12
                pill(self.width() - tw - 14, 10, tw, th, alpha=180)
                painter.setPen(QtGui.QPen(QtGui.QColor("#ff8b8b")))
                painter.drawText(self.width() - tw + 6, 10 + fm.ascent() + 6, flash_text)

        painter.end()

    def _frame_tick(self):
        dt = self._last_frame_time.elapsed() / 1000.0  # сек
        self._last_frame_time.restart()

        if dt > 0:
            cur_fps = 1.0 / dt
            self._fps = (self._fps * 0.85) + (cur_fps * 0.15)

        if self._follow_selected:
            sel = self.shared.get_selected_agent_id()
            if sel and sel in self.engine.agents:
                ent = self.engine.agents[sel]
                tx = ent.transform.pos.x
                tz = ent.transform.pos.z
                self.center_x += (tx - self.center_x) * self._follow_lerp
                self.center_z += (tz - self.center_z) * self._follow_lerp
                self._clamp_camera_center()

        self.engine.update(dt)
        self.update()

    # ---- picking и ввод мыши

    def _screen_to_world_plane(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self._mv is None or self._proj is None or self._viewport is None:
            return None

        gl_y = self._viewport[3] - y  # инверсия Y Qt -> OpenGL

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
        return (float(wx), float(wz))

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
                if aid is not None:
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

        if self._btns & Qt.RightButton:
            self.yaw_deg += delta.x() * 0.25
            self.pitch_deg = max(15.0, min(80.0, self.pitch_deg - delta.y() * 0.25))
            self.update()
            return

        if self._btns & Qt.MiddleButton:
            pan_speed = max(0.1, self.distance * 0.01)
            yaw = math.radians(self.yaw_deg)

            right_x = math.cos(yaw)
            right_z = math.sin(yaw)
            fwd_x = -math.sin(yaw)
            fwd_z = math.cos(yaw)

            self.center_x -= (right_x * delta.x() + fwd_x * delta.y()) * pan_speed * 0.02
            self.center_z -= (right_z * delta.x() + fwd_z * delta.y()) * pan_speed * 0.02
            self._clamp_camera_center()
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
            self.center_x = max(0.0, min(self.shared.world_w, self.shared.world_w * 0.5))
            self.center_z = max(0.0, min(self.shared.world_h, self.shared.world_h * 0.5))
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
                self._clamp_camera_center()
                self.update()

        elif e.key() == Qt.Key_Tab:
            self.shared.cycle_next_agent()
            sel = self.shared.get_selected_agent_id()
            if sel and sel in self.engine.agents:
                ent = self.engine.agents[sel]
                self.center_x = ent.transform.pos.x
                self.center_z = ent.transform.pos.z
                self._clamp_camera_center()
                self.update()

        elif e.key() == Qt.Key_H:
            self._show_help = not self._show_help
            self.update()

        else:
            super().keyPressEvent(e)


# =============================================================================
# 4) Правые панели (Monitor / Chat / Mind / Events)
# =============================================================================

class AgentMonitorWidget(QtWidgets.QWidget):
    """
    Монитор агента: аккуратные полоски, элайдинг длинного текста,
    кнопки с ховерами и компактные цвета.
    """
    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        self.lblWorld = ElidedLabel("Tick: 0 | Disconnected")
        self.lblWorld.setFont(ui_font(11, QtGui.QFont.Medium))
        self.lblWorld.setStyleSheet("color:#bfc3d9;")
        lay.addWidget(self.lblWorld)

        self.lblName = ElidedLabel("Агент: —")
        self.lblName.setFont(ui_font(14, QtGui.QFont.DemiBold))
        self.lblName.setStyleSheet("color:#eef;")
        lay.addWidget(self.lblName)

        # Combat badge
        self.lblCombat = ElidedLabel("Combat: –")
        self.lblCombat.setAlignment(Qt.AlignLeft)
        self.lblCombat.setFont(mono_font(11))
        self.lblCombat.setStyleSheet("""
            QLabel {
                color:#eaeaf5;
                background: rgba(40,40,55,0.6);
                border:1px solid #4a4a5f;
                border-radius: 10px;
                padding:4px 8px;
            }
        """)
        lay.addWidget(self.lblCombat)

        # прогресс-полоски
        def _mkbar(fmt_text: str, grad_from: str, grad_to: str) -> QtWidgets.QProgressBar:
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            bar.setFormat(fmt_text + ": %v")
            bar.setFont(ui_font(10))
            bar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            bar.setFixedHeight(18)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: rgba(32,32,42,0.72);
                    border: 1px solid #4a4a5f;
                    border-radius: 7px;
                    color: #dcdcf0;
                    text-align: center;
                    padding: 0 4px;
                }}
                QProgressBar::chunk {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                                stop:0 {grad_from}, stop:1 {grad_to});
                    border-radius: 7px;
                }}
            """)
            return bar

        self.hpBar     = _mkbar("HP",     "#3fb57f", "#66ddaa")
        self.fearBar   = _mkbar("Fear",   "#e08a35", "#ffb66b")
        self.energyBar = _mkbar("Energy", "#5e9ee6", "#8cc9ff")
        self.hungerBar = _mkbar("Hunger", "#d0524b", "#ff7b73")

        for b in (self.hpBar, self.fearBar, self.energyBar, self.hungerBar):
            lay.addWidget(b)

        # таблица с координатами и статусами
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        small_norm = "color:#b7bdd8; font-family:monospace; font-size:11px;"
        small_dang = "color:#ff9090; font-family:monospace; font-size:11px; font-weight:700;"

        self.lblPos = ElidedLabel("pos: (0,0)")
        self.lblGoal = ElidedLabel("goal: (0,0)")
        self.lblAge = ElidedLabel("age: 0 ticks")
        self.lblDrive = ElidedLabel("drive: –")
        self.lblSurvival = ElidedLabel("survival: –")
        self.lblState = ElidedLabel("state: –")
        self.lblDeathCause = ElidedLabel("death: –")

        for w in (self.lblPos, self.lblGoal, self.lblAge, self.lblDrive, self.lblSurvival, self.lblState, self.lblDeathCause):
            w.setStyleSheet(small_norm)
            w.setFont(mono_font(11))

        grid.addWidget(self.lblPos,         0, 0)
        grid.addWidget(self.lblGoal,        1, 0)
        grid.addWidget(self.lblAge,         2, 0)
        grid.addWidget(self.lblDrive,       3, 0)
        grid.addWidget(self.lblSurvival,    4, 0)
        grid.addWidget(self.lblState,       5, 0)
        grid.addWidget(self.lblDeathCause,  6, 0)

        lay.addLayout(grid)

        # быстрые кнопки
        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(8)
        self.btnCenter = QtWidgets.QPushButton("Фокус (F)")
        self.btnNext = QtWidgets.QPushButton("Следующий (Tab)")
        for b in (self.btnCenter, self.btnNext):
            b.setCursor(Qt.PointingHandCursor)
            b.setFont(ui_font(10, QtGui.QFont.Medium))
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            b.setStyleSheet("""
                QPushButton {
                    background: rgba(40,40,55,0.6);
                    border:1px solid #4a4a5f;
                    color:#e9ecff;
                    padding:8px 12px;
                    border-radius:10px;
                }
                QPushButton:hover { border-color:#6a6a8a; }
                QPushButton:pressed { background: rgba(30,30,45,0.7); }
            """)
        btns.addWidget(self.btnCenter)
        btns.addWidget(self.btnNext)
        lay.addLayout(btns)

        lay.addStretch(1)

        self._styleSmallNormal = small_norm
        self._styleSmallDanger = small_dang

        self.btnCenter.clicked.connect(self._center_on_selected)
        self.btnNext.clicked.connect(self.shared.cycle_next_agent)

        self.shared.updated.connect(self.refresh)

    def _combat_symbol(self, state: Optional[str]) -> str:
        mapping = {"attack": "⚔", "block": "🛡", "evade": "↪", "run": "🏃", "idle": "–"}
        return mapping.get((state or "idle").lower(), "–")

    def _center_on_selected(self):
        # визуальная кнопка, фактический фокус — горячей клавишей F
        pass

    @Slot()
    def refresh(self):
        conn = "Connected" if self.shared.is_connected() else "Disconnected"
        self.lblWorld.setText(f"Tick: {self.shared.get_tick()} | {conn}")

        data = self.shared.get_selected_agent_debug()

        if not data:
            self.lblName.setText("Агент: —")
            self.lblCombat.setText("Combat: –")
            self.hpBar.setValue(0); self.fearBar.setValue(0); self.energyBar.setValue(0); self.hungerBar.setValue(0)
            for w, text in (
                (self.lblPos, "pos: (–,–)"),
                (self.lblGoal, "goal: (–,–)"),
                (self.lblAge, "age: –"),
                (self.lblDrive, "drive: –"),
                (self.lblSurvival, "survival: –"),
                (self.lblState, "state: –"),
                (self.lblDeathCause, "death: –"),
            ):
                w.setText(text)
                w.setStyleSheet(self._styleSmallNormal)
            return

        # базовая инфа
        self.lblName.setText(f"Агент: {data['name']} ({data['id']})")

        hp = max(0, min(100, int(data["health"])))
        fear_pct = max(0, min(100, int(data["fear"] * 100.0)))
        energy_pct = max(0, min(100, int(data.get("energy", 0.0))))
        hunger_pct = max(0, min(100, int(data.get("hunger", 0.0))))

        self.hpBar.setValue(hp)
        self.fearBar.setValue(fear_pct)
        self.energyBar.setValue(energy_pct)
        self.hungerBar.setValue(hunger_pct)

        self.lblPos.setText(f"pos: ({data['pos']['x']:.1f}, {data['pos']['y']:.1f})")
        self.lblGoal.setText(f"goal: ({data['goal']['x']:.1f}, {data['goal']['y']:.1f})")
        self.lblAge.setText(f"age: {data['age_ticks']} ticks")

        drive = data.get("mind_drive", None)
        self.lblDrive.setText(f"drive: {str(drive) if drive is not None else '–'}")

        surv = data.get("mind_survival_score", None)
        surv_text = "–" if surv is None else (f"{float(surv):.2f}" if isinstance(surv, (int, float, str)) else str(surv))
        self.lblSurvival.setText(f"survival: {surv_text}")

        alive_flag = data.get("alive", True)
        cause = data.get("cause_of_death", None)

        panic_bits: List[str] = []
        if data["fear"] >= 0.6: panic_bits.append("PANIC")
        if data["health"] <= 30: panic_bits.append("LOW_HP")
        if data.get("hunger", 0.0) >= 70.0: panic_bits.append("STARVING")
        if data.get("energy", 100.0) <= 30.0: panic_bits.append("EXHAUSTED")

        panic_suffix = (" !" + "!".join(panic_bits)) if panic_bits else ""

        if alive_flag:
            self.lblState.setText(f"state: ALIVE{panic_suffix}")
            self.lblDeathCause.setText("death: –")
            self.lblState.setStyleSheet(self._styleSmallDanger if panic_bits else self._styleSmallNormal)
            self.lblDeathCause.setStyleSheet(self._styleSmallNormal)
        else:
            self.lblState.setText("state: DEAD")
            self.lblDeathCause.setText(f"death: {cause}" if cause else "death: (unknown)")
            self.lblState.setStyleSheet(self._styleSmallDanger)
            self.lblDeathCause.setStyleSheet(self._styleSmallDanger)

        # combat бейдж
        c_state = (data.get("combat_state") or "idle").lower()
        c_sym = self._combat_symbol(c_state)
        c_skill = data.get("combat_skill")
        skill_txt = f"{int(float(c_skill)*100)}%" if isinstance(c_skill, (int, float)) else "–"
        # подсветка при попадании
        hit_style = """
            QLabel { color:#ffecec; background: rgba(60,20,20,0.65);
                     border:1px solid #e05a5a; border-radius: 10px; padding:4px 8px; }"""
        norm_style = """
            QLabel { color:#eaeaf5; background: rgba(40,40,55,0.6);
                     border:1px solid #4a4a5f; border-radius: 10px; padding:4px 8px; }"""
        self.lblCombat.setStyleSheet(hit_style if data.get("combat_just_hit") else norm_style)
        self.lblCombat.setText(f"{c_sym}  Combat: {c_state.upper()}  •  skill {skill_txt}")


class ChatWidget(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.chatView = QtWidgets.QTextEdit()
        self.chatView.setReadOnly(True)
        self.chatView.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.chatView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chatView.setFont(mono_font(11))
        self.chatView.setDocumentTitle("Chat")
        self.chatView.setStyleSheet("""
            QTextEdit {
                background-color: rgba(20,20,30,0.72);
                border: 1px solid #444;
                color: #8fda8f;
                border-radius: 10px;
                padding:6px;
            }
        """)
        lay.addWidget(self.chatView, 1)

        self.shared.updated.connect(self.refresh)

    @Slot()
    def refresh(self):
        lines = self.shared.get_chat_lines()
        self.chatView.setPlainText("\n".join(lines[-400:]))
        self.chatView.moveCursor(QtGui.QTextCursor.End)


class MindWidget(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        self.lblHeader = QtWidgets.QLabel("Мысленная активность выбранного агента")
        self.lblHeader.setFont(ui_font(12, QtGui.QFont.DemiBold))
        self.lblHeader.setStyleSheet("color:#eee;")
        outer.addWidget(self.lblHeader)

        self.liveBrain = OrganicBrainView()
        self.liveBrain.setMinimumHeight(340)
        outer.addWidget(self.liveBrain, 0)

        # краткая статистика
        assoc_frame = QtWidgets.QFrame()
        assoc_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(18,18,26,0.8);
                border: 1px solid #333;
                border-radius: 12px;
            }
            QLabel {
                color:#c8cbe2;
                font-size:11px;
                font-family:monospace;
            }
        """)
        assoc_lay = QtWidgets.QGridLayout(assoc_frame)
        assoc_lay.setContentsMargins(10, 10, 10, 10)
        assoc_lay.setHorizontalSpacing(10)
        assoc_lay.setVerticalSpacing(6)

        self.lblDangerZones = ElidedLabel("danger_zones_count: –")
        self.lblHazardsKnown = ElidedLabel("hazards_known: –")
        self.lblDriveSummary = ElidedLabel("drive: –")
        self.lblSurvivalScore = ElidedLabel("survival_score: –")

        for row, w in enumerate((self.lblDangerZones, self.lblHazardsKnown, self.lblDriveSummary, self.lblSurvivalScore)):
            w.setFont(mono_font(11))
            assoc_lay.addWidget(w, row, 0)

        outer.addWidget(assoc_frame, 0)

        # правила и убеждения
        brain_frame = QtWidgets.QFrame()
        brain_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(18,18,26,0.8);
                border: 1px solid #333;
                border-radius: 12px;
            }
            QTextEdit {
                background-color: rgba(10,10,18,0.84);
                border: 1px solid #444;
                color: #d0ffd0;
                font-family: monospace;
                font-size: 11px;
                border-radius: 10px;
                padding:6px;
            }
            QLabel {
                color:#e0e3ff;
                font-size:12px;
                font-weight:600;
            }
        """)
        brain_lay = QtWidgets.QVBoxLayout(brain_frame)
        brain_lay.setContentsMargins(10, 10, 10, 10)
        brain_lay.setSpacing(8)

        self.lblRulesTitle = QtWidgets.QLabel("Текущие поведенческие правила:")
        self.lblRulesTitle.setFont(ui_font(12, QtGui.QFont.Medium))
        self.rulesView = QtWidgets.QTextEdit()
        self.rulesView.setReadOnly(True)
        self.rulesView.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.rulesView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.rulesView.setFont(mono_font(11))

        self.lblBeliefTitle = QtWidgets.QLabel("Активные убеждения:")
        self.lblBeliefTitle.setFont(ui_font(12, QtGui.QFont.Medium))
        self.beliefsView = QtWidgets.QTextEdit()
        self.beliefsView.setReadOnly(True)
        self.beliefsView.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.beliefsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.beliefsView.setFont(mono_font(11))

        brain_lay.addWidget(self.lblRulesTitle)
        brain_lay.addWidget(self.rulesView, 0)
        brain_lay.addWidget(self.lblBeliefTitle)
        brain_lay.addWidget(self.beliefsView, 1)

        outer.addWidget(brain_frame, 1)

        # память
        mem_frame = QtWidgets.QFrame()
        mem_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(18,18,26,0.8);
                border: 1px solid #333;
                border-radius: 12px;
            }
            QTextEdit {
                background-color: rgba(10,10,18,0.84);
                border: 1px solid #444;
                color: #d0d0ff;
                font-family: monospace;
                font-size: 11px;
                border-radius: 10px;
                padding:6px;
            }
            QLabel {
                color:#e0e3ff;
                font-size:12px;
                font-weight:600;
            }
        """)
        mem_lay = QtWidgets.QVBoxLayout(mem_frame)
        mem_lay.setContentsMargins(10, 10, 10, 10)
        mem_lay.setSpacing(8)

        self.lblMemTitle = QtWidgets.QLabel("Последние мысли / опыт тела:")
        self.lblMemTitle.setFont(ui_font(12, QtGui.QFont.Medium))
        self.memView = QtWidgets.QTextEdit()
        self.memView.setReadOnly(True)
        self.memView.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.memView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.memView.setFont(mono_font(11))

        mem_lay.addWidget(self.lblMemTitle)
        mem_lay.addWidget(self.memView, 1)

        outer.addWidget(mem_frame, 1)
        outer.addStretch(1)

        self.shared.updated.connect(self.refresh)

    @Slot()
    def refresh(self):
        data = self.shared.get_selected_agent_debug()
        if not data:
            self.liveBrain.clear()
            self.lblDangerZones.setText("danger_zones_count: –")
            self.lblHazardsKnown.setText("hazards_known: –")
            self.lblDriveSummary.setText("drive: –")
            self.lblSurvivalScore.setText("survival_score: –")
            self.rulesView.setPlainText("(нет данных)")
            self.beliefsView.setPlainText("(нет данных)")
            self.memView.setPlainText("(агент не выбран)")
            return

        self.liveBrain.update_from_debug(data, tick=self.shared.get_tick())
        self.lblDangerZones.setText(f"danger_zones_count: {data.get('danger_zones_count', 0)}")
        self.lblHazardsKnown.setText(f"hazards_known: {data.get('hazards_known', 0)}")
        drive = data.get("mind_drive", "–")
        self.lblDriveSummary.setText(f"drive: {drive}")

        surv = data.get("mind_survival_score", None)
        surv_text = "–" if surv is None else (f"{float(surv):.2f}" if isinstance(surv, (int, float, str)) else str(surv))
        self.lblSurvivalScore.setText(f"survival_score: {surv_text}")

        rules = data.get("mind_behavior_rules", {})
        if isinstance(rules, dict) and rules:
            self.rulesView.setPlainText("\n".join(f"{k} = {v}" for k, v in rules.items()))
        else:
            self.rulesView.setPlainText("(нет данных о правилах поведения)")

        beliefs = data.get("mind_beliefs", [])
        if isinstance(beliefs, list) and beliefs:
            lines = []
            for b in beliefs[-200:]:
                if isinstance(b, dict):
                    cond = b.get("if", "?")
                    concl = b.get("then", "?")
                    strength = b.get("strength", "?")
                    lines.append(f"IF {cond} THEN {concl} [{strength}]")
                else:
                    lines.append(str(b))
            self.beliefsView.setPlainText("\n".join(lines))
        else:
            self.beliefsView.setPlainText("(нет активных убеждений)")

        tail = data.get("mind_memory_tail") or data.get("memory_tail") or []
        mem_lines: List[str] = []
        if isinstance(tail, list):
            for ev in tail[-400:]:
                if not isinstance(ev, dict):
                    mem_lines.append(str(ev)); continue
                etype = ev.get("type", "?"); tick = ev.get("tick", "?")
                lvl = ev.get("level", ""); actor = ev.get("actor", ""); pos = ev.get("pos", None); edata = ev.get("data", {})
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    try: pos_str = f"@({float(pos[0]):.1f},{float(pos[1]):.1f})"
                    except Exception: pos_str = f"@{pos}"
                else:
                    pos_str = ""
                actor_str = f"[{actor}]" if actor else ""
                lvl_str = f"[{lvl.upper()}]" if lvl else ""
                mem_lines.append(f"[t={tick}]{lvl_str}{actor_str} {etype}{pos_str}: {edata}")

        self.memView.setPlainText("\n".join(mem_lines) if mem_lines else "(нет свежей памяти)")


class WorldEventsWidget(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.eventsView = QtWidgets.QTextEdit()
        self.eventsView.setReadOnly(True)
        self.eventsView.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.eventsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.eventsView.setFont(mono_font(11))
        self.eventsView.setStyleSheet("""
            QTextEdit {
                background-color: rgba(20,20,30,0.72);
                border: 1px solid #444;
                color: #ffd28f;
                border-radius: 10px;
                padding:6px;
            }
        """)
        lay.addWidget(self.eventsView, 1)

        self.shared.updated.connect(self.refresh)

    @Slot()
    def refresh(self):
        lines = self.shared.get_world_events_lines()
        self.eventsView.setPlainText("\n".join(lines[-400:]))
        self.eventsView.moveCursor(QtGui.QTextCursor.End)


class RightPanel(QtWidgets.QWidget):
    def __init__(self, shared: SharedState, parent=None):
        super().__init__(parent)
        self.shared = shared

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Вкладки сразу в прокручиваемой области, чтобы ничего не резалось на меньших ширинах
        wrapper = QtWidgets.QFrame()
        wrapper.setStyleSheet("QFrame { background-color: transparent; }")
        wrap_lay = QtWidgets.QVBoxLayout(wrapper)
        wrap_lay.setContentsMargins(8, 8, 8, 8)
        wrap_lay.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #2c2c38;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #14141d, stop:1 #0f0f16);
                border-radius: 12px;
            }
            QTabBar::tab {
                background-color: #232330;
                color:#cfd3ee;
                padding:7px 12px;
                margin-right:4px;
                border-top-left-radius:8px;
                border-top-right-radius:8px;
                font-size:11px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color:#343447;
                color:#fff;
            }
            QWidget { background-color: transparent; color:#cfd3ee; }
        """)

        self.monitor_tab = AgentMonitorWidget(shared)
        self.chat_tab = ChatWidget(shared)
        self.mind_tab = MindWidget(shared)
        self.events_tab = WorldEventsWidget(shared)

        self.tabs.addTab(self.monitor_tab, "Monitor")
        self.tabs.addTab(self.chat_tab, "Chat")
        self.tabs.addTab(self.mind_tab, "Mind")
        self.tabs.addTab(self.events_tab, "Events")

        wrap_lay.addWidget(self.tabs, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(wrapper)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")

        layout.addWidget(scroll, 1)

        # тень
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        self.setGraphicsEffect(shadow)


# =============================================================================
# 5) Главное окно
# =============================================================================

class MainWindow3D(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Мини-Матрица — 3D наблюдатель (God Console)")
        self.resize(1560, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0b12; color: #eef; }
            QLabel { color: #eef; }
            QSplitter::handle { background: #1a1a23; width: 6px; border-radius: 3px; }
            QStatusBar { color:#c7cbe4; background-color:#11121a; }
            QToolTip { color:#eaeaf5; background:#151522; border:1px solid #34344a; }
        """)

        # движок
        self.engine = MiniMatrixEngine()

        # окружение
        try:
            village_meshes = build_lowpoly_village(
                world_w=self.engine.world.width,
                world_h=self.engine.world.height,
            )
        except TypeError:
            village_meshes = build_lowpoly_village(
                self.engine.world.width,
                self.engine.world.height,
            )
        if hasattr(self.engine, "load_static_environment"):
            try:
                self.engine.load_static_environment(village_meshes)
            except Exception as e:
                print("[env] load_static_environment failed:", e)

        # shared
        self.shared = SharedState(self.engine)

        # WS клиент
        self.ws_client = WebSocketClient()
        self.ws_client.world_state_received.connect(self.shared.update_from_snapshot)
        self.ws_client.connection_status.connect(self.on_connection_status)
        self.ws_client.message_log.connect(self.on_message_log)

        # 3D-вью
        self.view3d = World3DView(self.shared)
        self.view3d.setMinimumSize(900, 720)
        self.view3d.requestSetGoal.connect(self.on_request_set_goal)
        self.view3d.setFocus()

        # рамка вокруг 3D
        frame3d = QtWidgets.QFrame()
        frame3d.setFrameShape(QtWidgets.QFrame.StyledPanel)
        frame3d.setStyleSheet("""
            QFrame {
                background: #000;
                border: 1px solid #2b2b37;
                border-radius: 14px;
            }
        """)
        lay3d = QtWidgets.QVBoxLayout(frame3d)
        lay3d.setContentsMargins(10, 10, 10, 10)
        lay3d.setSpacing(8)

        # верхняя панель действий (пилюли)
        topbar = QtWidgets.QHBoxLayout()
        topbar.setSpacing(8)

        self.connPill = make_pill_label("● Disconnected", ok=False)

        self.autoFollow = QtWidgets.QCheckBox("Автоследование")
        self.autoFollow.setChecked(False)
        self.autoFollow.stateChanged.connect(lambda s: self.view3d.set_follow_selected(bool(s)))
        self.autoFollow.setStyleSheet("""
            QCheckBox { color:#cfd3ee; }
            QCheckBox::indicator { width:16px; height:16px; }
        """)
        self.autoFollow.setFont(ui_font(10))

        self.helpToggle = QtWidgets.QCheckBox("Подсказки (H)")
        self.helpToggle.setChecked(True)
        self.helpToggle.stateChanged.connect(lambda s: setattr(self.view3d, "_show_help", bool(s)))
        self.helpToggle.setStyleSheet(self.autoFollow.styleSheet())
        self.helpToggle.setFont(ui_font(10))

        topbar.addWidget(self.connPill, 0, Qt.AlignLeft)
        topbar.addStretch(1)
        topbar.addWidget(self.autoFollow, 0, Qt.AlignRight)
        topbar.addWidget(self.helpToggle, 0, Qt.AlignRight)

        lay3d.addLayout(topbar)
        lay3d.addWidget(self.view3d, 1)

        # правая панель
        self.right_panel = RightPanel(self.shared)
        self.right_panel.setMinimumWidth(420)
        self.right_panel.setMaximumWidth(560)

        # сплиттер
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(frame3d)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1150, 410])

        # центральная сцена
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # тень под 3D блок
        shadow3d = QGraphicsDropShadowEffect(self)
        shadow3d.setBlurRadius(30)
        shadow3d.setOffset(0, 6)
        shadow3d.setColor(QtGui.QColor(0, 0, 0, 140))
        frame3d.setGraphicsEffect(shadow3d)

        outer.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # статусбар
        self.statusBar().showMessage("Ready", 1500)

        # подключаемся к серверу
        self.ws_client.connect(WS_URL)

    @Slot(bool)
    def on_connection_status(self, ok: bool):
        self.shared.set_connected(ok)
        if ok:
            self.connPill.setText("● Connected")
            self.connPill.setStyleSheet("""
                QLabel {
                    color:#c9ffe3; background: rgba(20,40,30,0.55);
                    border:1px solid #2b6345; padding:6px 10px; border-radius: 12px;
                    font-weight:600;
                }
            """)
        else:
            self.connPill.setText("● Disconnected")
            self.connPill.setStyleSheet("""
                QLabel {
                    color:#ffb8b8; background: rgba(50,18,18,0.55);
                    border:1px solid #6a2b2b; padding:6px 10px; border-radius: 12px;
                    font-weight:600;
                }
            """)
        self.statusBar().showMessage("Connected" if ok else "Disconnected", 2500)

    @Slot(str)
    def on_message_log(self, text: str):
        self.statusBar().showMessage(text, 4000)

    @Slot(str, float, float)
    def on_request_set_goal(self, agent_id: str, x: float, z: float):
        x = max(0.0, min(self.shared.world_w, x))
        z = max(0.0, min(self.shared.world_h, z))
        self.ws_client.send_set_goal(agent_id, x, z)
        self.statusBar().showMessage(f"goal for {agent_id} → ({x:.1f}, {z:.1f})", 3000)


# =============================================================================
# 6) main
# =============================================================================

def main():
    # Qt6 уже с HiDPI по умолчанию, но на всякий:
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(ui_font(10))

    win = MainWindow3D()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
