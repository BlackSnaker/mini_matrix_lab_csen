from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x


def _as_pct(v: Any, default: float = 0.0) -> float:
    x = _as_float(v, default)
    if x <= 1.5:
        x *= 100.0
    return _clamp(x, 0.0, 100.0)


def _belief_strength(b: Any) -> float:
    if isinstance(b, dict):
        return _clamp(_as_float(b.get("strength", 0.0), 0.0), 0.0, 1.0)
    return _clamp(_as_float(getattr(b, "strength", 0.0), 0.0), 0.0, 1.0)


def _drive_activity(drive: str) -> float:
    key = str(drive or "").strip().lower()
    if key in ("explore", "find_food", "combat", "counterattack"):
        return 0.88
    if key in ("heal", "seek_safety", "retreat"):
        return 0.66
    if key in ("stay_with_ally", "rest"):
        return 0.48
    if key in ("idle", "dead", ""):
        return 0.24
    return 0.54


def _level_weight(level: str) -> float:
    key = str(level or "").strip().lower()
    if key == "critical":
        return 1.0
    if key == "warning":
        return 0.7
    return 0.35


def _rule_count(rules: Any) -> int:
    if isinstance(rules, dict):
        return len(rules)
    return 0


def _lang_pack(lang: str) -> Dict[str, str]:
    if lang == "en":
        return {
            "title": "Living Brain",
            "evolution": "Evolution Trace",
            "stage": "Stage",
            "plasticity": "Plasticity",
            "complexity": "Complexity",
            "stress": "Stress",
            "beliefs": "Beliefs",
            "memories": "Memories",
            "hazards": "Hazards",
            "empty": "Select an agent to inspect its evolving brain.",
            "seed": "Seed",
            "forming": "Forming",
            "adaptive": "Adaptive",
            "rewiring": "Rewiring",
            "specialized": "Specialized",
        }
    return {
        "title": "Живой мозг",
        "evolution": "След эволюции",
        "stage": "Стадия",
        "plasticity": "Пластичность",
        "complexity": "Сложность",
        "stress": "Напряжение",
        "beliefs": "Убеждения",
        "memories": "Память",
        "hazards": "Угрозы",
        "empty": "Выберите агента, чтобы увидеть его живой мозг и динамику развития.",
        "seed": "Зарождение",
        "forming": "Формирование",
        "adaptive": "Адаптация",
        "rewiring": "Перестройка",
        "specialized": "Специализация",
    }


@dataclass
class BrainFrame:
    agent_id: str
    agent_name: str
    tick: int
    age_ticks: int
    generation: int
    fear: float
    health: float
    energy: float
    hunger: float
    survival: float
    activity: float
    plasticity: float
    complexity: float
    stress: float
    coherence: float
    curiosity: float
    beliefs_count: int
    memories_count: int
    hazards_count: int
    danger_count: int
    trauma_count: int
    drive: str
    last_thought: str
    belief_strengths: List[float] = field(default_factory=list)
    memory_levels: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BrainSurfaceVertex:
    pos: QtGui.QVector3D
    normal: QtGui.QVector3D
    frontal: float
    parietal: float
    temporal: float
    occipital: float
    limbic: float
    cerebellum: float


def _mix_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = _clamp(t, 0.0, 1.0)
    return QtGui.QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def _with_alpha(color: QtGui.QColor, alpha: int) -> QtGui.QColor:
    out = QtGui.QColor(color)
    out.setAlpha(int(_clamp(alpha, 0, 255)))
    return out


def _extract_brain_frame(info: Dict[str, Any], tick: Optional[int]) -> BrainFrame:
    mind = dict(info.get("mind_block", {}) or {})
    beliefs = list(info.get("mind_beliefs", mind.get("beliefs", [])) or [])
    memories = list(info.get("mind_memory_tail") or info.get("memory_tail") or mind.get("memory_tail", []) or [])
    rules = dict(info.get("mind_behavior_rules", mind.get("behavior_rules", {})) or {})
    trauma_map = list(info.get("mind_trauma_map", mind.get("trauma_map", [])) or [])

    fear = _clamp(_as_float(info.get("fear", 0.0), 0.0), 0.0, 1.0)
    health = _clamp(_as_float(info.get("health", 100.0), 100.0), 0.0, 100.0)
    energy = _as_pct(info.get("energy", 100.0), 100.0)
    hunger = _as_pct(info.get("hunger", 0.0), 0.0)
    survival = _clamp(_as_float(info.get("mind_survival_score", mind.get("survival_score", 0.0)), 0.0), 0.0, 1.0)
    drive = str(info.get("mind_drive", mind.get("current_drive", "idle")) or "idle")
    last_thought = str(info.get("mind_last_thought", mind.get("last_thought", "")) or "")
    age_ticks = int(_as_float(info.get("age_ticks", mind.get("age_ticks", 0)), 0.0))
    generation = int(_as_float(info.get("generation", 0), 0.0))
    hazards = int(_as_float(info.get("hazards_known", 0), 0.0))
    danger = int(_as_float(info.get("danger_zones_count", 0), 0.0))
    curiosity = _clamp(_as_float(info.get("mind_curiosity_charge", mind.get("curiosity_charge", 0.0)), 0.0), 0.0, 1.0)
    beliefs_count = len(beliefs)
    memories_count = len(memories)
    trauma_count = len(trauma_map)

    belief_strengths = [_belief_strength(b) for b in beliefs[:16]]
    if not belief_strengths:
        belief_strengths = [0.22]

    memory_levels: List[str] = []
    critical_weight = 0.0
    for ev in memories[-16:]:
        if isinstance(ev, dict):
            level = str(ev.get("level", "info"))
        else:
            level = str(getattr(ev, "level", "info"))
        memory_levels.append(level)
        critical_weight += _level_weight(level)
    critical_weight = min(1.0, critical_weight / 8.0)

    mean_strength = sum(belief_strengths) / max(1, len(belief_strengths))
    age_norm = min(1.0, age_ticks / 1200.0)
    rule_factor = min(1.0, _rule_count(rules) / 8.0)
    memory_factor = min(1.0, memories_count / 18.0)
    hazards_factor = min(1.0, (hazards + danger) / 16.0)
    trauma_factor = min(1.0, trauma_count / 10.0)

    activity = _clamp(
        0.34 * _drive_activity(drive)
        + 0.20 * mean_strength
        + 0.16 * critical_weight
        + 0.14 * curiosity
        + 0.16 * memory_factor,
        0.0,
        1.0,
    )
    plasticity = _clamp(
        0.28 * (energy / 100.0)
        + 0.22 * (1.0 - hunger / 100.0)
        + 0.20 * (1.0 - fear)
        + 0.18 * curiosity
        + 0.12 * (1.0 - trauma_factor),
        0.0,
        1.0,
    )
    complexity = _clamp(
        0.18
        + 0.24 * min(1.0, beliefs_count / 12.0)
        + 0.18 * rule_factor
        + 0.14 * age_norm
        + 0.12 * min(1.0, generation / 4.0)
        + 0.14 * mean_strength,
        0.0,
        1.0,
    )
    stress = _clamp(
        0.42 * fear
        + 0.18 * (1.0 - survival)
        + 0.16 * (hunger / 100.0)
        + 0.12 * hazards_factor
        + 0.12 * trauma_factor,
        0.0,
        1.0,
    )
    coherence = _clamp(
        0.40 * survival
        + 0.24 * (1.0 - fear)
        + 0.18 * mean_strength
        + 0.18 * (1.0 - stress),
        0.0,
        1.0,
    )

    frame_tick = int(_as_float(tick if tick is not None else info.get("tick", age_ticks), age_ticks))

    return BrainFrame(
        agent_id=str(info.get("id")),
        agent_name=str(info.get("name", info.get("id", "agent"))),
        tick=frame_tick,
        age_ticks=age_ticks,
        generation=generation,
        fear=fear,
        health=health,
        energy=energy,
        hunger=hunger,
        survival=survival,
        activity=activity,
        plasticity=plasticity,
        complexity=complexity,
        stress=stress,
        coherence=coherence,
        curiosity=curiosity,
        beliefs_count=beliefs_count,
        memories_count=memories_count,
        hazards_count=hazards,
        danger_count=danger,
        trauma_count=trauma_count,
        drive=drive,
        last_thought=last_thought,
        belief_strengths=belief_strengths,
        memory_levels=memory_levels,
    )


class OrganicBrainView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

        self._lang = "ru"
        self._current: Optional[BrainFrame] = None
        self._history: Dict[str, Deque[BrainFrame]] = {}
        self._last_signature: Dict[str, Tuple[int, int, int, int, str]] = {}
        self._phase = 0.0
        self._brain_path_cache_key: Optional[Tuple[int, int, int, int]] = None
        self._brain_path_cache: Optional[Tuple[QtGui.QPainterPath, QtGui.QPainterPath]] = None
        self._node_layout_cache_key: Optional[Tuple[int, int, int, int]] = None
        self._node_layout_cache: Optional[List[Tuple[QtCore.QPointF, float, QtGui.QColor]]] = None

        self._anim = QtCore.QTimer(self)
        self._anim.setTimerType(QtCore.Qt.PreciseTimer)
        self._anim.setInterval(33)
        self._anim.timeout.connect(self._on_anim_tick)
        self._anim.start()

    def set_language(self, lang: str) -> None:
        self._lang = "en" if str(lang).lower() == "en" else "ru"
        self.update()

    def clear(self) -> None:
        self._current = None
        self.update()

    def update_from_debug(self, info: Dict[str, Any], *, tick: Optional[int] = None) -> None:
        if not info or not info.get("id"):
            self.clear()
            return

        frame = self._extract_frame(info, tick)
        history = self._history.setdefault(frame.agent_id, deque(maxlen=180))
        sig = (
            int(frame.tick),
            int(frame.age_ticks),
            int(frame.beliefs_count),
            int(frame.memories_count),
            str(frame.drive),
        )
        if self._last_signature.get(frame.agent_id) != sig:
            history.append(frame)
            self._last_signature[frame.agent_id] = sig
        self._current = frame
        self.update()

    def _on_anim_tick(self) -> None:
        self._phase = (self._phase + 0.16) % 1000.0
        if self._current is not None and self.isVisible():
            self.update()

    def _extract_frame(self, info: Dict[str, Any], tick: Optional[int]) -> BrainFrame:
        return _extract_brain_frame(info, tick)

    def _stage_key(self, frame: BrainFrame) -> str:
        if frame.age_ticks < 24 or frame.complexity < 0.24:
            return "seed"
        if frame.age_ticks < 120 or frame.complexity < 0.42:
            return "forming"
        if frame.plasticity > 0.76 and frame.activity > 0.56:
            return "rewiring"
        if frame.complexity > 0.72 and frame.coherence > 0.58:
            return "specialized"
        return "adaptive"

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        self._draw_backdrop(painter, rect)

        if self._current is None:
            self._draw_empty(painter, rect)
            return

        frame = self._current
        labels = _lang_pack(self._lang)

        self._draw_header(painter, rect, frame, labels)

        brain_rect = QtCore.QRectF(rect.left() + 18.0, rect.top() + 48.0, rect.width() - 36.0, rect.height() * 0.54)
        strip_rect = QtCore.QRectF(rect.left() + 18.0, brain_rect.bottom() + 6.0, rect.width() - 36.0, 40.0)
        timeline_rect = QtCore.QRectF(rect.left() + 18.0, rect.bottom() - 80.0, rect.width() - 36.0, 60.0)

        left_path, right_path = self._brain_paths(brain_rect)
        self._draw_hemisphere(painter, left_path, brain_rect, frame, True)
        self._draw_hemisphere(painter, right_path, brain_rect, frame, False)
        self._draw_corpus_callosum(painter, brain_rect, frame)
        self._draw_gyri(painter, left_path, brain_rect, frame, True)
        self._draw_gyri(painter, right_path, brain_rect, frame, False)
        self._draw_connections(painter, brain_rect, frame)
        self._draw_memory_sparks(painter, brain_rect, frame)
        self._draw_stress_marks(painter, brain_rect, frame)
        self._draw_brainstem(painter, brain_rect, frame)
        self._draw_stats_strip(painter, strip_rect, frame, labels)
        self._draw_timeline(painter, timeline_rect, frame, labels)

    def _draw_backdrop(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        bg = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg.setColorAt(0.0, QtGui.QColor(12, 18, 28))
        bg.setColorAt(0.5, QtGui.QColor(8, 10, 18))
        bg.setColorAt(1.0, QtGui.QColor(5, 7, 12))
        painter.setBrush(bg)
        painter.setPen(QtGui.QPen(QtGui.QColor(44, 58, 78), 1.2))
        painter.drawRoundedRect(rect, 18.0, 18.0)

    def _draw_empty(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        labels = _lang_pack(self._lang)
        brain_rect = QtCore.QRectF(rect.left() + 24.0, rect.top() + 58.0, rect.width() - 48.0, rect.height() * 0.52)
        left_path, right_path = self._brain_paths(brain_rect)
        painter.setPen(QtGui.QPen(QtGui.QColor(72, 86, 112, 120), 1.2))
        painter.setBrush(QtGui.QColor(18, 24, 36, 90))
        painter.drawPath(left_path)
        painter.drawPath(right_path)
        painter.setPen(QtGui.QColor(190, 200, 220))
        title_font = painter.font()
        title_font.setPointSize(11)
        title_font.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(title_font)
        painter.drawText(rect.adjusted(18, 14, -18, -14), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, labels["title"])

        body_font = painter.font()
        body_font.setPointSize(10)
        body_font.setWeight(QtGui.QFont.Normal)
        painter.setFont(body_font)
        painter.setPen(QtGui.QColor(144, 156, 180))
        painter.drawText(rect.adjusted(26, 0, -26, -16), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, labels["empty"])

    def _draw_header(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame, labels: Dict[str, str]) -> None:
        title_font = painter.font()
        title_font.setPointSize(11)
        title_font.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(236, 242, 255))
        painter.drawText(rect.adjusted(18, 12, -18, -12), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, labels["title"])

        meta_font = painter.font()
        meta_font.setPointSize(10)
        meta_font.setWeight(QtGui.QFont.Medium)
        painter.setFont(meta_font)
        painter.setPen(QtGui.QColor(163, 177, 205))
        meta = f"{frame.agent_name}  |  drive={frame.drive or 'idle'}  |  gen={frame.generation}  |  age={frame.age_ticks}"
        painter.drawText(rect.adjusted(18, 30, -18, -12), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, meta)

        stage_key = self._stage_key(frame)
        pill = QtCore.QRectF(rect.right() - 160.0, rect.top() + 12.0, 142.0, 24.0)
        pill_grad = QtGui.QLinearGradient(pill.topLeft(), pill.bottomRight())
        pill_grad.setColorAt(0.0, QtGui.QColor(30, 64, 78))
        pill_grad.setColorAt(1.0, QtGui.QColor(16, 36, 52))
        painter.setBrush(pill_grad)
        painter.setPen(QtGui.QPen(QtGui.QColor(84, 162, 192), 1.0))
        painter.drawRoundedRect(pill, 12.0, 12.0)
        painter.setPen(QtGui.QColor(218, 247, 255))
        painter.drawText(pill, QtCore.Qt.AlignCenter, f"{labels['stage']}: {labels[stage_key]}")

    def _brain_paths(self, rect: QtCore.QRectF) -> Tuple[QtGui.QPainterPath, QtGui.QPainterPath]:
        key = (
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )
        if self._brain_path_cache_key == key and self._brain_path_cache is not None:
            return self._brain_path_cache

        half_w = rect.width() * 0.47
        left_x = rect.left() + rect.width() * 0.03
        right_x = rect.center().x() - rect.width() * 0.01
        core_h = rect.height() * 0.76

        left = QtGui.QPainterPath()
        left.addEllipse(QtCore.QRectF(left_x, rect.top() + rect.height() * 0.08, half_w, core_h))
        left.addEllipse(QtCore.QRectF(left_x + 18.0, rect.top(), half_w * 0.72, rect.height() * 0.52))
        left.addEllipse(QtCore.QRectF(left_x + 22.0, rect.top() + rect.height() * 0.42, half_w * 0.54, rect.height() * 0.28))

        right = QtGui.QPainterPath()
        right.addEllipse(QtCore.QRectF(right_x, rect.top() + rect.height() * 0.08, half_w, core_h))
        right.addEllipse(QtCore.QRectF(right_x + half_w * 0.12, rect.top(), half_w * 0.72, rect.height() * 0.52))
        right.addEllipse(QtCore.QRectF(right_x + half_w * 0.26, rect.top() + rect.height() * 0.42, half_w * 0.54, rect.height() * 0.28))

        self._brain_path_cache_key = key
        self._brain_path_cache = (left.simplified(), right.simplified())
        return self._brain_path_cache

    def _draw_hemisphere(
        self,
        painter: QtGui.QPainter,
        path: QtGui.QPainterPath,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        is_left: bool,
    ) -> None:
        pulse = 0.86 + 0.14 * math.sin(self._phase + (0.0 if is_left else 1.2))
        aura = _clamp((0.24 + frame.complexity * 0.64) * pulse, 0.0, 1.0)
        if is_left:
            c1 = QtGui.QColor(74, 206, 212, int(165 + 50 * aura))
            c2 = QtGui.QColor(34, 86, 126, int(150 + 50 * aura))
        else:
            c1 = QtGui.QColor(94, 176, 248, int(150 + 45 * aura))
            c2 = QtGui.QColor(36, 74, 132, int(140 + 50 * aura))

        glow_pen = QtGui.QPen(QtGui.QColor(c1.red(), c1.green(), c1.blue(), 36 + int(120 * aura)), 8.0 + frame.activity * 6.0)
        painter.setPen(glow_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)

        grad = QtGui.QLinearGradient(rect.left(), rect.top(), rect.right(), rect.bottom())
        grad.setColorAt(0.0, c1)
        grad.setColorAt(1.0, c2)
        painter.setBrush(grad)
        painter.setPen(QtGui.QPen(QtGui.QColor(212, 232, 255, 46 + int(86 * frame.coherence)), 1.6 + frame.complexity * 1.8))
        painter.drawPath(path)

    def _draw_corpus_callosum(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        center = rect.center()
        bridge = QtGui.QPainterPath()
        bridge.moveTo(center.x() - rect.width() * 0.10, rect.top() + rect.height() * 0.52)
        bridge.cubicTo(
            center.x() - rect.width() * 0.04,
            rect.top() + rect.height() * 0.34,
            center.x() + rect.width() * 0.04,
            rect.top() + rect.height() * 0.34,
            center.x() + rect.width() * 0.10,
            rect.top() + rect.height() * 0.52,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor(228, 244, 255, 70 + int(110 * frame.coherence)), 3.0 + frame.coherence * 3.0))
        painter.drawPath(bridge)

    def _draw_gyri(
        self,
        painter: QtGui.QPainter,
        clip_path: QtGui.QPainterPath,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        is_left: bool,
    ) -> None:
        painter.save()
        painter.setClipPath(clip_path)
        painter.setBrush(QtCore.Qt.NoBrush)
        base_alpha = 44 + int(84 * frame.activity)
        painter.setPen(QtGui.QPen(QtGui.QColor(234, 246, 255, base_alpha), 1.15))
        x0 = rect.left() + (rect.width() * 0.08 if is_left else rect.width() * 0.54)
        x1 = rect.left() + (rect.width() * 0.46 if is_left else rect.width() * 0.92)
        bend = rect.width() * (0.10 if is_left else -0.10)
        for idx in range(7):
            y = rect.top() + rect.height() * (0.16 + idx * 0.10)
            wav = rect.height() * (0.04 + 0.02 * ((idx % 3) / 2.0))
            path = QtGui.QPainterPath(QtCore.QPointF(x0, y))
            path.cubicTo(
                QtCore.QPointF(x0 + bend, y - wav),
                QtCore.QPointF(x1 - bend, y + wav),
                QtCore.QPointF(x1, y),
            )
            painter.drawPath(path)
        painter.restore()

    def _node_layout(self, rect: QtCore.QRectF) -> List[Tuple[QtCore.QPointF, float, QtGui.QColor]]:
        key = (
            int(round(rect.x())),
            int(round(rect.y())),
            int(round(rect.width())),
            int(round(rect.height())),
        )
        if self._node_layout_cache_key == key and self._node_layout_cache is not None:
            return self._node_layout_cache

        base = [
            (0.24, 0.22, QtGui.QColor(88, 236, 214)),
            (0.36, 0.38, QtGui.QColor(102, 210, 255)),
            (0.28, 0.57, QtGui.QColor(70, 170, 255)),
            (0.42, 0.73, QtGui.QColor(255, 142, 92)),
            (0.50, 0.49, QtGui.QColor(236, 248, 255)),
            (0.76, 0.22, QtGui.QColor(98, 228, 255)),
            (0.64, 0.38, QtGui.QColor(118, 194, 255)),
            (0.72, 0.57, QtGui.QColor(92, 164, 255)),
            (0.58, 0.73, QtGui.QColor(255, 168, 86)),
        ]
        nodes: List[Tuple[QtCore.QPointF, float, QtGui.QColor]] = []
        for nx, ny, color in base:
            pt = QtCore.QPointF(rect.left() + rect.width() * nx, rect.top() + rect.height() * ny)
            nodes.append((pt, 1.0, color))
        self._node_layout_cache_key = key
        self._node_layout_cache = nodes
        return nodes

    def _draw_connections(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        nodes = self._node_layout(rect)
        act = frame.activity
        for idx, strength in enumerate(frame.belief_strengths[:12]):
            a = nodes[idx % 4][0]
            b = nodes[5 + (idx % 4)][0]
            mid = QtCore.QPointF(rect.center().x(), (a.y() + b.y()) * 0.5 - rect.height() * (0.10 + 0.04 * (idx % 3)))
            path = QtGui.QPainterPath(a)
            path.quadTo(mid, b)

            alpha = int(40 + 180 * _clamp(strength * 0.7 + act * 0.3, 0.0, 1.0))
            width = 1.2 + strength * 4.0 + frame.complexity * 1.4
            glow = QtGui.QPen(QtGui.QColor(98, 228, 255, alpha // 2), width + 2.4)
            core = QtGui.QPen(QtGui.QColor(216, 248, 255, alpha), width)
            painter.setPen(glow)
            painter.drawPath(path)
            painter.setPen(core)
            painter.drawPath(path)

        activations = [
            max(frame.activity, frame.plasticity),
            max(frame.complexity, frame.coherence),
            min(1.0, frame.memories_count / 12.0),
            max(frame.stress, min(1.0, frame.hazards_count / 8.0)),
            frame.coherence,
            max(frame.activity, frame.curiosity),
            min(1.0, frame.beliefs_count / 10.0),
            min(1.0, frame.memories_count / 10.0),
            max(frame.stress, min(1.0, frame.danger_count / 8.0)),
        ]

        for idx, ((pt, _scale, color), intensity) in enumerate(zip(nodes, activations)):
            glow_radius = 7.0 + intensity * 12.0
            aura = QtGui.QRadialGradient(pt, glow_radius)
            aura.setColorAt(0.0, QtGui.QColor(color.red(), color.green(), color.blue(), 210))
            aura.setColorAt(0.4, QtGui.QColor(color.red(), color.green(), color.blue(), 86))
            aura.setColorAt(1.0, QtGui.QColor(color.red(), color.green(), color.blue(), 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(aura)
            painter.drawEllipse(pt, glow_radius, glow_radius)
            painter.setBrush(QtGui.QColor(236, 248, 255, 220))
            painter.drawEllipse(pt, 2.4 + intensity * 2.8, 2.4 + intensity * 2.8)

    def _draw_memory_sparks(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        if not frame.memory_levels:
            return
        anchor_left = QtCore.QPointF(rect.left() + rect.width() * 0.26, rect.top() + rect.height() * 0.58)
        anchor_right = QtCore.QPointF(rect.left() + rect.width() * 0.74, rect.top() + rect.height() * 0.58)
        for idx, level in enumerate(frame.memory_levels[-12:]):
            side = -1.0 if idx % 2 == 0 else 1.0
            base = anchor_left if side < 0 else anchor_right
            dist = 10.0 + idx * 4.5
            phase = self._phase * 0.3 + idx * 0.7
            dx = math.cos(phase) * dist * side
            dy = math.sin(phase) * dist * 0.6
            pt = QtCore.QPointF(base.x() + dx, base.y() + dy)
            if level == "critical":
                color = QtGui.QColor(255, 108, 108, 220)
            elif level == "warning":
                color = QtGui.QColor(255, 184, 88, 214)
            else:
                color = QtGui.QColor(112, 228, 255, 190)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(pt, 2.0 + idx * 0.12, 2.0 + idx * 0.12)

    def _draw_stress_marks(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        count = max(frame.trauma_count, int(round(frame.stress * 4.0)))
        if count <= 0:
            return
        positions = [
            (0.33, 0.46),
            (0.63, 0.44),
            (0.40, 0.67),
            (0.59, 0.70),
            (0.48, 0.30),
        ]
        for idx in range(min(count, len(positions))):
            px, py = positions[idx]
            pt = QtCore.QPointF(rect.left() + rect.width() * px, rect.top() + rect.height() * py)
            radius = 10.0 + frame.stress * 10.0 + idx * 1.6
            grad = QtGui.QRadialGradient(pt, radius)
            grad.setColorAt(0.0, QtGui.QColor(255, 90, 90, 160))
            grad.setColorAt(0.45, QtGui.QColor(255, 122, 64, 70))
            grad.setColorAt(1.0, QtGui.QColor(255, 80, 80, 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(grad)
            painter.drawEllipse(pt, radius, radius)

    def _draw_brainstem(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        stem = QtCore.QRectF(rect.center().x() - 16.0, rect.bottom() - 22.0, 32.0, 36.0)
        grad = QtGui.QLinearGradient(stem.topLeft(), stem.bottomLeft())
        grad.setColorAt(0.0, QtGui.QColor(184, 224, 255, 180))
        grad.setColorAt(1.0, QtGui.QColor(90, 122, 168, 160))
        painter.setPen(QtGui.QPen(QtGui.QColor(220, 238, 255, 88), 1.1))
        painter.setBrush(grad)
        painter.drawRoundedRect(stem, 10.0, 10.0)

        pulse = 0.45 + 0.55 * math.sin(self._phase * 1.25)
        node = QtCore.QPointF(stem.center().x(), stem.center().y() + 4.0)
        grad2 = QtGui.QRadialGradient(node, 10.0 + 6.0 * pulse)
        grad2.setColorAt(0.0, QtGui.QColor(102, 218, 255, 220))
        grad2.setColorAt(1.0, QtGui.QColor(102, 218, 255, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(grad2)
        painter.drawEllipse(node, 10.0 + 6.0 * pulse, 10.0 + 6.0 * pulse)

    def _draw_stats_strip(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        labels: Dict[str, str],
    ) -> None:
        bg = QtGui.QColor(14, 18, 28, 180)
        painter.setPen(QtGui.QPen(QtGui.QColor(48, 60, 86), 1.0))
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, 12.0, 12.0)

        items = [
            (labels["plasticity"], frame.plasticity, QtGui.QColor(82, 208, 255)),
            (labels["complexity"], frame.complexity, QtGui.QColor(84, 238, 188)),
            (labels["stress"], frame.stress, QtGui.QColor(255, 168, 84)),
            (labels["beliefs"], min(1.0, frame.beliefs_count / 10.0), QtGui.QColor(180, 214, 255)),
            (labels["memories"], min(1.0, frame.memories_count / 12.0), QtGui.QColor(124, 206, 255)),
            (labels["hazards"], min(1.0, (frame.hazards_count + frame.danger_count) / 12.0), QtGui.QColor(255, 126, 96)),
        ]

        gap = 8.0
        pill_w = (rect.width() - gap * (len(items) - 1)) / len(items)
        x = rect.left()
        for label, value, color in items:
            pill = QtCore.QRectF(x, rect.top(), pill_w, rect.height())
            painter.setPen(QtGui.QPen(QtGui.QColor(54, 74, 98), 1.0))
            painter.setBrush(QtGui.QColor(18, 24, 36, 175))
            painter.drawRoundedRect(pill, 10.0, 10.0)

            bar = QtCore.QRectF(pill.left() + 10.0, pill.bottom() - 12.0, (pill.width() - 20.0) * _clamp(value, 0.0, 1.0), 4.0)
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawRoundedRect(bar, 2.0, 2.0)

            font = painter.font()
            font.setPointSize(8)
            font.setWeight(QtGui.QFont.Medium)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(222, 232, 248))
            painter.drawText(pill.adjusted(8, 6, -8, -16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, label)
            painter.setPen(QtGui.QColor(color))
            painter.drawText(pill.adjusted(8, 16, -8, -8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, f"{int(round(value * 100.0))}%")
            x += pill_w + gap

    def _draw_timeline(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        labels: Dict[str, str],
    ) -> None:
        painter.setPen(QtGui.QPen(QtGui.QColor(54, 68, 92), 1.0))
        painter.setBrush(QtGui.QColor(10, 14, 22, 184))
        painter.drawRoundedRect(rect, 14.0, 14.0)

        title_font = painter.font()
        title_font.setPointSize(9)
        title_font.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(214, 226, 244))
        painter.drawText(rect.adjusted(10, 6, -10, -6), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, labels["evolution"])

        history = list(self._history.get(frame.agent_id, deque()))
        if len(history) < 2:
            return

        chart = rect.adjusted(10.0, 20.0, -10.0, -8.0)
        max_points = min(90, len(history))
        series = history[-max_points:]

        painter.setPen(QtGui.QPen(QtGui.QColor(38, 50, 72), 1.0))
        for frac in (0.25, 0.5, 0.75):
            y = chart.top() + chart.height() * frac
            painter.drawLine(QtCore.QPointF(chart.left(), y), QtCore.QPointF(chart.right(), y))

        self._draw_series(painter, chart, [f.complexity for f in series], QtGui.QColor(88, 238, 188))
        self._draw_series(painter, chart, [f.plasticity for f in series], QtGui.QColor(84, 204, 255))
        self._draw_series(painter, chart, [f.stress for f in series], QtGui.QColor(255, 176, 92))

    def _draw_series(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        values: List[float],
        color: QtGui.QColor,
    ) -> None:
        if len(values) < 2:
            return
        path = QtGui.QPainterPath()
        step_x = rect.width() / max(1, len(values) - 1)
        for idx, val in enumerate(values):
            x = rect.left() + idx * step_x
            y = rect.bottom() - rect.height() * _clamp(val, 0.0, 1.0)
            pt = QtCore.QPointF(x, y)
            if idx == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        painter.setPen(QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue(), 70), 4.0))
        painter.drawPath(path)
        painter.setPen(QtGui.QPen(color, 1.8))
        painter.drawPath(path)


class BrainModel3DView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)

        self._lang = "ru"
        self._current: Optional[BrainFrame] = None
        self._phase = 0.0
        self._yaw = -0.68
        self._pitch = 0.34
        self._roll = 0.0
        self._zoom = 1.0
        self._distance = 6.6
        self._last_pos: Optional[QtCore.QPoint] = None

        self._vertices: List[BrainSurfaceVertex] = []
        self._faces: List[Tuple[int, int, int]] = []
        self._surface_dirty = True
        self._surface_faces: List[Tuple[QtGui.QPolygonF, QtGui.QColor, QtGui.QPen]] = []
        self._build_mesh()

        self._anim = QtCore.QTimer(self)
        self._anim.setTimerType(QtCore.Qt.PreciseTimer)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._on_anim_tick)
        self._anim.start()

    def _invalidate_surface_cache(self) -> None:
        self._surface_dirty = True
        self._surface_faces = []

    def set_language(self, lang: str) -> None:
        self._lang = "en" if str(lang).lower() == "en" else "ru"
        self.update()

    def clear(self) -> None:
        self._current = None
        self._invalidate_surface_cache()
        self.update()

    def update_from_debug(self, info: Dict[str, Any], *, tick: Optional[int] = None) -> None:
        if not info or not info.get("id"):
            self.clear()
            return
        self._current = _extract_brain_frame(info, tick)
        self._invalidate_surface_cache()
        self.update()

    def _on_anim_tick(self) -> None:
        self._phase = (self._phase + 0.07) % (math.pi * 1000.0)
        if self._current is not None and self.isVisible():
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._last_pos = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_pos is not None and (event.buttons() & QtCore.Qt.LeftButton):
            now = event.position().toPoint()
            delta = now - self._last_pos
            self._yaw += delta.x() * 0.008
            self._pitch = _clamp(self._pitch + delta.y() * 0.006, -1.2, 1.2)
            self._last_pos = now
            self._invalidate_surface_cache()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y() / 120.0
        self._zoom = _clamp(self._zoom + delta * 0.08, 0.68, 1.48)
        self._invalidate_surface_cache()
        self.update()
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._invalidate_surface_cache()
        super().resizeEvent(event)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        self._draw_backdrop(painter, rect)

        labels = _lang_pack(self._lang)
        title = labels["title"] + " 3D"
        subtitle = "drag: orbit  |  wheel: zoom" if self._lang == "en" else "ЛКМ: орбита  |  колесо: зум"
        self._draw_header(painter, rect, title, subtitle)

        if self._current is None:
            self._draw_empty(painter, rect, labels["empty"])
            return

        frame = self._current
        view_rect = rect.adjusted(12.0, 44.0, -12.0, -16.0)
        self._draw_shadow(painter, view_rect)
        self._draw_ground_reflection(painter, view_rect, frame)

        region_activity = self._region_activity(frame)
        for poly, brush_color, pen in self._surface_face_batch(view_rect, frame, region_activity):
            painter.setPen(pen)
            painter.setBrush(brush_color)
            painter.drawPolygon(poly)

        self._draw_major_sulci(painter, view_rect, frame)
        self._draw_internal_paths(painter, view_rect, frame, region_activity)
        self._draw_hotspots(painter, view_rect, frame, region_activity)
        self._draw_overlay_metrics(painter, rect, frame)

    def _surface_face_batch(
        self,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        region_activity: Dict[str, float],
    ) -> List[Tuple[QtGui.QPolygonF, QtGui.QColor, QtGui.QPen]]:
        if not self._surface_dirty and self._surface_faces:
            return self._surface_faces

        model_scale = 1.18 + 0.08 * frame.complexity
        transformed: List[QtGui.QVector3D] = []
        normals: List[QtGui.QVector3D] = []
        projected: List[QtCore.QPointF] = []
        heats: List[float] = []

        for vertex in self._vertices:
            pos = QtGui.QVector3D(
                vertex.pos.x() * model_scale,
                vertex.pos.y() * model_scale,
                vertex.pos.z() * model_scale,
            )
            transformed_pos = self._rotate_vector(pos)
            transformed_normal = self._rotate_normal(vertex.normal)
            transformed.append(transformed_pos)
            normals.append(transformed_normal)
            projected.append(self._project(transformed_pos, rect))
            heats.append(self._vertex_heat(vertex, region_activity))

        light_dir = QtGui.QVector3D(-0.24, 0.56, 1.0).normalized()
        draw_faces: List[Tuple[float, QtGui.QPolygonF, QtGui.QColor, QtGui.QPen]] = []
        for ia, ib, ic in self._faces:
            na = normals[ia]
            nb = normals[ib]
            nc = normals[ic]
            avg_normal = (na + nb + nc) * (1.0 / 3.0)
            if avg_normal.lengthSquared() < 1e-6:
                continue
            avg_normal.normalize()
            if avg_normal.z() <= -0.28:
                continue

            pa = projected[ia]
            pb = projected[ib]
            pc = projected[ic]
            poly = QtGui.QPolygonF([pa, pb, pc])
            avg_z = (transformed[ia].z() + transformed[ib].z() + transformed[ic].z()) / 3.0
            light = _clamp(QtGui.QVector3D.dotProduct(avg_normal, light_dir) * 0.75 + 0.25, 0.0, 1.0)
            heat = (heats[ia] + heats[ib] + heats[ic]) / 3.0
            limbic = (self._vertices[ia].limbic + self._vertices[ib].limbic + self._vertices[ic].limbic) / 3.0
            underside = _clamp((-avg_normal.y()) * 0.46 + max(0.0, -avg_normal.z()) * 0.22, 0.0, 1.0)
            fissure = _clamp(
                limbic * 0.32
                + max(0.0, 0.18 - abs((transformed[ia].x() + transformed[ib].x() + transformed[ic].x()) / 3.0)) * 1.4,
                0.0,
                1.0,
            )
            base = QtGui.QColor(160, 126, 131, 212)
            tissue = _mix_color(base, QtGui.QColor(224, 194, 184, 228), light)
            tissue = _mix_color(tissue, QtGui.QColor(132, 96, 104, 218), underside * 0.54 + fissure * 0.18)
            specular = _clamp(light * light * (0.45 + 0.35 * avg_normal.z()), 0.0, 1.0)
            tissue = _mix_color(tissue, QtGui.QColor(245, 225, 220, 236), specular * 0.25)
            activation_color = _mix_color(
                QtGui.QColor(255, 207, 160, 220),
                QtGui.QColor(255, 116, 88, 230),
                _clamp(frame.stress * 0.68 + limbic * 0.32, 0.0, 1.0),
            )
            face_color = _mix_color(tissue, activation_color, _clamp(heat * 0.34 + frame.activity * 0.03, 0.0, 0.45))
            face_color.setAlpha(int(176 + 58 * light))
            edge_color = _mix_color(QtGui.QColor(78, 58, 62), face_color, 0.24)
            edge_color.setAlpha(92)
            draw_faces.append((avg_z, poly, face_color, QtGui.QPen(edge_color, 0.55)))

        draw_faces.sort(key=lambda item: item[0])
        self._surface_faces = [(poly, brush_color, pen) for _, poly, brush_color, pen in draw_faces]
        self._surface_dirty = False
        return self._surface_faces

    def _build_mesh(self) -> None:
        self._append_hemisphere(side=-1.0, center_x=-0.86, radii=(1.36, 0.98, 1.46), lat_steps=26, lon_steps=40)
        self._append_hemisphere(side=1.0, center_x=0.86, radii=(1.36, 0.98, 1.46), lat_steps=26, lon_steps=40)
        self._append_cerebellum(center=QtGui.QVector3D(0.0, -0.78, -1.36), radii=(0.86, 0.48, 0.58), lat_steps=16, lon_steps=24)
        self._append_brainstem(center=QtGui.QVector3D(0.0, -1.24, -0.66), radii=(0.24, 0.64, 0.24), lat_steps=12, lon_steps=18)

    def _append_hemisphere(
        self,
        *,
        side: float,
        center_x: float,
        radii: Tuple[float, float, float],
        lat_steps: int,
        lon_steps: int,
    ) -> None:
        rx, ry, rz = radii
        start = len(self._vertices)
        cols = lon_steps + 1
        for i in range(lat_steps + 1):
            theta = -math.pi * 0.5 + math.pi * i / lat_steps
            for j in range(lon_steps + 1):
                phi = -math.pi + 2.0 * math.pi * j / lon_steps
                ux = math.cos(theta) * math.cos(phi)
                uy = math.sin(theta)
                uz = math.cos(theta) * math.sin(phi)
                medial = max(0.0, ux * -side)
                lateral = max(0.0, ux * side)
                front_factor = _clamp((uz + 1.0) * 0.5, 0.0, 1.0)
                rear_factor = _clamp((1.0 - uz) * 0.5, 0.0, 1.0)
                lower_factor = _clamp((-uy + 1.0) * 0.5, 0.0, 1.0)
                upper_factor = _clamp((uy + 1.0) * 0.5, 0.0, 1.0)
                temporal_notch = _clamp(1.0 - abs(uy + 0.16) * 1.28, 0.0, 1.0)
                sylvian = _clamp(1.0 - abs(uy + 0.02) * 1.9, 0.0, 1.0) * _clamp((uz + 0.2) * 0.8, 0.0, 1.0)

                fold = (
                    0.072 * math.sin(6.0 * phi + 2.1 * theta)
                    + 0.044 * math.sin(12.0 * phi - 4.8 * theta)
                    + 0.018 * math.cos(19.0 * phi + 1.7 * theta)
                    + 0.012 * math.sin(27.0 * phi - 2.0 * theta)
                )
                fold *= 0.76 + 0.24 * lateral

                radial_x = 1.0 - 0.28 * medial + 0.16 * front_factor - 0.08 * rear_factor
                radial_y = 1.0 + 0.12 * upper_factor - 0.22 * lower_factor * (0.5 + 0.5 * lateral)
                radial_z = 1.0 + 0.16 * front_factor - 0.10 * rear_factor

                x = center_x + rx * ux * radial_x
                x -= side * medial * (0.22 + 0.09 * upper_factor)
                x -= side * sylvian * 0.08

                y = ry * uy * radial_y
                y -= lower_factor * (0.14 + 0.08 * front_factor)
                y += upper_factor * front_factor * 0.05

                z = rz * uz * radial_z
                z += front_factor * (0.22 + 0.08 * upper_factor)
                z -= rear_factor * 0.14

                x += ux * fold * 0.42
                y += uy * fold * 0.17
                z += uz * fold * 0.46

                temporal_pull = temporal_notch * lower_factor
                x += side * temporal_pull * 0.10
                y -= temporal_pull * 0.10
                z += front_factor * temporal_pull * 0.06

                frontal = _clamp((uz + 1.0) * 0.5, 0.0, 1.0) * (0.48 + 0.52 * max(0.0, uy + 0.12))
                occipital = _clamp((1.0 - uz) * 0.5, 0.0, 1.0) * (0.52 + 0.48 * max(0.0, uy + 0.04))
                parietal = _clamp((uy + 1.0) * 0.5, 0.0, 1.0) * (0.40 + 0.60 * (1.0 - abs(uz) * 0.65))
                temporal = _clamp(1.0 - abs(uy + 0.18) * 1.18, 0.0, 1.0) * (0.36 + 0.64 * lateral)
                limbic = _clamp(1.0 - abs(ux) * 0.92, 0.0, 1.0) * _clamp(1.0 - abs(uy) * 1.08, 0.0, 1.0) * _clamp(1.0 - abs(uz) * 0.92, 0.0, 1.0)

                normal = QtGui.QVector3D(ux / rx, uy / ry, uz / rz)
                if normal.lengthSquared() > 1e-6:
                    normal.normalize()
                self._vertices.append(
                    BrainSurfaceVertex(
                        pos=QtGui.QVector3D(x, y, z),
                        normal=normal,
                        frontal=frontal,
                        parietal=parietal,
                        temporal=temporal,
                        occipital=occipital,
                        limbic=limbic,
                        cerebellum=0.0,
                    )
                )

        for i in range(lat_steps):
            for j in range(lon_steps):
                a = start + i * cols + j
                b = a + 1
                c = a + cols
                d = c + 1
                self._faces.append((a, c, b))
                self._faces.append((b, c, d))

    def _append_cerebellum(
        self,
        *,
        center: QtGui.QVector3D,
        radii: Tuple[float, float, float],
        lat_steps: int,
        lon_steps: int,
    ) -> None:
        rx, ry, rz = radii
        start = len(self._vertices)
        cols = lon_steps + 1
        for i in range(lat_steps + 1):
            theta = -math.pi * 0.5 + math.pi * i / lat_steps
            for j in range(lon_steps + 1):
                phi = -math.pi + 2.0 * math.pi * j / lon_steps
                ux = math.cos(theta) * math.cos(phi)
                uy = math.sin(theta)
                uz = math.cos(theta) * math.sin(phi)
                fold = 0.085 * math.sin(8.0 * phi + theta * 3.0)
                x = center.x() + rx * ux * (1.0 + fold)
                y = center.y() + ry * uy * (1.0 + fold * 0.22)
                z = center.z() + rz * uz * (1.0 + fold * 0.55)
                normal = QtGui.QVector3D(ux / rx, uy / ry, uz / rz)
                if normal.lengthSquared() > 1e-6:
                    normal.normalize()
                self._vertices.append(
                    BrainSurfaceVertex(
                        pos=QtGui.QVector3D(x, y, z),
                        normal=normal,
                        frontal=0.0,
                        parietal=0.0,
                        temporal=0.0,
                        occipital=0.18,
                        limbic=0.12,
                        cerebellum=1.0,
                    )
                )

        for i in range(lat_steps):
            for j in range(lon_steps):
                a = start + i * cols + j
                b = a + 1
                c = a + cols
                d = c + 1
                self._faces.append((a, c, b))
                self._faces.append((b, c, d))

    def _append_brainstem(
        self,
        *,
        center: QtGui.QVector3D,
        radii: Tuple[float, float, float],
        lat_steps: int,
        lon_steps: int,
    ) -> None:
        rx, ry, rz = radii
        start = len(self._vertices)
        cols = lon_steps + 1
        for i in range(lat_steps + 1):
            v = i / lat_steps
            theta = -math.pi * 0.5 + math.pi * i / lat_steps
            for j in range(lon_steps + 1):
                phi = -math.pi + 2.0 * math.pi * j / lon_steps
                ux = math.cos(theta) * math.cos(phi)
                uy = math.sin(theta)
                uz = math.cos(theta) * math.sin(phi)
                bulge = math.exp(-((v - 0.34) * 4.4) ** 2) * 0.22
                taper = 1.0 - 0.22 * v
                x = center.x() + rx * ux * (taper + bulge)
                y = center.y() + ry * uy - v * 0.16
                z = center.z() + rz * uz * (taper + bulge * 0.6)
                normal = QtGui.QVector3D(ux / max(0.01, rx), uy / max(0.01, ry), uz / max(0.01, rz))
                if normal.lengthSquared() > 1e-6:
                    normal.normalize()
                self._vertices.append(
                    BrainSurfaceVertex(
                        pos=QtGui.QVector3D(x, y, z),
                        normal=normal,
                        frontal=0.0,
                        parietal=0.0,
                        temporal=0.0,
                        occipital=0.08,
                        limbic=0.16,
                        cerebellum=0.82,
                    )
                )

        for i in range(lat_steps):
            for j in range(lon_steps):
                a = start + i * cols + j
                b = a + 1
                c = a + cols
                d = c + 1
                self._faces.append((a, c, b))
                self._faces.append((b, c, d))

    def _draw_backdrop(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(7, 12, 18))
        grad.setColorAt(0.45, QtGui.QColor(10, 16, 26))
        grad.setColorAt(1.0, QtGui.QColor(4, 7, 14))
        painter.setPen(QtGui.QPen(QtGui.QColor(35, 58, 82), 1.2))
        painter.setBrush(grad)
        painter.drawRoundedRect(rect, 18.0, 18.0)

        vignette = QtGui.QRadialGradient(rect.center(), rect.width() * 0.72)
        vignette.setColorAt(0.0, QtGui.QColor(24, 42, 64, 0))
        vignette.setColorAt(0.72, QtGui.QColor(6, 10, 16, 0))
        vignette.setColorAt(1.0, QtGui.QColor(2, 3, 7, 180))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(vignette)
        painter.drawRoundedRect(rect, 18.0, 18.0)

    def _draw_header(self, painter: QtGui.QPainter, rect: QtCore.QRectF, title: str, subtitle: str) -> None:
        title_font = painter.font()
        title_font.setPointSize(11)
        title_font.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(234, 242, 255))
        painter.drawText(rect.adjusted(16, 10, -16, -10), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, title)

        sub_font = painter.font()
        sub_font.setPointSize(9)
        sub_font.setWeight(QtGui.QFont.Medium)
        painter.setFont(sub_font)
        painter.setPen(QtGui.QColor(138, 165, 198))
        painter.drawText(rect.adjusted(16, 28, -16, -10), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, subtitle)

    def _draw_empty(self, painter: QtGui.QPainter, rect: QtCore.QRectF, text: str) -> None:
        painter.setPen(QtGui.QColor(154, 172, 198))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(24, 48, -24, -24), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, text)

    def _draw_shadow(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        shadow_rect = QtCore.QRectF(
            rect.center().x() - rect.width() * 0.21,
            rect.bottom() - rect.height() * 0.20,
            rect.width() * 0.42,
            rect.height() * 0.12,
        )
        shadow = QtGui.QRadialGradient(shadow_rect.center(), shadow_rect.width() * 0.6)
        shadow.setColorAt(0.0, QtGui.QColor(0, 0, 0, 120))
        shadow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(shadow)
        painter.drawEllipse(shadow_rect)

    def _draw_ground_reflection(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        glow_rect = QtCore.QRectF(
            rect.center().x() - rect.width() * 0.26,
            rect.bottom() - rect.height() * 0.34,
            rect.width() * 0.52,
            rect.height() * 0.26,
        )
        glow = QtGui.QRadialGradient(glow_rect.center(), glow_rect.width() * 0.45)
        shimmer = 0.88 + 0.12 * math.sin(self._phase * 1.9)
        active = int((34 + 34 * frame.activity + 28 * frame.plasticity) * shimmer)
        glow.setColorAt(0.0, QtGui.QColor(255, 186, 164, active))
        glow.setColorAt(0.55, QtGui.QColor(255, 186, 164, active // 3))
        glow.setColorAt(1.0, QtGui.QColor(255, 186, 164, 0))
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(glow_rect)

    def _region_activity(self, frame: BrainFrame) -> Dict[str, float]:
        beliefs = min(1.0, frame.beliefs_count / 12.0)
        memories = min(1.0, frame.memories_count / 14.0)
        hazards = min(1.0, (frame.hazards_count + frame.danger_count) / 12.0)
        return {
            "frontal": _clamp(0.48 * frame.activity + 0.28 * frame.curiosity + 0.24 * frame.plasticity, 0.0, 1.0),
            "parietal": _clamp(0.52 * frame.complexity + 0.26 * frame.coherence + 0.22 * beliefs, 0.0, 1.0),
            "temporal": _clamp(0.46 * memories + 0.22 * beliefs + 0.18 * frame.activity + 0.14 * frame.stress, 0.0, 1.0),
            "occipital": _clamp(0.28 * frame.activity + 0.24 * frame.curiosity + 0.26 * frame.coherence + 0.22 * frame.complexity, 0.0, 1.0),
            "limbic": _clamp(0.54 * frame.stress + 0.18 * frame.fear + 0.16 * hazards + 0.12 * (1.0 - frame.survival), 0.0, 1.0),
            "cerebellum": _clamp(0.38 * frame.coherence + 0.34 * frame.activity + 0.28 * frame.plasticity, 0.0, 1.0),
        }

    def _vertex_heat(self, vertex: BrainSurfaceVertex, region_activity: Dict[str, float]) -> float:
        return _clamp(
            vertex.frontal * region_activity["frontal"]
            + vertex.parietal * region_activity["parietal"]
            + vertex.temporal * region_activity["temporal"]
            + vertex.occipital * region_activity["occipital"]
            + vertex.limbic * region_activity["limbic"]
            + vertex.cerebellum * region_activity["cerebellum"],
            0.0,
            1.0,
        )

    def _rotate_vector(self, vec: QtGui.QVector3D) -> QtGui.QVector3D:
        yaw = self._yaw
        pitch = self._pitch
        roll = self._roll

        x, y, z = vec.x(), vec.y(), vec.z()

        cy, sy = math.cos(yaw), math.sin(yaw)
        x, z = x * cy + z * sy, -x * sy + z * cy

        cp, sp = math.cos(pitch), math.sin(pitch)
        y, z = y * cp - z * sp, y * sp + z * cp

        cr, sr = math.cos(roll), math.sin(roll)
        x, y = x * cr - y * sr, x * sr + y * cr

        return QtGui.QVector3D(x, y, z)

    def _rotate_normal(self, normal: QtGui.QVector3D) -> QtGui.QVector3D:
        rotated = self._rotate_vector(normal)
        if rotated.lengthSquared() > 1e-6:
            rotated.normalize()
        return rotated

    def _project(self, vec: QtGui.QVector3D, rect: QtCore.QRectF) -> QtCore.QPointF:
        depth = max(1.2, self._distance - vec.z())
        scale = min(rect.width(), rect.height()) * 0.90 * self._zoom
        persp = scale / depth
        return QtCore.QPointF(
            rect.center().x() + vec.x() * persp,
            rect.center().y() - vec.y() * persp,
        )

    def _brain_point(self, x: float, y: float, z: float) -> QtCore.QPointF:
        return self._project(self._rotate_vector(QtGui.QVector3D(x, y, z)), self.rect().adjusted(12, 44, -12, -16))

    def _draw_major_sulci(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        painter.save()
        painter.setBrush(QtCore.Qt.NoBrush)
        alpha = int(28 + 44 * frame.complexity + 22 * frame.activity)
        groove_pen = QtGui.QPen(QtGui.QColor(94, 66, 72, alpha), 1.0)
        painter.setPen(groove_pen)

        sulci_specs = [
            [(-1.62, 0.36, 0.82), (-1.24, 0.12, 0.50), (-0.94, -0.10, 0.18), (-0.58, -0.18, -0.16)],
            [(1.62, 0.36, 0.82), (1.24, 0.12, 0.50), (0.94, -0.10, 0.18), (0.58, -0.18, -0.16)],
            [(-1.10, 0.78, 0.42), (-0.74, 0.56, 0.16), (-0.46, 0.42, -0.22), (-0.30, 0.34, -0.64)],
            [(1.10, 0.78, 0.42), (0.74, 0.56, 0.16), (0.46, 0.42, -0.22), (0.30, 0.34, -0.64)],
            [(-0.14, 0.96, 0.64), (-0.10, 0.62, 0.24), (-0.08, 0.24, -0.18), (-0.06, -0.04, -0.52)],
            [(0.14, 0.96, 0.64), (0.10, 0.62, 0.24), (0.08, 0.24, -0.18), (0.06, -0.04, -0.52)],
        ]
        for spec in sulci_specs:
            path = QtGui.QPainterPath(self._project(self._rotate_vector(QtGui.QVector3D(*spec[0])), rect))
            for point in spec[1:]:
                path.lineTo(self._project(self._rotate_vector(QtGui.QVector3D(*point)), rect))
            painter.drawPath(path)

        fissure_pen = QtGui.QPen(QtGui.QColor(72, 42, 48, int(76 + frame.coherence * 40)), 1.5)
        painter.setPen(fissure_pen)
        fissure = QtGui.QPainterPath(self._project(self._rotate_vector(QtGui.QVector3D(0.0, 1.02, 0.92)), rect))
        fissure.cubicTo(
            self._project(self._rotate_vector(QtGui.QVector3D(0.0, 0.72, 0.40)), rect),
            self._project(self._rotate_vector(QtGui.QVector3D(0.0, 0.22, -0.08)), rect),
            self._project(self._rotate_vector(QtGui.QVector3D(0.0, -0.10, -0.78)), rect),
        )
        painter.drawPath(fissure)
        painter.restore()

    def _draw_hotspots(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        region_activity: Dict[str, float],
    ) -> None:
        points = [
            (QtGui.QVector3D(-1.85, 0.16, 1.16), region_activity["frontal"], QtGui.QColor(90, 222, 255)),
            (QtGui.QVector3D(1.85, 0.16, 1.16), region_activity["frontal"], QtGui.QColor(90, 222, 255)),
            (QtGui.QVector3D(-1.56, -0.46, 0.22), region_activity["temporal"], QtGui.QColor(110, 199, 255)),
            (QtGui.QVector3D(1.56, -0.46, 0.22), region_activity["temporal"], QtGui.QColor(110, 199, 255)),
            (QtGui.QVector3D(0.0, 1.22, 0.08), region_activity["parietal"], QtGui.QColor(116, 255, 204)),
            (QtGui.QVector3D(0.0, 0.08, -1.62), region_activity["occipital"], QtGui.QColor(138, 206, 255)),
            (QtGui.QVector3D(0.0, -0.12, 0.18), region_activity["limbic"], QtGui.QColor(255, 118, 92)),
            (QtGui.QVector3D(0.0, -0.78, -1.42), region_activity["cerebellum"], QtGui.QColor(104, 234, 255)),
        ]
        for idx, (point, intensity, color) in enumerate(points):
            intensity = _clamp(intensity, 0.0, 1.0)
            if intensity <= 0.08:
                continue
            pulse = 0.72 + 0.28 * math.sin(self._phase * (1.2 + idx * 0.04) + idx * 0.7)
            transformed = self._rotate_vector(point)
            screen = self._project(transformed, rect)
            depth = max(1.0, self._distance - transformed.z())
            radius = (10.0 + 18.0 * intensity * pulse) / depth * 2.6
            glow = QtGui.QRadialGradient(screen, radius)
            glow.setColorAt(0.0, _with_alpha(color, 210))
            glow.setColorAt(0.35, _with_alpha(color, 96))
            glow.setColorAt(1.0, _with_alpha(color, 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(screen, radius, radius)

    def _draw_internal_paths(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        frame: BrainFrame,
        region_activity: Dict[str, float],
    ) -> None:
        nodes = [
            QtGui.QVector3D(-1.10, 0.30, 0.74),
            QtGui.QVector3D(-0.86, 0.62, -0.02),
            QtGui.QVector3D(-0.92, -0.26, 0.05),
            QtGui.QVector3D(-0.34, -0.02, 0.12),
            QtGui.QVector3D(1.10, 0.30, 0.74),
            QtGui.QVector3D(0.86, 0.62, -0.02),
            QtGui.QVector3D(0.92, -0.26, 0.05),
            QtGui.QVector3D(0.34, -0.02, 0.12),
            QtGui.QVector3D(0.0, -0.60, -1.02),
        ]
        strengths = list(frame.belief_strengths[:10])
        if len(strengths) < 10:
            strengths.extend([frame.activity] * (10 - len(strengths)))
        connections = [
            (0, 4, max(frame.coherence, strengths[0])),
            (1, 5, max(region_activity["parietal"], strengths[1])),
            (2, 6, max(region_activity["temporal"], strengths[2])),
            (3, 7, max(region_activity["limbic"], strengths[3])),
            (0, 3, max(region_activity["frontal"], strengths[4])),
            (4, 7, max(region_activity["frontal"], strengths[5])),
            (1, 3, max(region_activity["parietal"], strengths[6])),
            (5, 7, max(region_activity["parietal"], strengths[7])),
            (3, 8, max(region_activity["limbic"], region_activity["cerebellum"], strengths[8])),
            (7, 8, max(region_activity["limbic"], region_activity["cerebellum"], strengths[9])),
        ]
        for idx, (ia, ib, intensity) in enumerate(connections):
            intensity = _clamp(intensity, 0.0, 1.0)
            if intensity <= 0.12:
                continue
            a3 = self._rotate_vector(nodes[ia])
            b3 = self._rotate_vector(nodes[ib])
            mid3 = self._rotate_vector((nodes[ia] + nodes[ib]) * 0.5 + QtGui.QVector3D(0.0, 0.36 + 0.14 * intensity, 0.0))
            a = self._project(a3, rect)
            b = self._project(b3, rect)
            mid = self._project(mid3, rect)

            path = QtGui.QPainterPath(a)
            path.quadTo(mid, b)
            line_color = _mix_color(QtGui.QColor(100, 214, 255), QtGui.QColor(255, 132, 96), frame.stress * 0.55)
            glow = QtGui.QPen(_with_alpha(line_color, int(30 + 75 * intensity)), 2.4 + 2.4 * intensity)
            core = QtGui.QPen(_with_alpha(QtGui.QColor(238, 247, 255), int(110 + 100 * intensity)), 0.85 + 1.2 * intensity)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(glow)
            painter.drawPath(path)
            painter.setPen(core)
            painter.drawPath(path)

            t = (self._phase * (0.24 + idx * 0.01) + idx * 0.15) % 1.0
            omt = 1.0 - t
            pulse_point = QtCore.QPointF(
                omt * omt * a.x() + 2.0 * omt * t * mid.x() + t * t * b.x(),
                omt * omt * a.y() + 2.0 * omt * t * mid.y() + t * t * b.y(),
            )
            radius = 2.0 + 3.6 * intensity
            halo = QtGui.QRadialGradient(pulse_point, radius * 2.8)
            halo.setColorAt(0.0, QtGui.QColor(246, 252, 255, 230))
            halo.setColorAt(0.55, _with_alpha(line_color, 110))
            halo.setColorAt(1.0, _with_alpha(line_color, 0))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(pulse_point, radius * 2.8, radius * 2.8)
            painter.setBrush(QtGui.QColor(250, 252, 255, 230))
            painter.drawEllipse(pulse_point, radius, radius)

    def _draw_overlay_metrics(self, painter: QtGui.QPainter, rect: QtCore.QRectF, frame: BrainFrame) -> None:
        labels = _lang_pack(self._lang)
        panel = QtCore.QRectF(rect.right() - 208.0, rect.top() + 48.0, 188.0, 92.0)
        painter.setPen(QtGui.QPen(QtGui.QColor(50, 72, 98), 1.0))
        painter.setBrush(QtGui.QColor(9, 14, 22, 168))
        painter.drawRoundedRect(panel, 14.0, 14.0)

        lines = [
            f"{labels['plasticity']}: {int(round(frame.plasticity * 100.0))}%",
            f"{labels['complexity']}: {int(round(frame.complexity * 100.0))}%",
            f"{labels['stress']}: {int(round(frame.stress * 100.0))}%",
            f"drive: {frame.drive or 'idle'}",
        ]
        font = painter.font()
        font.setPointSize(9)
        font.setWeight(QtGui.QFont.Medium)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(220, 231, 248))
        y = panel.top() + 14.0
        for line in lines:
            painter.drawText(QtCore.QRectF(panel.left() + 12.0, y, panel.width() - 24.0, 18.0), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, line)
            y += 18.0
