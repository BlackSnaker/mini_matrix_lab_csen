from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Slot

from brain_visualizer import BrainModel3DView, OrganicBrainView
from mind_trainer_gui import BeliefGraphView, MindTrainerInteractive


def _safe_float(v: Any, fallback: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return fallback
    if x != x or x in (float("inf"), float("-inf")):
        return fallback
    return x


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _resource_pct(v: Any, fallback: float = 0.0) -> int:
    x = _safe_float(v, fallback)
    if x <= 1.5:
        x *= 100.0
    return int(_clamp(x, 0.0, 100.0))


def _iter_world_agents(world: Any) -> List[Any]:
    if world is None:
        return []
    agents_obj = getattr(world, "agents", {}) or {}
    if isinstance(agents_obj, dict):
        return list(agents_obj.values())
    return list(agents_obj)


def _belief_key(belief: Dict[str, Any]) -> str:
    cond = str(belief.get("if", "?"))
    concl = str(belief.get("then", "?"))
    return f"{cond} -> {concl}"


def _event_signature(ev: Dict[str, Any]) -> str:
    tick = ev.get("tick", "?")
    etype = ev.get("type", ev.get("etype", "event"))
    level = ev.get("level", "info")
    data = ev.get("data", {})
    try:
        data_txt = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        data_txt = str(data)
    return f"{tick}|{level}|{etype}|{data_txt}"


def _format_memory_event(ev: Dict[str, Any]) -> str:
    tick = ev.get("tick", "?")
    etype = ev.get("type", ev.get("etype", "event"))
    level = str(ev.get("level", "") or "").upper()
    data = ev.get("data", {})
    if isinstance(data, dict) and data:
        payload = ", ".join(f"{k}={v}" for k, v in data.items())
    elif data:
        payload = str(data)
    else:
        payload = "-"
    level_part = f"[{level}] " if level else ""
    return f"[t={tick}] {level_part}{etype}: {payload}"


def _make_agent_lineup(num_agents: int) -> List[Dict[str, str]]:
    base = [
        {
            "id": "a1",
            "name": "Echo",
            "persona": (
                "Ты Echo. Осторожный, тревожный выживальщик. "
                "Тебе страшно умереть, ты переживаешь за Nova. "
                "Ты стараешься держать всех в безопасности и предупреждать об угрозах."
            ),
        },
        {
            "id": "a2",
            "name": "Nova",
            "persona": (
                "Ты Nova. Смелая исследовательница. Ты любишь разведку, "
                "но не хочешь, чтобы Echo пострадал. Ты звучишь уверенно и тёпло, даже когда больно."
            ),
        },
        {
            "id": "a3",
            "name": "Astra",
            "persona": "Ты Astra. Наблюдательница и аналитик. Ты замечаешь закономерности и быстро учишься.",
        },
    ]
    if num_agents <= len(base):
        return base[:num_agents]

    out = list(base)
    for idx in range(len(base), num_agents):
        out.append(
            {
                "id": f"agent_{idx}",
                "name": f"A{idx}",
                "persona": "scout/explorer",
            }
        )
    return out


@dataclass
class BrainSnapshot:
    epoch: int
    tick: int
    drive: str
    thought: str
    survival: float
    fear: float
    curiosity: float
    trauma_count: int
    beliefs: Dict[str, float] = field(default_factory=dict)
    rules: Dict[str, float] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)


class BrainOnlyWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        num_agents: int,
        max_ticks_per_epoch: int,
        seed: int,
        interval_ms: int,
        fresh_start: bool,
    ):
        super().__init__()
        self.setWindowTitle("Brain Module")
        self.resize(1440, 960)
        self.setStyleSheet(
            """
            QMainWindow { background-color:#0b1017; color:#eef3ff; }
            QLabel { color:#eef3ff; }
            QFrame[card="true"] {
                background-color:#111827;
                border:1px solid #233246;
                border-radius:16px;
            }
            QComboBox, QSpinBox, QToolButton {
                background-color:#162131;
                color:#eef3ff;
                border:1px solid #2a4059;
                border-radius:10px;
                padding:6px 10px;
            }
            QTextEdit {
                background-color:#0f1724;
                color:#dbe7ff;
                border:1px solid #26364d;
                border-radius:12px;
                font-family:monospace;
                font-size:11px;
                padding:8px;
            }
            QTabWidget::pane {
                border:1px solid #223247;
                border-radius:12px;
                background:#0f1724;
            }
            QTabBar::tab {
                background:#132033;
                color:#cfe0ff;
                border:1px solid #24354b;
                padding:7px 12px;
                border-bottom:none;
                border-top-left-radius:8px;
                border-top-right-radius:8px;
            }
            QTabBar::tab:selected { background:#1b2d45; }
            """
        )

        self.trainer = MindTrainerInteractive(
            num_agents=num_agents,
            max_ticks_per_epoch=max_ticks_per_epoch,
            seed=seed,
            disaster_interval_ticks=400,
            relief_after_disaster=80,
            fresh_start=fresh_start,
            agent_lineup=_make_agent_lineup(num_agents),
        )
        self._snapshots: Dict[str, BrainSnapshot] = {}
        self._history: Dict[str, Deque[str]] = {}
        self._last_frame_key: Dict[str, Tuple[int, int]] = {}

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.setInterval(max(16, int(interval_ms)))
        self.timer.timeout.connect(self._on_timer_tick)
        self.running = False
        self._panel_dirty = False

        self._panel_refresh_timer = QtCore.QTimer(self)
        self._panel_refresh_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._panel_refresh_timer.setInterval(100)
        self._panel_refresh_timer.timeout.connect(self._flush_panel_refresh)
        self._panel_refresh_timer.start()

        self._build_ui()
        self._connect_signals()
        self._rebuild_agent_list()
        self._refresh_header()
        self._refresh_selected_agent()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        controls_card = QtWidgets.QFrame()
        controls_card.setProperty("card", True)
        controls = QtWidgets.QHBoxLayout(controls_card)
        controls.setContentsMargins(12, 12, 12, 12)
        controls.setSpacing(10)

        title_col = QtWidgets.QVBoxLayout()
        title_col.setSpacing(2)
        self.lblTitle = QtWidgets.QLabel("Модуль мозга")
        title_font = QtGui.QFont()
        title_font.setFamilies(["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"])
        title_font.setPointSize(16)
        title_font.setWeight(QtGui.QFont.DemiBold)
        self.lblTitle.setFont(title_font)
        self.lblEpochTick = QtWidgets.QLabel("Epoch 0 | Tick 0")
        self.lblEpochTick.setStyleSheet("color:#9db3d4;")
        title_col.addWidget(self.lblTitle)
        title_col.addWidget(self.lblEpochTick)
        controls.addLayout(title_col)

        controls.addStretch(1)

        self.comboAgent = QtWidgets.QComboBox()
        self.comboAgent.setMinimumWidth(220)
        controls.addWidget(self.comboAgent)

        self.btnRunPause = QtWidgets.QToolButton()
        self.btnRunPause.setText("Старт")
        controls.addWidget(self.btnRunPause)

        self.btnStep = QtWidgets.QToolButton()
        self.btnStep.setText("Шаг")
        controls.addWidget(self.btnStep)

        self.btnNextEpoch = QtWidgets.QToolButton()
        self.btnNextEpoch.setText("Новая эпоха")
        controls.addWidget(self.btnNextEpoch)

        self.btnSave = QtWidgets.QToolButton()
        self.btnSave.setText("Сохранить мозги")
        controls.addWidget(self.btnSave)

        self.spinSpeed = QtWidgets.QSpinBox()
        self.spinSpeed.setRange(16, 1000)
        self.spinSpeed.setSingleStep(4)
        self.spinSpeed.setSuffix(" ms")
        self.spinSpeed.setValue(self.timer.interval())
        controls.addWidget(self.spinSpeed)

        root.addWidget(controls_card)

        info_card = QtWidgets.QFrame()
        info_card.setProperty("card", True)
        info_layout = QtWidgets.QGridLayout(info_card)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.setHorizontalSpacing(16)
        info_layout.setVerticalSpacing(6)

        self.lblAgent = QtWidgets.QLabel("Агент: -")
        self.lblDrive = QtWidgets.QLabel("Drive: -")
        self.lblThought = QtWidgets.QLabel("Мысль: -")
        self.lblThought.setWordWrap(True)
        self.lblVitals = QtWidgets.QLabel("HP/Fear/Energy/Hunger: -")
        self.lblMind = QtWidgets.QLabel("Survival/Curiosity/Trauma: -")

        info_layout.addWidget(self.lblAgent, 0, 0)
        info_layout.addWidget(self.lblDrive, 0, 1)
        info_layout.addWidget(self.lblVitals, 1, 0)
        info_layout.addWidget(self.lblMind, 1, 1)
        info_layout.addWidget(self.lblThought, 2, 0, 1, 2)

        root.addWidget(info_card)

        brain_split = QtWidgets.QSplitter(Qt.Horizontal)
        brain_split.setChildrenCollapsible(False)
        self.brain3D = BrainModel3DView()
        self.liveBrain = OrganicBrainView()
        self.liveBrain.setMinimumWidth(360)
        brain_split.addWidget(self.brain3D)
        brain_split.addWidget(self.liveBrain)
        brain_split.setStretchFactor(0, 3)
        brain_split.setStretchFactor(1, 2)
        brain_split.setSizes([860, 520])
        root.addWidget(brain_split, 0)

        lower = QtWidgets.QSplitter(Qt.Horizontal)
        lower.setChildrenCollapsible(False)

        self.tabs = QtWidgets.QTabWidget()
        self.changeView = QtWidgets.QTextEdit()
        self.changeView.setReadOnly(True)
        self.stateView = QtWidgets.QTextEdit()
        self.stateView.setReadOnly(True)
        self.signalView = QtWidgets.QTextEdit()
        self.signalView.setReadOnly(True)
        self.belief2D = BeliefGraphView()

        self.tabs.addTab(self.changeView, "Изменения")
        self.tabs.addTab(self.stateView, "Состояние")
        self.tabs.addTab(self.signalView, "Память / Обучение")
        self.tabs.addTab(self.belief2D, "Убеждения 2D")

        lower.addWidget(self.tabs)
        lower.setSizes([900])
        root.addWidget(lower, 1)

        self.setCentralWidget(central)
        self.statusBar().setStyleSheet("background-color:#111827; color:#b7c8e4;")
        self.statusBar().showMessage("Готово")

    def _connect_signals(self) -> None:
        self.comboAgent.currentIndexChanged.connect(self._refresh_selected_agent)
        self.btnRunPause.clicked.connect(self._toggle_running)
        self.btnStep.clicked.connect(self._step_once)
        self.btnNextEpoch.clicked.connect(self._next_epoch)
        self.btnSave.clicked.connect(self._save_brains)
        self.spinSpeed.valueChanged.connect(self._set_interval)
        self.trainer.world_changed.connect(self._on_world_changed)
        self.trainer.epoch_changed.connect(self._on_epoch_changed)
        self.trainer.agent_list_changed.connect(self._rebuild_agent_list)

    def _selected_agent(self) -> Optional[Any]:
        agent_id = self.comboAgent.currentData()
        if not agent_id or not self.trainer.world:
            return None
        if hasattr(self.trainer.world, "get_agent_by_id"):
            return self.trainer.world.get_agent_by_id(agent_id)
        for ag in _iter_world_agents(self.trainer.world):
            if getattr(ag, "id", None) == agent_id:
                return ag
        return None

    def _brain_public(self, ag: Any) -> Dict[str, Any]:
        brain = getattr(ag, "brain", None)
        if brain is None or not hasattr(brain, "export_public_state_for_ui"):
            return {}
        data = brain.export_public_state_for_ui()
        return dict(data) if isinstance(data, dict) else {}

    def _debug_payload(self, ag: Any, public: Dict[str, Any]) -> Dict[str, Any]:
        brain = getattr(ag, "brain", None)
        fear_raw = _safe_float(public.get("fear_level", getattr(brain, "fear_level", getattr(ag, "fear", 0.0))), 0.0)
        survival = _safe_float(public.get("survival_score", getattr(brain, "survival_score", 0.0)), 0.0)
        return {
            "id": str(getattr(ag, "id", "agent")),
            "name": str(getattr(ag, "name", getattr(ag, "id", "agent"))),
            "tick": int(self.trainer.monitor.tick),
            "age_ticks": int(getattr(ag, "age_ticks", public.get("age_ticks", 0)) or 0),
            "fear": fear_raw,
            "health": float(_clamp(_safe_float(getattr(ag, "health", 0.0), 0.0), 0.0, 100.0)),
            "energy": _resource_pct(getattr(ag, "energy", public.get("energy", 0.0)), 0.0),
            "hunger": _resource_pct(getattr(ag, "hunger", public.get("hunger", 0.0)), 0.0),
            "hazards_known": int(len(getattr(ag, "known_hazards", []) or [])),
            "danger_zones_count": int(len(getattr(ag, "danger_zones", []) or [])),
            "memory_tail": list(getattr(ag, "memory_tail", []) or []),
            "mind_block": public,
            "mind_survival_score": survival,
            "mind_drive": public.get("current_drive", getattr(brain, "current_drive", "idle")),
            "mind_behavior_rules": public.get("behavior_rules", {}),
            "mind_beliefs": public.get("beliefs", []),
            "mind_memory_tail": public.get("memory_tail", []),
            "mind_trauma_map": public.get("trauma_map", []),
            "mind_curiosity_charge": public.get("curiosity_charge", getattr(brain, "curiosity_charge", 0.0) if brain else 0.0),
            "mind_last_thought": public.get("last_thought", getattr(brain, "last_thought", "")),
            "generation": int(getattr(ag, "generation", 0) or 0),
            "parents": list(getattr(ag, "parents", []) or []),
            "lineage_role": str(getattr(ag, "lineage_role", "balanced") or "balanced"),
            "tags": list(getattr(ag, "tags", []) or []),
        }

    def _state_text(self, ag: Any, public: Dict[str, Any]) -> str:
        brain = getattr(ag, "brain", None)
        rules = dict(public.get("behavior_rules", {}) or {})
        beliefs = list(public.get("beliefs", []) or [])
        trauma = list(public.get("trauma_map", []) or [])
        skills = public.get("skills")

        lines = [
            f"agent_id           = {getattr(ag, 'id', '-')}",
            f"name               = {getattr(ag, 'name', '-')}",
            f"epoch              = {self.trainer.monitor.epoch}",
            f"tick               = {self.trainer.monitor.tick}",
            f"age_ticks          = {getattr(ag, 'age_ticks', 0)}",
            f"generation         = {getattr(ag, 'generation', 0)}",
            f"parents            = {list(getattr(ag, 'parents', []) or [])}",
            f"lineage_role       = {getattr(ag, 'lineage_role', 'balanced')}",
            "",
            f"drive              = {public.get('current_drive', getattr(brain, 'current_drive', '-'))}",
            f"last_thought       = {public.get('last_thought', getattr(brain, 'last_thought', '-'))}",
            f"survival_score     = {_safe_float(public.get('survival_score', 0.0), 0.0):.2f}",
            f"fear_level         = {_safe_float(public.get('fear_level', getattr(ag, 'fear', 0.0)), 0.0):.2f}",
            f"curiosity_charge   = {_safe_float(public.get('curiosity_charge', 0.0), 0.0):.2f}",
            f"health             = {_safe_float(getattr(ag, 'health', 0.0), 0.0):.1f}",
            f"energy_pct         = {_resource_pct(getattr(ag, 'energy', public.get('energy', 0.0)), 0.0)}",
            f"hunger_pct         = {_resource_pct(getattr(ag, 'hunger', public.get('hunger', 0.0)), 0.0)}",
            f"trauma_spots       = {len(trauma)}",
            f"belief_count       = {len(beliefs)}",
            f"memory_events      = {len(list(public.get('memory_tail', []) or []))}",
            "",
            "behavior_rules:",
        ]
        if rules:
            for key, value in rules.items():
                lines.append(f"  {key:<24} = {_safe_float(value, 0.0):.2f}")
        else:
            lines.append("  -")

        lines.append("")
        lines.append("beliefs:")
        if beliefs:
            for belief in beliefs[:16]:
                cond = belief.get("if", "?")
                concl = belief.get("then", "?")
                strength = _safe_float(belief.get("strength", 0.0), 0.0)
                lines.append(f"  {cond} -> {concl} [{strength:.2f}]")
        else:
            lines.append("  -")

        if skills:
            lines.append("")
            lines.append(f"skills             = {skills}")
        return "\n".join(lines)

    def _signal_text(self, agent_id: str, public: Dict[str, Any]) -> str:
        lines: List[str] = ["memory_tail:"]
        memories = list(public.get("memory_tail", []) or [])
        if memories:
            for ev in memories[-16:]:
                if isinstance(ev, dict):
                    lines.append(f"  {_format_memory_event(ev)}")
                else:
                    lines.append(f"  {ev}")
        else:
            lines.append("  -")

        report = self.trainer.get_agent_learning_report(agent_id, tail=8)
        latest = report.get("latest")
        tail = list(report.get("tail", []) or [])

        lines.append("")
        lines.append("learning_signals:")
        if latest:
            lines.append(f"  latest_tick = {latest.get('tick', '?')}")
        if tail:
            for entry in tail[-5:]:
                lines.append(f"  [t={entry.get('tick', '?')}]")
                learned = list(entry.get("learned", []) or [])
                strengthened = list(entry.get("strengthened_links", []) or [])
                changed = list(entry.get("changed_beliefs", []) or [])
                if learned:
                    lines.append("    learned:")
                    for row in learned[:5]:
                        lines.append(f"      - {row}")
                if strengthened:
                    lines.append("    strengthened:")
                    for row in strengthened[:5]:
                        lines.append(f"      - {row}")
                if changed:
                    lines.append("    beliefs:")
                    for row in changed[:5]:
                        lines.append(f"      - {row}")
        else:
            lines.append("  -")
        return "\n".join(lines)

    def _snapshot_from_public(self, public: Dict[str, Any]) -> BrainSnapshot:
        beliefs: Dict[str, float] = {}
        for belief in list(public.get("beliefs", []) or []):
            if isinstance(belief, dict):
                beliefs[_belief_key(belief)] = _safe_float(belief.get("strength", 0.0), 0.0)

        rules = {
            str(key): _safe_float(value, 0.0)
            for key, value in dict(public.get("behavior_rules", {}) or {}).items()
        }
        memories = [dict(ev) for ev in list(public.get("memory_tail", []) or []) if isinstance(ev, dict)]
        return BrainSnapshot(
            epoch=int(self.trainer.monitor.epoch),
            tick=int(self.trainer.monitor.tick),
            drive=str(public.get("current_drive", "idle") or "idle"),
            thought=str(public.get("last_thought", "") or ""),
            survival=_safe_float(public.get("survival_score", 0.0), 0.0),
            fear=_safe_float(public.get("fear_level", 0.0), 0.0),
            curiosity=_safe_float(public.get("curiosity_charge", 0.0), 0.0),
            trauma_count=len(list(public.get("trauma_map", []) or [])),
            beliefs=beliefs,
            rules=rules,
            memories=memories,
        )

    def _append_history(self, agent_id: str, lines: List[str]) -> None:
        if not lines:
            return
        hist = self._history.setdefault(agent_id, deque(maxlen=320))
        for line in lines:
            hist.append(line)

    def _describe_changes(
        self,
        agent_id: str,
        prev: Optional[BrainSnapshot],
        curr: BrainSnapshot,
    ) -> List[str]:
        prefix = f"[e={curr.epoch} t={curr.tick}]"
        if prev is None:
            return [f"{prefix} старт наблюдения за мозгом {agent_id}"]

        lines: List[str] = []
        if prev.drive != curr.drive:
            lines.append(f"{prefix} drive: {prev.drive} -> {curr.drive}")

        if curr.thought and curr.thought != prev.thought:
            lines.append(f"{prefix} thought: {curr.thought}")

        for label, old, new, threshold in (
            ("survival", prev.survival, curr.survival, 0.02),
            ("fear", prev.fear, curr.fear, 0.02),
            ("curiosity", prev.curiosity, curr.curiosity, 0.02),
        ):
            if abs(new - old) >= threshold:
                lines.append(f"{prefix} {label}: {old:.2f} -> {new:.2f}")

        if prev.trauma_count != curr.trauma_count:
            lines.append(f"{prefix} trauma_spots: {prev.trauma_count} -> {curr.trauma_count}")

        added_beliefs = [key for key in curr.beliefs if key not in prev.beliefs]
        removed_beliefs = [key for key in prev.beliefs if key not in curr.beliefs]
        changed_beliefs: List[Tuple[float, str, float, float]] = []
        for key, new_value in curr.beliefs.items():
            if key not in prev.beliefs:
                continue
            old_value = prev.beliefs[key]
            delta = abs(new_value - old_value)
            if delta >= 0.05:
                changed_beliefs.append((delta, key, old_value, new_value))
        changed_beliefs.sort(reverse=True)

        for key in added_beliefs[:4]:
            lines.append(f"{prefix} new belief: {key} [{curr.beliefs[key]:.2f}]")
        for key in removed_beliefs[:4]:
            lines.append(f"{prefix} removed belief: {key}")
        for _, key, old_value, new_value in changed_beliefs[:5]:
            lines.append(f"{prefix} belief weight: {key} {old_value:.2f} -> {new_value:.2f}")

        rule_changes: List[Tuple[float, str, float, float]] = []
        for key, new_value in curr.rules.items():
            old_value = prev.rules.get(key)
            if old_value is None:
                lines.append(f"{prefix} new rule: {key} = {new_value:.2f}")
                continue
            delta = abs(new_value - old_value)
            if delta >= 0.03:
                rule_changes.append((delta, key, old_value, new_value))
        rule_changes.sort(reverse=True)
        for _, key, old_value, new_value in rule_changes[:5]:
            lines.append(f"{prefix} rule: {key} {old_value:.2f} -> {new_value:.2f}")

        prev_memory = {_event_signature(ev) for ev in prev.memories}
        for ev in curr.memories:
            sig = _event_signature(ev)
            if sig not in prev_memory:
                lines.append(f"{prefix} memory: {_format_memory_event(ev)}")

        latest_learning = self.trainer.get_agent_learning_report(agent_id, tail=1).get("latest")
        if latest_learning and int(latest_learning.get("tick", -1)) == curr.tick:
            learned = list(latest_learning.get("learned", []) or [])
            strengthened = list(latest_learning.get("strengthened_links", []) or [])
            changed = list(latest_learning.get("changed_beliefs", []) or [])
            if learned:
                lines.append(f"{prefix} learned: {', '.join(learned[:3])}")
            if strengthened:
                lines.append(f"{prefix} strengthened: {', '.join(strengthened[:3])}")
            if changed:
                lines.append(f"{prefix} belief_changes: {', '.join(changed[:3])}")

        return lines

    @Slot()
    def _rebuild_agent_list(self) -> None:
        prev_id = self.comboAgent.currentData()
        self.comboAgent.blockSignals(True)
        self.comboAgent.clear()
        for ag in _iter_world_agents(self.trainer.world):
            self.comboAgent.addItem(f"{getattr(ag, 'name', ag.id)} ({ag.id})", getattr(ag, "id", ""))
        idx = self.comboAgent.findData(prev_id)
        if idx < 0 and self.comboAgent.count() > 0:
            idx = 0
        if idx >= 0:
            self.comboAgent.setCurrentIndex(idx)
        self.comboAgent.blockSignals(False)
        self._refresh_selected_agent()

    @Slot()
    def _on_world_changed(self) -> None:
        self._refresh_header()
        self._refresh_selected_agent_core(update_panels=False)

    @Slot()
    def _on_epoch_changed(self) -> None:
        self._refresh_header()
        self.statusBar().showMessage(f"Эпоха {self.trainer.monitor.epoch} запущена")
        self._refresh_selected_agent()

    def _refresh_header(self) -> None:
        self.lblEpochTick.setText(f"Epoch {self.trainer.monitor.epoch} | Tick {self.trainer.monitor.tick}")

    def _refresh_selected_agent_core(self, *, update_panels: bool) -> None:
        ag = self._selected_agent()
        if ag is None:
            self.lblAgent.setText("Агент: -")
            self.lblDrive.setText("Drive: -")
            self.lblThought.setText("Мысль: -")
            self.lblVitals.setText("HP/Fear/Energy/Hunger: -")
            self.lblMind.setText("Survival/Curiosity/Trauma: -")
            self.stateView.setPlainText("Нет выбранного агента")
            self.signalView.setPlainText("Нет выбранного агента")
            self.changeView.setPlainText("Нет выбранного агента")
            self.brain3D.clear()
            self.liveBrain.clear()
            self.belief2D.update_from_brain(None)
            self._panel_dirty = False
            return

        public = self._brain_public(ag)
        brain = getattr(ag, "brain", None)
        payload = self._debug_payload(ag, public)
        self.brain3D.update_from_debug(payload, tick=self.trainer.monitor.tick)
        self.liveBrain.update_from_debug(payload, tick=self.trainer.monitor.tick)

        hp = _safe_float(getattr(ag, "health", 0.0), 0.0)
        fear = _safe_float(public.get("fear_level", getattr(ag, "fear", 0.0)), 0.0)
        energy = _resource_pct(getattr(ag, "energy", public.get("energy", 0.0)), 0.0)
        hunger = _resource_pct(getattr(ag, "hunger", public.get("hunger", 0.0)), 0.0)
        survival = _safe_float(public.get("survival_score", 0.0), 0.0)
        curiosity = _safe_float(public.get("curiosity_charge", 0.0), 0.0)
        trauma_count = len(list(public.get("trauma_map", []) or []))
        drive = str(public.get("current_drive", getattr(brain, "current_drive", "-")) or "-")
        thought = str(public.get("last_thought", getattr(brain, "last_thought", "-")) or "-")

        self.lblAgent.setText(f"Агент: {getattr(ag, 'name', ag.id)} ({ag.id})")
        self.lblDrive.setText(f"Drive: {drive}")
        self.lblThought.setText(f"Мысль: {thought}")
        self.lblVitals.setText(f"HP {hp:.1f} | Fear {fear:.2f} | Energy {energy} | Hunger {hunger}")
        self.lblMind.setText(f"Survival {survival:.2f} | Curiosity {curiosity:.2f} | Trauma {trauma_count}")

        agent_id = str(getattr(ag, "id", "agent"))
        curr_snapshot = self._snapshot_from_public(public)
        frame_key = (curr_snapshot.epoch, curr_snapshot.tick)
        last_key = self._last_frame_key.get(agent_id)
        if last_key != frame_key:
            prev_snapshot = self._snapshots.get(agent_id)
            self._append_history(agent_id, self._describe_changes(agent_id, prev_snapshot, curr_snapshot))
            self._snapshots[agent_id] = curr_snapshot
            self._last_frame_key[agent_id] = frame_key
            self._panel_dirty = True

        if not update_panels:
            self.statusBar().showMessage(f"{getattr(ag, 'name', ag.id)} | drive={drive} | tick={self.trainer.monitor.tick}")
            return

        self.belief2D.update_from_brain(brain)
        self.stateView.setPlainText(self._state_text(ag, public))
        self.signalView.setPlainText(self._signal_text(agent_id, public))

        history = list(self._history.get(agent_id, deque()))
        self.changeView.setPlainText("\n".join(history[-240:]) if history else "Изменений пока нет")
        self.changeView.moveCursor(QtGui.QTextCursor.End)
        self.statusBar().showMessage(f"{getattr(ag, 'name', ag.id)} | drive={drive} | tick={self.trainer.monitor.tick}")
        self._panel_dirty = False

    @Slot()
    def _refresh_selected_agent(self) -> None:
        self._refresh_selected_agent_core(update_panels=True)

    @Slot()
    def _flush_panel_refresh(self) -> None:
        if self._panel_dirty:
            self._refresh_selected_agent_core(update_panels=True)

    @Slot()
    def _toggle_running(self) -> None:
        self.running = not self.running
        if self.running:
            self.timer.start()
            self.btnRunPause.setText("Пауза")
            self.statusBar().showMessage("Мозг проигрывается")
        else:
            self.timer.stop()
            self.btnRunPause.setText("Старт")
            self.statusBar().showMessage("Пауза")

    @Slot()
    def _step_once(self) -> None:
        self.trainer.step_tick()

    @Slot()
    def _next_epoch(self) -> None:
        self.trainer.force_next_epoch()

    @Slot()
    def _save_brains(self) -> None:
        self.trainer.save_brains_now("./trained_brains")
        self.statusBar().showMessage("Мозги сохранены в ./trained_brains")

    @Slot()
    def _on_timer_tick(self) -> None:
        self.trainer.step_tick()

    @Slot(int)
    def _set_interval(self, value: int) -> None:
        self.timer.setInterval(max(16, int(value)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone brain-only viewer")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents in the trainer world")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for training world")
    parser.add_argument("--max-ticks", type=int, default=2000, help="Max ticks per epoch")
    parser.add_argument("--interval-ms", type=int, default=16, help="Playback timer interval in milliseconds")
    parser.add_argument("--fresh-start", action="store_true", help="Start from fresh empty brains")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = BrainOnlyWindow(
        num_agents=max(1, int(args.agents)),
        max_ticks_per_epoch=max(10, int(args.max_ticks)),
        seed=int(args.seed),
        interval_ms=max(16, int(args.interval_ms)),
        fresh_start=bool(args.fresh_start),
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
