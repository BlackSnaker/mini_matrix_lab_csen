from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from mind_trainer import MindTrainer, _anneal_exploration, _iter_agents_of, save_all_brains


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    description: str
    agents: int
    epochs: int
    ticks: int
    disaster_interval: int
    relief_after: int
    max_animals: int
    monitor_interval: int
    learning_interval: int
    ui_interval: float


PROFILES: Dict[str, TrainingProfile] = {
    "turbo": TrainingProfile(
        name="turbo",
        description="Maximum speed, sparse telemetry, headless survival grind.",
        agents=4,
        epochs=12,
        ticks=1200,
        disaster_interval=260,
        relief_after=60,
        max_animals=18,
        monitor_interval=32,
        learning_interval=16,
        ui_interval=0.14,
    ),
    "balanced": TrainingProfile(
        name="balanced",
        description="Good training quality with responsive terminal dashboard.",
        agents=5,
        epochs=14,
        ticks=1500,
        disaster_interval=300,
        relief_after=70,
        max_animals=24,
        monitor_interval=24,
        learning_interval=10,
        ui_interval=0.16,
    ),
    "survival": TrainingProfile(
        name="survival",
        description="Heavier curriculum for stronger long-term survival behavior.",
        agents=6,
        epochs=18,
        ticks=1800,
        disaster_interval=320,
        relief_after=80,
        max_animals=28,
        monitor_interval=20,
        learning_interval=8,
        ui_interval=0.18,
    ),
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _fmt_ratio(value: float) -> str:
    return f"{max(0.0, value) * 100.0:5.1f}%"


def _fmt_float(value: float) -> str:
    return f"{value:6.2f}"


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if result != result or result in (float("inf"), float("-inf")):
        return default
    return result


def _as_pct(value: Any, default: float = 0.0) -> float:
    result = _as_float(value, default)
    if result <= 1.5:
        result *= 100.0
    return max(0.0, min(100.0, result))


def _belief_strength(value: Any) -> float:
    if isinstance(value, dict):
        return _clamp01(_as_float(value.get("strength", 0.0), 0.0))
    return _clamp01(_as_float(getattr(value, "strength", 0.0), 0.0))


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


@dataclass(frozen=True)
class TerminalBrainFrame:
    agent_id: str
    agent_name: str
    generation: int
    age_ticks: int
    drive: str
    survival: float
    fear: float
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
    last_thought: str


def _extract_terminal_brain_frame(agent: Any) -> Optional[TerminalBrainFrame]:
    if agent is None or not hasattr(agent, "serialize_public_state"):
        return None

    info = dict(agent.serialize_public_state() or {})
    mind = dict(info.get("mind", {}) or {})
    beliefs = list(mind.get("beliefs", []) or [])
    memories = list(info.get("memory_tail") or mind.get("memory_tail", []) or [])
    rules = dict(mind.get("behavior_rules", {}) or {})
    trauma_map = list(mind.get("trauma_map", []) or [])

    fear = _clamp01(_as_float(info.get("fear", 0.0), 0.0))
    energy = _as_pct(info.get("energy", 100.0), 100.0)
    hunger = _as_pct(info.get("hunger", 0.0), 0.0)
    survival = _clamp01(_as_float(mind.get("survival_score", 0.0), 0.0))
    drive = str(mind.get("current_drive", "idle") or "idle")
    last_thought = str(mind.get("last_thought", "") or "")
    age_ticks = int(_as_float(info.get("age_ticks", 0), 0.0))
    generation = int(_as_float(info.get("generation", 0), 0.0))
    hazards = int(_as_float(info.get("hazards_known", 0), 0.0))
    danger = int(_as_float(info.get("danger_zones_count", 0), 0.0))
    curiosity = _clamp01(_as_float(mind.get("curiosity_charge", 0.0), 0.0))
    beliefs_count = len(beliefs)
    memories_count = len(memories)
    trauma_count = len(trauma_map)

    belief_strengths = [_belief_strength(belief) for belief in beliefs[:16]]
    if not belief_strengths:
        belief_strengths = [0.22]

    critical_weight = 0.0
    for memory in memories[-16:]:
        if isinstance(memory, dict):
            level = str(memory.get("level", "info"))
        else:
            level = str(getattr(memory, "level", "info"))
        critical_weight += _level_weight(level)
    critical_weight = min(1.0, critical_weight / 8.0)

    mean_strength = sum(belief_strengths) / max(1, len(belief_strengths))
    age_norm = min(1.0, age_ticks / 1200.0)
    rule_factor = min(1.0, _rule_count(rules) / 8.0)
    memory_factor = min(1.0, memories_count / 18.0)
    hazards_factor = min(1.0, (hazards + danger) / 16.0)
    trauma_factor = min(1.0, trauma_count / 10.0)

    activity = _clamp01(
        0.34 * _drive_activity(drive)
        + 0.20 * mean_strength
        + 0.16 * critical_weight
        + 0.14 * curiosity
        + 0.16 * memory_factor
    )
    plasticity = _clamp01(
        0.28 * (energy / 100.0)
        + 0.22 * (1.0 - hunger / 100.0)
        + 0.20 * (1.0 - fear)
        + 0.18 * curiosity
        + 0.12 * (1.0 - trauma_factor)
    )
    complexity = _clamp01(
        0.18
        + 0.24 * min(1.0, beliefs_count / 12.0)
        + 0.18 * rule_factor
        + 0.14 * age_norm
        + 0.12 * min(1.0, generation / 4.0)
        + 0.14 * mean_strength
    )
    stress = _clamp01(
        0.42 * fear
        + 0.18 * (1.0 - survival)
        + 0.16 * (hunger / 100.0)
        + 0.12 * hazards_factor
        + 0.12 * trauma_factor
    )
    coherence = _clamp01(
        0.40 * survival
        + 0.24 * (1.0 - fear)
        + 0.18 * mean_strength
        + 0.18 * (1.0 - stress)
    )

    return TerminalBrainFrame(
        agent_id=str(info.get("id", getattr(agent, "id", "agent"))),
        agent_name=str(info.get("name", getattr(agent, "name", "agent"))),
        generation=generation,
        age_ticks=age_ticks,
        drive=drive,
        survival=survival,
        fear=fear,
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
        last_thought=last_thought,
    )


def _brain_region_activity(frame: TerminalBrainFrame) -> Dict[str, float]:
    beliefs = min(1.0, frame.beliefs_count / 12.0)
    memories = min(1.0, frame.memories_count / 14.0)
    hazards = min(1.0, (frame.hazards_count + frame.danger_count) / 12.0)
    return {
        "frontal": _clamp01(0.48 * frame.activity + 0.28 * frame.curiosity + 0.24 * frame.plasticity),
        "parietal": _clamp01(0.52 * frame.complexity + 0.26 * frame.coherence + 0.22 * beliefs),
        "temporal": _clamp01(0.46 * memories + 0.22 * beliefs + 0.18 * frame.activity + 0.14 * frame.stress),
        "occipital": _clamp01(0.28 * frame.activity + 0.24 * frame.curiosity + 0.26 * frame.coherence + 0.22 * frame.complexity),
        "limbic": _clamp01(0.54 * frame.stress + 0.18 * frame.fear + 0.16 * hazards + 0.12 * (1.0 - frame.survival)),
        "cerebellum": _clamp01(0.38 * frame.coherence + 0.34 * frame.activity + 0.28 * frame.plasticity),
    }


class TurboMindTrainer(MindTrainer):
    def __init__(
        self,
        *,
        monitor_interval_ticks: int,
        learning_interval_ticks: int,
        ui_interval_sec: float,
        checkpoint_interval_ticks: int,
        export_dir: str,
        continuous_mode: bool,
        progress_hook: Optional[Callable[["TurboMindTrainer"], None]] = None,
        disable_jsonl: bool = True,
        disable_csv: bool = True,
        disable_snapshots: bool = True,
        keep_verbose_logs: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monitor_interval_ticks = max(1, int(monitor_interval_ticks))
        self.learning_interval_ticks = max(0, int(learning_interval_ticks))
        self.ui_interval_sec = max(0.05, float(ui_interval_sec))
        self.checkpoint_interval_ticks = max(0, int(checkpoint_interval_ticks))
        self.export_dir = export_dir
        self.continuous_mode = bool(continuous_mode)
        self.progress_hook = progress_hook
        self.disable_jsonl = bool(disable_jsonl)
        self.disable_csv = bool(disable_csv)
        self.disable_snapshots = bool(disable_snapshots)
        self.keep_verbose_logs = bool(keep_verbose_logs)
        self.verbose = bool(keep_verbose_logs)

        self.run_started_at: float = 0.0
        self.epoch_started_at: float = 0.0
        self.total_ticks_done: int = 0
        self._last_ui_emit_at: float = 0.0
        self.stop_requested: bool = False
        self.stop_reason: str = ""
        self.checkpoint_count: int = 0
        self.last_checkpoint_tick: int = 0

        if self.disable_snapshots:
            self.snapshot_every = 0

    def _log(self, msg: str):
        if self.keep_verbose_logs:
            super()._log(msg)

    def _append_monitor_jsonl(self, row):
        if self.disable_jsonl:
            return
        super()._append_monitor_jsonl(row)

    def _append_learning_jsonl(self, agent_id, row):
        if self.disable_jsonl:
            return
        super()._append_learning_jsonl(agent_id, row)

    def _export_monitor_csv(self):
        if self.disable_csv:
            return
        super()._export_monitor_csv()

    def _maybe_snapshot_world(self, t: int):
        if self.disable_snapshots:
            return
        super()._maybe_snapshot_world(t)

    def _emit_progress(self, force: bool = False):
        if self.progress_hook is None:
            return
        now = time.perf_counter()
        if not force and (now - self._last_ui_emit_at) < self.ui_interval_sec:
            return
        self._last_ui_emit_at = now
        self.progress_hook(self)

    def request_stop(self, reason: str = "stop requested"):
        self.stop_requested = True
        self.stop_reason = reason
        self.monitor.note = f"{reason}; saving brains before exit"
        self._emit_progress(force=True)

    def _save_runtime_checkpoint(self, reason: str):
        if self._world is None:
            return
        save_all_brains(self._world, self.export_dir)
        self.checkpoint_count += 1
        self.last_checkpoint_tick = self.total_ticks_done
        self.monitor.note = reason
        self._emit_progress(force=True)

    def _auto_save_generation(self):
        if self._world is None:
            return
        self._save_runtime_checkpoint(f"checkpoint saved after epoch {self.monitor.epoch + 1}")

    def _spawn_world(self, epoch_idx: int):
        super()._spawn_world(epoch_idx)
        self.epoch_started_at = time.perf_counter()
        self._emit_progress(force=True)

    def _episode_tick_loop(self):
        assert self._world is not None
        dead_marked = False
        last_monitor_tick = -1

        for t in range(self.max_ticks_per_episode):
            if self.stop_requested:
                self.monitor.note = self.stop_reason or "stop requested"
                break

            _anneal_exploration(self._world, t, self.max_ticks_per_episode)
            self._maybe_inject_disaster(t)
            self._maybe_inject_relief(t)

            self._world.tick()
            self.total_ticks_done += 1
            self.monitor.tick = t

            if self.learning_interval_ticks > 0 and (t % self.learning_interval_ticks == 0):
                MindTrainer._collect_learning_signals(self, t)

            if self.checkpoint_interval_ticks > 0 and (self.total_ticks_done % self.checkpoint_interval_ticks == 0):
                self._save_runtime_checkpoint(
                    f"autosave checkpoint @ tick {self.total_ticks_done:,}"
                )

            if t % 60 == 0:
                self._nudge_idle_agents()

            if t == 0 or (t % self.monitor_interval_ticks == 0) or (t == self.max_ticks_per_episode - 1):
                MindTrainer._collect_monitor_stats(self)
                last_monitor_tick = t
                if not dead_marked and self.monitor.alive_ratio <= 0.0:
                    self.monitor.note = "early stop: all agents dead"
                    dead_marked = True
                    self._emit_progress(force=True)
                    break

            self._emit_progress(force=False)
            if self.stop_requested:
                self.monitor.note = self.stop_reason or "stop requested"
                break

        if last_monitor_tick != self.monitor.tick:
            MindTrainer._collect_monitor_stats(self)

        dead_count = sum(
            1 for agent in _iter_agents_of(self._world)
            if not getattr(agent, "is_alive", lambda: True)()
        )
        if self.stop_requested:
            self.monitor.note = self.stop_reason or "graceful stop requested"
        elif dead_count > 0 and not dead_marked:
            self.monitor.note = f"episode end: {dead_count} dead"
        elif not dead_marked:
            self.monitor.note = "episode end: all survived"

        self._emit_progress(force=True)

    def train(self):
        self.run_started_at = time.perf_counter()
        self.epoch_started_at = self.run_started_at
        self.total_ticks_done = 0
        self._last_ui_emit_at = 0.0
        self.stop_requested = False
        self.stop_reason = ""
        self.checkpoint_count = 0
        self.last_checkpoint_tick = 0
        last_world = None
        try:
            epoch_idx = 0
            while True:
                if not self.continuous_mode and epoch_idx >= self.epochs:
                    break
                self._spawn_world(epoch_idx)
                self._episode_tick_loop()
                self._auto_save_generation()
                last_world = self._world
                if self.stop_requested:
                    break
                epoch_idx += 1

            self._log("training complete")
            self._log(f"final monitor: {self.monitor.to_dict()}")
            return last_world
        finally:
            if self._world is not None:
                try:
                    self._save_runtime_checkpoint("final brain save")
                except Exception:
                    pass
            self._export_monitor_csv()
            self._emit_progress(force=True)


class TerminalDashboard:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLACK_BG = "\033[40m"
    GREEN = "\033[38;5;34m"
    GREEN_DIM = "\033[38;5;22m"
    GREEN_SOFT = "\033[38;5;40m"
    GREEN_BRIGHT = "\033[38;5;46m"
    GREEN_GLOW = "\033[38;5;119m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]{}<>/\\|*+-=~:;."
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, profile: TrainingProfile):
        self.profile = profile
        self.is_tty = sys.stdout.isatty()
        self.last_plain_emit: float = 0.0
        self._cursor_hidden = False

    def start(self):
        if self.is_tty:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
            self._cursor_hidden = True

    def stop(self):
        if self._cursor_hidden:
            sys.stdout.write("\033[?25h" + self.RESET)
            sys.stdout.flush()
            self._cursor_hidden = False

    def _color(self, text: str, code: str) -> str:
        if not self.is_tty:
            return text
        return f"{self.BLACK_BG}{code}{text}{self.RESET}"

    def _bar(self, frac: float, width: int, color: str) -> str:
        filled = int(round(_clamp01(frac) * width))
        core = "=" * filled + "." * max(0, width - filled)
        return self._color(core, color)

    def _strip_ansi(self, text: str) -> str:
        return self._ANSI_RE.sub("", text)

    def _visual_len(self, text: str) -> int:
        return len(self._strip_ansi(text))

    def _fit(self, text: str, width: int) -> str:
        plain = self._strip_ansi(text)
        if len(plain) <= width:
            return text + " " * (width - len(plain))
        if width <= 3:
            return plain[:width]
        return plain[: width - 3] + "..."

    def _matrix_char(self, slot: int, phase: int, salt: int = 0) -> str:
        idx = (slot * 17 + phase * 13 + salt * 19 + (slot // 3) * 7) % len(self.GLYPHS)
        return self.GLYPHS[idx]

    def _matrix_rain(self, width: int, phase: int, salt: int = 0) -> str:
        parts: List[str] = []
        for col in range(max(1, width)):
            if (col + phase + salt) % 11 != 0:
                parts.append(" ")
                continue
            ch = self._matrix_char(col, phase, salt)
            if (col + phase + salt) % 33 == 0:
                parts.append(self._color(ch, self.GREEN_GLOW + self.BOLD))
            elif (col + phase + salt) % 22 == 0:
                parts.append(self._color(ch, self.GREEN_BRIGHT))
            else:
                parts.append(self._color(ch, self.GREEN_DIM))
        return "".join(parts)

    def _style_line(self, text: str) -> str:
        if text.startswith(":: "):
            return self._color(text, self.GREEN_GLOW + self.BOLD)
        if text.startswith("note:") or text.startswith("signal:"):
            return self._color(text, self.GREEN_BRIGHT)
        if text.startswith("path:") or text.startswith("mode:"):
            return self._color(text, self.GREEN_GLOW)
        if text.startswith("  "):
            return self._color(text, self.GREEN_SOFT)
        if "dead" in text:
            return self._color(text, self.GREEN_BRIGHT)
        return self._color(text, self.GREEN_SOFT)

    def _frame_border(self, inner_width: int, label: str, phase: int) -> str:
        label_text = f" {label.upper()} "
        dash_count = max(0, inner_width - len(label_text))
        if not self.is_tty:
            return "+" + label_text + "-" * dash_count + "+"
        left = self._color("+", self.GREEN_DIM)
        center = self._color(label_text, self.GREEN_GLOW + self.BOLD)
        right = self._color("-" * dash_count + "+", self.GREEN_DIM)
        return left + center + right

    def _frame_line(self, text: str, inner_width: int, row_idx: int, phase: int) -> str:
        content = self._style_line(self._fit(text, inner_width))
        border = self._color("|", self.GREEN_DIM)
        return f"{border} {content} {border}"

    def _center_text(self, text: str, width: int) -> str:
        pad = max(0, (width - self._visual_len(text)) // 2)
        return (" " * pad) + text

    def _brain_outline(self, text: str) -> str:
        return self._color(text, self.GREEN_DIM)

    def _brain_cluster(self, symbol: str, level: float, *, count: int = 3, hot: bool = False) -> str:
        value = _clamp01(level)
        if hot and value >= 0.82:
            return self._color(symbol.upper() * count, self.RED + self.BOLD)
        if hot and value >= 0.58:
            return self._color(symbol.upper() * count, self.YELLOW + self.BOLD)
        if value >= 0.84:
            return self._color(symbol.upper() * count, self.GREEN_GLOW + self.BOLD)
        if value >= 0.60:
            return self._color(symbol.upper() * count, self.GREEN_BRIGHT)
        if value >= 0.34:
            return self._color(symbol.lower() * count, self.GREEN_SOFT)
        if value >= 0.16:
            return self._color(symbol.lower() * count, self.GREEN_DIM)
        return self._color("." * count, self.GREEN_DIM)

    def _brain_meter(self, label: str, value: float, *, hot: bool = False) -> str:
        if hot and value >= 0.82:
            color = self.RED
        elif hot and value >= 0.58:
            color = self.YELLOW
        elif value >= 0.76:
            color = self.GREEN_GLOW
        elif value >= 0.48:
            color = self.GREEN_BRIGHT
        else:
            color = self.GREEN_SOFT
        return f"{label:<10}[{self._bar(value, 10, color)}] {int(round(_clamp01(value) * 100.0)):>3d}%"

    def _ranked_agents(self, trainer: TurboMindTrainer) -> List[Dict[str, Any]]:
        world = getattr(trainer, "_world", None)
        if world is None:
            return []

        rows = []
        for agent in _iter_agents_of(world):
            brain = getattr(agent, "brain", None)
            rows.append(
                {
                    "agent": agent,
                    "name": getattr(agent, "name", getattr(agent, "agent_id", "agent")),
                    "alive": 1 if getattr(agent, "is_alive", lambda: True)() else 0,
                    "survival": float(getattr(brain, "survival_score", 0.0) or 0.0),
                    "hp": float(getattr(agent, "health", 0.0) or 0.0),
                    "fear": float(getattr(agent, "fear", 0.0) or 0.0),
                    "drive": str(getattr(brain, "current_drive", "")) if brain is not None else "",
                }
            )
        rows.sort(key=lambda row: (row["alive"], row["survival"], row["hp"]), reverse=True)
        return rows

    def _top_agents(self, ranked_rows: List[Dict[str, Any]]) -> List[str]:
        out = []
        for idx, row in enumerate(ranked_rows[:3], start=1):
            state = "alive" if row["alive"] else "dead "
            out.append(
                f"{idx}. {row['name']:<10} {state}  surv={row['survival']:.2f}  "
                f"hp={row['hp']:.0f}  fear={row['fear']:.2f}  drive={row['drive'] or '-'}"
            )
        return out or ["no agents yet"]

    def _focus_brain(self, ranked_rows: List[Dict[str, Any]]) -> Optional[TerminalBrainFrame]:
        if not ranked_rows:
            return None
        return _extract_terminal_brain_frame(ranked_rows[0].get("agent"))

    def _brain_model_lines(self, frame: TerminalBrainFrame, inner_width: int) -> List[str]:
        region = _brain_region_activity(frame)
        frontal = self._brain_cluster("F", region["frontal"])
        parietal = self._brain_cluster("P", region["parietal"])
        temporal = self._brain_cluster("T", region["temporal"])
        occipital = self._brain_cluster("O", region["occipital"])
        limbic = self._brain_cluster("L", region["limbic"], hot=True)
        cerebellum = self._brain_cluster("C", region["cerebellum"], count=4)
        fissure = self._brain_outline("||")

        raw_lines = [
            self._center_text(self._brain_outline("            .-=================-."), inner_width),
            self._center_text(
                self._brain_outline("        .-~~  ")
                + frontal + self._brain_outline(" ")
                + parietal + self._brain_outline(" ") + fissure + self._brain_outline(" ")
                + parietal + self._brain_outline(" ") + frontal
                + self._brain_outline("  ~~-."),
                inner_width,
            ),
            self._center_text(
                self._brain_outline("     .-~    ")
                + frontal + self._brain_outline(" ")
                + temporal + self._brain_outline("  ") + fissure + self._brain_outline("  ")
                + temporal + self._brain_outline(" ") + occipital
                + self._brain_outline("    ~-."),
                inner_width,
            ),
            self._center_text(
                self._brain_outline("    /       ")
                + limbic + self._brain_outline(" ")
                + temporal + self._brain_outline("  ") + fissure + self._brain_outline("  ")
                + temporal + self._brain_outline(" ") + limbic
                + self._brain_outline("       \\"),
                inner_width,
            ),
            self._center_text(
                self._brain_outline("    \\         ")
                + cerebellum + self._brain_outline(" ") + fissure + self._brain_outline(" ")
                + cerebellum
                + self._brain_outline("         /"),
                inner_width,
            ),
            self._center_text(self._brain_outline("      `-.___           ||           ___.-'"), inner_width),
        ]
        return raw_lines

    def render(self, trainer: TurboMindTrainer, final: bool = False):
        elapsed = time.perf_counter() - max(1e-6, trainer.run_started_at or time.perf_counter())
        ticks_done = trainer.total_ticks_done
        tps = ticks_done / max(1e-6, elapsed)
        if trainer.continuous_mode:
            total_goal_ticks = None
            eta_text = "--:--"
            epoch_goal_label = "inf"
        else:
            total_goal_ticks = max(1, trainer.epochs * trainer.max_ticks_per_episode)
            eta_text = _fmt_duration((total_goal_ticks - ticks_done) / max(1e-6, tps))
            epoch_goal_label = str(trainer.epochs)

        if not self.is_tty:
            now = time.perf_counter()
            if not final and (now - self.last_plain_emit) < max(1.0, trainer.ui_interval_sec * 4.0):
                return
            self.last_plain_emit = now
            line = (
                f"[brain-forge] profile={self.profile.name} "
                f"epoch={trainer.monitor.epoch + 1}/{epoch_goal_label} "
                f"tick={trainer.monitor.tick + 1}/{trainer.max_ticks_per_episode} "
                f"alive={_fmt_ratio(trainer.monitor.alive_ratio)} "
                f"survival={trainer.monitor.avg_survival:.2f} "
                f"tps={tps:,.0f} "
                f"note={trainer.monitor.note}"
            )
            print(line)
            return

        width = max(86, min(122, shutil.get_terminal_size((104, 32)).columns))
        phase = int(time.perf_counter() * 14.0)
        inner_width = max(56, width - 4)
        lines: List[str] = []
        ranked_rows = self._ranked_agents(trainer)
        focus_brain = self._focus_brain(ranked_rows)
        title = "Brain Forge // Matrix Console"
        status = "Final Summary" if final else "Live Training"
        lines.append(self._matrix_rain(width, phase, salt=1))
        lines.append(self._frame_border(inner_width, title, phase))
        lines.append(
            self._frame_line(
                f":: {status} :: profile={self.profile.name}  epoch={trainer.monitor.epoch + 1}/{epoch_goal_label}  "
                f"tick={trainer.monitor.tick + 1}/{trainer.max_ticks_per_episode}",
                inner_width,
                0,
                phase,
            )
        )
        lines.append(
            self._frame_line(
                f"mode: {'continuous until Ctrl+C' if trainer.continuous_mode else 'scheduled run'}  "
                f"elapsed={_fmt_duration(elapsed)}  eta={eta_text}  ticks/s={tps:,.0f}  total={ticks_done:,}",
                inner_width,
                1,
                phase,
            )
        )
        lines.append(self._frame_border(inner_width, "Progress", phase + 1))
        if trainer.continuous_mode:
            lines.append(
                self._frame_line(
                    f"signal: training runs continuously until interrupted by operator",
                    inner_width,
                    2,
                    phase,
                )
            )
        else:
            lines.append(
                self._frame_line(
                    f"epoch progress  [{self._bar((trainer.monitor.epoch + 1) / max(1, trainer.epochs), 28, self.GREEN_SOFT)}]  "
                    f"{trainer.monitor.epoch + 1}/{trainer.epochs}",
                    inner_width,
                    2,
                    phase,
                )
            )
        lines.append(
            self._frame_line(
                f"tick progress   [{self._bar((trainer.monitor.tick + 1) / max(1, trainer.max_ticks_per_episode), 28, self.GREEN_BRIGHT)}]  "
                f"{trainer.monitor.tick + 1}/{trainer.max_ticks_per_episode}",
                inner_width,
                3,
                phase,
            )
        )
        lines.append(
            self._frame_line(
                f"alive ratio     [{self._bar(trainer.monitor.alive_ratio, 28, self.GREEN_BRIGHT)}]  {_fmt_ratio(trainer.monitor.alive_ratio)}",
                inner_width,
                4,
                phase,
            )
        )
        lines.append(
            self._frame_line(
                f"learning={trainer.monitor.learning_events}  beliefs={trainer.monitor.belief_changes}  "
                f"peer lessons={trainer.monitor.peer_shared_lessons}  checkpoints={trainer.checkpoint_count}",
                inner_width,
                5,
                phase,
            )
        )
        lines.append(self._frame_border(inner_width, "Vitals", phase + 2))
        lines.append(
            self._frame_line(
                f"hp={_fmt_float(trainer.monitor.avg_hp)}  energy={_fmt_float(trainer.monitor.avg_energy)}  "
                f"hunger={_fmt_float(trainer.monitor.avg_hunger)}  survival={_fmt_float(trainer.monitor.avg_survival)}",
                inner_width,
                6,
                phase,
            )
        )
        lines.append(
            self._frame_line(
                f"fear={_fmt_float(trainer.monitor.avg_fear)}  curiosity={_fmt_float(trainer.monitor.avg_curiosity)}  "
                f"panic={_fmt_ratio(trainer.monitor.panic_ratio)}  tamed={_fmt_ratio(trainer.monitor.tamed_ratio)}  "
                f"cling={_fmt_ratio(trainer.monitor.cling_ratio)}",
                inner_width,
                7,
                phase,
            )
        )
        lines.append(
            self._frame_line(
                f"hazards={trainer.monitor.avg_known_hazards:5.2f}  trauma={trainer.monitor.avg_trauma_spots:5.2f}",
                inner_width,
                8,
                phase,
            )
        )
        lines.append(self._frame_border(inner_width, "Brain Model", phase + 3))
        if focus_brain is None:
            lines.append(self._frame_line("signal: brain model will appear after first active agent spawn", inner_width, 9, phase))
        else:
            lines.append(
                self._frame_line(
                    f":: {focus_brain.agent_name} :: drive={focus_brain.drive or 'idle'}  gen={focus_brain.generation}  "
                    f"age={focus_brain.age_ticks}  survival={focus_brain.survival:.2f}",
                    inner_width,
                    9,
                    phase,
                )
            )
            for idx, brain_line in enumerate(self._brain_model_lines(focus_brain, inner_width), start=10):
                lines.append(self._frame_line(brain_line, inner_width, idx, phase))
            region = _brain_region_activity(focus_brain)
            lines.append(
                self._frame_line(
                    self._brain_meter("frontal", region["frontal"])
                    + "  "
                    + self._brain_meter("parietal", region["parietal"]),
                    inner_width,
                    16,
                    phase,
                )
            )
            lines.append(
                self._frame_line(
                    self._brain_meter("temporal", region["temporal"])
                    + "  "
                    + self._brain_meter("occipital", region["occipital"]),
                    inner_width,
                    17,
                    phase,
                )
            )
            lines.append(
                self._frame_line(
                    self._brain_meter("limbic", region["limbic"], hot=True)
                    + "  "
                    + self._brain_meter("cerebell.", region["cerebellum"]),
                    inner_width,
                    18,
                    phase,
                )
            )
            lines.append(
                self._frame_line(
                    f"beliefs={focus_brain.beliefs_count}  memories={focus_brain.memories_count}  "
                    f"hazards={focus_brain.hazards_count + focus_brain.danger_count}  trauma={focus_brain.trauma_count}  "
                    f"plasticity={focus_brain.plasticity:.2f}  coherence={focus_brain.coherence:.2f}",
                    inner_width,
                    19,
                    phase,
                )
            )
            lines.append(
                self._frame_line(
                    "thought: " + (focus_brain.last_thought or "..."),
                    inner_width,
                    20,
                    phase,
                )
            )
        lines.append(self._frame_border(inner_width, "Top Agents", phase + 4))
        lines.append(self._frame_line(":: best survival candidates ::", inner_width, 21, phase))
        for idx, row in enumerate(self._top_agents(ranked_rows), start=22):
            lines.append(self._frame_line("  " + row, inner_width, idx, phase))
        lines.append(self._frame_border(inner_width, "Operator", phase + 5))
        note = trainer.monitor.note or "running"
        lines.append(self._frame_line("note: " + note, inner_width, 25, phase))
        lines.append(self._frame_line("signal: Ctrl+C -> graceful stop -> save brains -> exit", inner_width, 26, phase))
        lines.append(self._frame_line("path: exports -> ./trained_brains   runtime brains -> ./brains", inner_width, 27, phase))
        lines.append(self._frame_border(inner_width, "Ready", phase + 6))

        sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brain_forge.py",
        description="Fast headless terminal trainer for agent brains.",
    )
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), default="balanced")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--agents", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--monitor-interval", type=int)
    parser.add_argument("--learning-interval", type=int)
    parser.add_argument("--ui-interval", type=float)
    parser.add_argument("--checkpoint-every", type=int, default=400)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--logs-dir", default="./trainer_logs")
    parser.add_argument("--snapshots-dir", default="./trainer_snapshots")
    parser.add_argument("--export-dir", default="./trained_brains")
    parser.add_argument("--keep-jsonl", action="store_true")
    parser.add_argument("--keep-csv", action="store_true")
    parser.add_argument("--verbose-trainer", action="store_true")
    return parser


def _resolve_profile(args: argparse.Namespace) -> TrainingProfile:
    base = PROFILES[args.profile]
    return TrainingProfile(
        name=base.name,
        description=base.description,
        agents=args.agents or base.agents,
        epochs=args.epochs or base.epochs,
        ticks=args.ticks or base.ticks,
        disaster_interval=base.disaster_interval,
        relief_after=base.relief_after,
        max_animals=base.max_animals,
        monitor_interval=args.monitor_interval or base.monitor_interval,
        learning_interval=args.learning_interval or base.learning_interval,
        ui_interval=args.ui_interval or base.ui_interval,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    profile = _resolve_profile(args)
    continuous_mode = bool(args.continuous or args.epochs is None)

    dashboard = TerminalDashboard(profile)
    trainer = TurboMindTrainer(
        num_agents=profile.agents,
        epochs=profile.epochs,
        max_ticks_per_episode=profile.ticks,
        seed=args.seed,
        fresh_start=bool(args.fresh_start),
        disaster_interval_ticks=profile.disaster_interval,
        relief_after_disaster=profile.relief_after,
        max_animals_per_world=profile.max_animals,
        snapshot_every=args.snapshot_every,
        logs_dir=args.logs_dir,
        snapshots_dir=args.snapshots_dir,
        monitor_interval_ticks=profile.monitor_interval,
        learning_interval_ticks=profile.learning_interval,
        ui_interval_sec=profile.ui_interval,
        checkpoint_interval_ticks=args.checkpoint_every,
        export_dir=args.export_dir,
        continuous_mode=continuous_mode,
        progress_hook=dashboard.render,
        disable_jsonl=not args.keep_jsonl,
        disable_csv=not args.keep_csv,
        disable_snapshots=(args.snapshot_every <= 0),
        keep_verbose_logs=bool(args.verbose_trainer),
    )

    stop_signal_hits = {"count": 0}
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_graceful_stop(signum, _frame):
        stop_signal_hits["count"] += 1
        sig_name = signal.Signals(signum).name
        if stop_signal_hits["count"] == 1:
            trainer.request_stop(f"{sig_name.lower()} received")
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _request_graceful_stop)
    signal.signal(signal.SIGTERM, _request_graceful_stop)

    dashboard.start()
    gc_state = gc.isenabled()
    if gc_state:
        gc.disable()

    try:
        trainer.train()
        trainer.export_trained_brains(args.export_dir)
        dashboard.render(trainer, final=True)
        return 0
    except KeyboardInterrupt:
        trainer.monitor.note = "interrupted by user"
        try:
            trainer.export_trained_brains(args.export_dir)
        except Exception:
            pass
        dashboard.render(trainer, final=True)
        return 130
    finally:
        if gc_state:
            gc.enable()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        dashboard.stop()


if __name__ == "__main__":
    raise SystemExit(main())
