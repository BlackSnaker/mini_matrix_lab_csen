#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from brain_io import save_brain
from mind_core import ConsciousnessBlock
from training_room import (
    DEFAULT_ROOM_AGENT_ID,
    LESSON_IDLE,
    LESSON_SPARRING,
    LESSON_WOLF,
    TrainingRoomManager,
)
from world import Agent, World


DOJO_REPORTS_DIR = "dojo_reports"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    if result != result or result in (float("inf"), float("-inf")):
        return float(default)
    return result


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pct(value: Any, default: float = 0.0) -> float:
    raw = _safe_float(value, default)
    if raw <= 1.5:
        raw *= 100.0
    return _clamp(raw, 0.0, 100.0)


def _fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _fmt_tick(tick: int) -> str:
    return f"{max(0, int(tick)):04d}"


def _trim(text: str, width: int) -> str:
    raw = str(text or "")
    if width <= 0:
        return ""
    if len(raw) <= width:
        return raw
    if width <= 3:
        return raw[:width]
    return raw[: width - 3] + "..."


def _bar(value: float, width: int, *, fill: str = "#", empty: str = ".") -> str:
    ratio = _clamp(float(value), 0.0, 1.0)
    done = int(round(ratio * max(1, width)))
    return fill * done + empty * max(0, width - done)


def _lesson_label(mode: str) -> str:
    key = str(mode or LESSON_IDLE)
    if key == LESSON_SPARRING:
        return "Morpheus Sparring"
    if key == LESSON_WOLF:
        return "Wolf Drill"
    return "Idle"


def _profile_text(agent: Any) -> Dict[str, Any]:
    brain = getattr(agent, "brain", None)
    return {
        "drive": str(getattr(brain, "current_drive", "idle") or "idle"),
        "survival": _safe_float(getattr(brain, "survival_score", 0.0), 0.0),
        "curiosity": _safe_float(getattr(brain, "curiosity_charge", 0.0), 0.0),
        "thought": str(getattr(brain, "last_thought", "") or ""),
        "beliefs": len(list(getattr(brain, "beliefs", []) or [])),
        "memories": len(list(getattr(brain, "memory_tail", []) or [])),
    }


def _agent_spec(agent_id: str) -> Dict[str, str]:
    key = str(agent_id or DEFAULT_ROOM_AGENT_ID).strip() or DEFAULT_ROOM_AGENT_ID
    if key == "agent_1":
        return {
            "id": key,
            "name": "Agent 1",
            "persona": (
                "Ты Agent 1. Первый агент лаборатории CSEN. "
                "Комната Морфеуса для тебя безопасная среда для боевого обучения. "
                "Ты учишься держать стойку, принимать удар и отвечать волкам."
            ),
        }
    return {
        "id": key,
        "name": key,
        "persona": (
            "Ты учебный агент CSEN. Комната Морфеуса для тебя безопасная среда "
            "для боевой подготовки и развития устойчивости."
        ),
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class DojoArgs:
    agent_id: str
    ticks: int
    lesson: str
    switch_every: int
    seed: int
    ui_interval: float
    save_every: int
    room_autosave: int
    world_size: float
    continuous: bool
    until_mastery: bool
    mastery_skill: float
    mastery_mentor_hits: int
    mastery_wolf_hits: int
    plain: bool
    fresh_start: bool
    progress_out: str
    report_out: str


class MorpheusDojoRunner:
    def __init__(self, args: DojoArgs) -> None:
        self.args = args
        self._stop_requested = False
        self._plain_last_emit = 0.0
        self._render_last_at = 0.0
        self._last_save_tick = -10**9
        self._start_monotonic = time.perf_counter()
        self._seen_event_keys: Deque[Tuple[Any, ...]] = deque()
        self._activity_lines: Deque[str] = deque(maxlen=8)
        self._seen_event_key_set: set[Tuple[Any, ...]] = set()
        self._last_phase_index = -1

        random.seed(int(args.seed))

        spec = _agent_spec(args.agent_id)
        self.world = World(width=float(args.world_size), height=float(args.world_size))
        self.lab_agent = Agent(
            agent_id=spec["id"],
            name=spec["name"],
            x=float(args.world_size) * 0.5,
            y=float(args.world_size) * 0.5,
            goal_x=float(args.world_size) * 0.5,
            goal_y=float(args.world_size) * 0.5,
            persona=spec["persona"],
        )
        if args.fresh_start:
            self.lab_agent.brain = ConsciousnessBlock(agent_id=self.lab_agent.id, persona=spec["persona"])
        self.world.add_agent(self.lab_agent)

        self.room = TrainingRoomManager(seed=int(args.seed), autosave_ticks=max(1, int(args.room_autosave)))
        self.room.attach_world(self.world, preferred_agent_id=self.lab_agent.id, announce=False)
        self._apply_curriculum(force=True, announce=False)

        self.start_skill = _safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0)
        self.start_survival = _safe_float(getattr(getattr(self.lab_agent, "brain", None), "survival_score", 0.0), 0.0)
        self.max_skill = self.start_skill
        self.min_health = _safe_float(getattr(self.lab_agent, "health", 100.0), 100.0)
        self.max_fear = _safe_float(getattr(self.lab_agent, "fear", 0.0), 0.0)

        self.mentor_hits_on_lab = 0
        self.lab_hits_on_mentor = 0
        self.lab_hits_on_wolf = 0
        self.wolf_hits_on_lab = 0
        self.save_count = 0
        self.last_room_brain_path = self.room.last_room_brain_path()
        self.progress_path = Path(args.progress_out)
        self.report_path = Path(args.report_out)

        self._push_activity("Dojo initialized. Agent is inside Morpheus room.")
        self._push_activity(f"Curriculum started: {_lesson_label(self.room.lesson_mode)}")
        self._write_progress(status="running")

    def request_stop(self) -> None:
        self._stop_requested = True

    def _push_activity(self, text: str) -> None:
        self._activity_lines.appendleft(str(text))

    def _event_key(self, ev: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            int(ev.get("tick", -1) or -1),
            str(ev.get("type", "") or ""),
            str(ev.get("who", "") or ""),
            str(ev.get("target", "") or ""),
            str(ev.get("victim_id", "") or ""),
            str(ev.get("victim_species", "") or ""),
            round(_safe_float(ev.get("damage", 0.0), 0.0), 2),
        )

    def _remember_event_key(self, key: Tuple[Any, ...]) -> bool:
        if key in self._seen_event_key_set:
            return False
        self._seen_event_key_set.add(key)
        self._seen_event_keys.append(key)
        while len(self._seen_event_keys) > 256:
            old = self._seen_event_keys.popleft()
            self._seen_event_key_set.discard(old)
        return True

    def _curriculum_mode_for_tick(self, tick: int) -> str:
        mode = str(self.args.lesson or "cycle").strip().lower()
        if mode == "sparring":
            return LESSON_SPARRING
        if mode == "wolf":
            return LESSON_WOLF
        if self.args.until_mastery:
            skill = _safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0)
            hp = _safe_float(getattr(self.lab_agent, "health", 100.0), 100.0)
            if skill < min(0.55, self.args.mastery_skill * 0.12):
                return LESSON_SPARRING
            if hp < 40.0:
                return LESSON_SPARRING
        phase_len = max(1, int(self.args.switch_every))
        phase_index = max(0, int(tick)) // phase_len
        return LESSON_SPARRING if phase_index % 2 == 0 else LESSON_WOLF

    def _apply_curriculum(self, *, force: bool = False, announce: bool = False) -> None:
        tick = int(getattr(self.world, "tick_count", 0))
        desired = self._curriculum_mode_for_tick(tick)
        if not force and desired == str(self.room.lesson_mode or LESSON_IDLE):
            return
        if desired == LESSON_SPARRING:
            self.room.start_sparring(self.world, announce=announce)
        elif desired == LESSON_WOLF:
            self.room.start_wolf_drill(self.world, announce=announce)
        else:
            self.room.stop_combat_lesson(self.world, announce=announce)
        if self.args.lesson == "cycle":
            phase_len = max(1, int(self.args.switch_every))
            phase_index = tick // phase_len
            if phase_index != self._last_phase_index:
                self._last_phase_index = phase_index
                switch_at = (phase_index + 1) * phase_len
                self._push_activity(
                    f"Phase {phase_index + 1:02d}: {_lesson_label(desired)} until tick {switch_at}"
                )

    def _mastery_state(self) -> Dict[str, Any]:
        skill_now = _safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0)
        lesson = str(self.args.lesson or "cycle").strip().lower()
        need_mentor = lesson in ("cycle", "sparring")
        need_wolf = lesson in ("cycle", "wolf")

        components: List[Tuple[str, float]] = [
            ("skill", _clamp(skill_now / max(0.001, float(self.args.mastery_skill)), 0.0, 1.0)),
        ]
        pending: List[str] = []
        if skill_now < float(self.args.mastery_skill):
            pending.append(f"combat_skill {skill_now:.2f}/{self.args.mastery_skill:.2f}")

        if need_mentor:
            mentor_ratio = _clamp(
                float(self.lab_hits_on_mentor) / max(1, int(self.args.mastery_mentor_hits)),
                0.0,
                1.0,
            )
            components.append(("mentor", mentor_ratio))
            if self.lab_hits_on_mentor < int(self.args.mastery_mentor_hits):
                pending.append(f"mentor counters {self.lab_hits_on_mentor}/{self.args.mastery_mentor_hits}")

        if need_wolf:
            wolf_ratio = _clamp(
                float(self.lab_hits_on_wolf) / max(1, int(self.args.mastery_wolf_hits)),
                0.0,
                1.0,
            )
            components.append(("wolf", wolf_ratio))
            if self.lab_hits_on_wolf < int(self.args.mastery_wolf_hits):
                pending.append(f"wolf hits {self.lab_hits_on_wolf}/{self.args.mastery_wolf_hits}")

        ratio = sum(value for _name, value in components) / max(1, len(components))
        completed = not pending
        return {
            "enabled": bool(self.args.until_mastery),
            "completed": bool(completed),
            "ratio": round(ratio, 4),
            "pending": pending,
            "components": {name: round(value, 4) for name, value in components},
            "skill_target": round(float(self.args.mastery_skill), 4),
            "mentor_target": int(self.args.mastery_mentor_hits if need_mentor else 0),
            "wolf_target": int(self.args.mastery_wolf_hits if need_wolf else 0),
            "needs_mentor": bool(need_mentor),
            "needs_wolf": bool(need_wolf),
        }

    def _scan_events(self) -> None:
        for ev in list(getattr(self.world, "event_log", []) or []):
            if not isinstance(ev, dict):
                continue
            key = self._event_key(ev)
            if not self._remember_event_key(key):
                continue
            etype = str(ev.get("type", "") or "")
            tick = int(ev.get("tick", getattr(self.world, "tick_count", 0)) or 0)
            if etype == "dojo_hit":
                who = str(ev.get("who", "") or "")
                target = str(ev.get("target", "") or "")
                damage = _safe_float(ev.get("damage", 0.0), 0.0)
                if who == "Morpheus" and target == self.lab_agent.name:
                    self.mentor_hits_on_lab += 1
                elif target == "Morpheus" and who == self.lab_agent.name:
                    self.lab_hits_on_mentor += 1
                self._push_activity(f"[t={_fmt_tick(tick)}] {who} -> {target} for {damage:.1f}")
            elif etype == "agent_attack" and str(ev.get("who", "") or "") == self.lab_agent.id:
                species = str(ev.get("victim_species", "beast") or "beast")
                damage = _safe_float(ev.get("damage", 0.0), 0.0)
                self.lab_hits_on_wolf += 1
                self._push_activity(f"[t={_fmt_tick(tick)}] {self.lab_agent.name} hit {species} for {damage:.1f}")
            elif etype == "animal_attack" and str(ev.get("victim_id", "") or "") == self.lab_agent.id:
                species = str(ev.get("attacker_species", "wolf") or "wolf")
                damage = _safe_float(ev.get("damage", 0.0), 0.0)
                self.wolf_hits_on_lab += 1
                self._push_activity(f"[t={_fmt_tick(tick)}] {species} hit {self.lab_agent.name} for {damage:.1f}")

    def _save_brains(self, *, reason: str) -> None:
        brain = getattr(self.lab_agent, "brain", None)
        if brain is not None:
            try:
                save_brain(brain)
            except Exception as exc:
                self._push_activity(f"Runtime brain save failed: {exc}")
        try:
            self.last_room_brain_path = self.room._save_room_brain(self.world, reason=reason, force=True)
        except Exception as exc:
            self._push_activity(f"Room brain save failed: {exc}")
        self.save_count += 1
        self._last_save_tick = int(getattr(self.world, "tick_count", 0))
        if self.last_room_brain_path:
            self._push_activity(f"Brain saved -> {self.last_room_brain_path}")

    def _summary_payload(self, *, status: str) -> Dict[str, Any]:
        tick = int(getattr(self.world, "tick_count", 0))
        room_state = self.room.combat_status(self.world)
        brain_state = _profile_text(self.lab_agent)
        mastery = self._mastery_state()
        mentor = self.room._get_agent(self.world, self.room.mentor_agent_id)
        wolf = self.room._get_training_wolf(self.world)
        lab_x = _safe_float(getattr(self.lab_agent, "x", 0.0), 0.0)
        lab_y = _safe_float(getattr(self.lab_agent, "y", 0.0), 0.0)
        mentor_dist = None
        wolf_dist = None
        if mentor is not None:
            mentor_dist = ((lab_x - _safe_float(getattr(mentor, "x", 0.0), 0.0)) ** 2 + (lab_y - _safe_float(getattr(mentor, "y", 0.0), 0.0)) ** 2) ** 0.5
        if wolf is not None:
            wolf_dist = ((lab_x - _safe_float(getattr(wolf, "x", 0.0), 0.0)) ** 2 + (lab_y - _safe_float(getattr(wolf, "y", 0.0), 0.0)) ** 2) ** 0.5
        return {
            "status": str(status),
            "agent_id": self.lab_agent.id,
            "agent_name": self.lab_agent.name,
            "tick": tick,
            "ticks_target": int(self.args.ticks),
            "continuous": bool(self.args.continuous),
            "until_mastery": bool(self.args.until_mastery),
            "lesson": str(self.args.lesson),
            "active_lesson_mode": str(room_state.get("lesson_mode") or LESSON_IDLE),
            "switch_every": int(self.args.switch_every),
            "seed": int(self.args.seed),
            "world_size": float(self.args.world_size),
            "lab": {
                "health": round(_safe_float(getattr(self.lab_agent, "health", 0.0), 0.0), 2),
                "energy": round(_pct(getattr(self.lab_agent, "energy", 0.0), 0.0), 2),
                "hunger": round(_pct(getattr(self.lab_agent, "hunger", 0.0), 0.0), 2),
                "fear": round(_safe_float(getattr(self.lab_agent, "fear", 0.0), 0.0), 4),
                "combat_skill": round(_safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0), 4),
                "combat_skill_gain": round(_safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0) - self.start_skill, 4),
                "survival": round(brain_state["survival"], 4),
                "survival_gain": round(brain_state["survival"] - self.start_survival, 4),
                "drive": brain_state["drive"],
                "curiosity": round(brain_state["curiosity"], 4),
                "beliefs": int(brain_state["beliefs"]),
                "memories": int(brain_state["memories"]),
                "thought": brain_state["thought"],
            },
            "opponents": {
                "mentor_health": round(_safe_float(getattr(mentor, "health", 0.0), 0.0), 2) if mentor is not None else 0.0,
                "wolf_hp": round(_safe_float(getattr(wolf, "hp", 0.0), 0.0), 2) if wolf is not None else 0.0,
                "mentor_distance": round(_safe_float(mentor_dist, 0.0), 3) if mentor_dist is not None else None,
                "wolf_distance": round(_safe_float(wolf_dist, 0.0), 3) if wolf_dist is not None else None,
            },
            "combat_counts": {
                "mentor_hits_on_lab": int(self.mentor_hits_on_lab),
                "lab_hits_on_mentor": int(self.lab_hits_on_mentor),
                "lab_hits_on_wolf": int(self.lab_hits_on_wolf),
                "wolf_hits_on_lab": int(self.wolf_hits_on_lab),
            },
            "training_extremes": {
                "max_skill": round(self.max_skill, 4),
                "min_health": round(self.min_health, 2),
                "max_fear": round(self.max_fear, 4),
            },
            "saves": {
                "count": int(self.save_count),
                "last_tick": int(self._last_save_tick),
                "room_brain_path": self.last_room_brain_path,
            },
            "mastery": mastery,
            "room_state": room_state,
            "recent_activity": list(self._activity_lines),
            "chat_tail": list(getattr(self.world, "chat_log", [])[-8:]),
        }

    def _write_progress(self, *, status: str) -> None:
        _write_json(self.progress_path, self._summary_payload(status=status))

    def _write_report(self, *, status: str) -> None:
        _write_json(self.report_path, self._summary_payload(status=status))

    def _step(self) -> None:
        self._apply_curriculum(announce=False)
        self.room.maintain_world(self.world)
        self.world.tick()
        self.room.maintain_world(self.world)
        self._scan_events()

        skill = _safe_float(getattr(self.lab_agent, "combat_skill", 0.0), 0.0)
        hp = _safe_float(getattr(self.lab_agent, "health", 100.0), 100.0)
        fear = _safe_float(getattr(self.lab_agent, "fear", 0.0), 0.0)
        self.max_skill = max(self.max_skill, skill)
        self.min_health = min(self.min_health, hp)
        self.max_fear = max(self.max_fear, fear)

        tick = int(getattr(self.world, "tick_count", 0))
        if (tick - self._last_save_tick) >= max(1, int(self.args.save_every)):
            self._save_brains(reason="dojo_interval")
        if tick % max(1, int(self.args.room_autosave)) == 0:
            self.last_room_brain_path = self.room.last_room_brain_path() or self.last_room_brain_path

    def _render_plain(self, *, final: bool = False, status: str = "running") -> None:
        now = time.perf_counter()
        if not final and (now - self._plain_last_emit) < max(0.2, float(self.args.ui_interval)):
            return
        self._plain_last_emit = now
        summary = self._summary_payload(status=status)
        lab = summary["lab"]
        counts = summary["combat_counts"]
        mastery = summary["mastery"]
        tick = summary["tick"]
        target = summary["ticks_target"]
        tick_goal = "mastery" if self.args.until_mastery and self.args.continuous else (target if not self.args.continuous else "inf")
        mastery_text = ""
        if mastery.get("enabled"):
            mastery_text = (
                f" mastery={float(mastery.get('ratio', 0.0)) * 100.0:4.0f}%"
                f" pending={_trim('; '.join(list(mastery.get('pending', []))[:2]) or 'none', 44)}"
            )
        line = (
            f"[dojo] status={status} tick={tick}/{tick_goal} "
            f"lesson={summary['active_lesson_mode']} skill={lab['combat_skill']:.2f} "
            f"hp={lab['health']:.1f} fear={lab['fear']:.2f} drive={lab['drive']} "
            f"mentor_hits={counts['mentor_hits_on_lab']} wolf_hits={counts['wolf_hits_on_lab']} "
            f"thought={_trim(lab['thought'], 56)}{mastery_text}"
        )
        print(line)

    def _render_tty(self, *, final: bool = False, status: str = "running") -> None:
        now = time.perf_counter()
        if not final and (now - self._render_last_at) < max(0.05, float(self.args.ui_interval)):
            return
        self._render_last_at = now

        width = max(92, min(124, shutil.get_terminal_size((110, 36)).columns))
        inner = width - 4
        summary = self._summary_payload(status=status)
        lab = summary["lab"]
        opponents = summary["opponents"]
        counts = summary["combat_counts"]
        mastery = summary["mastery"]
        elapsed = time.perf_counter() - self._start_monotonic
        tick = int(summary["tick"])
        total = int(summary["ticks_target"])
        if self.args.until_mastery:
            target_progress = float(mastery.get("ratio", 0.0))
        else:
            target_progress = 0.0 if self.args.continuous else tick / max(1, total)
        lesson_mode = str(summary["active_lesson_mode"])
        phase_remaining = None
        if self.args.lesson == "cycle":
            phase_len = max(1, int(self.args.switch_every))
            phase_remaining = phase_len - (tick % phase_len)

        lines: List[str] = []
        lines.append("=" * width)
        lines.append(" CSEN MORPHEUS DOJO ".center(width, "="))
        lines.append("=" * width)
        lines.append(
            _trim(
                f"Agent: {summary['agent_name']} ({summary['agent_id']})  "
                f"Lesson: {_lesson_label(lesson_mode)}  "
                f"Status: {status}  "
                f"Elapsed: {_fmt_duration(elapsed)}",
                width,
            )
        )
        lines.append(
            _trim(
                f"Progress: [{_bar(target_progress, 28)}]  "
                f"tick {tick}/{'mastery' if self.args.until_mastery and self.args.continuous else (total if not self.args.continuous else 'inf')}  "
                f"saves={summary['saves']['count']}  "
                f"room_brain={summary['saves']['room_brain_path'] or '--'}",
                width,
            )
        )
        if mastery.get("enabled"):
            pending = "; ".join(list(mastery.get("pending", []))[:3]) or "none"
            lines.append(
                _trim(
                    f"Mastery: [{_bar(float(mastery.get('ratio', 0.0)), 28)}]  "
                    f"{float(mastery.get('ratio', 0.0)) * 100.0:5.1f}%  pending={pending}",
                    width,
                )
            )
        if phase_remaining is not None:
            lines.append(
                _trim(
                    f"Curriculum: cycle  switch_every={self.args.switch_every}  next_switch_in={phase_remaining} ticks",
                    width,
                )
            )
        else:
            lines.append(_trim(f"Curriculum: fixed mode `{self.args.lesson}`", width))
        lines.append("-" * width)
        lines.append(" Lab Agent ".center(width, "-"))
        lines.append(
            _trim(
                f"HP      [{_bar(lab['health'] / 100.0, 24)}] {lab['health']:6.1f}    "
                f"Energy  [{_bar(lab['energy'] / 100.0, 24)}] {lab['energy']:6.1f}",
                width,
            )
        )
        lines.append(
            _trim(
                f"Fear    [{_bar(lab['fear'], 24)}] {lab['fear']:6.2f}    "
                f"Skill   [{_bar(lab['combat_skill'] / 5.0, 24)}] {lab['combat_skill']:6.2f}  "
                f"(+{lab['combat_skill_gain']:.2f})",
                width,
            )
        )
        lines.append(
            _trim(
                f"Drive={lab['drive']}  Survival={lab['survival']:.2f}  Curiosity={lab['curiosity']:.2f}  "
                f"Beliefs={lab['beliefs']}  Memories={lab['memories']}",
                width,
            )
        )
        lines.append(_trim(f"Thought: {lab['thought'] or '...'}", width))
        lines.append("-" * width)
        lines.append(" Opponents ".center(width, "-"))
        lines.append(
            _trim(
                f"Morpheus HP={opponents['mentor_health']:.1f}  dist={opponents['mentor_distance'] if opponents['mentor_distance'] is not None else '--'}    "
                f"Wolf HP={opponents['wolf_hp']:.1f}  dist={opponents['wolf_distance'] if opponents['wolf_distance'] is not None else '--'}",
                width,
            )
        )
        lines.append(
            _trim(
                f"Morpheus -> Agent hits: {counts['mentor_hits_on_lab']}    "
                f"Agent -> Morpheus hits: {counts['lab_hits_on_mentor']}    "
                f"Agent -> Wolf hits: {counts['lab_hits_on_wolf']}    "
                f"Wolf -> Agent hits: {counts['wolf_hits_on_lab']}",
                width,
            )
        )
        lines.append("-" * width)
        lines.append(" Recent Activity ".center(width, "-"))
        for line in list(self._activity_lines)[:6]:
            lines.append(_trim(f"* {line}", width))
        while len(lines) < 18:
            lines.append("")
        lines.append("-" * width)
        lines.append(" Recent Chat ".center(width, "-"))
        chat_tail = list(getattr(self.world, "chat_log", [])[-5:])
        if chat_tail:
            for chat in chat_tail:
                lines.append(_trim(f"> {chat}", width))
        else:
            lines.append("> no chat yet")
        lines.append("-" * width)
        lines.append(_trim("Ctrl+C -> graceful stop, save brains, write final report.", width))
        lines.append(_trim(f"Progress file: {self.progress_path}", width))
        lines.append(_trim(f"Report file:   {self.report_path}", width))
        lines.append("=" * width)

        sys.stdout.write("\033[2J\033[H" + "\n".join(lines) + "\n")
        sys.stdout.flush()

    def render(self, *, final: bool = False, status: str = "running") -> None:
        self._write_progress(status=status)
        if self.args.plain or not sys.stdout.isatty():
            self._render_plain(final=final, status=status)
        else:
            self._render_tty(final=final, status=status)

    def run(self) -> int:
        final_status = "completed"
        try:
            while not self._stop_requested:
                mastery = self._mastery_state()
                if self.args.until_mastery and bool(mastery.get("completed")):
                    final_status = "mastered"
                    self._push_activity("Mastery threshold reached. Dojo cycle completed.")
                    break
                if not self.args.continuous and int(getattr(self.world, "tick_count", 0)) >= int(self.args.ticks):
                    final_status = "completed"
                    if self.args.until_mastery:
                        final_status = "mastery_cap_reached"
                        self._push_activity("Tick cap reached before full mastery.")
                    break
                self._step()
                self.render(status="running")
            if self._stop_requested:
                final_status = "interrupted"
        except KeyboardInterrupt:
            final_status = "interrupted"
        finally:
            self._save_brains(reason=f"dojo_{final_status}")
            self._write_progress(status=final_status)
            self._write_report(status=final_status)
            self.render(final=True, status=final_status)
        return 0


def _parse_args(argv: Sequence[str]) -> DojoArgs:
    parser = argparse.ArgumentParser(
        prog="morpheus_dojo.py",
        description="Headless terminal combat-training utility for the Morpheus room.",
    )
    parser.add_argument("--agent-id", default=DEFAULT_ROOM_AGENT_ID)
    parser.add_argument("--ticks", type=int, default=1800)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--until-mastery", action="store_true")
    parser.add_argument("--lesson", choices=("cycle", "sparring", "wolf"), default="cycle")
    parser.add_argument("--switch-every", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--ui-interval", type=float, default=0.12)
    parser.add_argument("--save-every", type=int, default=180)
    parser.add_argument("--room-autosave", type=int, default=120)
    parser.add_argument("--world-size", type=float, default=30.0)
    parser.add_argument("--mastery-skill", type=float, default=5.0)
    parser.add_argument("--mastery-mentor-hits", type=int, default=8)
    parser.add_argument("--mastery-wolf-hits", type=int, default=12)
    parser.add_argument("--plain", action="store_true")
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--progress-out")
    parser.add_argument("--report-out")
    ns = parser.parse_args(list(argv))

    agent_id = str(ns.agent_id or DEFAULT_ROOM_AGENT_ID).strip() or DEFAULT_ROOM_AGENT_ID
    report_default = Path(DOJO_REPORTS_DIR) / f"{agent_id}.dojo_report.json"
    progress_default = Path(DOJO_REPORTS_DIR) / f"{agent_id}.dojo_progress.json"
    return DojoArgs(
        agent_id=agent_id,
        ticks=max(1, int(ns.ticks)),
        lesson=str(ns.lesson),
        switch_every=max(1, int(ns.switch_every)),
        seed=int(ns.seed),
        ui_interval=max(0.03, float(ns.ui_interval)),
        save_every=max(1, int(ns.save_every)),
        room_autosave=max(1, int(ns.room_autosave)),
        world_size=max(18.0, float(ns.world_size)),
        continuous=bool(ns.continuous),
        until_mastery=bool(ns.until_mastery),
        mastery_skill=_clamp(float(ns.mastery_skill), 0.01, 5.0),
        mastery_mentor_hits=max(1, int(ns.mastery_mentor_hits)),
        mastery_wolf_hits=max(1, int(ns.mastery_wolf_hits)),
        plain=bool(ns.plain),
        fresh_start=bool(ns.fresh_start),
        progress_out=str(ns.progress_out or progress_default),
        report_out=str(ns.report_out or report_default),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    runner = MorpheusDojoRunner(args)

    def _handle_signal(signum: int, _frame: Any) -> None:
        runner.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
