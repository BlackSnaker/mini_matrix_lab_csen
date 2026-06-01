from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import math
import threading

from ollama_coach import (
    CoachAdvice,
    OllamaCoach,
    apply_advice_to_agent,
    apply_goal_feedback,
    build_training_snapshot,
    infer_operator_control,
    infer_operator_goal,
    summarize_control,
)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    if math.isnan(out) or math.isinf(out):
        return fallback
    return out


def _iter_agents(world: Any) -> Iterable[Any]:
    agents = getattr(world, "agents", {}) or {}
    if isinstance(agents, dict):
        return list(agents.values())
    if isinstance(agents, (list, tuple, set)):
        return list(agents)
    return []


@dataclass(frozen=True)
class _Bounds:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.left + self.width * 0.5, self.top + self.height * 0.5)

    @property
    def safe_radius(self) -> float:
        return max(4.0, min(self.width, self.height) * 0.46)

    def clamp_point(self, x: float, y: float, *, padding: float = 1.5) -> Tuple[float, float]:
        return (
            max(self.left + padding, min(self.right - padding, float(x))),
            max(self.top + padding, min(self.bottom - padding, float(y))),
        )


class _WorldRoomProxy:
    room_id = "ollama_world"
    label = "Ollama_World"
    object_id = "ollama_world_zone"

    def bounds_for(self, world: Any) -> _Bounds:
        width = max(12.0, _safe_float(getattr(world, "width", 30.0), 30.0))
        height = max(12.0, _safe_float(getattr(world, "height", 30.0), 30.0))
        return _Bounds(left=1.0, top=1.0, right=width - 1.0, bottom=height - 1.0)

    def clamp_point_for_agent(self, world: Any, agent_id: Optional[str], x: float, y: float) -> Tuple[float, float]:
        return self.bounds_for(world).clamp_point(x, y)

    def is_confined(self, agent_id: Optional[str]) -> bool:
        return False


class OllamaBrainService:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        host: Optional[str] = None,
        auto_interval_ticks: int = 180,
    ) -> None:
        self.default_model = model
        self.default_host = host
        self.auto_interval_ticks = max(12, int(auto_interval_ticks))
        self._lock = threading.Lock()
        self._results: Dict[str, Dict[str, Any]] = {}
        self._inflight: Dict[str, bool] = {}
        self._plan_queues: Dict[str, list[Dict[str, Any]]] = {}
        self._wait_until: Dict[str, int] = {}
        self._last_recall_keys: Dict[str, str] = {}
        self._active_manual_command: Dict[str, Dict[str, Any]] = {}

    def attach_world(self, world: Any, *, training_room: Optional[Any] = None) -> None:
        try:
            setattr(world, "ollama_brain_bridge", self)
            setattr(world, "training_room_manager", training_room)
        except Exception:
            pass

    def configure_brain(
        self,
        brain: Any,
        *,
        enabled: Optional[bool] = None,
        auto_mode: Optional[bool] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        interval_ticks: Optional[int] = None,
        tick: Optional[int] = None,
    ) -> None:
        if brain is None:
            return
        if hasattr(brain, "configure_ollama"):
            brain.configure_ollama(
                enabled=enabled,
                auto_mode=auto_mode,
                model=model if model is not None else self.default_model,
                host=host if host is not None else self.default_host,
                interval_ticks=interval_ticks or self.auto_interval_ticks,
                tick=tick,
            )
            return
        if enabled is not None:
            setattr(brain, "ollama_enabled", bool(enabled))
        if auto_mode is not None:
            setattr(brain, "ollama_auto_mode", bool(auto_mode))
        setattr(brain, "ollama_model", model if model is not None else self.default_model)
        setattr(brain, "ollama_host", host if host is not None else self.default_host)
        setattr(brain, "ollama_interval_ticks", max(1, int(interval_ticks or self.auto_interval_ticks)))

    def queue_instruction(self, brain: Any, text: str, *, tick: Optional[int] = None) -> None:
        if brain is None:
            return
        if hasattr(brain, "queue_ollama_instruction"):
            brain.queue_ollama_instruction(text, tick=tick)
            return
        line = str(text or "").strip()
        if not line:
            return
        setattr(brain, "ollama_pending_instruction", line)
        setattr(brain, "ollama_last_operator_instruction", line)
        setattr(brain, "ollama_force_request", True)
        setattr(brain, "ollama_enabled", True)

    def force_request(self, brain: Any) -> None:
        if brain is None:
            return
        setattr(brain, "ollama_enabled", True)
        setattr(brain, "ollama_force_request", True)

    def _agent_key(self, agent: Any) -> str:
        return str(getattr(agent, "id", getattr(agent, "agent_id", "agent")))

    def _clear_plan_state(self, agent: Any) -> None:
        agent_id = self._agent_key(agent)
        self._plan_queues.pop(agent_id, None)
        self._wait_until.pop(agent_id, None)

    def _clear_manual_command_state(self, agent: Any) -> None:
        agent_id = self._agent_key(agent)
        self._active_manual_command.pop(agent_id, None)
        self._last_recall_keys.pop(agent_id, None)

    def _store_plan_queue(self, agent: Any, steps: Iterable[Dict[str, Any]]) -> None:
        agent_id = self._agent_key(agent)
        normalized: list[Dict[str, Any]] = []
        for step in list(steps or []):
            if not isinstance(step, dict):
                continue
            if str(step.get("type") or "") == "sequence":
                for nested in list(step.get("steps", []) or []):
                    if isinstance(nested, dict):
                        normalized.append(dict(nested))
                continue
            normalized.append(dict(step))
        if normalized:
            self._plan_queues[agent_id] = normalized
        else:
            self._plan_queues.pop(agent_id, None)

    def _plan_queue(self, agent: Any) -> list[Dict[str, Any]]:
        return list(self._plan_queues.get(self._agent_key(agent), []) or [])

    def _set_wait_until(self, agent: Any, tick: int, hold_ticks: int) -> None:
        self._wait_until[self._agent_key(agent)] = int(tick) + max(1, int(hold_ticks))

    def _describe_advice(self, advice: CoachAdvice) -> str:
        chunks: list[str] = []
        action = advice.raw.get("_action") if isinstance(advice.raw, dict) else None
        if isinstance(action, dict):
            chunks.append(f"action={summarize_control(action)}")
        if advice.goal is not None:
            chunks.append(f"goal=({float(advice.goal[0]):.1f}, {float(advice.goal[1]):.1f})")
        if advice.lesson:
            chunks.append(f"lesson={str(advice.lesson)[:72]}")
        if getattr(advice, "speech", ""):
            chunks.append(f"speech={str(advice.speech)[:72]}")
        if advice.thought:
            chunks.append(f"thought={str(advice.thought)[:64]}")
        if advice.belief:
            chunks.append(f"belief={str(advice.belief.get('then') or '')[:48]}")
        return " | ".join(chunks) if chunks else "empty response"

    def _set_active_manual_command(self, agent: Any, *, command: str, parsed: str, tick: int) -> None:
        agent_id = self._agent_key(agent)
        self._active_manual_command[agent_id] = {
            "command": str(command or "").strip()[:160],
            "parsed": str(parsed or "").strip()[:220],
            "started_tick": int(tick),
        }

    def _update_active_manual_parsed(self, agent: Any, parsed: str) -> None:
        agent_id = self._agent_key(agent)
        row = self._active_manual_command.get(agent_id)
        if not isinstance(row, dict):
            return
        text = str(parsed or "").strip()
        current = str(row.get("parsed") or "").strip()
        if text and (not current or current == "await_model"):
            row["parsed"] = text[:220]

    def _record_manual_command_example(
        self,
        world: Any,
        room: Any,
        agent: Any,
        brain: Any,
        *,
        success: bool,
        outcome: str,
        tick: int,
    ) -> None:
        agent_id = self._agent_key(agent)
        active = self._active_manual_command.get(agent_id)
        if not isinstance(active, dict):
            return
        command = str(active.get("command") or "").strip()
        parsed = str(active.get("parsed") or "").strip()
        if not command or not parsed:
            return
        snapshot = build_training_snapshot(agent, world, room, active_lesson=self._active_lesson_payload(brain))
        behavior_context = snapshot.get("behavior_context", {}) if isinstance(snapshot.get("behavior_context"), dict) else {}
        profile = behavior_context.get("current_profile", {}) if isinstance(behavior_context.get("current_profile"), dict) else {}
        row = {
            "command": command,
            "parsed": parsed,
            "lesson": str(getattr(brain, "ollama_active_lesson", "") or "")[:180],
            "thought": str(getattr(brain, "ollama_last_thought", "") or "")[:160],
            "outcome": str(outcome or "")[:180],
            "success": bool(success),
            "traits": list(profile.get("traits", []) or [])[:8],
            "drive": str(profile.get("drive") or "idle"),
        }
        appender = getattr(brain, "append_ollama_command_example", None)
        if callable(appender):
            try:
                appender(row, tick=tick)
            except Exception:
                pass

    def _operator_override_active(self, brain: Any, *, tick: int) -> bool:
        if brain is None:
            return False
        checker = getattr(brain, "is_ollama_operator_override_active", None)
        if callable(checker):
            try:
                return bool(checker(tick=tick))
            except Exception:
                pass
        try:
            until = int(getattr(brain, "ollama_operator_override_until_tick", -10**9))
        except Exception:
            return False
        return bool(getattr(brain, "ollama_authoritative_commands", True)) and int(tick) <= until

    def _mark_operator_override(self, brain: Any, *, tick: int, hold_ticks: int) -> None:
        if brain is None:
            return
        marker = getattr(brain, "mark_ollama_operator_override", None)
        if callable(marker):
            try:
                marker(tick=tick, hold_ticks=hold_ticks)
                return
            except Exception:
                pass
        try:
            current = int(getattr(brain, "ollama_operator_override_until_tick", -10**9))
            setattr(brain, "ollama_operator_override_until_tick", max(current, int(tick) + max(1, int(hold_ticks))))
        except Exception:
            pass

    def _clear_operator_override(self, brain: Any) -> None:
        if brain is None:
            return
        clearer = getattr(brain, "clear_ollama_operator_override", None)
        if callable(clearer):
            try:
                clearer()
                return
            except Exception:
                pass
        try:
            setattr(brain, "ollama_operator_override_until_tick", -10**9)
        except Exception:
            pass

    def _clear_manual_control(self, agent: Any) -> None:
        if agent is None:
            return
        clearer = getattr(agent, "clear_manual_control", None)
        if callable(clearer):
            try:
                clearer()
                return
            except Exception:
                pass
        try:
            setattr(agent, "_manual_control_until_tick", -10**9)
            setattr(agent, "_manual_control_source", None)
        except Exception:
            pass

    def _sync_authoritative_goal(self, world: Any, room: Any, agent: Any, brain: Any, *, tick: int) -> bool:
        if brain is None or agent is None:
            return False
        goal = getattr(brain, "ollama_active_goal", None)
        if not (isinstance(goal, tuple) and len(goal) == 2):
            return False
        if str(getattr(brain, "ollama_status", "") or "") != "guiding":
            return False
        command_source = str(getattr(brain, "ollama_active_command_source", "") or "")
        if command_source != "manual" and not self._operator_override_active(brain, tick=tick):
            return False
        gx = _safe_float(goal[0], _safe_float(getattr(agent, "goal_x", getattr(agent, "x", 0.0)), 0.0))
        gy = _safe_float(goal[1], _safe_float(getattr(agent, "goal_y", getattr(agent, "y", 0.0)), 0.0))
        if hasattr(room, "clamp_point_for_agent"):
            try:
                gx, gy = room.clamp_point_for_agent(world, getattr(agent, "id", None), gx, gy)
            except Exception:
                pass
        self._clear_manual_control(agent)
        bounds = (
            _safe_float(getattr(world, "width", gx), gx),
            _safe_float(getattr(world, "height", gy), gy),
        )
        setter = getattr(agent, "set_goal", None)
        if callable(setter):
            try:
                setter(gx, gy, world_size=bounds, reason="ollama_authoritative_goal", tick=tick)
                return True
            except Exception:
                pass
        try:
            agent.goal_x = float(gx)
            agent.goal_y = float(gy)
            return True
        except Exception:
            return False

    def _advance_plan_if_ready(self, world: Any, room: Any, agent: Any, brain: Any, *, tick: int, trigger: str) -> bool:
        agent_id = self._agent_key(agent)
        wait_until = self._wait_until.get(agent_id)
        active_goal = getattr(brain, "ollama_active_goal", None)
        if wait_until is None and isinstance(active_goal, tuple) and len(active_goal) == 2 and str(getattr(brain, "ollama_status", "") or "") == "guiding":
            return False
        if wait_until is not None and int(tick) < int(wait_until):
            return False
        if wait_until is not None:
            self._wait_until.pop(agent_id, None)
            self._append_log(brain, "system", "Этап ожидания завершён.", tick=tick)
        queue = self._plan_queues.get(agent_id) or []
        if not queue:
            active_source = str(getattr(brain, "ollama_active_command_source", "") or "")
            if str(getattr(brain, "ollama_status", "") or "") == "waiting":
                brain.ollama_status = "wait_complete"
                if active_source == "manual":
                    self._clear_operator_override(brain)
            return False
        next_control = dict(queue.pop(0))
        if queue:
            self._plan_queues[agent_id] = queue
        else:
            self._plan_queues.pop(agent_id, None)
        self._append_log(brain, "plan", f"{trigger}: {summarize_control(next_control)}", tick=tick)
        applied = self._apply_local_control(
            world,
            room,
            agent,
            brain,
            next_control,
            lesson=str(getattr(brain, "ollama_active_lesson", "") or ""),
            tick=tick,
        )
        if not applied and not queue:
            self._plan_queues.pop(agent_id, None)
        return applied

    def _hold_position_after_manual_goal(self, world: Any, agent: Any, brain: Any, *, tick: int) -> None:
        x = _safe_float(getattr(agent, "x", 0.0), 0.0)
        y = _safe_float(getattr(agent, "y", 0.0), 0.0)
        try:
            agent.goal_x = x
            agent.goal_y = y
            agent.vx = 0.0
            agent.vy = 0.0
        except Exception:
            pass
        self._clear_manual_control(agent)
        self._mark_operator_override(brain, tick=tick, hold_ticks=84)
        brain.ollama_status = "holding_position"
        brain.ollama_active_command_source = "manual"
        self._append_log(brain, "system", "Команда завершена. Удерживаю позицию до следующего приказа.", tick=tick)

    def before_world_tick(self, world: Any) -> None:
        self._drain_results(world)
        room_manager = getattr(world, "training_room_manager", None)
        tick = int(getattr(world, "tick_count", 0) or 0)
        for agent in _iter_agents(world):
            brain = getattr(agent, "brain", None)
            if brain is None:
                continue
            if not bool(getattr(brain, "ollama_enabled", False)):
                continue
            if not bool(getattr(agent, "is_alive", lambda: True)()):
                setattr(brain, "ollama_status", "agent_dead")
                continue

            pending_instruction = str(getattr(brain, "ollama_pending_instruction", "") or "").strip()
            force_request = bool(getattr(brain, "ollama_force_request", False))
            auto_mode = bool(getattr(brain, "ollama_auto_mode", False))
            interval_ticks = max(1, int(getattr(brain, "ollama_interval_ticks", self.auto_interval_ticks) or self.auto_interval_ticks))
            last_request_tick = int(getattr(brain, "ollama_last_request_tick", -10**9))

            room = self._room_for(world, agent, room_manager)
            if pending_instruction:
                self._clear_plan_state(agent)
            if self._advance_plan_if_ready(world, room, agent, brain, tick=tick, trigger="plan"):
                continue
            self._sync_authoritative_goal(world, room, agent, brain, tick=tick)
            want_request = bool(pending_instruction) or force_request
            if not want_request and self._operator_override_active(brain, tick=tick):
                continue
            if not want_request and auto_mode and (tick - last_request_tick) >= interval_ticks:
                want_request = True
            if not want_request:
                continue

            snapshot = build_training_snapshot(
                agent,
                world,
                room,
                active_lesson=self._active_lesson_payload(brain),
                operator_instruction=pending_instruction or None,
            )
            behavior_context = snapshot.get("behavior_context", {}) if isinstance(snapshot.get("behavior_context"), dict) else {}
            if pending_instruction and behavior_context:
                summary = str(behavior_context.get("summary") or "").strip()
                recall_key = str(pending_instruction)
                if summary and self._last_recall_keys.get(self._agent_key(agent)) != recall_key:
                    self._append_log(brain, "recall", summary, tick=tick)
                    self._last_recall_keys[self._agent_key(agent)] = recall_key
            elif not pending_instruction:
                self._last_recall_keys.pop(self._agent_key(agent), None)
            local_control = infer_operator_control(snapshot) if pending_instruction else None
            if pending_instruction and isinstance(local_control, dict):
                parsed_text = summarize_control(local_control)
                self._append_log(brain, "parsed", parsed_text, tick=tick)
                self._set_active_manual_command(agent, command=pending_instruction, parsed=parsed_text, tick=tick)
            if self._apply_local_control(world, room, agent, brain, local_control, lesson=pending_instruction, tick=tick):
                brain.ollama_pending_instruction = None
                brain.ollama_force_request = False
                continue

            if bool(getattr(brain, "ollama_request_inflight", False)):
                if pending_instruction:
                    brain.ollama_status = "operator_wait"
                continue

            local_goal = None
            if isinstance(local_control, dict) and str(local_control.get("type")) == "move":
                goal_raw = local_control.get("goal")
                if isinstance(goal_raw, tuple) and len(goal_raw) == 2:
                    local_goal = (float(goal_raw[0]), float(goal_raw[1]))
            elif pending_instruction:
                local_goal = infer_operator_goal(snapshot)
            if local_goal is not None:
                parsed_text = f"move({float(local_goal[0]):.1f}, {float(local_goal[1]):.1f})"
                self._append_log(brain, "parsed", parsed_text, tick=tick)
                self._set_active_manual_command(agent, command=pending_instruction, parsed=parsed_text, tick=tick)
                advice = CoachAdvice(
                    model="brain_local_command",
                    thought="Выполняю короткую команду движения.",
                    lesson=pending_instruction,
                    goal=local_goal,
                    belief=None,
                    behavior={},
                    reward_hint=0.14,
                    raw={"goal_source": "operator_relative"},
                )
                self._apply_advice(world, room, agent, brain, advice, command_source="manual")
                self._append_log(brain, "system", "Команда движения выполнена локально через мозг.", tick=tick)
                brain.ollama_pending_instruction = None
                brain.ollama_force_request = False
                continue

            self._spawn_request(
                agent=agent,
                brain=brain,
                snapshot=snapshot,
                tick=tick,
                source="manual" if pending_instruction else "auto",
                operator_instruction=pending_instruction or None,
            )

    def after_world_tick(self, world: Any) -> None:
        self._drain_results(world)
        tick = int(getattr(world, "tick_count", 0) or 0)
        for agent in _iter_agents(world):
            brain = getattr(agent, "brain", None)
            if brain is None:
                continue
            goal = getattr(brain, "ollama_active_goal", None)
            if not (isinstance(goal, tuple) and len(goal) == 2):
                continue
            gx, gy = float(goal[0]), float(goal[1])
            dist = math.hypot(float(getattr(agent, "x", gx)) - gx, float(getattr(agent, "y", gy)) - gy)
            if dist <= 1.35:
                active_source = str(getattr(brain, "ollama_active_command_source", "") or "")
                apply_goal_feedback(
                    agent,
                    world,
                    goal=goal,
                    lesson=str(getattr(brain, "ollama_active_lesson", "") or ""),
                    model=str(getattr(brain, "ollama_last_model", "ollama") or "ollama"),
                    reward=max(0.08, float(getattr(brain, "ollama_active_reward_hint", 0.12) or 0.12)),
                    success=True,
                )
                self._append_log(brain, "system", "Цель достигнута. Сигнал обучения записан.", tick=tick)
                if hasattr(brain, "clear_ollama_goal"):
                    brain.clear_ollama_goal()
                else:
                    brain.ollama_active_goal = None
                    brain.ollama_goal_started_tick = None
                    brain.ollama_active_command_source = None
                if active_source == "manual":
                    self._clear_operator_override(brain)
                brain.ollama_status = "goal_complete"
                room = self._room_for(world, agent, getattr(world, "training_room_manager", None))
                if self._advance_plan_if_ready(world, room, agent, brain, tick=tick, trigger="goal_complete"):
                    continue
                if active_source == "manual":
                    self._record_manual_command_example(world, room, agent, brain, success=True, outcome="goal_complete", tick=tick)
                    self._hold_position_after_manual_goal(world, agent, brain, tick=tick)
                    self._clear_manual_command_state(agent)
                continue

            started_tick = getattr(brain, "ollama_goal_started_tick", None)
            if started_tick is None:
                continue
            if tick > int(started_tick) and (tick - int(started_tick)) >= 220:
                active_source = str(getattr(brain, "ollama_active_command_source", "") or "")
                apply_goal_feedback(
                    agent,
                    world,
                    goal=goal,
                    lesson=str(getattr(brain, "ollama_active_lesson", "") or ""),
                    model=str(getattr(brain, "ollama_last_model", "ollama") or "ollama"),
                    reward=-0.04,
                    success=False,
                )
                self._append_log(brain, "system", "Цель не выполнена вовремя. Записан отрицательный feedback.", tick=tick)
                if hasattr(brain, "clear_ollama_goal"):
                    brain.clear_ollama_goal()
                else:
                    brain.ollama_active_goal = None
                    brain.ollama_goal_started_tick = None
                    brain.ollama_active_command_source = None
                self._clear_plan_state(agent)
                if active_source == "manual":
                    room = self._room_for(world, agent, getattr(world, "training_room_manager", None))
                    self._record_manual_command_example(world, room, agent, brain, success=False, outcome="goal_timeout", tick=tick)
                    self._clear_operator_override(brain)
                    self._clear_manual_command_state(agent)
                brain.ollama_status = "goal_timeout"
                brain.ollama_force_request = True

    def _room_for(self, world: Any, agent: Any, room_manager: Optional[Any]) -> Any:
        if room_manager is not None:
            try:
                if hasattr(room_manager, "is_confined") and room_manager.is_confined(getattr(agent, "id", None)):
                    return room_manager
            except Exception:
                pass
        return _WorldRoomProxy()

    def _active_lesson_payload(self, brain: Any) -> Optional[Dict[str, Any]]:
        lesson = str(getattr(brain, "ollama_active_lesson", "") or "").strip()
        goal = getattr(brain, "ollama_active_goal", None)
        if not lesson and not (isinstance(goal, tuple) and len(goal) == 2):
            return None
        payload: Dict[str, Any] = {
            "lesson": lesson,
            "model": str(getattr(brain, "ollama_last_model", "") or ""),
            "started_tick": getattr(brain, "ollama_goal_started_tick", None),
        }
        if isinstance(goal, tuple) and len(goal) == 2:
            payload["goal"] = {"x": float(goal[0]), "y": float(goal[1])}
        return payload

    def _spawn_request(
        self,
        *,
        agent: Any,
        brain: Any,
        snapshot: Dict[str, Any],
        tick: int,
        source: str,
        operator_instruction: Optional[str],
    ) -> None:
        agent_id = str(getattr(agent, "id", getattr(agent, "agent_id", "agent")))
        with self._lock:
            if self._inflight.get(agent_id):
                return
            self._inflight[agent_id] = True

        brain.ollama_request_inflight = True
        brain.ollama_last_request_tick = int(tick)
        brain.ollama_status = "thinking"
        if source == "manual" and operator_instruction:
            self._set_active_manual_command(agent, command=operator_instruction, parsed="await_model", tick=tick)
            self._append_log(brain, "request", f"manual -> {operator_instruction}", tick=tick)
        elif source != "manual":
            self._append_log(brain, "request", "auto lesson request", tick=tick)

        model = str(getattr(brain, "ollama_model", self.default_model) or self.default_model or "").strip() or None
        host = str(getattr(brain, "ollama_host", self.default_host) or self.default_host or "").strip() or None

        def _worker() -> None:
            try:
                coach = OllamaCoach(model=model, host=host)
                advice = coach.request_advice(snapshot)
                payload: Dict[str, Any] = {
                    "agent_id": agent_id,
                    "source": str(source),
                    "operator_instruction": operator_instruction,
                    "advice": advice,
                    "error": None,
                }
            except Exception as exc:
                payload = {
                    "agent_id": agent_id,
                    "source": str(source),
                    "operator_instruction": operator_instruction,
                    "advice": None,
                    "error": str(exc),
                }
            with self._lock:
                self._results[agent_id] = payload
                self._inflight.pop(agent_id, None)

        threading.Thread(target=_worker, name=f"ollama_brain_{agent_id}", daemon=True).start()

    def _drain_results(self, world: Any) -> None:
        with self._lock:
            if not self._results:
                return
            ready = list(self._results.values())
            self._results.clear()

        room_manager = getattr(world, "training_room_manager", None)
        for row in ready:
            agent_id = str(row.get("agent_id") or "")
            agent = self._find_agent(world, agent_id)
            if agent is None:
                continue
            brain = getattr(agent, "brain", None)
            if brain is None:
                continue
            brain.ollama_request_inflight = False
            if row.get("error"):
                err = str(row.get("error") or "unknown ollama error")
                if err.strip().casefold() == "timed out":
                    err = "Ollama request timed out"
                operator_instruction = str(row.get("operator_instruction") or "").strip()
                if operator_instruction:
                    err = f"{err}: {operator_instruction}"
                brain.ollama_last_error = err
                brain.ollama_status = "error"
                brain.ollama_force_request = False
                brain.ollama_pending_instruction = None
                self._append_log(brain, "error", err, tick=int(getattr(world, "tick_count", 0) or 0))
                if str(row.get("source") or "") == "manual":
                    room = self._room_for(world, agent, room_manager)
                    self._record_manual_command_example(
                        world,
                        room,
                        agent,
                        brain,
                        success=False,
                        outcome=err,
                        tick=int(getattr(world, "tick_count", 0) or 0),
                    )
                    self._clear_manual_command_state(agent)
                continue

            advice = row.get("advice")
            if advice is None:
                continue
            source = str(row.get("source") or "auto")
            current_tick = int(getattr(world, "tick_count", 0) or 0)
            if source != "manual" and (
                str(getattr(brain, "ollama_pending_instruction", "") or "").strip()
                or self._operator_override_active(brain, tick=current_tick)
            ):
                self._append_log(brain, "system", "Автоурок проигнорирован: действует приоритетный приказ оператора.", tick=current_tick)
                continue
            room = self._room_for(world, agent, room_manager)
            if isinstance(advice, CoachAdvice) and str(advice.model or "") != "brain_local_command":
                self._append_log(brain, "model", self._describe_advice(advice), tick=current_tick)
            self._apply_advice(world, room, agent, brain, advice, command_source=source)
            brain.ollama_pending_instruction = None
            brain.ollama_force_request = False

    def _apply_advice(self, world: Any, room: Any, agent: Any, brain: Any, advice: CoachAdvice, *, command_source: str = "auto") -> None:
        tick = int(getattr(world, "tick_count", 0) or 0)
        result = apply_advice_to_agent(agent, world, room, advice)
        parsed_text = ""
        if isinstance(advice.raw, dict) and isinstance(advice.raw.get("_action"), dict):
            parsed_text = summarize_control(advice.raw.get("_action"))
        elif result.get("goal") is not None:
            goal = result.get("goal")
            parsed_text = f"move({float(goal[0]):.2f}, {float(goal[1]):.2f})"
        if command_source == "manual" and parsed_text:
            self._update_active_manual_parsed(agent, parsed_text)
        action_result = self._apply_structured_action(
            world,
            room,
            agent,
            brain,
            advice.raw.get("_action") if isinstance(advice.raw, dict) else None,
            lesson=advice.lesson,
            tick=tick,
            command_source=command_source,
        )
        if result.get("goal") is None and isinstance(action_result.get("goal"), tuple):
            result["goal"] = action_result.get("goal")
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=320 if result.get("goal") is not None else 96)
        if result.get("goal") is not None:
            self._clear_manual_control(agent)
        brain.ollama_last_error = None
        brain.ollama_last_model = str(result.get("model") or advice.model)
        brain.ollama_last_thought = str(result.get("thought") or advice.thought or "")
        brain.ollama_last_response_tick = int(tick)
        brain.ollama_active_lesson = str(result.get("lesson") or advice.lesson or "")
        brain.ollama_active_goal = result.get("goal")
        brain.ollama_goal_started_tick = int(tick) if result.get("goal") is not None else None
        brain.ollama_active_reward_hint = float(result.get("reward_hint") or advice.reward_hint or 0.12)
        brain.ollama_active_command_source = str(command_source or "auto")
        brain.ollama_status = str(action_result.get("status") or ("guiding" if result.get("goal") is not None else "note"))
        speech_text = str(result.get("speech") or getattr(advice, "speech", "") or "").strip()
        if speech_text:
            self._append_log(brain, "agent", speech_text, tick=tick)
        elif brain.ollama_last_thought:
            brain.last_thought = brain.ollama_last_thought
            self._append_log(brain, "agent", brain.ollama_last_thought, tick=tick)
        goal = result.get("goal")
        lesson_text = brain.ollama_active_lesson
        if goal is not None:
            lesson_text = f"{lesson_text} -> goal=({goal[0]:.1f}, {goal[1]:.1f})"
        if lesson_text:
            self._append_log(brain, "ollama", lesson_text, tick=tick)

    def _apply_stop(self, world: Any, agent: Any, brain: Any, *, lesson: str, tick: int, command_source: str = "manual") -> None:
        self._clear_plan_state(agent)
        x = float(getattr(agent, "x", 0.0))
        y = float(getattr(agent, "y", 0.0))
        facing = (
            _safe_float(getattr(agent, "_manual_facing_x", 1.0), 1.0),
            _safe_float(getattr(agent, "_manual_facing_y", 0.0), 0.0),
        )
        apply_manual = getattr(agent, "apply_manual_control", None)
        if callable(apply_manual):
            try:
                apply_manual(
                    x,
                    y,
                    dt=0.05,
                    world_size=(float(getattr(world, "width", x)), float(getattr(world, "height", y))),
                    facing=facing,
                    tick=tick,
                    hold_ticks=18,
                    source="ollama:stop",
                )
            except Exception:
                pass
        try:
            agent.goal_x = x
            agent.goal_y = y
            agent.vx = 0.0
            agent.vy = 0.0
        except Exception:
            pass
        if hasattr(brain, "clear_ollama_goal"):
            brain.clear_ollama_goal()
        else:
            brain.ollama_active_goal = None
            brain.ollama_goal_started_tick = None
            brain.ollama_active_lesson = ""
        brain.ollama_last_error = None
        brain.ollama_last_model = "brain_local_command"
        brain.ollama_last_thought = "Останавливаюсь."
        brain.ollama_last_response_tick = int(tick)
        brain.ollama_status = "stopped"
        brain.ollama_active_command_source = str(command_source or "auto")
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=96)
        try:
            brain.last_thought = brain.ollama_last_thought
        except Exception:
            pass
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(brain, "agent", "Останавливаюсь.", tick=tick)

    def _apply_local_control(
        self,
        world: Any,
        room: Any,
        agent: Any,
        brain: Any,
        control: Optional[Dict[str, Any]],
        *,
        lesson: str,
        tick: int,
    ) -> bool:
        if not isinstance(control, dict):
            return False
        ctype = str(control.get("type") or "")
        if ctype == "sequence":
            steps = [dict(step) for step in list(control.get("steps", []) or []) if isinstance(step, dict)]
            if not steps:
                return False
            self._store_plan_queue(agent, steps[1:])
            first = steps[0]
            self._append_log(brain, "plan", summarize_control(control), tick=tick)
            return self._apply_local_control(world, room, agent, brain, first, lesson=lesson, tick=tick)
        if ctype == "set_move_speed":
            speed = _safe_float(control.get("speed"), _safe_float(getattr(agent, "move_speed", 3.0), 3.0))
            preset = str(control.get("preset") or "")
            self._apply_move_speed(agent, brain, speed=speed, lesson=lesson, tick=tick, preset=preset, command_source="manual")
            self._append_log(brain, "system", f"Скорость ходьбы изменена локально: {float(speed):.2f}.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="speed_tuned", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "stop":
            self._clear_plan_state(agent)
            self._apply_stop(world, agent, brain, lesson=lesson, tick=tick)
            self._append_log(brain, "system", "Команда остановки выполнена локально через мозг.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="stopped", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "move_to_landmark":
            landmark = str(control.get("landmark") or "")
            goal = self._landmark_goal(agent, world, room, landmark)
            if goal is None:
                return False
            advice = CoachAdvice(
                model="brain_local_command",
                thought="Иду к ориентиру комнаты.",
                lesson=lesson or f"Иду к {landmark}.",
                goal=goal,
                belief=None,
                behavior={},
                reward_hint=0.14,
                raw={"goal_source": "operator_landmark", "_action": {"name": "move_to_landmark", "args": {"landmark": landmark}}},
            )
            self._apply_advice(world, room, agent, brain, advice, command_source="manual")
            self._append_log(brain, "system", f"Команда движения к ориентиру `{landmark}` выполнена локально.", tick=tick)
            return True
        if ctype == "move":
            goal_raw = control.get("goal")
            if not (isinstance(goal_raw, tuple) and len(goal_raw) == 2):
                return False
            advice = CoachAdvice(
                model="brain_local_command",
                thought="Выполняю короткую команду движения.",
                lesson=lesson or "Локальный шаг.",
                goal=(float(goal_raw[0]), float(goal_raw[1])),
                belief=None,
                behavior={},
                reward_hint=0.12,
                raw={"goal_source": "operator_sequence"},
            )
            self._apply_advice(world, room, agent, brain, advice, command_source="manual")
            self._append_log(brain, "system", "Локальный шаг движения выполнен по плану.", tick=tick)
            return True
        if ctype == "face_landmark":
            landmark = str(control.get("landmark") or "")
            goal = self._landmark_goal(agent, world, room, landmark)
            if goal is None:
                return False
            self._apply_face(world, agent, brain, target=goal, lesson=lesson or f"Смотрю на {landmark}.", tick=tick, status="facing", command_source="manual")
            self._append_log(brain, "system", f"Поворот к ориентиру `{landmark}` выполнен локально.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="facing", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "face_direction":
            direction = str(control.get("direction") or "")
            self._apply_face_direction(world, agent, brain, direction=direction, lesson=lesson or f"Поворачиваюсь {direction}.", tick=tick, command_source="manual")
            self._append_log(brain, "system", f"Поворот `{direction}` выполнен локально.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="facing", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "wait":
            hold_ticks = max(12, int(control.get("ticks") or 48))
            self._apply_wait(world, agent, brain, hold_ticks=hold_ticks, lesson=lesson or "Стою и жду.", tick=tick, command_source="manual")
            self._append_log(brain, "system", "Команда ожидания выполнена локально.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="waiting", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "remember_note":
            text = str(control.get("text") or "").strip()
            if not text:
                return False
            self._apply_memory_note(agent, brain, text=text, tick=tick, lesson=lesson or text, command_source="manual")
            self._append_log(brain, "system", "Заметка сохранена в память локально.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="memorized", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        if ctype == "tune_emotion":
            self._apply_emotion_tone(
                agent,
                brain,
                preset=str(control.get("preset") or ""),
                fear=control.get("fear"),
                curiosity=control.get("curiosity"),
                energy_delta=control.get("energy_delta"),
                drive=control.get("drive"),
                thought=control.get("thought"),
                lesson=lesson or "Настраиваю эмоциональный фон.",
                tick=tick,
                command_source="manual",
            )
            self._append_log(brain, "system", "Эмоциональное состояние скорректировано локально.", tick=tick)
            self._record_manual_command_example(world, room, agent, brain, success=True, outcome="emotion_tuned", tick=tick)
            self._clear_manual_command_state(agent)
            return True
        return False

    def _apply_structured_action(
        self,
        world: Any,
        room: Any,
        agent: Any,
        brain: Any,
        action: Optional[Dict[str, Any]],
        *,
        lesson: str,
        tick: int,
        command_source: str,
    ) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return {}
        name = str(action.get("name") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if name == "stop":
            self._apply_stop(world, agent, brain, lesson=lesson, tick=tick, command_source=command_source)
            return {"status": "stopped"}
        if name == "face_landmark":
            goal = self._landmark_goal(agent, world, room, str(args.get("landmark") or ""))
            if goal is not None:
                self._apply_face(world, agent, brain, target=goal, lesson="", tick=tick, status="facing", command_source=command_source)
                return {"status": "facing"}
        if name == "face_direction":
            self._apply_face_direction(world, agent, brain, direction=str(args.get("direction") or ""), lesson="", tick=tick, command_source=command_source)
            return {"status": "facing"}
        if name == "set_move_speed":
            self._apply_move_speed(
                agent,
                brain,
                speed=_safe_float(args.get("speed"), _safe_float(getattr(agent, "move_speed", 3.0), 3.0)),
                lesson="",
                tick=tick,
                preset="",
                command_source=command_source,
            )
            return {"status": "speed_tuned"}
        if name == "wait":
            self._apply_wait(world, agent, brain, hold_ticks=max(12, int(args.get("ticks") or 48)), lesson="", tick=tick, command_source=command_source)
            return {"status": "waiting"}
        if name == "remember_note":
            text = str(args.get("text") or "").strip()
            if text:
                self._apply_memory_note(agent, brain, text=text, tick=tick, lesson="", command_source=command_source)
                return {"status": "memorized"}
        if name == "tune_emotion":
            self._apply_emotion_tone(
                agent,
                brain,
                preset=str(args.get("preset") or ""),
                fear=args.get("fear"),
                curiosity=args.get("curiosity"),
                energy_delta=args.get("energy_delta"),
                drive=args.get("drive"),
                thought=args.get("thought"),
                lesson=lesson,
                tick=tick,
                command_source=command_source,
            )
            return {"status": "emotion_tuned"}
        return {}

    def _landmark_goal(self, agent: Any, world: Any, room: Any, landmark: str) -> Optional[Tuple[float, float]]:
        snapshot = build_training_snapshot(agent, world, room)
        landmarks = ((snapshot.get("room") or {}).get("landmarks") or {})
        target = landmarks.get(str(landmark))
        if not isinstance(target, dict):
            return None
        return (_safe_float(target.get("x"), 0.0), _safe_float(target.get("y"), 0.0))

    def _apply_face(
        self,
        world: Any,
        agent: Any,
        brain: Any,
        *,
        target: Tuple[float, float],
        lesson: str,
        tick: int,
        status: str,
        command_source: str,
    ) -> None:
        if hasattr(brain, "clear_ollama_goal"):
            brain.clear_ollama_goal()
        else:
            brain.ollama_active_goal = None
            brain.ollama_goal_started_tick = None
            brain.ollama_active_lesson = ""
        ax = _safe_float(getattr(agent, "x", 0.0), 0.0)
        ay = _safe_float(getattr(agent, "y", 0.0), 0.0)
        dx = float(target[0]) - ax
        dy = float(target[1]) - ay
        length = math.hypot(dx, dy)
        facing = (1.0, 0.0) if length <= 1e-6 else (dx / length, dy / length)
        apply_manual = getattr(agent, "apply_manual_control", None)
        if callable(apply_manual):
            try:
                apply_manual(
                    ax,
                    ay,
                    dt=0.05,
                    world_size=(float(getattr(world, "width", ax)), float(getattr(world, "height", ay))),
                    facing=facing,
                    tick=tick,
                    hold_ticks=14,
                    source="ollama:face",
                )
            except Exception:
                pass
        try:
            agent.goal_x = ax
            agent.goal_y = ay
            agent.vx = 0.0
            agent.vy = 0.0
            agent._manual_facing_x = float(facing[0])
            agent._manual_facing_y = float(facing[1])
        except Exception:
            pass
        brain.ollama_last_model = str(getattr(brain, "ollama_last_model", None) or "brain_local_command")
        brain.ollama_last_thought = "Фокусируюсь на ориентире."
        brain.ollama_last_response_tick = int(tick)
        brain.ollama_status = str(status)
        brain.ollama_active_command_source = str(command_source or "auto")
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=72)
        try:
            brain.last_thought = brain.ollama_last_thought
        except Exception:
            pass
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(brain, "agent", brain.ollama_last_thought, tick=tick)

    def _apply_face_direction(self, world: Any, agent: Any, brain: Any, *, direction: str, lesson: str, tick: int, command_source: str) -> None:
        fx = _safe_float(getattr(agent, "_manual_facing_x", getattr(agent, "vx", 1.0)), 1.0)
        fy = _safe_float(getattr(agent, "_manual_facing_y", getattr(agent, "vy", 0.0)), 0.0)
        norm = math.hypot(fx, fy)
        if norm <= 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = fx / norm, fy / norm
        direction = str(direction or "forward")
        if direction == "backward":
            target = (-fx, -fy)
        elif direction == "left":
            target = (-fy, fx)
        elif direction == "right":
            target = (fy, -fx)
        else:
            target = (fx, fy)
        ax = _safe_float(getattr(agent, "x", 0.0), 0.0)
        ay = _safe_float(getattr(agent, "y", 0.0), 0.0)
        self._apply_face(world, agent, brain, target=(ax + target[0], ay + target[1]), lesson=lesson, tick=tick, status="facing", command_source=command_source)

    def _apply_wait(self, world: Any, agent: Any, brain: Any, *, hold_ticks: int, lesson: str, tick: int, command_source: str) -> None:
        if hasattr(brain, "clear_ollama_goal"):
            brain.clear_ollama_goal()
        else:
            brain.ollama_active_goal = None
            brain.ollama_goal_started_tick = None
            brain.ollama_active_lesson = ""
        x = float(getattr(agent, "x", 0.0))
        y = float(getattr(agent, "y", 0.0))
        facing = (
            _safe_float(getattr(agent, "_manual_facing_x", 1.0), 1.0),
            _safe_float(getattr(agent, "_manual_facing_y", 0.0), 0.0),
        )
        apply_manual = getattr(agent, "apply_manual_control", None)
        if callable(apply_manual):
            try:
                apply_manual(
                    x,
                    y,
                    dt=0.05,
                    world_size=(float(getattr(world, "width", x)), float(getattr(world, "height", y))),
                    facing=facing,
                    tick=tick,
                    hold_ticks=max(12, int(hold_ticks)),
                    source="ollama:wait",
                )
            except Exception:
                pass
        try:
            agent.goal_x = x
            agent.goal_y = y
            agent.vx = 0.0
            agent.vy = 0.0
        except Exception:
            pass
        brain.ollama_last_model = str(getattr(brain, "ollama_last_model", None) or "brain_local_command")
        brain.ollama_last_thought = "Жду и наблюдаю."
        brain.ollama_last_response_tick = int(tick)
        brain.ollama_status = "waiting"
        brain.ollama_active_command_source = str(command_source or "auto")
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=max(hold_ticks, 48))
        try:
            brain.last_thought = brain.ollama_last_thought
        except Exception:
            pass
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(brain, "agent", brain.ollama_last_thought, tick=tick)
        self._set_wait_until(agent, tick, hold_ticks)

    def _apply_move_speed(
        self,
        agent: Any,
        brain: Any,
        *,
        speed: float,
        lesson: str,
        tick: int,
        preset: str,
        command_source: str,
    ) -> None:
        target = round(_safe_float(speed, _safe_float(getattr(agent, "move_speed", 3.0), 3.0)), 3)
        target = _safe_float(max(0.85, min(6.5, target)), 3.0)
        try:
            agent.move_speed = float(target)
        except Exception:
            pass
        thought = "Иду медленнее." if preset == "slow" else ("Ускоряю шаг." if preset == "fast" else "Подстраиваю скорость шага.")
        try:
            brain.ollama_last_model = str(getattr(brain, "ollama_last_model", None) or "brain_local_command")
            brain.ollama_last_thought = thought
            brain.ollama_last_response_tick = int(tick)
            brain.ollama_status = "speed_tuned"
            brain.ollama_active_command_source = str(command_source or "auto")
            brain.last_thought = thought
        except Exception:
            pass
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=64)
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(brain, "agent", f"{thought} Скорость={float(target):.2f}.", tick=tick)

    def _apply_emotion_tone(
        self,
        agent: Any,
        brain: Any,
        *,
        preset: str,
        fear: Any,
        curiosity: Any,
        energy_delta: Any,
        drive: Any,
        thought: Any,
        lesson: str,
        tick: int,
        command_source: str,
    ) -> None:
        preset_key = str(preset or "").strip().casefold()
        target_fear = None if fear is None else _safe_float(fear, 0.1)
        target_curiosity = None if curiosity is None else _safe_float(curiosity, 0.5)
        delta_energy = _safe_float(energy_delta, 0.0) if energy_delta is not None else 0.0
        target_drive = str(drive or "").strip()
        thought_text = str(thought or "").strip()

        if preset_key == "calm":
            target_fear = 0.06 if target_fear is None else target_fear
            target_curiosity = 0.34 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "idle"
            thought_text = thought_text or "Успокаиваюсь и выравниваю состояние."
        elif preset_key == "focus":
            target_fear = 0.08 if target_fear is None else target_fear
            target_curiosity = 0.42 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "idle"
            thought_text = thought_text or "Собираюсь и держу фокус."
        elif preset_key == "curious":
            target_fear = 0.08 if target_fear is None else target_fear
            target_curiosity = 0.82 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "explore"
            thought_text = thought_text or "Мне интересно изучить комнату."
        elif preset_key == "brave":
            target_fear = 0.03 if target_fear is None else target_fear
            target_curiosity = 0.58 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "explore"
            thought_text = thought_text or "Действую увереннее."
        elif preset_key == "rest":
            target_fear = 0.04 if target_fear is None else target_fear
            target_curiosity = 0.22 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "rest"
            delta_energy = delta_energy if abs(delta_energy) > 1e-6 else 8.0
            thought_text = thought_text or "Делаю паузу и восстанавливаюсь."
        elif preset_key == "energize":
            target_fear = 0.08 if target_fear is None else target_fear
            target_curiosity = 0.74 if target_curiosity is None else target_curiosity
            target_drive = target_drive or "explore"
            delta_energy = delta_energy if abs(delta_energy) > 1e-6 else 10.0
            thought_text = thought_text or "Чувствую прилив энергии."

        if target_fear is not None:
            target_fear = _safe_float(max(0.0, min(1.0, target_fear)), 0.08)
            try:
                brain.fear_level = float(target_fear)
            except Exception:
                pass
            try:
                agent.fear = float(target_fear)
            except Exception:
                pass
        if target_curiosity is not None:
            target_curiosity = _safe_float(max(0.0, min(1.0, target_curiosity)), 0.5)
            try:
                brain.curiosity_charge = float(target_curiosity)
            except Exception:
                pass
            try:
                if hasattr(brain, "behavior_rules"):
                    shift = 0.0
                    if target_curiosity >= 0.7:
                        shift = 0.08
                    elif target_curiosity <= 0.3:
                        shift = -0.05
                    if shift != 0.0:
                        brain.behavior_rules.exploration_bias = max(
                            0.0,
                            min(1.0, float(getattr(brain.behavior_rules, "exploration_bias", 0.2)) + shift),
                        )
            except Exception:
                pass
        if abs(delta_energy) > 1e-6:
            try:
                agent.energy = max(0.0, min(100.0, _safe_float(getattr(agent, "energy", 100.0), 100.0) + float(delta_energy)))
            except Exception:
                pass
            try:
                brain.energy = max(0.0, min(100.0, _safe_float(getattr(brain, "energy", 100.0), 100.0) + float(delta_energy)))
            except Exception:
                pass
        if target_drive:
            try:
                brain.current_drive = str(target_drive)
            except Exception:
                pass
        if not thought_text:
            thought_text = "Подстраиваю внутреннее состояние."
        try:
            brain.ollama_last_model = str(getattr(brain, "ollama_last_model", None) or "brain_local_command")
            brain.ollama_last_thought = thought_text[:140]
            brain.ollama_last_response_tick = int(tick)
            brain.ollama_status = "emotion_tuned"
            brain.ollama_active_command_source = str(command_source or "auto")
            brain.last_thought = brain.ollama_last_thought
        except Exception:
            pass
        if command_source == "manual":
            self._mark_operator_override(brain, tick=tick, hold_ticks=72)
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(
            brain,
            "agent",
            f"{thought_text[:120]} fear={getattr(brain, 'fear_level', getattr(agent, 'fear', 0.0)):.2f} curiosity={getattr(brain, 'curiosity_charge', 0.5):.2f}.",
            tick=tick,
        )

    def _apply_memory_note(self, agent: Any, brain: Any, *, text: str, tick: int, lesson: str, command_source: str) -> None:
        note = str(text or "").strip()
        if not note:
            return
        if brain is not None and hasattr(brain, "add_memory"):
            try:
                brain.add_memory({"type": "ollama_note", "tick": int(tick), "data": {"text": note}})
            except Exception:
                pass
        if brain is not None and hasattr(brain, "add_belief"):
            try:
                brain.add_belief({"if": "operator_note", "then": note[:120], "strength": 0.56})
            except Exception:
                pass
        try:
            brain.ollama_last_model = str(getattr(brain, "ollama_last_model", None) or "brain_local_command")
            brain.ollama_last_thought = "Запоминаю это."
            brain.ollama_last_response_tick = int(tick)
            brain.ollama_status = "memorized"
            brain.ollama_active_command_source = str(command_source or "auto")
            if command_source == "manual":
                self._mark_operator_override(brain, tick=tick, hold_ticks=64)
            brain.last_thought = brain.ollama_last_thought
        except Exception:
            pass
        if lesson:
            self._append_log(brain, "ollama", lesson, tick=tick)
        self._append_log(brain, "agent", "Запоминаю это.", tick=tick)

    def _append_log(self, brain: Any, role: str, text: str, *, tick: Optional[int] = None) -> None:
        if brain is None:
            return
        if hasattr(brain, "_append_ollama_log"):
            brain._append_ollama_log(role, text, tick=tick)

    def _find_agent(self, world: Any, agent_id: str) -> Optional[Any]:
        if hasattr(world, "get_agent_by_id"):
            try:
                out = world.get_agent_by_id(agent_id)
                if out is not None:
                    return out
            except Exception:
                pass
        for agent in _iter_agents(world):
            if str(getattr(agent, "id", getattr(agent, "agent_id", ""))) == str(agent_id):
                return agent
        return None
