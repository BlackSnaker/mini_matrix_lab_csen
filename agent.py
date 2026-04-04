# agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, cast
import math
import random

import config
from memory import AgentMemory
from mind_core import ConsciousnessBlock

try:
    from brain_io import load_brain  # type: ignore
except Exception:
    load_brain = None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


# ---------- SAFE HELPERS: TIME/DT/WORLD SIZE ----------

def _world_time_int(world) -> int:
    """
    Безопасно получить «тик/время» мира, не превращая метод в int(<method>).
    Приоритеты: get_time() → time → ticks → tick_count → current_tick → frame/step → t → tick (как поле).
    """
    candidates = ("get_time", "time", "ticks", "tick_count", "current_tick",
                  "frame", "step", "t", "tick")
    for name in candidates:
        v = getattr(world, name, None)
        if v is None:
            continue
        if callable(v) and name == "tick":  # не вызываем tick()
            continue
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        try:
            return int(v)
        except Exception:
            continue
    return 0


def _world_dt(world) -> float:
    """Безопасно получить dt (сек/тик). Если его нет — вернуть 1.0."""
    v = getattr(world, "dt", 1.0)
    if callable(v):
        try:
            v = v()
        except Exception:
            v = 1.0
    try:
        return float(v)
    except Exception:
        return 1.0


def _world_size(world) -> Optional[Tuple[float, float]]:
    """Пытаемся вытащить (W, H) из world/world.map/arena и т.п."""
    for obj_name in ("", "world", "map", "arena"):
        obj = world if not obj_name else getattr(world, obj_name, None)
        if obj is None:
            continue
        W = getattr(obj, "width", None) or getattr(obj, "w", None)
        H = getattr(obj, "height", None) or getattr(obj, "h", None)
        try:
            if W is not None and H is not None:
                return float(W), float(H)
        except Exception:
            continue
    return None


def _pct_value(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v <= 1.5:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _tick_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _belief_public_view(b: Any) -> Dict[str, Any]:
    if isinstance(b, dict):
        return {
            "if": str(b.get("if") or b.get("condition") or ""),
            "then": str(b.get("then") or b.get("conclusion") or ""),
            "strength": float(b.get("strength", 0.0) or 0.0),
        }
    return {
        "if": str(getattr(b, "condition", "")),
        "then": str(getattr(b, "conclusion", "")),
        "strength": float(getattr(b, "strength", 0.0) or 0.0),
    }


def _memory_public_view(ev: Any) -> Dict[str, Any]:
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
            "tick": _tick_or_none(ev.get("tick")),
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
        "tick": _tick_or_none(getattr(ev, "tick", None)),
        "level": str(getattr(ev, "level", "info")),
        "actor": getattr(ev, "actor", None),
        "pos": pos if isinstance(pos, tuple) and len(pos) == 2 else None,
        "data": dict(getattr(ev, "data", {}) or {}),
    }


# ---------- INTROSPECTION SHIM FOR BRAIN (beliefs + neural graph) ----------

def _ensure_brain_introspection_api(brain: Any) -> None:
    """
    Добавляет недостающие поля/методы мозгу, чтобы UI получил:
      - beliefs (get_beliefs/add_belief)
      - memory_tail (appendable как объекты с .tick/.etype/.data)
      - export_neural2d / export_neural3d / export_graph
      - debug_weights
    """

    # --- Лёгкий адаптер события памяти под .tick/.etype/.data ---
    class _MemEv:
        __slots__ = ("tick", "etype", "data")
        def __init__(self, tick: Optional[int], etype: str, data: Dict[str, Any]):
            self.tick = tick
            self.etype = etype
            self.data = data

    def _as_event(ev: Any) -> _MemEv:
        # Уже объект?
        if hasattr(ev, "tick") and hasattr(ev, "etype") and hasattr(ev, "data"):
            return cast(_MemEv, ev)
        # Словарь?
        if isinstance(ev, dict):
            tick = ev.get("tick")
            etype = ev.get("etype") or ev.get("type") or ev.get("event") or ev.get("name") or "event"
            data = ev.get("data")
            if not isinstance(data, dict):
                # Если нет вложенного "data", берём весь словарь, но без служебных ключей
                data = {k: v for k, v in ev.items() if k not in ("tick", "etype", "type", "event", "name")}
            return _MemEv(tick=tick, etype=str(etype), data=data)
        # Что-то иное → завернём как есть
        return _MemEv(tick=None, etype=type(ev).__name__, data={"value": ev})

    # beliefs store
    if not hasattr(brain, "beliefs"):
        brain.beliefs = []  # type: List[Dict[str, Any]]
    if not hasattr(brain, "get_beliefs"):
        def get_beliefs() -> List[Dict[str, Any]]:
            return list(brain.beliefs)  # type: ignore[attr-defined]
        brain.get_beliefs = get_beliefs  # type: ignore[attr-defined]
    if not hasattr(brain, "add_belief"):
        def add_belief(b: Dict[str, Any]) -> None:
            bl = cast(List[Dict[str, Any]], getattr(brain, "beliefs", []))
            bl.append(b)
            if len(bl) > 128:
                del bl[:-128]
            brain.beliefs = bl
        brain.add_belief = add_belief  # type: ignore[attr-defined]

    # memory_tail: нормализуем существующее и оборачиваем добавления
    mt = cast(List[Any], getattr(brain, "memory_tail", []))
    try:
        brain.memory_tail = [_as_event(e) for e in mt]  # type: ignore[attr-defined]
    except Exception:
        brain.memory_tail = []  # type: ignore[attr-defined]

    if not hasattr(brain, "add_memory"):
        def add_memory(ev: Dict[str, Any]) -> None:
            mt_local = cast(List[Any], getattr(brain, "memory_tail", []))
            mt_local.append(_as_event(ev))
            # ограничиваем хвост
            if len(mt_local) > 256:
                del mt_local[:-256]
            brain.memory_tail = mt_local  # type: ignore[attr-defined]
        brain.add_memory = add_memory  # type: ignore[attr-defined]

    # weights exposure
    if not hasattr(brain, "debug_weights"):
        def debug_weights() -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k in ("avoid_hazard_radius", "healing_zone_seek_priority",
                      "curiosity", "seek_food_priority", "rest_priority"):
                if hasattr(brain, k):
                    try:
                        out[k] = float(getattr(brain, k))
                    except Exception:
                        pass
            return out
        brain.debug_weights = debug_weights  # type: ignore[attr-defined]

    # neural graph (миникарта)
    def _export_graph_common() -> Dict[str, Any]:
        w = {}
        try:
            w = dict(brain.debug_weights())
        except Exception:
            pass

        nodes = [
            {"id": "in_pain", "label": "pain", "type": "input"},
            {"id": "in_hunger", "label": "hunger", "type": "input"},
            {"id": "in_safe", "label": "safe_zone", "type": "input"},
            {"id": "in_fear", "label": "fear", "type": "input"},
            {"id": "act_avoid", "label": "avoid", "type": "action"},
            {"id": "act_seek_heal", "label": "seek_heal", "type": "action"},
            {"id": "act_seek_food", "label": "seek_food", "type": "action"},
            {"id": "act_explore", "label": "explore", "type": "action"},
            {"id": "act_rest", "label": "rest", "type": "action"},
        ]

        def _w(name: str, default: float) -> float:
            try:
                return float(w.get(name, default))
            except Exception:
                return default

        edges = [
            {"source": "in_pain",   "target": "act_avoid",     "w": 0.7},
            {"source": "in_pain",   "target": "act_seek_heal", "w": 0.6},
            {"source": "in_fear",   "target": "act_avoid",     "w": 0.5},
            {"source": "in_hunger", "target": "act_seek_food", "w": 0.6},
            {"source": "in_safe",   "target": "act_rest",      "w": 0.4},
            {"source": "in_safe",   "target": "act_explore",   "w": 0.2 + 0.6*_w("curiosity", 0.0)},
            {"source": "in_fear",   "target": "act_seek_heal", "w": 0.2 + 0.8*_w("healing_zone_seek_priority", 0.5)},
            {"source": "in_hunger", "target": "act_rest",      "w": 0.1 + 0.4*_w("rest_priority", 0.0)},
        ]

        graph = {"nodes": nodes, "edges": edges,
                 "ts": _world_time_int(getattr(brain, "_last_world", object()))}
        return graph

    if not hasattr(brain, "export_neural2d"):
        brain.export_neural2d = _export_graph_common  # type: ignore[attr-defined]
    if not hasattr(brain, "export_neural3d"):
        brain.export_neural3d = _export_graph_common  # type: ignore[attr-defined]
    if not hasattr(brain, "export_graph"):
        brain.export_graph = _export_graph_common     # type: ignore[attr-defined]
    if not hasattr(brain, "neural_graph"):
        brain.neural_graph = _export_graph_common()   # type: ignore[attr-defined]


@dataclass
class Agent:
    """
    Агент с мозгом, авто-обороной и fallback-движением (если мир сам не двигает),
    + шим для визуализации belief/нейрографа.
    """

    # идентификаторы / профиль
    agent_id: str
    name: str
    persona: str = "caring/supportive"

    # положение и цель
    x: float = 0.0
    y: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    # витальные параметры
    health: float = 100.0         # 0..100
    energy: float = 1.0           # 0..1
    hunger: float = 0.0           # 0..1 (0 — сыт)
    fear: float = 0.0             # 0..1
    alive: bool = True
    age_ticks: int = 0
    cause_of_death: Optional[str] = None

    # мозг
    brain: Optional[ConsciousnessBlock] = None
    memory: AgentMemory = field(default_factory=AgentMemory, repr=False)
    known_hazards: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    danger_zones: List[Tuple[float, float, float]] = field(default_factory=list, repr=False)

    # боёвка
    attack_power: float = field(default_factory=lambda: float(getattr(config, "AGENT_BASE_ATTACK_POWER", 8.0)))
    attack_range: float = field(default_factory=lambda: float(getattr(config, "AGENT_MELEE_RANGE", 1.6)))
    attack_cooldown: int = field(default_factory=lambda: int(getattr(config, "AGENT_ATTACK_COOLDOWN", 18)))
    # NEW: накопленный боевой навык. Чем выше, тем чаще агент решается на контратаку.
    combat_skill: float = 0.0     # 0..5, растёт в бою и снижает склонность к панике
    last_attacker_id: Optional[str] = None
    took_damage_tick: int = -10**9
    _last_attack_tick: int = -10**9
    _aggro_memory: Dict[str, int] = field(default_factory=dict)

    # движение (fallback)
    move_speed: float = field(default_factory=lambda: float(getattr(config, "AGENT_BASE_SPEED", 3.0)))
    arrive_eps: float = 0.6  # радиус «прибыли к цели»
    _manual_control_until_tick: int = field(default=-10**9, repr=False)
    _manual_control_source: Optional[str] = field(default=None, repr=False)
    _manual_facing_x: float = field(default=1.0, repr=False)
    _manual_facing_y: float = field(default=0.0, repr=False)

    # локальные флаги, чтобы не спамить одинаковые beliefs
    _belief_flags: Dict[str, bool] = field(default_factory=dict)

    # lineage / social training metadata
    generation: int = 0
    parents: List[str] = field(default_factory=list)
    lineage_role: str = "balanced"
    born_tick: int = 0
    tags: List[str] = field(default_factory=list)
    wolf_tame_skill: float = 0.0
    wolf_tame_successes: int = 0

    # служебное
    id: str = field(init=False)  # == agent_id
    _last_brain_error: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        self.id = self.agent_id
        if self.brain is None:
            self.brain = self._init_brain()
        self._apply_persona_defaults()
        self.alive = bool(self.alive) and self.health > 0.0
        self._sync_memory_from_brain()

    # ----------- BRAIN INIT -----------
    def _init_brain(self) -> ConsciousnessBlock:
        if load_brain is not None:
            try:
                br = load_brain(self.agent_id)
                if isinstance(br, ConsciousnessBlock):
                    try:
                        if hasattr(br, "set_persona"):
                            br.set_persona(self.persona)
                        elif hasattr(br, "persona"):
                            setattr(br, "persona", self.persona)
                    except Exception:
                        pass
                    _ensure_brain_introspection_api(br)
                    return br
            except Exception:
                pass
        try:
            br = ConsciousnessBlock(agent_id=self.agent_id, persona=self.persona)
        except TypeError:
            br = ConsciousnessBlock(agent_id=self.agent_id)
            try:
                if hasattr(br, "set_persona"):
                    br.set_persona(self.persona)
                elif hasattr(br, "persona"):
                    setattr(br, "persona", self.persona)
            except Exception:
                pass
        _ensure_brain_introspection_api(br)
        return br

    # ----------- PERSONA TWEAKS -----------
    def _apply_persona_defaults(self):
        p = (self.persona or "").lower()
        if "scout" in p or "развед" in p:
            self.energy = min(1.0, self.energy + 0.2)
            self.attack_range = max(self.attack_range, 1.7)
            self.move_speed *= 1.1
        elif "protective" in p or "защит" in p:
            self.attack_power *= 1.1
            self.attack_cooldown = max(10, int(self.attack_cooldown * 0.9))
        elif "loner" in p or "одино" in p:
            self.fear = min(1.0, self.fear + 0.1)

    def _sync_memory_from_brain(self) -> None:
        if self.memory.get_recent(1):
            return
        tail = list(getattr(getattr(self, "brain", None), "memory_tail", []) or [])
        if tail:
            self.memory.extend_from_any(tail)

    def _remember_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        *,
        tick: Optional[int] = None,
        private: bool = False,
        level: Optional[str] = None,
        actor: Optional[str] = None,
        pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        payload = dict(data or {})
        if pos is not None:
            payload.setdefault("pos", pos)
        if actor:
            payload.setdefault("actor", actor)
        tick_val = _tick_or_none(tick)
        self.memory.remember(
            event_type,
            payload,
            tick=tick_val,
            private=private,
            level=level,
            actor=actor,
            pos=pos,
        )
        try:
            if self.brain and hasattr(self.brain, "record_event") and tick_val is not None:
                self.brain.record_event(tick_val, event_type, payload)
            elif self.brain and hasattr(self.brain, "add_memory"):
                ev: Dict[str, Any] = {
                    "type": event_type,
                    "tick": tick_val,
                    "data": payload,
                    "private": private,
                }
                if level is not None:
                    ev["level"] = level
                if actor is not None:
                    ev["actor"] = actor
                if pos is not None:
                    ev["pos"] = pos
                self.brain.add_memory(ev)
        except Exception:
            pass

    def _update_velocity(self, old_x: float, old_y: float, dt: float) -> None:
        if dt <= 1e-6:
            self.vx = 0.0
            self.vy = 0.0
            return
        self.vx = (self.x - old_x) / dt
        self.vy = (self.y - old_y) / dt

    def _refresh_world_knowledge(self, world) -> None:
        now = _world_time_int(world)

        def _upsert(hid: str, *, kind: str, x: float, y: float, radius: float, danger: float) -> None:
            self.known_hazards[hid] = {
                "id": hid,
                "kind": kind,
                "x": float(x),
                "y": float(y),
                "radius": max(0.5, float(radius)),
                "danger": max(0.0, float(danger)),
                "last_seen": int(now),
            }

        for obj in getattr(world, "objects", []) or []:
            kind = str(getattr(obj, "kind", "")).lower()
            if "hazard" not in kind and "danger" not in kind and "threat" not in kind:
                continue
            ox = float(getattr(obj, "x", 0.0))
            oy = float(getattr(obj, "y", 0.0))
            radius = float(getattr(obj, "radius", 2.0))
            level = float(getattr(obj, "danger_level", radius))
            hid = str(getattr(obj, "obj_id", f"hazard@{round(ox, 1)}:{round(oy, 1)}"))
            _upsert(hid, kind=kind or "hazard", x=ox, y=oy, radius=radius, danger=max(radius, level))

        animals = getattr(world, "animals", {}) or {}
        if isinstance(animals, dict):
            animal_items = list(animals.values())
        else:
            animal_items = list(animals)
        for ani in animal_items:
            if ani is None:
                continue
            species = getattr(ani, "species", None)
            aggressive = bool(getattr(species, "aggressive", False))
            alive = bool(getattr(ani, "is_alive", lambda: float(getattr(ani, "hp", 1.0)) > 0.0)())
            if not aggressive or not alive:
                continue
            ax = float(getattr(ani, "x", 0.0))
            ay = float(getattr(ani, "y", 0.0))
            hid = str(getattr(ani, "uid", f"animal@{round(ax, 1)}:{round(ay, 1)}"))
            _upsert(hid, kind="animal", x=ax, y=ay, radius=2.5, danger=3.0)

        cutoff = int(now) - 600
        self.known_hazards = {
            hid: hz for hid, hz in self.known_hazards.items()
            if int(hz.get("last_seen", now)) >= cutoff
        }

        zones: List[Tuple[float, float, float]] = []

        def _append_zone(px: float, py: float, weight: float) -> None:
            for i, (zx, zy, zw) in enumerate(zones):
                if _dist2(px, py, zx, zy) <= 9.0:
                    zones[i] = ((zx + px) * 0.5, (zy + py) * 0.5, max(zw, weight))
                    return
            zones.append((px, py, weight))

        for hz in self.known_hazards.values():
            _append_zone(float(hz["x"]), float(hz["y"]), max(1.0, float(hz["radius"])))

        for tm in getattr(getattr(self, "brain", None), "trauma_map", []) or []:
            pos = tm.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                continue
            _append_zone(float(pos[0]), float(pos[1]), max(1.0, float(tm.get("intensity", 0.0)) * 6.0))

        self.danger_zones = zones[-30:]

    def _public_memory_tail(self) -> List[Dict[str, Any]]:
        tail = self.memory.dump_public_view(tail=8)
        if tail:
            return tail
        out: List[Dict[str, Any]] = []
        for ev in list(getattr(getattr(self, "brain", None), "memory_tail", []) or [])[-8:]:
            out.append(_memory_public_view(ev))
        return out

    def _public_mind_state(self) -> Dict[str, Any]:
        brain = self.brain
        if brain is None:
            return {}
        try:
            data = dict(brain.export_public_state_for_ui() or {})
        except Exception:
            data = {}
        beliefs = data.get("beliefs", getattr(brain, "beliefs", [])) or []
        mem_tail = data.get("memory_tail", getattr(brain, "memory_tail", [])) or []
        data["beliefs"] = [_belief_public_view(b) for b in list(beliefs)[-20:]]
        data["memory_tail"] = [_memory_public_view(ev) for ev in list(mem_tail)[-20:]]
        return data

    # ----------- COMBAT -----------
    def can_attack(self, now_tick: int) -> bool:
        return (now_tick - self._last_attack_tick) >= int(self.attack_cooldown)

    def mark_attack(self, now_tick: int) -> None:
        self._last_attack_tick = int(now_tick)

    def receive_damage(self, amount: float, attacker_id: Optional[str], world_tick: int) -> None:
        dmg = float(max(0.0, amount))
        if dmg <= 0.0 or not self.is_alive():
            return

        health_before = float(self.health)
        self.health = max(0.0, health_before - dmg)
        self.last_attacker_id = attacker_id
        self.took_damage_tick = int(world_tick)

        self.fear = _clamp01(self.fear + min(0.6, dmg / 100.0 * 0.8))
        self.energy = _clamp01(self.energy - min(0.25, dmg / 100.0 * 0.3))

        if attacker_id:
            self._aggro_memory[attacker_id] = int(world_tick)

        if self.health <= 0.0 and not self.cause_of_death:
            self.cause_of_death = f"by:{attacker_id or 'unknown'}"
        self.alive = self.health > 0.0

        self._remember_event(
            "damage",
            {
                "amount": dmg,
                "by": attacker_id,
                "hp_before": health_before,
                "hp_after": self.health,
            },
            tick=world_tick,
            actor=attacker_id,
            level="critical" if self.health <= 0.0 else "warning",
        )
        if not self.alive:
            self._remember_event(
                "death",
                {"reason": self.cause_of_death or "damage"},
                tick=world_tick,
                level="critical",
            )

        try:
            if self.brain and hasattr(self.brain, "on_pain"):
                self.brain.on_pain(source_id=attacker_id, amount=dmg, pos=(self.x, self.y), tick=world_tick)
        except Exception:
            pass

    def on_animal_attack(self, world, attacker, damage: float, health_before: float) -> None:
        a_uid = getattr(attacker, "uid", None)
        species = getattr(attacker, "species", None)
        species_id = getattr(species, "species_id", None)
        now = _world_time_int(world)

        if a_uid:
            self.last_attacker_id = a_uid
            self._aggro_memory[a_uid] = now

        self.fear = _clamp01(self.fear + min(0.5, float(damage) / 100.0))
        self.alive = self.health > 0.0
        self._remember_event(
            "animal_attack",
            {
                "damage": float(damage),
                "species": species_id,
                "hp_before": float(health_before),
                "hp_after": float(self.health),
            },
            tick=now,
            actor=a_uid or species_id,
            pos=(float(self.x), float(self.y)),
            level="critical" if self.health <= 0.0 else "warning",
        )

        # убеждение: «Если рядом зверь → отступай/ищи безопасную зону»
        self._push_belief(
            cond="near(animal)",
            concl="retreat_to_safe_zone",
            strength=0.7,
            now=now
        )

        try:
            if self._ollama_operator_override_active(world):
                pass
            elif not self._in_safe_zone(world) and getattr(attacker, "x", None) is not None:
                ax, ay = float(getattr(attacker, "x", 0.0)), float(getattr(attacker, "y", 0.0))
                bounds = _world_size(world)
                if self._should_counterattack_after_hit():
                    # Осмысленная контратака: держим врага в зоне мили, а не уходим автоматически.
                    # NEW: вместо «всегда отступать» опытный/уверенный агент может пойти в бой.
                    self.set_goal(ax, ay, world_size=bounds, reason="counterattack", tick=now)
                    self._push_belief(
                        cond="animal_attacks_often",
                        concl="counterattack_if_ready",
                        strength=0.6,
                        now=now,
                    )
                else:
                    vx, vy = self.x - ax, self.y - ay
                    n = math.hypot(vx, vy) or 1.0
                    vx, vy = vx / n, vy / n
                    retreat = 6.0 + 6.0 * random.random()
                    gx = self.x + vx * retreat
                    gy = self.y + vy * retreat
                    self.set_goal(gx, gy, world_size=bounds, reason="retreat", tick=now)
        except Exception:
            pass

        if self.health <= 0.0 and not self.cause_of_death:
            self.cause_of_death = f"animal:{species_id or (a_uid or 'unknown')}"

        try:
            if self.brain and hasattr(self.brain, "on_threat"):
                self.brain.on_threat(
                    kind="animal",
                    attacker_id=a_uid,
                    species=species_id,
                    damage=float(damage),
                    pos=(self.x, self.y),
                    tick=now,
                )
            elif self.brain and hasattr(self.brain, "add_memory"):
                self.brain.add_memory({
                    "type": "attack_received",
                    "tick": now,
                    "actor": a_uid or "animal",
                    "level": "high",
                    "data": {"damage": float(damage), "species": species_id},
                })
        except Exception:
            pass

        try:
            if self.brain and hasattr(self.brain, "note_combat_feedback"):
                # Отрицательная обратная связь за полученный урон.
                # NEW: передаём в мозг сигнал, что текущий обмен в бою был невыгодным.
                self.brain.note_combat_feedback(
                    hp_delta=-float(max(0.0, damage)),
                    dist_delta=0.0,
                    done=not self.is_alive(),
                )
        except Exception:
            pass

    def _should_counterattack_after_hit(self) -> bool:
        """
        Решение «держаться и драться» вместо автоматического бегства.
        Чем выше combat_skill и здоровье — тем выше шанс контратаки.
        """
        # NEW: простая эвристика уверенности (здоровье + навык + низкий страх).
        hp_ratio = _clamp01(float(self.health) / 100.0)
        fear = _clamp01(float(self.fear))
        skill = max(0.0, min(5.0, float(self.combat_skill)))
        confidence = (0.55 * hp_ratio) + (0.35 * (skill / 5.0)) + (0.25 * (1.0 - fear))
        return bool(confidence >= 0.62)

    # ----------- LIFE -----------
    def is_alive(self) -> bool:
        return bool(self.alive) and self.health > 0.0

    def set_goal(
        self,
        x: float,
        y: float,
        world_size: Optional[Tuple[float, float]] = None,
        reason: Optional[str] = None,
        tick: Optional[int] = None,
    ) -> None:
        gx, gy = float(x), float(y)
        if world_size:
            W, H = world_size
            gx = max(0.0, min(W, gx))
            gy = max(0.0, min(H, gy))
        changed = (abs(self.goal_x - gx) > 1e-6) or (abs(self.goal_y - gy) > 1e-6)
        self.goal_x, self.goal_y = gx, gy
        if changed and reason:
            event_type = "external_command" if "command" in reason else "new_goal"
            self._remember_event(
                event_type,
                {"goal": (gx, gy), "reason": reason},
                tick=tick,
                actor="self",
                pos=(gx, gy),
            )

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - float(x), self.y - float(y))

    def mark_manual_control(
        self,
        *,
        tick: Optional[int],
        hold_ticks: int = 2,
        source: str = "player",
    ) -> None:
        base_tick = _tick_or_none(tick)
        if base_tick is None:
            base_tick = 0
        self._manual_control_until_tick = max(
            int(self._manual_control_until_tick),
            int(base_tick) + max(1, int(hold_ticks)),
        )
        self._manual_control_source = str(source or "manual")

    def is_manual_control_active(self, world: Any = None, *, tick: Optional[int] = None) -> bool:
        now = _tick_or_none(tick)
        if now is None and world is not None:
            now = _world_time_int(world)
        if now is None:
            now = 0
        return int(now) <= int(self._manual_control_until_tick)

    def clear_manual_control(self) -> None:
        self._manual_control_until_tick = -10**9
        self._manual_control_source = None

    def apply_manual_control(
        self,
        x: float,
        y: float,
        *,
        dt: float,
        world_size: Optional[Tuple[float, float]] = None,
        facing: Optional[Tuple[float, float]] = None,
        tick: Optional[int] = None,
        hold_ticks: int = 2,
        source: str = "player",
    ) -> float:
        gx, gy = float(x), float(y)
        if world_size:
            W, H = world_size
            gx = max(0.0, min(float(W), gx))
            gy = max(0.0, min(float(H), gy))
        old_x, old_y = float(self.x), float(self.y)
        self.x = gx
        self.y = gy
        self.goal_x = gx
        self.goal_y = gy
        self._update_velocity(old_x, old_y, max(float(dt), 1e-6))
        fx, fy = (float(self.vx), float(self.vy))
        if facing is not None:
            fx, fy = float(facing[0]), float(facing[1])
        n = math.hypot(fx, fy)
        if n > 1e-6:
            self._manual_facing_x = fx / n
            self._manual_facing_y = fy / n
        self.mark_manual_control(tick=tick, hold_ticks=hold_ticks, source=source)
        return math.hypot(self.x - old_x, self.y - old_y)

    def _in_safe_zone(self, world) -> bool:
        try:
            for obj in getattr(world, "objects", []) or []:
                if getattr(obj, "kind", "") == "safe":
                    if _dist2(self.x, self.y, float(getattr(obj, "x", 0.0)), float(getattr(obj, "y", 0.0))) <= (float(getattr(obj, "radius", 0.0)) ** 2):
                        return True
        except Exception:
            pass
        return False

    def brain_tick(self, world_ctx: Any = None) -> None:
        try:
            if self.brain and hasattr(self.brain, "tick_update"):
                # сохраним мир внутрь мозга — для штампа времени в графе
                try:
                    setattr(self.brain, "_last_world", world_ctx)
                except Exception:
                    pass
                if hasattr(self.brain, "absorb_recent_memory_summary"):
                    self.brain.absorb_recent_memory_summary(self.memory.summarize_recent())
                self.brain.tick_update(agent_ref=self, world_ref=world_ctx)
                self._last_brain_error = None
        except Exception as exc:
            err = str(exc)
            if err != self._last_brain_error:
                self._remember_event(
                    "brain_error",
                    {"error": err},
                    tick=_world_time_int(world_ctx) if world_ctx is not None else None,
                    private=True,
                    level="warning",
                )
            self._last_brain_error = err

    def _ollama_operator_override_active(self, world: Any = None) -> bool:
        brain = getattr(self, "brain", None)
        if brain is None:
            return False
        tick = _world_time_int(world) if world is not None else None
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
        now = 0 if tick is None else int(tick)
        return bool(getattr(brain, "ollama_authoritative_commands", True)) and now <= until

    def soft_needs_update(self, dt: float = 1.0, in_safe: bool = False) -> None:
        self.age_ticks += 1
        self.hunger = _clamp01(self.hunger + 0.002 * dt)
        if self.fear > 0.4:
            self.energy = _clamp01(self.energy - 0.01 * dt)
        else:
            self.energy = _clamp01(self.energy + 0.005 * dt)
        if in_safe:
            self.health = min(100.0, self.health + 0.03 * 100.0 * dt / 60.0)
            self.fear = _clamp01(self.fear - 0.02 * dt)

    # ----------- FALLBACK MOVE + SIMPLE REWARD -----------
    def _random_goal_nearby(self, world, radius: float = 12.0) -> Tuple[float, float]:
        angle = random.random() * math.tau
        r = radius * (0.4 + 0.6 * random.random())
        gx = self.x + math.cos(angle) * r
        gy = self.y + math.sin(angle) * r
        bounds = _world_size(world)
        if bounds:
            W, H = bounds
            gx = max(0.0, min(W, gx))
            gy = max(0.0, min(H, gy))
        return gx, gy

    def _fallback_move(self, world) -> float:
        """
        Если мир сам двигает агентов (world.handles_agent_motion / move_agent) — отдаём управление ему.
        Иначе — простая «движуха» к цели + выбор новой цели по прибытию.
        Возвращаем пройденную дистанцию (для простого поощрения).
        """
        dt = _world_dt(world)
        old_x, old_y = self.x, self.y
        operator_override = self._ollama_operator_override_active(world)

        if self.is_manual_control_active(world):
            self.goal_x = self.x
            self.goal_y = self.y
            return 0.0

        if getattr(world, "handles_agent_motion", False):
            self.vx = 0.0
            self.vy = 0.0
            return 0.0
        if hasattr(world, "move_agent"):
            try:
                moved = float(world.move_agent(self, dt) or 0.0)
                self._update_velocity(old_x, old_y, dt)
                if moved > 1e-3:
                    self._remember_event(
                        "move",
                        {"from": (old_x, old_y), "to": (self.x, self.y)},
                        tick=_world_time_int(world),
                        pos=(self.x, self.y),
                    )
                return moved
            except Exception:
                pass

        bounds = _world_size(world)

        # цель достигнута → выбираем новую
        if self.distance_to(self.goal_x, self.goal_y) <= self.arrive_eps or (self.goal_x == self.x and self.goal_y == self.y):
            if operator_override:
                self.goal_x = self.x
                self.goal_y = self.y
                self.vx = 0.0
                self.vy = 0.0
                return 0.0
            if self.last_attacker_id and getattr(world, "animals", None) and self.last_attacker_id in getattr(world, "animals", {}):
                ani = world.animals[self.last_attacker_id]
                ax, ay = float(getattr(ani, "x", self.x)), float(getattr(ani, "y", self.y))
                vx, vy = self.x - ax, self.y - ay
                n = math.hypot(vx, vy) or 1.0
                vx, vy = vx / n, vy / n
                retreat = 8.0 + 10.0 * random.random()
                gx, gy = self.x + vx * retreat, self.y + vy * retreat
                if bounds:
                    W, H = bounds
                    gx = max(0.0, min(W, gx))
                    gy = max(0.0, min(H, gy))
                self.set_goal(gx, gy, world_size=bounds)
            else:
                gx, gy = self._random_goal_nearby(world, radius=12.0)
                self.set_goal(gx, gy, world_size=bounds)

        # шаг к цели
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        dist = math.hypot(dx, dy)
        if dist <= 1e-6:
            self.vx = 0.0
            self.vy = 0.0
            return 0.0

        speed = self.move_speed
        speed *= (0.5 + 0.5 * self.energy)
        speed *= (1.0 - 0.3 * _clamp01(self.hunger))
        speed *= (1.0 - 0.25 * _clamp01(self.fear))

        step = max(0.0, speed * dt)
        if step >= dist:
            self.x, self.y = self.goal_x, self.goal_y
            moved = dist
        else:
            k = step / dist
            self.x += dx * k
            self.y += dy * k
            if bounds:
                W, H = bounds
                self.x = max(0.0, min(W, self.x))
                self.y = max(0.0, min(H, self.y))
            moved = step

        self._update_velocity(old_x, old_y, dt)
        if moved > 1e-3:
            self._remember_event(
                "move",
                {"from": (old_x, old_y), "to": (self.x, self.y)},
                tick=_world_time_int(world),
                pos=(self.x, self.y),
            )
        return moved

    def _brain_step_reward(self, world, moved: float) -> None:
        """
        Простейшее «поощрение» мозга: немного за движение/живость, штраф за голод/страх.
        Также пишем снимок состояния, чтобы стимульнуть построение памяти/связей.
        """
        reward = 0.0
        reward += min(0.02, moved * 0.01)  # лёгкая награда за движение
        reward += 0.002                    # базовая «жизнь»
        reward -= 0.003 * _clamp01(self.hunger)
        reward -= 0.003 * _clamp01(self.fear)

        snapshot = {
            "tick": _world_time_int(world),
            "pos": (self.x, self.y),
            "goal": (self.goal_x, self.goal_y),
            "health": self.health,
            "energy": self.energy,
            "hunger": self.hunger,
            "fear": self.fear,
            "in_safe": self._in_safe_zone(world),
        }

        try:
            if self.brain and hasattr(self.brain, "on_step"):
                self.brain.on_step(state=snapshot, reward=float(reward))
            elif self.brain and hasattr(self.brain, "reinforce"):
                self.brain.reinforce(float(reward))
            elif self.brain and hasattr(self.brain, "add_memory"):
                self.brain.add_memory({"type": "step", "data": {"reward": float(reward), **snapshot}})
        except Exception:
            pass

    # ----------- BELIEFS AUTO-UPDATE -----------
    def _push_belief(self, *, cond: str, concl: str, strength: float, now: int) -> None:
        key = f"{cond}->{concl}"
        if self._belief_flags.get(key):
            return
        try:
            if self.brain and hasattr(self.brain, "add_belief"):
                self.brain.add_belief({"if": cond, "then": concl, "strength": float(strength), "tick": now})
            self._belief_flags[key] = True
        except Exception:
            pass

    def _update_beliefs(self, world) -> None:
        now = _world_time_int(world)
        if self.health <= 35.0:
            self._push_belief(cond="low_hp", concl="seek_safe_zone", strength=0.8, now=now)
        if self.hunger >= 0.7:
            self._push_belief(cond="high_hunger", concl="seek_food", strength=0.7, now=now)
        if self.fear >= 0.6:
            self._push_belief(cond="high_fear", concl="avoid_threat", strength=0.6, now=now)

    # ----------- MAIN TICK -----------
    def tick(self, world) -> None:
        if not self.is_alive():
            return

        dt = _world_dt(world)
        in_safe = self._in_safe_zone(world)

        self.soft_needs_update(dt=dt, in_safe=in_safe)
        self._refresh_world_knowledge(world)
        self.brain_tick(world_ctx=world)

        moved = self._fallback_move(world)  # движение (если мир не двигает)
        self._auto_defend_melee(world)      # авто-оборона
        self._brain_step_reward(world, moved)
        self._update_beliefs(world)         # подпитываем beliefs для UI

        # поддержим поле brain.neural_graph для UI, если хотят читать напрямую
        try:
            if self.brain and hasattr(self.brain, "export_graph"):
                self.brain.neural_graph = self.brain.export_graph()
        except Exception:
            pass

    # ----------- THREAT PICK + MELEE -----------
    def _select_threat(self, world) -> Optional[Any]:
        animals = getattr(world, "animals", {}) or {}
        if not animals:
            return None

        now = _world_time_int(world)

        if self.last_attacker_id and self.last_attacker_id in animals:
            ani = animals[self.last_attacker_id]
            if getattr(ani, "is_alive", lambda: True)():
                if _dist2(self.x, self.y, float(getattr(ani, "x", 0.0)), float(getattr(ani, "y", 0.0))) <= (self.attack_range + 0.2) ** 2:
                    return ani

        best = None
        best_d2 = (self.attack_range + 0.2) ** 2
        for ani in animals.values():
            try:
                if not getattr(ani, "is_alive", lambda: True)():
                    continue
                species = getattr(ani, "species", None)
                aggressive = bool(getattr(species, "aggressive", True))
                if not aggressive:
                    last = self._aggro_memory.get(getattr(ani, "uid", ""), -10**9)
                    if now - last > 200:
                        continue
                d2 = _dist2(self.x, self.y, float(getattr(ani, "x", 0.0)), float(getattr(ani, "y", 0.0)))
                if d2 <= best_d2:
                    best_d2 = d2
                    best = ani
            except Exception:
                continue
        return best

    def _auto_defend_melee(self, world) -> None:
        if not self.is_alive():
            return
        if self._ollama_operator_override_active(world):
            return
        now = _world_time_int(world)
        target = self._select_threat(world)
        if target is None or not self.can_attack(now):
            return

        dmg = float(self.attack_power) * (0.85 + 0.3 * random.random())
        try:
            if hasattr(target, "receive_damage"):
                target.receive_damage(dmg, attacker_id=self.id, world_tick=now)
        except Exception:
            pass
        self.mark_attack(now)
        try:
            if hasattr(world, "add_event"):
                world.add_event({
                    "type": "agent_melee",
                    "tick": now,
                    "who": self.name,
                    "agent_id": self.id,
                    "target": getattr(target, "uid", "animal"),
                    "damage": round(dmg, 2),
                })
            elif hasattr(world, "add_chat_line"):
                world.add_chat_line(f"[combat] {self.name} ударил зверя на {dmg:.1f}")
        except Exception:
            pass

    # ----------- PUBLIC SNAPSHOT -----------
    def serialize_public_state(self) -> Dict[str, Any]:
        facing_x = float(self._manual_facing_x)
        facing_y = float(self._manual_facing_y)
        if math.hypot(self.vx, self.vy) > 1e-6:
            n = math.hypot(self.vx, self.vy)
            facing_x = float(self.vx) / n
            facing_y = float(self.vy) / n
        out = {
            "id": self.id,
            "name": self.name,
            "alive": self.is_alive(),
            "pos": {"x": float(self.x), "y": float(self.y)},
            "vel": {"x": float(self.vx), "y": float(self.vy)},
            "goal": {"x": float(self.goal_x), "y": float(self.goal_y)},
            "facing": {"x": facing_x, "y": facing_y},
            "health": float(self.health),
            "energy": _pct_value(self.energy, 100.0),
            "hunger": _pct_value(self.hunger, 0.0),
            "fear": float(_clamp01(self.fear)),
            "age_ticks": int(self.age_ticks),
            "cause_of_death": self.cause_of_death,
            "manual_control_source": self._manual_control_source,
            "danger_zones_count": int(len(self.danger_zones)),
            "hazards_known": int(len(self.known_hazards)),
            "memory_tail": self._public_memory_tail(),
            "mind": self._public_mind_state(),
            "tags": list(self.tags),
            "generation": int(self.generation),
            "parents": list(self.parents),
            "lineage_role": str(self.lineage_role),
        }
        return out

    def to_public_snapshot(self) -> Dict[str, Any]:
        return self.serialize_public_state()
