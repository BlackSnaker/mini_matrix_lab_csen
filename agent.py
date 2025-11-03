# agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, cast
import math
import random

import config
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

    # витальные параметры
    health: float = 100.0         # 0..100
    energy: float = 1.0           # 0..1
    hunger: float = 0.0           # 0..1 (0 — сыт)
    fear: float = 0.0             # 0..1
    age_ticks: int = 0
    cause_of_death: Optional[str] = None

    # мозг
    brain: Optional[ConsciousnessBlock] = None

    # боёвка
    attack_power: float = field(default_factory=lambda: float(getattr(config, "AGENT_BASE_ATTACK_POWER", 8.0)))
    attack_range: float = field(default_factory=lambda: float(getattr(config, "AGENT_MELEE_RANGE", 1.6)))
    attack_cooldown: int = field(default_factory=lambda: int(getattr(config, "AGENT_ATTACK_COOLDOWN", 18)))
    last_attacker_id: Optional[str] = None
    took_damage_tick: int = -10**9
    _last_attack_tick: int = -10**9
    _aggro_memory: Dict[str, int] = field(default_factory=dict)

    # движение (fallback)
    move_speed: float = field(default_factory=lambda: float(getattr(config, "AGENT_BASE_SPEED", 3.0)))
    arrive_eps: float = 0.6  # радиус «прибыли к цели»

    # локальные флаги, чтобы не спамить одинаковые beliefs
    _belief_flags: Dict[str, bool] = field(default_factory=dict)

    # служебное
    id: str = field(init=False)  # == agent_id

    def __post_init__(self):
        self.id = self.agent_id
        if self.brain is None:
            self.brain = self._init_brain()
        self._apply_persona_defaults()

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

        try:
            if self.brain and hasattr(self.brain, "on_pain"):
                self.brain.on_pain(source_id=attacker_id, amount=dmg)
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

        # убеждение: «Если рядом зверь → отступай/ищи безопасную зону»
        self._push_belief(
            cond="near(animal)",
            concl="retreat_to_safe_zone",
            strength=0.7,
            now=now
        )

        try:
            if not self._in_safe_zone(world) and getattr(attacker, "x", None) is not None:
                ax, ay = float(getattr(attacker, "x", 0.0)), float(getattr(attacker, "y", 0.0))
                vx, vy = self.x - ax, self.y - ay
                n = math.hypot(vx, vy) or 1.0
                vx, vy = vx / n, vy / n
                retreat = 6.0 + 6.0 * random.random()
                gx = self.x + vx * retreat
                gy = self.y + vy * retreat
                bounds = _world_size(world)
                self.set_goal(gx, gy, world_size=bounds)
        except Exception:
            pass

        if self.health <= 0.0 and not self.cause_of_death:
            self.cause_of_death = f"animal:{species_id or (a_uid or 'unknown')}"

        try:
            if self.brain and hasattr(self.brain, "on_threat"):
                self.brain.on_threat(kind="animal", attacker_id=a_uid, species=species_id, damage=float(damage))
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

    # ----------- LIFE -----------
    def is_alive(self) -> bool:
        return self.health > 0.0

    def set_goal(self, x: float, y: float, world_size: Optional[Tuple[float, float]] = None) -> None:
        gx, gy = float(x), float(y)
        if world_size:
            W, H = world_size
            gx = max(0.0, min(W, gx))
            gy = max(0.0, min(H, gy))
        self.goal_x, self.goal_y = gx, gy

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - float(x), self.y - float(y))

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
                self.brain.tick_update(agent_ref=self, world_ref=world_ctx)
        except Exception:
            pass

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
        if getattr(world, "handles_agent_motion", False):
            return 0.0
        if hasattr(world, "move_agent"):
            try:
                return float(world.move_agent(self, _world_dt(world)) or 0.0)
            except Exception:
                pass

        dt = _world_dt(world)
        bounds = _world_size(world)

        # цель достигнута → выбираем новую
        if self.distance_to(self.goal_x, self.goal_y) <= self.arrive_eps or (self.goal_x == self.x and self.goal_y == self.y):
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
            return 0.0

        speed = self.move_speed
        speed *= (0.5 + 0.5 * self.energy)
        speed *= (1.0 - 0.3 * _clamp01(self.hunger))
        speed *= (1.0 - 0.25 * _clamp01(self.fear))

        step = max(0.0, speed * dt)
        if step >= dist:
            self.x, self.y = self.goal_x, self.goal_y
            return dist
        else:
            k = step / dist
            self.x += dx * k
            self.y += dy * k
            if bounds:
                W, H = bounds
                self.x = max(0.0, min(W, self.x))
                self.y = max(0.0, min(H, self.y))
            return step

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
    def to_public_snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "alive": self.is_alive(),
            "pos": {"x": float(self.x), "y": float(self.y)},
            "goal": {"x": float(self.goal_x), "y": float(self.goal_y)},
            "health": float(self.health),
            "energy": float(self.energy * 100.0),
            "hunger": float(self.hunger * 100.0),
            "fear": float(self.fear),
            "age_ticks": int(self.age_ticks),
            "cause_of_death": self.cause_of_death,
        }
