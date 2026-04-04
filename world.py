# world.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
import random

import config

# серверная модель зверя
from animals import Animal as AnimalSim
from wolf_taming_system import WolfTamingSystem
from agent_reproduction_system import AgentReproductionSystem
# Агент теперь в отдельном модуле, но оставляем совместимость импорта: from world import Agent
from agent import Agent  # re-export для server.py

__all__ = [
    "World",
    "WorldObject",
    "push_global_event",
    "set_global_event_sink",
    "Agent",
]

# =============================================================================
# Глобальные константы/баланс (+ версия синхронизации для движка)
# =============================================================================

ENGINE_SYNC_VERSION = 3                                   # инкрементируй при изменении формата
ENGINE3D_SCALE = getattr(config, "ENGINE3D_SCALE", 1.0)   # масштаб для 3D-пакета

COMM_RADIUS = 15.0
MAX_CHAT_LOG = 50
MAX_EVENT_LOG = 100  # лог мировых событий (для HUD/истории)

MERGE_DANGER_DIST = 5.0
MAX_DANGER_POINTS = 30

SPEAK_EVERY_N_TICKS = 6
PANIC_THRESHOLD = 0.6
CRITICAL_HEALTH = 30.0
LIMP_HEALTH = 50.0
DEAD_HEALTH = 0.0

FEAR_DECAY = 0.9
RANDOM_SPIKE_PROB = 0.02
RANDOM_SPIKE_VALUE = 0.5

ENERGY_MAX = 100.0
HUNGER_MAX = 100.0
ENERGY_DRAIN_PER_TICK = 0.4
HUNGER_GAIN_PER_TICK = 0.3
LOW_ENERGY = 30.0
HIGH_HUNGER = 70.0

SAFE_ACTIVITIES_HEAL = ("heal", "calm", "sleep", "rest", "repair_self")
SAFE_ACTIVITIES_EAT = ("eat", "restock_food")
SAFE_ACTIVITIES_REST = ("rest", "sleep", "calm")

# тело/«разведение» между агентами
AGENT_BODY_RADIUS = 1.0

AVOID_RADIUS = 4.0          # личное пространство для антистолпотворения
SEPARATION_STRENGTH = 5.0   # сила разлёта

COLLISION_RADIUS = AGENT_BODY_RADIUS * 2.0  # ~2.0
BLOCK_DOT_THRESHOLD = 0.7
SIDE_STEP_STRENGTH = 1.5

POST_RESOLVE_PUSH = 1.0     # раздвижение после шага

# -----------------------------------------------------------------------------
# ЖИВОТНЫЕ / ПИТОМЦЫ (клиентская логика реакции агента)
# -----------------------------------------------------------------------------

ANIMAL_VIEW_RADIUS = 18.0    # радиус, на котором агент замечает зверя
TAME_RADIUS = 2.5            # дистанция для попытки приручения

# -----------------------------------------------------------------------------
# Флаги/радиусы из config с безопасными дефолтами
# -----------------------------------------------------------------------------

ALLY_HELP_RADIUS = getattr(config, "ALLY_HELP_RADIUS", 6.0)
PETS_DEFEND_OWNER = getattr(config, "PETS_DEFEND_OWNER", True)
AGENT_FIGHTBACK_ENABLED = getattr(config, "AGENT_FIGHTBACK_ENABLED", True)

# =============================================================================
# Глобальный синк событий для совместимости (push_global_event из других модулей)
# =============================================================================

_GLOBAL_WORLD_EVENT_SINK: Optional["World"] = None
_PENDING_EVENTS: List[Dict[str, Any]] = []


def set_global_event_sink(world: "World") -> None:
    """
    Назначить текущий мир глобальным приёмником событий и слить отложенные.
    """
    global _GLOBAL_WORLD_EVENT_SINK
    _GLOBAL_WORLD_EVENT_SINK = world
    if _PENDING_EVENTS:
        for ev in _PENDING_EVENTS:
            world.add_event(ev)
        _PENDING_EVENTS.clear()


def push_global_event(etype: str, **payload) -> None:
    """
    Универсальный вызов из внешних модулей.
    Поддерживает старый формат: push_global_event("brain_updated <agent_id>")
    """
    # Разбор упрощённого legacy-формата
    if not payload and isinstance(etype, str) and etype.startswith("brain_updated "):
        payload = {"agent_id": etype.split(" ", 1)[1]}
        etype = "brain_updated"

    ev = {"type": etype, **payload}
    if _GLOBAL_WORLD_EVENT_SINK is not None:
        _GLOBAL_WORLD_EVENT_SINK.push_global_event(etype, **payload)
    else:
        _PENDING_EVENTS.append(ev)


# =============================================================================
# Вспомогалки для 3D
# =============================================================================

def _xy_to_xz(x: float, y: float) -> Tuple[float, float, float]:
    """
    Отображение 2D-мира (x, y) → 3D (x, Yup, z). Высоту держим 0.0.
    Масштабируем через ENGINE3D_SCALE при желании.
    """
    s = ENGINE3D_SCALE
    return (x * s, 0.0, y * s)


def _compute_yaw_deg(vx: float, vy: float, dx_fallback: float, dy_fallback: float) -> float:
    """
    Яв — поворот вокруг вертикальной оси (Y) в градусах.
    Мы живём в XZ-плоскости: z=world.y, поэтому берём atan2(dx, dy).
    Если скорость почти 0 — берём направление на цель (fallback).
    """
    ax = vx
    ay = vy
    if abs(ax) + abs(ay) < 1e-6:
        ax = dx_fallback
        ay = dy_fallback
    if abs(ax) + abs(ay) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(ax, ay))


def _speed(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)


# =============================================================================
# Совместимость: гарантируем у Agent метод serialize_public_state
# =============================================================================

from typing import cast as _cast  # noqa: F401  (может понадобиться позже)

def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _pct_metric(x, default=0.0):
    v = _num(x, default)
    if v <= 1.5:
        v *= 100.0
    return max(0.0, min(100.0, v))


def _belief_public_view(b: Any) -> Dict[str, Any]:
    if isinstance(b, dict):
        return {
            "if": str(b.get("if") or b.get("condition") or ""),
            "then": str(b.get("then") or b.get("conclusion") or ""),
            "strength": _num(b.get("strength", 0.0), 0.0),
        }
    return {
        "if": str(getattr(b, "condition", "")),
        "then": str(getattr(b, "conclusion", "")),
        "strength": _num(getattr(b, "strength", 0.0), 0.0),
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
            "tick": ev.get("tick"),
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
        "tick": getattr(ev, "tick", None),
        "level": str(getattr(ev, "level", "info")),
        "actor": getattr(ev, "actor", None),
        "pos": pos if isinstance(pos, tuple) and len(pos) == 2 else None,
        "data": dict(getattr(ev, "data", {}) or {}),
    }

def _agent_default_public(self: Agent) -> Dict[str, Any]:
    """Публичное, безопасное представление агента для стрима/клиента."""
    try:
        yaw = _compute_yaw_deg(
            _num(getattr(self, "vx", 0.0)),
            _num(getattr(self, "vy", 0.0)),
            _num(getattr(self, "goal_x", getattr(self, "x", 0.0))) - _num(getattr(self, "x", 0.0)),
            _num(getattr(self, "goal_y", getattr(self, "y", 0.0))) - _num(getattr(self, "y", 0.0)),
        )
    except Exception:
        yaw = 0.0

    memory_tail: List[Dict[str, Any]] = []
    mem = getattr(self, "memory", None)
    if mem is not None and hasattr(mem, "dump_public_view"):
        try:
            memory_tail = list(mem.dump_public_view(tail=8))
        except Exception:
            memory_tail = []
    if not memory_tail:
        try:
            memory_tail = [
                _memory_public_view(ev)
                for ev in list(getattr(getattr(self, "brain", None), "memory_tail", []) or [])[-8:]
            ]
        except Exception:
            memory_tail = []

    mind: Dict[str, Any] = {}
    brain = getattr(self, "brain", None)
    if brain is not None:
        try:
            mind = dict(brain.export_public_state_for_ui() or {})
        except Exception:
            mind = {}
        try:
            beliefs_src = mind.get("beliefs", getattr(brain, "beliefs", [])) or []
            memory_src = mind.get("memory_tail", getattr(brain, "memory_tail", [])) or []
            mind["beliefs"] = [_belief_public_view(b) for b in list(beliefs_src)[-20:]]
            mind["memory_tail"] = [_memory_public_view(ev) for ev in list(memory_src)[-20:]]
        except Exception:
            pass

    out = {
        "id": getattr(self, "id", None),
        "name": getattr(self, "name", f"agent-{getattr(self, 'id', 'unknown')}"),

        "pos": {"x": _num(getattr(self, "x", 0.0)), "y": _num(getattr(self, "y", 0.0))},
        "goal": {"x": _num(getattr(self, "goal_x", getattr(self, "x", 0.0))),
                 "y": _num(getattr(self, "goal_y", getattr(self, "y", 0.0)))},

        "vel": {"x": _num(getattr(self, "vx", 0.0)), "y": _num(getattr(self, "vy", 0.0))},
        "yaw": yaw,

        # Унифицируем метрики состояния
        "health": _num(getattr(self, "health", getattr(self, "hp", 100.0))),
        "energy": _pct_metric(getattr(self, "energy", 100.0), 100.0),
        "hunger": _pct_metric(getattr(self, "hunger", 0.0), 0.0),
        "fear":   _num(getattr(self, "fear", 0.0)),
        "alive":  bool(self.is_alive()) if hasattr(self, "is_alive") else bool(getattr(self, "alive", True)),
        "age_ticks": int(getattr(self, "age_ticks", 0)),
        "danger_zones_count": int(len(getattr(self, "danger_zones", []) or [])),
        "hazards_known": int(len(getattr(self, "known_hazards", {}) or {})),
        "memory_tail": memory_tail,
        "mind": mind,

        # Необязательные поля для HUD
        "tags": list(getattr(self, "tags", [])) if hasattr(self, "tags") else [],
    }
    if hasattr(self, "generation"):
        out["generation"] = int(getattr(self, "generation", 0))
    if hasattr(self, "parents"):
        try:
            out["parents"] = list(getattr(self, "parents", []) or [])
        except Exception:
            out["parents"] = []
    if hasattr(self, "lineage_role"):
        out["lineage_role"] = str(getattr(self, "lineage_role", "balanced"))
    return out

# Если в Agent ещё нет метода — подмешиваем дефолтный
if not hasattr(Agent, "serialize_public_state"):
    setattr(Agent, "serialize_public_state", _agent_default_public)  # type: ignore[attr-defined]


# =============================================================================
# WorldObject
# =============================================================================

@dataclass(slots=True)
class WorldObject:
    """
    Объект окружения.
    kind:
       - "hazard": опасно (огонь, яд, радиация...)
       - "safe": безопасно / лечит / успокаивает / база
       - "neutral": ресурсная точка, ориентир, лут

    Доп. поля:
      - resource_tag: "food", "meds", "scrap" и т.д.
      - resource_abundance: насколько точка богата (0..1)
    """
    obj_id: str
    name: str
    kind: str
    x: float
    y: float
    radius: float

    danger_level: float = 0.0
    comfort_level: float = 0.0

    resource_tag: Optional[str] = None
    resource_abundance: float = 0.0

    def serialize_public(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.obj_id,
            "name": self.name,
            "kind": self.kind,
            "pos": {"x": self.x, "y": self.y},
            "pos3d": dict(zip(("x", "y", "z"), _xy_to_xz(self.x, self.y))),
            "radius": self.radius,
        }
        if self.kind == "hazard":
            data["approx_danger"] = round(self.danger_level, 2)
        if self.kind in ("safe", "neutral"):
            if self.comfort_level > 0.0:
                data["approx_comfort"] = round(self.comfort_level, 2)
            if self.resource_tag:
                data["resource_tag"] = self.resource_tag
                data["resource_abundance"] = round(self.resource_abundance, 2)
        return data


# =============================================================================
# World
# =============================================================================

class World:
    """
    Мир (серверная симуляция).
    Держит агентов, животных, объекты, активити-зоны, чат, лог событий.

    ВАЖНО:
    - animals: словарь {animal_id -> AnimalSim}
    - agents: словарь {agent_id -> Agent}
    - есть get_agent_by_id и _agent_log_attack_from_animal — их ждёт AnimalSim.tick()

    Новое:
    - export_for_engine3d(): готовит sync-пакет для 3D-движка (позиции, yaw, pos3d, цели, HUD).
    - fightback: агенты могут бить агрессивных зверей в мили, с кулдауном и модификаторами.
    - reproduction: агенты ищут партнёров и могут рождать новое поколение.
    """

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.tick_count = 0

        self.agents: Dict[str, Agent] = {}
        self.animals: Dict[str, AnimalSim] = {}

        self.objects: List[WorldObject] = []
        self.activities: Optional[Dict[str, Dict[str, Any]]] = None

        self.chat_log: List[str] = []
        self.event_log: List[Dict[str, Any]] = []
        self.chat = self.chat_log
        self.events = self.event_log

        # Кулдауны ударов агентов (tick номер, раньше — нельзя)
        self._agent_next_attack_tick: Dict[str, int] = {}
        # Отдельный модуль: обучение приручению волков + защита хозяев питомцами.
        self.wolf_taming = WolfTamingSystem(self)
        # Отдельный модуль: поиск партнёра + рождение + наследование опыта.
        self.agent_reproduction = AgentReproductionSystem(self)

        # Назначаем глобальный синк (и вливаем отложенные события)
        set_global_event_sink(self)

    # -----------------------------------------------------------------
    # нормализация/итерация животных
    # -----------------------------------------------------------------

    def _normalize_animals_container(self) -> None:
        """
        Гарантирует, что self.animals — dict(uid -> AnimalSim).
        Переводит из list/tuple/set при необходимости, чинит ключи != uid.
        """
        c = self.animals
        # Если уже dict — проверим ключи
        if isinstance(c, dict):
            fixed: Dict[str, AnimalSim] = {}
            for k, ani in list(c.items()):
                if ani is None:
                    continue
                uid = getattr(ani, "uid", None)
                if uid is None:
                    continue
                fixed[uid] = ani
            self.animals = fixed
            return

        # Если коллекция — соберём dict
        if isinstance(c, (list, tuple, set)):
            d: Dict[str, AnimalSim] = {}
            for ani in c:
                if ani is None:
                    continue
                uid = getattr(ani, "uid", None)
                if uid is None:
                    continue
                d[uid] = ani
            self.animals = d
            return

        # Иначе — пустой dict
        self.animals = {}

    def iter_animals(self) -> List[AnimalSim]:
        """
        Безопасно возвращает список животных независимо от формата контейнера.
        """
        if isinstance(self.animals, dict):
            return list(self.animals.values())
        if isinstance(self.animals, (list, tuple, set)):
            return [a for a in self.animals if a is not None]
        return []

    # -----------------------------------------------------------------
    # базовые операции над миром
    # -----------------------------------------------------------------

    def add_agent(self, agent: Agent):
        self.agents[agent.id] = agent

    def add_object(self, obj: WorldObject):
        self.objects.append(obj)

    def add_animal(self, animal: AnimalSim):
        # Починим контейнер на всякий случай и положим по ключу uid
        self._normalize_animals_container()
        self.animals[animal.uid] = animal

    def set_activity_registry(self, registry: Dict[str, Dict[str, Any]]):
        self.activities = registry

    def add_chat_line(self, text: str):
        self.chat_log.append(text)
        if len(self.chat_log) > MAX_CHAT_LOG:
            self.chat_log.pop(0)

    def add_event(self, ev: Dict[str, Any]):
        ev = dict(ev)
        ev["tick"] = self.tick_count
        self.event_log.append(ev)
        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log.pop(0)

    def push_global_event(self, etype: str, **payload) -> None:
        """
        Единая точка для глобальных событий: пишет в event_log и, при нужных
        типах, дублирует понятную строку в чат.
        """
        self.add_event({"type": etype, **payload})
        if etype == "brain_updated":
            aid = payload.get("agent_id")
            agent = self.agents.get(aid)
            who = agent.name if agent else aid
            self.add_chat_line(f"[brain] обновлён мозг {who or '<?>'}")

    def push_event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.push_global_event(kind, **dict(payload or {}))

    def announce_death(self, agent: Agent, reason: str):
        self.add_chat_line(f"[DEATH] {agent.name} погиб ({reason}) в t={self.tick_count}")
        self.add_event({"type": "death", "who": agent.id, "name": agent.name, "reason": reason})

    def set_agent_goal(self, agent_id: str, x: float, y: float) -> bool:
        a = self.agents.get(agent_id)
        if a is None:
            return False
        if hasattr(a, "set_goal"):
            try:
                a.set_goal(
                    x,
                    y,
                    world_size=(float(self.width), float(self.height)),
                    reason="external_command",
                    tick=self.tick_count,
                )
            except Exception:
                a.goal_x = max(0.0, min(float(self.width), float(x)))
                a.goal_y = max(0.0, min(float(self.height), float(y)))
        else:
            a.goal_x = max(0.0, min(float(self.width), float(x)))
            a.goal_y = max(0.0, min(float(self.height), float(y)))
        # лог и память остаются ответственностью агента/мозга в agent.py
        self.add_chat_line(f"[CMD] {a.name} получил приказ двигаться к ({float(a.goal_x):.1f}, {float(a.goal_y):.1f})")
        self.add_event({"type": "command_goal", "who": a.id, "name": a.name, "goal": (float(a.goal_x), float(a.goal_y))})
        return True

    def drive_agent(
        self,
        agent_id: str,
        x: float,
        y: float,
        dt: float,
        *,
        facing: Optional[Tuple[float, float]] = None,
        source: str = "player",
        hold_ticks: int = 3,
    ) -> bool:
        a = self.agents.get(agent_id)
        if a is None or not getattr(a, "is_alive", lambda: True)():
            return False
        if hasattr(a, "apply_manual_control"):
            try:
                a.apply_manual_control(
                    x,
                    y,
                    dt=max(float(dt), 1e-6),
                    world_size=(float(self.width), float(self.height)),
                    facing=facing,
                    tick=self.tick_count,
                    hold_ticks=max(1, int(hold_ticks)),
                    source=source,
                )
                return True
            except Exception:
                pass
        old_x = float(getattr(a, "x", 0.0))
        old_y = float(getattr(a, "y", 0.0))
        nx = max(0.0, min(float(self.width), float(x)))
        ny = max(0.0, min(float(self.height), float(y)))
        a.x = nx
        a.y = ny
        a.goal_x = nx
        a.goal_y = ny
        denom = max(float(dt), 1e-6)
        try:
            a.vx = (nx - old_x) / denom
            a.vy = (ny - old_y) / denom
        except Exception:
            pass
        return True

    # -----------------------------------------------------------------
    # функции, нужные зверям
    # -----------------------------------------------------------------

    def get_agent_by_id(self, agent_id: Optional[str]) -> Optional[Agent]:
        if agent_id is None:
            return None
        return self.agents.get(agent_id)

    def _agent_log_attack_from_animal(self, attacker: AnimalSim, victim: Agent, damage: float, health_before: float):
        # делегируем в агент (agent.py) через его public API if exists
        try:
            victim.on_animal_attack(self, attacker, damage, health_before)
        except Exception:
            # минимальный логлайн, если в агенте нет обработчика
            self.add_chat_line(f"[бой] {attacker.species.name} укусил(а) {victim.name} на {damage:.1f} урона!")
            if victim.health <= 0.0 and health_before > 0.0:
                victim.cause_of_death = getattr(victim, "cause_of_death", None) or f"animal:{attacker.species.species_id}"
                self.add_event({
                    "type": "death", "who": victim.id, "name": victim.name,
                    "reason": f"animal:{attacker.species.species_id}",
                })
        # событие для HUD
        self.add_event({
            "type": "animal_attack",
            "attacker_species": attacker.species.species_id,
            "victim_id": victim.id, "victim_name": victim.name,
            "damage": round(damage, 1),
        })

    # -----------------------------------------------------------------
    # Вспомогалки боёвки (ответ агента)
    # -----------------------------------------------------------------

    def _count_allies_near(self, me: Agent, radius: float) -> int:
        cnt = 0
        r2 = radius * radius
        mx, my = me.x, me.y
        for other in self.agents.values():
            if other is me or not other.is_alive():
                continue
            dx = other.x - mx
            dy = other.y - my
            if dx * dx + dy * dy <= r2:
                cnt += 1
        return cnt

    def _has_pet_near_owner(self, owner: Agent, radius: float) -> bool:
        r2 = radius * radius
        ox, oy = owner.x, owner.y
        for ani in self.iter_animals():
            if getattr(ani, "tamed_by", None) == owner.id and self._is_animal_alive(ani):
                dx = ani.x - ox
                dy = ani.y - oy
                if dx * dx + dy * dy <= r2:
                    return True
        return False

    def _nearest_aggressive_animal(self, me: Agent) -> Optional[Tuple[AnimalSim, float]]:
        best: Optional[Tuple[AnimalSim, float]] = None
        for ani in self.iter_animals():
            # безопасная проверка жизни зверя
            try:
                alive = bool(ani.is_alive()) if hasattr(ani, "is_alive") else (float(getattr(ani, "hp", 1.0)) > 0.0)
            except Exception:
                alive = True
            if not alive:
                continue
            # public флаг агрессии (см. animals.build_public_state/ species.aggressive)
            try:
                is_aggr = bool(getattr(ani.species, "aggressive", False))
            except Exception:
                is_aggr = False
            if not is_aggr:
                continue
            d = math.hypot(ani.x - me.x, ani.y - me.y)
            if best is None or d < best[1]:
                best = (ani, d)
        return best

    def _animal_can_take_damage(self, ani: AnimalSim) -> bool:
        return hasattr(ani, "apply_damage") or hasattr(ani, "hp")

    def _agent_can_attack_now(self, a: Agent) -> bool:
        if not bool(AGENT_FIGHTBACK_ENABLED):
            return False
        if not a.is_alive():
            return False
        next_tick = self._agent_next_attack_tick.get(a.id, 0)
        return self.tick_count >= next_tick

    def _effective_fear(self, a: Agent) -> float:
        eff_fear = float(getattr(a, "fear", 0.0))
        # союзники рядом повышают смелость
        allies = self._count_allies_near(a, float(ALLY_HELP_RADIUS))
        if allies:
            eff_fear = max(0.0, eff_fear - 0.08 * min(allies, 3))
        # питомец рядом тоже снижает страх
        if bool(PETS_DEFEND_OWNER) and self._has_pet_near_owner(a, float(ALLY_HELP_RADIUS)):
            eff_fear = max(0.0, eff_fear - 0.12)
        return eff_fear

    def _animal_apply_damage(self, ani: AnimalSim, dmg: float, source_id: str, source_name: str) -> bool:
        """
        Пытаемся нанести урон зверю. Возвращаем True, если зверь умер.
        """
        died = False
        # Нормально — если у AnimalSim есть метод, иначе — фолбэк по hp
        if hasattr(ani, "apply_damage") and callable(getattr(ani, "apply_damage")):
            try:
                died = bool(ani.apply_damage(dmg, by=source_id))
            except TypeError:
                died = bool(ani.apply_damage(dmg))  # на случай другой сигнатуры
        else:
            # аккуратно уменьшаем hp, если поле есть
            hp_before = float(getattr(ani, "hp", 100.0))
            hp_after = max(0.0, hp_before - float(dmg))
            try:
                setattr(ani, "hp", hp_after)
            except Exception:
                pass
            died = (hp_after <= 0.0)

        # Логи
        self.add_chat_line(f"[бой] {source_name} ударил(а) {getattr(ani.species, 'name', 'beast')} на {dmg:.1f} урона!")
        self.add_event({
            "type": "agent_attack",
            "who": source_id,
            "name": source_name,
            "victim_species": getattr(ani.species, "species_id", "?"),
            "damage": round(dmg, 1),
        })
        if died:
            self.add_chat_line(f"[бой] {getattr(ani.species, 'name', 'beast')} повержен(а).")
            self.add_event({
                "type": "animal_killed",
                "by": source_id,
                "by_name": source_name,
                "species": getattr(ani.species, "species_id", "?"),
                "animal_id": getattr(ani, "uid", None),
            })
        return died

    def _agent_melee_strike(self, a: Agent, ani: AnimalSim) -> None:
        """
        Реализует один мили-удар с модификаторами состояния.
        """
        # NEW: база урона берётся из агента и усиливается его боевым навыком.
        base = float(getattr(a, "attack_power", getattr(config, "AGENT_BASE_ATTACK_POWER", 8.0)))
        skill = max(0.0, min(5.0, float(getattr(a, "combat_skill", 0.0))))
        base *= (1.0 + 0.08 * skill)

        # NEW: поддержка двух шкал энергии (0..1 и 0..100), чтобы не ломать баланс урона.
        # Модификаторы: энергия/здоровье (0..1), страх (пенальти), союз/питомец (бонус)
        energy_raw = float(getattr(a, "energy", 100.0))
        if energy_raw <= 1.0:
            energy_ratio = max(0.0, min(1.0, energy_raw))
        else:
            energy_ratio = max(0.0, min(1.0, energy_raw / 100.0))
        health_ratio = max(0.0, min(1.0, float(getattr(a, "health", 100.0)) / 100.0))
        fear = self._effective_fear(a)

        stamina_mult = 0.7 + 0.3 * energy_ratio
        health_mult = 0.6 + 0.4 * health_ratio
        fear_mult = max(0.5, 1.0 - fear)  # чем больше страх — тем меньше урон

        ally_bonus = 1.0 + 0.10 * min(self._count_allies_near(a, float(ALLY_HELP_RADIUS)), 3)
        pet_bonus = 1.15 if (bool(PETS_DEFEND_OWNER) and self._has_pet_near_owner(a, float(ALLY_HELP_RADIUS))) else 1.0

        dmg = base * stamina_mult * health_mult * fear_mult * ally_bonus * pet_bonus
        dmg = max(1.0, round(dmg, 1))

        died = self._animal_apply_damage(ani, dmg, a.id, a.name)
        # NEW: рост навыка за боевой опыт (больше за добивание), плюс снижение страха за успех.
        if died:
            a.combat_skill = min(5.0, float(getattr(a, "combat_skill", 0.0)) + 0.12)
            a.fear = max(0.0, float(getattr(a, "fear", 0.0)) - 0.06)
        else:
            a.combat_skill = min(5.0, float(getattr(a, "combat_skill", 0.0)) + 0.02)

        try:
            br = getattr(a, "brain", None)
            if br and hasattr(br, "note_combat_feedback"):
                # NEW: положительная обратная связь в мозг за активное боевое действие.
                br.note_combat_feedback(
                    hp_delta=0.0,
                    dist_delta=-1.0,  # факт входа в мили для удара
                    done=bool(died),
                )
        except Exception:
            pass

        # Кулдаун
        cd = int(getattr(config, "AGENT_ATTACK_COOLDOWN", 18))
        self._agent_next_attack_tick[a.id] = self.tick_count + max(1, cd)

        # Если зверь умер — сразу убираем из контейнера
        if died:
            uid = getattr(ani, "uid", None)
            if uid and isinstance(self.animals, dict) and uid in self.animals:
                del self.animals[uid]

    def _process_agent_fightback(self, a: Agent) -> None:
        """
        Условие боя агента: агрессивный зверь в мили-радиусе + страх ниже порога + есть готовность (кулдаун).
        """
        if not self._agent_can_attack_now(a):
            return

        # Ограничение по страху
        eff_fear = self._effective_fear(a)
        if eff_fear > float(getattr(config, "AGENT_FEAR_FIGHT_THRESHOLD", 0.65)):
            return

        target = self._nearest_aggressive_animal(a)
        if not target:
            return
        ani, dist = target

        melee_range = float(getattr(config, "AGENT_MELEE_RANGE", 1.6))
        if dist > melee_range:
            return

        # Удар
        self._agent_melee_strike(a, ani)

    # -----------------------------------------------------------------
    # вспомогалки для агентов (соседи и т.п.)
    # -----------------------------------------------------------------

    def get_neighbors(self, me: Agent, radius: float) -> List[Agent]:
        res: List[Agent] = []
        r2 = radius * radius
        mx, my = me.x, me.y
        for other in self.agents.values():
            if other is me or not other.is_alive():
                continue
            dx = other.x - mx
            dy = other.y - my
            if dx * dx + dy * dy <= r2:
                res.append(other)
        return res

    # -----------------------------------------------------------------
    # основной тик мира
    # -----------------------------------------------------------------

    def _is_animal_alive(self, ani: AnimalSim) -> bool:
        try:
            if hasattr(ani, "is_alive"):
                return bool(ani.is_alive())
            return float(getattr(ani, "hp", 1.0)) > 0.0
        except Exception:
            return True

    def tick(self):
        brain_bridge = getattr(self, "ollama_brain_bridge", None)
        if brain_bridge is not None and hasattr(brain_bridge, "before_world_tick"):
            try:
                brain_bridge.before_world_tick(self)
            except Exception as e:
                self.add_event({
                    "type": "ollama_brain_before_tick_error",
                    "err": str(e),
                })

        # Починим контейнер животных перед тиками
        self._normalize_animals_container()

        # 1) звери
        for ani in list(self.iter_animals()):
            try:
                if hasattr(ani, "tick") and callable(getattr(ani, "tick")):
                    ani.tick(self)
                elif hasattr(ani, "step") and callable(getattr(ani, "step")):
                    # фолбэк для реализаций с step()
                    ani.step(self)
                else:
                    # мягкий no-op: если есть скорость, сдвинем в пределах мира
                    vx = float(getattr(ani, "vx", 0.0))
                    vy = float(getattr(ani, "vy", 0.0))
                    if vx or vy:
                        try:
                            ani.x = max(0.0, min(self.width,  float(getattr(ani, "x", 0.0)) + vx))
                            ani.y = max(0.0, min(self.height, float(getattr(ani, "y", 0.0)) + vy))
                        except Exception:
                            pass
            except Exception as e:
                # не даём упасть всему миру из-за одного зверя
                self.add_event({
                    "type": "animal_tick_error",
                    "uid": getattr(ani, "uid", None),
                    "err": str(e),
                })

        # чистка умерших (после тиков зверей)
        if isinstance(self.animals, dict):
            for uid, a in list(self.animals.items()):
                if not self._is_animal_alive(a):
                    del self.animals[uid]

        # 1.5) приручение волков и защита хозяев приручёнными волками
        try:
            if getattr(self, "wolf_taming", None):
                self.wolf_taming.step()
        except Exception as e:
            self.add_event({
                "type": "wolf_taming_error",
                "err": str(e),
            })

        # 2) агенты: жизнь + движение + речь + мозг
        for agent in list(self.agents.values()):
            agent.tick(self)
            # NEW: после шага — возможен ответный мили-удар
            self._process_agent_fightback(agent)

        # 2.5) размножение агентов и рождение нового поколения
        try:
            if getattr(self, "agent_reproduction", None):
                self.agent_reproduction.step()
        except Exception as e:
            self.add_event({
                "type": "agent_reproduction_error",
                "err": str(e),
            })

        # 3) глобальный тик
        self.tick_count += 1
        if brain_bridge is not None and hasattr(brain_bridge, "after_world_tick"):
            try:
                brain_bridge.after_world_tick(self)
            except Exception as e:
                self.add_event({
                    "type": "ollama_brain_after_tick_error",
                    "err": str(e),
                })

    # -----------------------------------------------------------------
    # снапшоты (старый + новый sync для 3D)
    # -----------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Старый формат: оставлен для обратной совместимости.
        """
        global_events_list: List[str] = []
        for ev in self.event_log[-20:]:
            etype = ev.get("type", "event")
            who = ev.get("name") or ev.get("who") or ev.get("victim_name")
            tick = ev.get("tick", self.tick_count)
            if who:
                global_events_list.append(f"[t={tick}] {etype}: {who}")
            else:
                global_events_list.append(f"[t={tick}] {etype}")

        animals_out: List[Dict[str, Any]] = []
        for ani in self.iter_animals():
            # безопасный билд public_state
            try:
                pub = ani.build_public_state()
            except Exception:
                sp = getattr(ani, "species", None)
                px = float(getattr(ani, "x", 0.0))
                py = float(getattr(ani, "y", 0.0))
                pub = {
                    "id": getattr(ani, "uid", "ani"),
                    "species": getattr(sp, "species_id", getattr(sp, "name", "beast")) if sp else "beast",
                    "name": getattr(sp, "name", "beast") if sp else "beast",
                    "temperament": getattr(sp, "temperament", None) if sp else None,
                    "pos": {"x": px, "y": py},
                    "hp": float(getattr(ani, "hp", 50.0)),
                    "age_ticks": int(getattr(ani, "age_ticks", 0)),
                    "owner_id": getattr(ani, "tamed_by", None),
                    "last_action": getattr(ani, "last_action", None),
                    "is_alive": (float(getattr(ani, "hp", 1.0)) > 0.0),
                }
            vx = float(getattr(ani, "vx", 0.0))
            vy = float(getattr(ani, "vy", 0.0))
            pub["vel"] = {"x": vx, "y": vy}
            pub["health"] = float(getattr(ani, "hp", pub.get("hp", 0.0)))
            pub["hp"] = float(getattr(ani, "hp", pub.get("hp", 0.0)))
            pub["age_ticks"] = int(getattr(ani, "age_ticks", pub.get("age_ticks", 0)))
            pub["tamed"] = bool(getattr(ani, "tamed_by", None))
            animals_out.append(pub)

        return {
            "tick": self.tick_count,
            "agents": [a.serialize_public_state() for a in self.agents.values()],
            "animals": animals_out,
            "world": {"width": self.width, "height": self.height},
            "objects": [o.serialize_public() for o in self.objects],
            "chat": list(self.chat_log[-20:]),
            "events": list(self.event_log[-20:]),
            "global_events": global_events_list,
        }

    def export_for_engine3d(self) -> Dict[str, Any]:
        """
        Новый формат для 3D-движка:
          - координаты и для 2D, и для 3D (pos / pos3d),
          - yaw и speed для интерполяции ориентации,
          - debug-поля: debug_last_thought / goal_dbg / age_dbg,
          - компактный HUD (chat, events_compact),
          - safe_point и карта мира.
        """
        # компактные HUD-события
        events_compact = []
        for ev in self.event_log[-16:]:
            e = {
                "t": ev.get("tick", self.tick_count),
                "type": ev.get("type", "event"),
            }
            if "name" in ev:
                e["name"] = ev["name"]
            if "who" in ev:
                e["who"] = ev["who"]
            if "reason" in ev:
                e["reason"] = ev["reason"]
            events_compact.append(e)

        # питомцы для каждого агента (для HUD)
        pets_by_owner: Dict[str, List[str]] = {}
        for ani in self.iter_animals():
            if getattr(ani, "tamed_by", None):
                pets_by_owner.setdefault(ani.tamed_by, []).append(getattr(ani, "uid", ""))

        # агенты для 3D
        agents_out: List[Dict[str, Any]] = []
        for a in self.agents.values():
            base = a.serialize_public_state() if hasattr(a, "serialize_public_state") else _agent_default_public(a)
            yaw = _compute_yaw_deg(getattr(a, "vx", 0.0), getattr(a, "vy", 0.0), a.goal_x - a.x, a.goal_y - a.y)
            spd = _speed(getattr(a, "vx", 0.0), getattr(a, "vy", 0.0))
            pos = base.get("pos", {"x": a.x, "y": a.y})
            goal = base.get("goal", {"x": a.goal_x, "y": a.goal_y})
            pos3 = _xy_to_xz(float(pos.get("x", a.x)), float(pos.get("y", a.y)))
            goal3 = _xy_to_xz(float(goal.get("x", a.goal_x)), float(goal.get("y", a.goal_y)))
            danger_cloud = [{
                "x": dx,
                "y": dy,
                "pos3d": dict(zip(("x", "y", "z"), _xy_to_xz(dx, dy))),
                "w": w,
            } for (dx, dy, w) in getattr(a, "danger_zones", [])[-24:]]

            mind_public = dict(base.get("mind", {}) or {})
            last_thought = getattr(getattr(a, "brain", None), "last_thought", None)
            agent_row = dict(base)
            agent_row.update({
                "pos3d": {"x": pos3[0], "y": pos3[1], "z": pos3[2]},
                "speed": spd,
                "yaw": yaw,
                "goal3d": {"x": goal3[0], "y": goal3[1], "z": goal3[2]},
                "pets": pets_by_owner.get(a.id, []),
                "danger_cloud": danger_cloud,
                "mind": mind_public,
                "debug_last_thought": last_thought,
                "goal_dbg": (goal.get("x", a.goal_x), goal.get("y", a.goal_y)),
                "age_dbg": base.get("age_ticks", getattr(a, "age_ticks", 0)),
            })
            agents_out.append(agent_row)

        # животные для 3D
        animals_out: List[Dict[str, Any]] = []
        for ani in self.iter_animals():
            # безопасный билд public_state
            try:
                pub = ani.build_public_state()
            except Exception:
                sp = getattr(ani, "species", None)
                px = float(getattr(ani, "x", 0.0))
                py = float(getattr(ani, "y", 0.0))
                pub = {
                    "id": getattr(ani, "uid", "ani"),
                    "species": getattr(sp, "species_id", getattr(sp, "name", "beast")) if sp else "beast",
                    "name": getattr(sp, "name", "beast") if sp else "beast",
                    "temperament": getattr(sp, "temperament", None) if sp else None,
                    "pos": {"x": px, "y": py},
                    "hp": float(getattr(ani, "hp", 50.0)),
                    "age_ticks": int(getattr(ani, "age_ticks", 0)),
                    "owner_id": getattr(ani, "tamed_by", None),
                    "last_action": getattr(ani, "last_action", None),
                    "is_alive": (float(getattr(ani, "hp", 1.0)) > 0.0),
                }
            vx = float(getattr(ani, "vx", 0.0))
            vy = float(getattr(ani, "vy", 0.0))
            yaw = _compute_yaw_deg(vx, vy, 0.0, 1.0)
            spd = _speed(vx, vy)
            pos3 = _xy_to_xz(pub["pos"]["x"], pub["pos"]["y"])
            animals_out.append({
                "id": pub["id"],
                "species": pub.get("species"),
                "name": pub.get("name"),
                "temperament": pub.get("temperament"),
                "pos": pub["pos"],
                "pos3d": {"x": pos3[0], "y": pos3[1], "z": pos3[2]},
                "vel": {"x": vx, "y": vy},
                "speed": spd,
                "yaw": yaw,
                "hp": float(pub.get("hp", getattr(ani, "hp", 0.0))),
                "health": float(pub.get("hp", getattr(ani, "hp", 0.0))),
                "age_ticks": int(pub.get("age_ticks", getattr(ani, "age_ticks", 0))),
                "tamed": bool(pub.get("owner_id", getattr(ani, "tamed_by", None))),
                "owner_id": pub.get("owner_id", getattr(ani, "tamed_by", None)),
                "last_action": pub.get("last_action"),
                "is_alive": bool(pub.get("is_alive", True)),
            })

        # объекты (hazard/safe) с pos3d уже включены в serialize_public()
        objects_out = [o.serialize_public() for o in self.objects]

        safe_x, safe_y = getattr(config, "SAFE_POINT", (self.width * 0.5, self.height * 0.5))
        sp3 = _xy_to_xz(safe_x, safe_y)

        payload = {
            "version": ENGINE_SYNC_VERSION,
            "frame": self.tick_count,  # можно использовать как frame_id
            "tick": self.tick_count,
            "world": {
                "width": self.width,
                "height": self.height,
                "scale": ENGINE3D_SCALE,
                "safe_point": {"x": safe_x, "y": safe_y, "pos3d": {"x": sp3[0], "y": sp3[1], "z": sp3[2]}},
            },
            "agents": agents_out,
            "animals": animals_out,
            "objects": objects_out,
            "chat": list(self.chat_log[-12:]),
            "events": list(self.event_log[-16:]),
            "global_events": [
                f"[t={ev.get('tick', self.tick_count)}] {ev.get('type', 'event')}: "
                f"{ev.get('name') or ev.get('who') or ev.get('victim_name') or ev.get('target') or ''}".rstrip(": ")
                for ev in self.event_log[-16:]
            ],
            "events_compact": events_compact,
        }
        return payload
