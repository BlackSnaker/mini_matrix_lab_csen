# world.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
import random

import config

# серверная модель зверя
from animals import Animal as AnimalSim
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

ENGINE_SYNC_VERSION = 3                               # инкрементируй при изменении формата
ENGINE3D_SCALE = getattr(config, "ENGINE3D_SCALE", 1.0)  # масштаб для 3D-пакета

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

from typing import cast as _cast

def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

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

    return {
        "id": getattr(self, "id", None),
        "name": getattr(self, "name", f"agent-{getattr(self, 'id', 'unknown')}"),

        "pos": {"x": _num(getattr(self, "x", 0.0)), "y": _num(getattr(self, "y", 0.0))},
        "goal": {"x": _num(getattr(self, "goal_x", getattr(self, "x", 0.0))),
                 "y": _num(getattr(self, "goal_y", getattr(self, "y", 0.0)))},

        "vel": {"x": _num(getattr(self, "vx", 0.0)), "y": _num(getattr(self, "vy", 0.0))},
        "yaw": yaw,

        # Унифицируем метрики состояния
        "health": _num(getattr(self, "health", getattr(self, "hp", 100.0))),
        "energy": _num(getattr(self, "energy", 100.0)),
        "hunger": _num(getattr(self, "hunger", 0.0)),
        "fear":   _num(getattr(self, "fear", 0.0)),
        "alive":  bool(getattr(self, "alive", True)),
        "age_ticks": int(getattr(self, "age_ticks", 0)),

        # Необязательные поля для HUD
        "tags": list(getattr(self, "tags", [])) if hasattr(self, "tags") else [],
    }

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

        # Кулдауны ударов агентов (tick номер, раньше — нельзя)
        self._agent_next_attack_tick: Dict[str, int] = {}

        # Назначаем глобальный синк (и вливаем отложенные события)
        if _GLOBAL_WORLD_EVENT_SINK is None:
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

    def announce_death(self, agent: Agent, reason: str):
        self.add_chat_line(f"[DEATH] {agent.name} погиб ({reason}) в t={self.tick_count}")
        self.add_event({"type": "death", "who": agent.id, "name": agent.name, "reason": reason})

    def set_agent_goal(self, agent_id: str, x: float, y: float) -> bool:
        a = self.agents.get(agent_id)
        if a is None:
            return False
        a.goal_x = x
        a.goal_y = y
        # лог и память остаются ответственностью агента/мозга в agent.py
        self.add_chat_line(f"[CMD] {a.name} получил приказ двигаться к ({x:.1f}, {y:.1f})")
        self.add_event({"type": "command_goal", "who": a.id, "name": a.name, "goal": (x, y)})
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
            if ani.tamed_by == owner.id and ani.is_alive():
                dx = ani.x - ox
                dy = ani.y - oy
                if dx * dx + dy * dy <= r2:
                    return True
        return False

    def _nearest_aggressive_animal(self, me: Agent) -> Optional[Tuple[AnimalSim, float]]:
        best: Optional[Tuple[AnimalSim, float]] = None
        for ani in self.iter_animals():
            if not ani.is_alive():
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
        self.add_chat_line(f"[бой] {source_name} ударил(а) {ani.species.name} на {dmg:.1f} урона!")
        self.add_event({
            "type": "agent_attack",
            "who": source_id,
            "name": source_name,
            "victim_species": getattr(ani.species, "species_id", "?"),
            "damage": round(dmg, 1),
        })
        if died:
            self.add_chat_line(f"[бой] {ani.species.name} повержен(а).")
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
        # Базовый урон
        base = float(getattr(config, "AGENT_BASE_ATTACK_POWER", 8.0))

        # Модификаторы: энергия/здоровье (0..1), страх (пенальти), союз/питомец (бонус)
        energy_ratio = max(0.0, min(1.0, float(getattr(a, "energy", 100.0)) / 100.0))
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

    def tick(self):
        # Починим контейнер животных перед тиками
        self._normalize_animals_container()

        # 1) звери
        for ani in list(self.iter_animals()):
            ani.tick(self)

        # чистка умерших (после тиков зверей)
        if isinstance(self.animals, dict):
            dead_ids = [uid for uid, a in self.animals.items() if not a.is_alive()]
            for uid in dead_ids:
                del self.animals[uid]

        # 2) агенты: жизнь + движение + речь + мозг
        for agent in list(self.agents.values()):
            agent.tick(self)
            # NEW: после шага — возможен ответный мили-удар
            self._process_agent_fightback(agent)

        # 3) глобальный тик
        self.tick_count += 1

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
            pub = ani.build_public_state()
            vx = getattr(ani, "vx", 0.0)
            vy = getattr(ani, "vy", 0.0)
            pub["vel"] = {"x": vx, "y": vy}
            pub["health"] = ani.hp
            pub["hp"] = ani.hp
            pub["age_ticks"] = ani.age_ticks
            pub["tamed"] = (ani.tamed_by is not None)
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
                pets_by_owner.setdefault(ani.tamed_by, []).append(ani.uid)

        # агенты для 3D
        agents_out: List[Dict[str, Any]] = []
        for a in self.agents.values():
            yaw = _compute_yaw_deg(getattr(a, "vx", 0.0), getattr(a, "vy", 0.0), a.goal_x - a.x, a.goal_y - a.y)
            spd = _speed(getattr(a, "vx", 0.0), getattr(a, "vy", 0.0))
            pos3 = _xy_to_xz(a.x, a.y)
            goal3 = _xy_to_xz(a.goal_x, a.goal_y)
            danger_cloud = [{
                "x": dx,
                "y": dy,
                "pos3d": dict(zip(("x", "y", "z"), _xy_to_xz(dx, dy))),
                "w": w,
            } for (dx, dy, w) in getattr(a, "danger_zones", [])[-24:]]

            mind_public = a.brain.export_public_state_for_ui() if getattr(a, "brain", None) else {}
            last_thought = getattr(getattr(a, "brain", None), "last_thought", None)

            agents_out.append({
                "id": a.id,
                "name": a.name,
                "pos": {"x": a.x, "y": a.y},
                "pos3d": {"x": pos3[0], "y": pos3[1], "z": pos3[2]},
                "vel": {"x": getattr(a, "vx", 0.0), "y": getattr(a, "vy", 0.0)},
                "speed": spd,
                "yaw": yaw,                     # в градусах (Y-up)
                "goal": {"x": a.goal_x, "y": a.goal_y},
                "goal3d": {"x": goal3[0], "y": goal3[1], "z": goal3[2]},
                "fear": getattr(a, "fear", 0.0),
                "health": getattr(a, "health", 100.0),
                "energy": getattr(a, "energy", 100.0),
                "hunger": getattr(a, "hunger", 0.0),
                "alive": a.is_alive(),
                "age_ticks": getattr(a, "age_ticks", 0),
                "pets": pets_by_owner.get(a.id, []),
                "danger_cloud": danger_cloud,   # для теплокарты боли
                "mind": mind_public,            # для HUD-панелей
                # DEBUG для HUD SidePanel:
                "debug_last_thought": last_thought,
                "goal_dbg": (a.goal_x, a.goal_y),
                "age_dbg": getattr(a, "age_ticks", 0),
            })

        # животные для 3D
        animals_out: List[Dict[str, Any]] = []
        for ani in self.iter_animals():
            pub = ani.build_public_state()
            vx = getattr(ani, "vx", 0.0)
            vy = getattr(ani, "vy", 0.0)
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
                "hp": pub.get("hp"),
                "health": pub.get("hp"),
                "age_ticks": pub.get("age_ticks", getattr(ani, "age_ticks", 0)),
                "tamed": bool(pub.get("owner_id")),
                "owner_id": pub.get("owner_id"),
                "last_action": pub.get("last_action"),
                "is_alive": pub.get("is_alive", True),
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
            "events_compact": events_compact,
        }
        return payload
