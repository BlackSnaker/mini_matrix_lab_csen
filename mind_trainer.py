# mind_trainer.py
#
# Оффлайн-тренер "мозга" агентов — улучшенная версия.
#
# Новое / улучшения:
#  1) Детализированный curriculum: чередование спокойных фаз, катастроф и
#     «лагерей-передышек» с постепенным наращиванием сложности.
#  2) Анти-AFK «пинок»: если агент почти не двигался N тиков, ему подкидывается
#     мягкая внешняя цель (world.set_agent_goal) в безопасной области.
#  3) «Аннеалинг» любопытства: exploration_bias и curiosity_charge плавно
#     подгоняются во время эпохи, чтобы агенты и не закисали в лагере,
#     и не лезли прямо в ад.
#  4) Мониторинг расширен: история метрик (monitor_history), учёт приручений
#     (tamed_ratio), среднее число известных угроз (avg_known_hazards).
#  5) Снапшоты мира (export_for_engine3d) — по шагам (snapshot_every) в JSON.
#  6) Безопасности: катастрофы стараются не появляться прямо в SAFE_POINT или
#     внутри «safe» зон, Relief-лагеря — не в ядре опасностей.
#  7) Грейсфул сейв: train() делает сейв мозгов даже при исключениях.
#  8) Кэп на животных (max_animals_per_world), чтобы мир не «перенаселялся».
#  9) Логи эпох в JSONL (monitor.jsonl) + финальный CSV с историей.
#
# Запуск:
#   python mind_trainer.py
#
# Артефакты:
#   ./trained_brains/<id>.mind.json   — снимки сознаний
#   ./brains/<id>.json                — runtime-формат (v3) через brain_io
#   ./trainer_logs/epoch_#/monitor.jsonl — поток метрик
#   ./trainer_snapshots/epoch_#/t_XXXX.json — снапшоты 3D-синх-пакетов
#
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import random
import json
import os
import csv

import config
from world import World, Agent, WorldObject
from mind_core import ConsciousnessBlock
from brain_io import save_brain  # пишет brains/<id>.json (v3)

# новая система зверей
from animals import Animal as AnimalSim, AnimalSpecies


# =============================================================================
# Вспомогательные numeric-хелперы (безопасные float'ы, клампы)
# =============================================================================

def _safe_float(v: Any, fallback: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return fallback
    if x != x:  # NaN
        return fallback
    if x in (float("inf"), float("-inf")):
        return fallback
    return x


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx*dx + dy*dy


# =============================================================================
# Универсальные итераторы по агентам/животным (dict | list)
# =============================================================================

def _iter_agents_of(world: World) -> List[Agent]:
    agents_attr = getattr(world, "agents", [])
    if isinstance(agents_attr, dict):
        return list(agents_attr.values())
    return list(agents_attr)

def _iter_animals_of(world: World) -> List[AnimalSim]:
    animals_attr = getattr(world, "animals", [])
    if isinstance(animals_attr, dict):
        return list(animals_attr.values())
    return list(animals_attr)

def _agent_id(ag: Agent) -> str:
    return getattr(ag, "id", None) or getattr(ag, "agent_id", "")


# =============================================================================
# Мониторинг прогресса обучения (диагностика / телеметрия для UI тренера)
# =============================================================================

@dataclass
class TrainingMonitorState:
    epoch: int = 0             # индекс эпохи (0..epochs-1)
    tick: int = 0              # текущий тик внутри эпохи

    avg_age: float = 0.0       # средний age_ticks агентов
    avg_fear: float = 0.0      # средний страх агентов (0..1)
    avg_hp: float = 0.0        # среднее здоровье
    avg_energy: float = 0.0    # средняя энергия
    avg_hunger: float = 0.0    # средний голод
    avg_curiosity: float = 0.0 # среднее любопытство (curiosity_charge)
    avg_survival: float = 0.0  # средний survival_score мозгов

    avg_trauma_spots: float = 0.0  # среднее количество травм в trauma_map
    avg_known_hazards: float = 0.0 # среднее число «известных угроз» на агента

    alive_ratio: float = 0.0   # доля агентов, у которых alive == True
    panic_ratio: float = 0.0   # доля агентов с fear_level > 0.7
    cling_ratio: float = 0.0   # доля агентов с drive == "stay_with_ally"
    tamed_ratio: float = 0.0   # доля агентов, у которых есть питомцы

    note: str = ""             # краткий комментарий про текущее состояние

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Утилиты сериализации мозга и массового сейва
# =============================================================================

def serialize_brain_for_save(brain: ConsciousnessBlock) -> Dict[str, Any]:
    return brain.to_dict()


def save_all_brains(world: World, out_dir: str):
    """
    Сохранить мозг каждого агента в два места:
      1) ./trained_brains/<id>.mind.json — снимок эпохи (для анализа)
      2) ./brains/<id>.json              — runtime-формат v3 (brain_io)
    """
    os.makedirs(out_dir, exist_ok=True)

    for ag in _iter_agents_of(world):
        brain = getattr(ag, "brain", None)
        if brain is None:
            continue

        agid = _agent_id(ag) or "unknown"

        # (1) оффлайн-снимок (для анализа)
        dump = serialize_brain_for_save(brain)
        path = os.path.join(out_dir, f"{agid}.mind.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[mind_trainer] WARN: can't write {path}: {e}")

        # (2) runtime-кэш эволюции линии (brains/<id>.json)
        try:
            save_brain(brain)
        except Exception as e:
            print(f"[mind_trainer] WARN: save_brain() failed for {agid}: {e}")


# =============================================================================
# Балансные костыли обучения
# =============================================================================

def _boost_exploration_bias(ag: Agent):
    """
    Заставляем в тренере реально гулять по карте (не AFK в лагере).
    """
    try:
        br = getattr(ag, "brain", None)
        if br is None:
            return

        rules = getattr(br, "behavior_rules", None)
        if rules is not None:
            if hasattr(rules, "exploration_bias"):
                current = _safe_float(getattr(rules, "exploration_bias", 0.0), 0.0)
                rules.exploration_bias = _clamp(max(current, 0.9), 0.0, 1.0)
            if hasattr(rules, "healing_zone_seek_priority"):
                cur_hsp = _safe_float(getattr(rules, "healing_zone_seek_priority", 0.5), 0.5)
                rules.healing_zone_seek_priority = _clamp(cur_hsp, 0.0, 0.6)

        if hasattr(br, "curiosity_charge"):
            cur_c = _safe_float(getattr(br, "curiosity_charge", 0.0), 0.0)
            br.curiosity_charge = _clamp(max(cur_c, 0.8), 0.0, 1.0)
    except Exception as e:
        print(f"[mind_trainer] WARN: can't boost exploration for {getattr(ag, 'id', '?')}: {e}")


def _anneal_exploration(world: World, t: int, T: int):
    """
    Плавно подруливаем любопытство в течение эпохи:
      - в начале чуть притормаживаем (0.75), к середине -> 0.95, к концу -> 0.85
    """
    if T <= 0:
        return
    phase = t / float(T)
    target = 0.75 + 0.35 * (1.0 - abs(2.0 * phase - 1.0))  # ∩-образная кривая (0..1)
    target = _clamp(target, 0.65, 0.98)

    for ag in _iter_agents_of(world):
        br = getattr(ag, "brain", None)
        if not br or not hasattr(br, "behavior_rules"):
            continue
        rules = br.behavior_rules
        try:
            if hasattr(rules, "exploration_bias"):
                rules.exploration_bias = 0.8 * rules.exploration_bias + 0.2 * target
            if hasattr(br, "curiosity_charge"):
                br.curiosity_charge = _clamp(0.9 * br.curiosity_charge + 0.1 * target, 0.0, 1.0)
        except Exception:
            pass


def _spawn_training_animals(world: World, seed: int):
    """
    Начальные «биомы» животных (дружелюбные / агрессивные).
    """
    rnd = random.Random(seed ^ 0xA11FA11F)

    animal_blueprints = [
        dict(
            uid="ani_friendly_0",
            species=AnimalSpecies(
                species_id="fox",
                name="Лиска",
                base_hp=45.0,
                aggressive=False,
                tamable=True,
                tame_difficulty=0.4,
                bite_damage=4.0,
                fear_radius=4.0,
                follow_distance=2.0,
                aggro_radius=10.0,
            ),
        ),
        dict(
            uid="ani_friendly_1",
            species=AnimalSpecies(
                species_id="dog",
                name="Пепелок",
                base_hp=50.0,
                aggressive=False,
                tamable=True,
                tame_difficulty=0.2,
                bite_damage=6.0,
                fear_radius=4.0,
                follow_distance=2.0,
                aggro_radius=10.0,
            ),
        ),
        dict(
            uid="ani_hostile_0",
            species=AnimalSpecies(
                species_id="beast",
                name="Грыз",
                base_hp=80.0,
                aggressive=True,
                tamable=False,
                tame_difficulty=1.0,
                bite_damage=12.0,
                fear_radius=8.0,
                follow_distance=2.0,
                aggro_radius=12.0,
            ),
        ),
        dict(
            uid="ani_hostile_1",
            species=AnimalSpecies(
                species_id="wolf",
                name="Клык",
                base_hp=70.0,
                aggressive=True,
                tamable=False,
                tame_difficulty=1.0,
                bite_damage=10.0,
                fear_radius=7.0,
                follow_distance=2.0,
                aggro_radius=12.0,
            ),
        ),
    ]

    for bp in animal_blueprints:
        ax = rnd.uniform(0.0, world.width)
        ay = rnd.uniform(0.0, world.height)
        ani = AnimalSim(uid=bp["uid"], species=bp["species"], x=ax, y=ay)
        world.add_animal(ani)


# =============================================================================
# Вспомогалка: мягко добавить/слить POI в activity_registry мира
# =============================================================================

def _merge_activity_spot(
    world: World,
    obj: WorldObject,
    activity_tags: List[str],
):
    base_registry = getattr(world, "activities", None) or {}
    new_registry = dict(base_registry)

    new_registry[obj.obj_id] = {
        "name": obj.name,
        "activity_tags": list(activity_tags),
        "comfort_level": getattr(obj, "comfort_level", 0.0),
        "danger_level": getattr(obj, "danger_level", 0.0),
        "area": {
            "x": getattr(obj, "x", 0.0),
            "y": getattr(obj, "y", 0.0),
            "radius": getattr(obj, "radius", 0.0),
        },
    }

    try:
        world.set_activity_registry(new_registry)
    except Exception as e:
        print("[mind_trainer] WARN: world.set_activity_registry failed:", e)


# =============================================================================
# Генерация тренировочной арены (карта эпохи)
# =============================================================================

def _make_training_world(
    seed: int,
    num_agents: int,
    fresh_start: bool,
    agent_lineup: Optional[List[Dict[str, str]]] = None,
) -> World:
    random.seed(seed)

    w = getattr(config, "WORLD_WIDTH", 100.0)
    h = getattr(config, "WORLD_HEIGHT", 100.0)

    world = World(width=w, height=h)

    # --- Опасные зоны
    for i in range(2):
        hx = random.uniform(10.0, w - 10.0)
        hy = random.uniform(10.0, h - 10.0)
        radius = random.uniform(4.0, 8.0)

        hazard = WorldObject(
            obj_id=f"hazard_{i}",
            name=f"Огонь_{i}",
            kind="hazard",
            x=hx,
            y=hy,
            radius=radius,
            danger_level=0.7,
            comfort_level=0.0,
        )
        world.add_object(hazard)

    # --- Безопасные зоны
    safe_spots: List[WorldObject] = []
    for i in range(2):
        sx = random.uniform(10.0, w - 10.0)
        sy = random.uniform(10.0, h - 10.0)
        radius = random.uniform(5.0, 9.0)

        safe = WorldObject(
            obj_id=f"safe_{i}",
            name=f"Убежище_{i}",
            kind="safe",
            x=sx,
            y=sy,
            radius=radius,
            danger_level=0.0,
            comfort_level=0.8,
        )
        world.add_object(safe)
        safe_spots.append(safe)

        _merge_activity_spot(
            world,
            safe,
            activity_tags=["heal", "rest", "eat", "calm", "sleep", "repair_self", "restock_food"],
        )

    # --- SAFE_POINT
    if safe_spots:
        rally = safe_spots[0]
        try:
            rx = _clamp(float(rally.x), 0.0, world.width)
            ry = _clamp(float(rally.y), 0.0, world.height)
            config.SAFE_POINT = (rx, ry)
        except Exception:
            config.SAFE_POINT = (world.width * 0.5, world.height * 0.5)
    else:
        config.SAFE_POINT = (world.width * 0.5, world.height * 0.5)

    # --- Состав агентов
    if agent_lineup and len(agent_lineup) > 0:
        spawn_specs = agent_lineup
    else:
        spawn_specs = []
        for i in range(num_agents):
            spawn_specs.append({
                "id": f"agent_{i}",
                "name": f"A{i}",
                "persona": random.choice(["caring/supportive", "protective", "loner", "scout/explorer"]),
            })

    # --- Создание агентов
    for spec in spawn_specs:
        ax = random.uniform(0.0, w)
        ay = random.uniform(0.0, h)
        gx = random.uniform(0.0, w)
        gy = random.uniform(0.0, h)

        persona_seed = spec.get("persona") or random.choice(
            ["caring/supportive", "protective", "loner", "scout/explorer"]
        )

        ag = Agent(
            agent_id=spec["id"],
            name=spec["name"],
            x=ax,
            y=ay,
            goal_x=gx,
            goal_y=gy,
            persona=persona_seed,
        )

        if fresh_start:
            ag.brain = ConsciousnessBlock(agent_id=_agent_id(ag) or spec["id"])

        _boost_exploration_bias(ag)
        world.add_agent(ag)

    # --- Животные
    _spawn_training_animals(world, seed=seed)

    return world


# =============================================================================
# MindTrainer — главный сценарист/учитель сознаний
# =============================================================================

@dataclass
class MindTrainer:
    """
    Улучшенный тренер:
      - curriculum из чередующихся стрессов/передышек,
      - контроль сложности,
      - снапшоты мира,
      - более богатые метрики.
    """

    num_agents: int = 3
    max_ticks_per_episode: int = 2000
    epochs: int = 3
    seed: int = 42

    disaster_interval_ticks: int = 400    # каждые N тиков → новая опасная зона
    relief_after_disaster: int = 80       # через N тиков после катастрофы → лагерь

    fresh_start: bool = False
    agent_lineup: Optional[List[Dict[str, str]]] = None

    # дополнительные настройки
    max_animals_per_world: int = 32
    snapshot_every: int = 0                 # 0 → выключено
    logs_dir: str = "./trainer_logs"
    snapshots_dir: str = "./trainer_snapshots"

    # runtime state
    monitor: TrainingMonitorState = field(default_factory=TrainingMonitorState)
    _world: Optional[World] = None
    _last_disaster_tick: Optional[int] = None
    _rng: random.Random = field(default_factory=random.Random)

    # история метрик по всем эпохам
    monitor_history: List[Dict[str, Any]] = field(default_factory=list)

    # логирование
    verbose: bool = True

    # -------------------------------------------------
    # Служебное логирование
    # -------------------------------------------------

    def _log(self, msg: str):
        if not self.verbose:
            return
        ep = getattr(self.monitor, "epoch", "?")
        tk = getattr(self.monitor, "tick", "?")
        print(f"[trainer e{ep} t{tk}] {msg}")

    # -------------------------------------------------
    # Автосейв поколения
    # -------------------------------------------------

    def _auto_save_generation(self):
        if self._world is None:
            return
        save_all_brains(self._world, "./trained_brains")
        self._log("brains saved for this epoch")

    # -------------------------------------------------
    # Создание/перезапуск мира на новую эпоху
    # -------------------------------------------------

    def _spawn_world(self, epoch_idx: int):
        world_seed = self.seed + epoch_idx * 997
        self._rng.seed(world_seed)

        self._world = _make_training_world(
            seed=world_seed,
            num_agents=self.num_agents,
            fresh_start=self.fresh_start,
            agent_lineup=self.agent_lineup,
        )

        self.monitor.epoch = epoch_idx
        self.monitor.tick = 0
        self.monitor.note = f"spawned world with seed={world_seed}"

        self._last_disaster_tick = None
        self._log(f"world spawned (seed={world_seed})")

    # -------------------------------------------------
    # Геометрические проверки (куда ставить катастрофы/лагеря)
    # -------------------------------------------------

    def _is_near_safe_point(self, x: float, y: float, min_dist: float = 8.0) -> bool:
        sx, sy = getattr(config, "SAFE_POINT", (self._world.width * 0.5, self._world.height * 0.5))
        return (_dist2(x, y, sx, sy) <= (min_dist * min_dist))

    def _is_inside_any_safe_zone(self, x: float, y: float) -> bool:
        if not self._world:
            return False
        for obj in self._world.objects:
            if obj.kind == "safe":
                if _dist2(x, y, obj.x, obj.y) <= (obj.radius * obj.radius):
                    return True
        return False

    def _too_many_animals(self) -> bool:
        if not self._world:
            return False
        return len(_iter_animals_of(self._world)) >= self.max_animals_per_world

    # -------------------------------------------------
    # Катастрофа: вброс новой угрозы посреди эпохи
    # -------------------------------------------------

    def _maybe_inject_disaster(self, t: int):
        if (
            not self._world
            or not self.disaster_interval_ticks
            or self.disaster_interval_ticks <= 0
        ):
            return

        if t <= 0 or (t % self.disaster_interval_ticks != 0):
            return

        w = self._world.width
        h = self._world.height

        # Ищем позицию, не попадающую в SAFE_POINT и внутрь safe-зон
        for _ in range(20):
            hx = self._rng.uniform(5.0, w - 5.0)
            hy = self._rng.uniform(5.0, h - 5.0)
            if self._is_near_safe_point(hx, hy, min_dist=8.0):
                continue
            if self._is_inside_any_safe_zone(hx, hy):
                continue
            break
        else:
            # fallback — в центре, но аккуратно
            hx = w * 0.66
            hy = h * 0.66

        radius = self._rng.uniform(4.0, 8.0)
        danger_lvl = 0.9 + 0.1 * (t / max(1, self.max_ticks_per_episode))  # лёгкий рост адовости

        hazard = WorldObject(
            obj_id=f"disaster_{t}_{len(self._world.objects)}",
            name=f"Токсичная_утечка_{t}",
            kind="hazard",
            x=hx,
            y=hy,
            radius=radius,
            danger_level=_clamp(danger_lvl, 0.5, 1.2),
            comfort_level=0.0,
        )
        self._world.add_object(hazard)

        # агрессивный зверь возле катастрофы — фактор паники
        if not self._too_many_animals():
            ax = _clamp(hx + self._rng.uniform(-3.0, 3.0), 0.0, w)
            ay = _clamp(hy + self._rng.uniform(-3.0, 3.0), 0.0, h)

            hostile_species = AnimalSpecies(
                species_id=f"beast_evt_{t}",
                name=f"Хищник_{t}",
                base_hp=90.0,
                aggressive=True,
                tamable=False,
                tame_difficulty=1.0,
                bite_damage=15.0,
                fear_radius=9.0,
                follow_distance=2.0,
                aggro_radius=12.0,
            )
            hostile_animal = AnimalSim(uid=f"ani_hostile_evt_{t}", species=hostile_species, x=ax, y=ay)
            self._world.add_animal(hostile_animal)

        if hasattr(self._world, "add_chat_line"):
            try:
                self._world.add_chat_line("[trainer] внезапная опасная зона! Все в лагерь!")
            except Exception:
                pass

        self.monitor.note = "disaster injected"
        self._last_disaster_tick = t
        self._log(f"disaster injected at ({hx:.1f},{hy:.1f}) r~{radius:.1f}")

    # -------------------------------------------------
    # Передышка: лагерь-после-бури
    # -------------------------------------------------

    def _maybe_inject_relief(self, t: int):
        if (
            not self._world
            or self._last_disaster_tick is None
            or self.relief_after_disaster is None
            or self.relief_after_disaster <= 0
        ):
            return

        if (t - self._last_disaster_tick) != self.relief_after_disaster:
            return

        w = self._world.width
        h = self._world.height

        # Ищем позицию лагеря вне ядра опасностей
        for _ in range(20):
            sx = self._rng.uniform(5.0, w - 5.0)
            sy = self._rng.uniform(5.0, h - 5.0)
            if self._is_inside_any_safe_zone(sx, sy):
                continue
            break
        else:
            sx, sy = w * 0.33, h * 0.33

        radius = self._rng.uniform(5.0, 9.0)

        refuge = WorldObject(
            obj_id=f"refuge_{t}_{len(self._world.objects)}",
            name=f"Лагерь_после_бури_{t}",
            kind="safe",
            x=sx,
            y=sy,
            radius=radius,
            danger_level=0.0,
            comfort_level=1.0,
        )
        self._world.add_object(refuge)

        _merge_activity_spot(
            self._world,
            refuge,
            activity_tags=["heal", "rest", "eat", "calm", "sleep", "repair_self", "restock_food"],
        )

        # дружелюбный зверёк у лагеря
        if not self._too_many_animals():
            ax = _clamp(sx + self._rng.uniform(-2.0, 2.0), 0.0, w)
            ay = _clamp(sy + self._rng.uniform(-2.0, 2.0), 0.0, h)

            friendly_species = AnimalSpecies(
                species_id=f"dog_evt_{t}",
                name=f"Дружок_{t}",
                base_hp=50.0,
                aggressive=False,
                tamable=True,
                tame_difficulty=0.2,
                bite_damage=6.0,
                fear_radius=4.0,
                follow_distance=2.0,
                aggro_radius=10.0,
            )
            friendly_animal = AnimalSim(uid=f"ani_friendly_evt_{t}", species=friendly_species, x=ax, y=ay)
            self._world.add_animal(friendly_animal)

        if hasattr(self._world, "add_chat_line"):
            try:
                self._world.add_chat_line("[trainer] безопасный лагерь создан. Отдыхайте вместе, можно приручить зверя.")
            except Exception:
                pass

        self.monitor.note = "relief injected"
        self._log(f"relief camp spawned at ({sx:.1f},{sy:.1f}) r~{radius:.1f}")

    # -------------------------------------------------
    # Анти-AFK «пинок» и лёгкий автогилдинг
    # -------------------------------------------------

    def _nudge_idle_agents(self, tick_window: int = 120, min_shift: float = 6.0):
        """
        Если агент «топчется» — даём мягкую внешнюю цель рядом с текущим местом,
        но стараемся выбирать точку не внутри опасных зон.
        """
        if not self._world:
            return
        w, h = self._world.width, self._world.height

        for ag in _iter_agents_of(self._world):
            # Берём из памяти последнюю «move» запись, если есть
            events = list(getattr(getattr(ag, "memory", None), "events", []))
            tail = events[-min(tick_window, len(events)):]
            recent_moves = [e for e in tail if e.get("kind") == "move"]
            if len(recent_moves) < 2:
                continue
            p0 = recent_moves[0].get("data", {}).get("to")
            p1 = recent_moves[-1].get("data", {}).get("to")
            if not p0 or not p1:
                continue
            dist2 = _dist2(p0[0], p0[1], p1[0], p1[1])
            if dist2 < (min_shift * min_shift):
                # Подкидываем цель поближе, но со сдвигом
                for _ in range(8):
                    gx = _clamp(ag.x + self._rng.uniform(-10.0, 10.0), 0.0, w)
                    gy = _clamp(ag.y + self._rng.uniform(-10.0, 10.0), 0.0, h)
                    # Простая эвристика: не вбегай прямо в «ядро» известных угроз
                    too_dangerous = False
                    for hz in getattr(ag, "known_hazards", {}) .values():
                        hx, hy, hr = hz["x"], hz["y"], hz["radius"]
                        if _dist2(gx, gy, hx, hy) <= (hr + 1.5) * (hr + 1.5):
                            too_dangerous = True
                            break
                    if not too_dangerous:
                        self._world.set_agent_goal(_agent_id(ag) or ag.name, gx, gy)
                        break

    # -------------------------------------------------
    # Подсчёт метрик мониторинга
    # -------------------------------------------------

    def _collect_monitor_stats(self):
        if self._world is None:
            return

        ages: List[float] = []
        fears: List[float] = []
        hps: List[float] = []
        enes: List[float] = []
        hung: List[float] = []
        survs: List[float] = []
        curios: List[float] = []
        trauma_counts: List[float] = []
        cling_flags: List[float] = []
        panic_flags: List[float] = []
        alive_flags: List[float] = []
        hazards_known: List[float] = []

        # «питомцы» у кого есть
        pets_owner_ids = set()
        for ani in _iter_animals_of(self._world):
            if getattr(ani, "tamed_by", None):
                pets_owner_ids.add(ani.tamed_by)

        agent_ids = [_agent_id(a) or a.name for a in _iter_agents_of(self._world)]

        for ag in _iter_agents_of(self._world):
            ages.append(_safe_float(getattr(ag, "age_ticks", 0), 0.0))
            fears.append(_safe_float(getattr(ag, "fear", 0.0), 0.0))
            hps.append(_safe_float(getattr(ag, "health", 0.0), 0.0))
            enes.append(_safe_float(getattr(ag, "energy", 0.0), 0.0))
            hung.append(_safe_float(getattr(ag, "hunger", 0.0), 0.0))

            alive_fn = getattr(ag, "is_alive", None)
            alive = True if alive_fn is None else bool(alive_fn())
            alive_flags.append(1.0 if alive else 0.0)

            hazards_known.append(float(len(getattr(ag, "known_hazards", {}))))

            br = getattr(ag, "brain", None)
            if br is not None:
                survs.append(_safe_float(getattr(br, "survival_score", 0.0), 0.0))
                curios.append(_safe_float(getattr(br, "curiosity_charge", 0.0), 0.0))

                trauma_map = getattr(br, "trauma_map", []) or []
                trauma_counts.append(float(len(trauma_map)))

                drive_now = getattr(br, "current_drive", "")
                cling_flags.append(1.0 if drive_now == "stay_with_ally" else 0.0)

                panic_flags.append(
                    1.0 if _safe_float(getattr(br, "fear_level", 0.0), 0.0) > 0.7 else 0.0
                )
            else:
                survs.append(0.0)
                curios.append(0.0)
                trauma_counts.append(0.0)
                cling_flags.append(0.0)
                panic_flags.append(0.0)

        def _avg(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        self.monitor.avg_age = _avg(ages)
        self.monitor.avg_fear = _avg(fears)
        self.monitor.avg_hp = _avg(hps)
        self.monitor.avg_energy = _avg(enes)
        self.monitor.avg_hunger = _avg(hung)
        self.monitor.avg_curiosity = _avg(curios)
        self.monitor.avg_survival = _avg(survs)
        self.monitor.avg_trauma_spots = _avg(trauma_counts)
        self.monitor.avg_known_hazards = _avg(hazards_known)
        self.monitor.cling_ratio = _avg(cling_flags)
        self.monitor.panic_ratio = _avg(panic_flags)
        self.monitor.alive_ratio = _avg(alive_flags)
        self.monitor.tamed_ratio = (len(pets_owner_ids) / max(1, len(agent_ids)))

        # сохраняем историю (для графиков/аналитики)
        rec = self.monitor.to_dict()
        self.monitor_history.append(dict(rec))
        self._append_monitor_jsonl(rec)

    # -------------------------------------------------
    # Логи мониторинга и снапшоты
    # -------------------------------------------------

    def _epoch_logs_dir(self, epoch: int) -> str:
        d = os.path.join(self.logs_dir, f"epoch_{epoch}")
        os.makedirs(d, exist_ok=True)
        return d

    def _append_monitor_jsonl(self, row: Dict[str, Any]):
        try:
            d = self._epoch_logs_dir(self.monitor.epoch)
            path = os.path.join(d, "monitor.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            print("[mind_trainer] WARN: monitor jsonl append failed:", e)

    def _maybe_snapshot_world(self, t: int):
        if not self.snapshot_every or self.snapshot_every <= 0 or not self._world:
            return
        if t % self.snapshot_every != 0:
            return
        try:
            d = os.path.join(self.snapshots_dir, f"epoch_{self.monitor.epoch}")
            os.makedirs(d, exist_ok=True)
            payload = self._world.export_for_engine3d()
            path = os.path.join(d, f"t_{t:04d}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            print("[mind_trainer] WARN: snapshot failed:", e)

    # -------------------------------------------------
    # Одна эпоха (эпизод) тренировки
    # -------------------------------------------------

    def _episode_tick_loop(self):
        assert self._world is not None
        dead_marked = False

        for t in range(self.max_ticks_per_episode):
            # плавная подстройка поведения
            _anneal_exploration(self._world, t, self.max_ticks_per_episode)

            # катастрофа по расписанию
            self._maybe_inject_disaster(t)

            # после катастрофы через relief_after_disaster тиков → лагерь
            self._maybe_inject_relief(t)

            # один шаг симуляции целиком
            self._world.tick()

            # лёгкий анти-AFK раз в 60 тиков
            if t % 60 == 0:
                self._nudge_idle_agents()

            # монитор
            self.monitor.tick = t
            self._collect_monitor_stats()

            # снапшот по расписанию
            self._maybe_snapshot_world(t)

            # ранний выход: если 0 живых и ещё не помечали
            if not dead_marked and self.monitor.alive_ratio <= 0.0:
                self.monitor.note = "early stop: all agents dead"
                self._log(self.monitor.note)
                dead_marked = True
                break

        # эпоха кончилась — краткая сводка
        dead_count = sum(
            1 for a in _iter_agents_of(self._world) if not getattr(a, "is_alive", lambda: True)()
        )
        if dead_count > 0 and not dead_marked:
            self.monitor.note = f"episode end: {dead_count} dead"
        elif not dead_marked:
            self.monitor.note = "episode end: all survived"

        self._log(self.monitor.note)

    # -------------------------------------------------
    # Публичный API тренера
    # -------------------------------------------------

    def _export_monitor_csv(self):
        """
        Выгрузить всю monitor_history в CSV для быстрой визуализации.
        """
        try:
            os.makedirs(self.logs_dir, exist_ok=True)
            path = os.path.join(self.logs_dir, "monitor_history.csv")
            if not self.monitor_history:
                return
            keys = sorted(set().union(*[row.keys() for row in self.monitor_history]))
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in self.monitor_history:
                    w.writerow(row)
        except Exception as e:
            print("[mind_trainer] WARN: export monitor csv failed:", e)

    def train(self) -> World:
        """
        Главная точка входа.
        """
        last_world = None

        try:
            for ep in range(self.epochs):
                # новый мир с новым сидом
                self._spawn_world(ep)

                # крутим симуляцию эпохи
                self._episode_tick_loop()

                # автосейв поколения
                self._auto_save_generation()

                # для внешнего доступа
                last_world = self._world

            self._log("training complete")
            self._log(f"final monitor: {self.monitor.to_dict()}")
            return last_world
        finally:
            # гарантированный сейв логов/метрик
            self._export_monitor_csv()
            if self._world is not None:
                try:
                    save_all_brains(self._world, "./trained_brains")
                except Exception:
                    pass

    def export_trained_brains(self, out_dir: str):
        if self._world is None:
            return
        save_all_brains(self._world, out_dir)
        self._log(f"brains exported to '{out_dir}' + ./brains/*.json")


# =============================================================================
# CLI / standalone запуск
# =============================================================================

if __name__ == "__main__":
    """
    Локальный оффлайн-запуск (без визуалки).
    """
    trainer = MindTrainer(
        num_agents=3,
        max_ticks_per_episode=1200,  # немного меньше по умолчанию — быстрее итерации
        epochs=3,
        seed=1234,
        disaster_interval_ticks=300,  # чаще стрессы
        relief_after_disaster=80,
        fresh_start=False,  # НЕ обнулять мозги между эпохами
        agent_lineup=[
            {
                "id": "a1",
                "name": "Echo",
                "persona": (
                    "Ты Echo. Ты осторожный, тревожный выживальщик. "
                    "Тебе страшно умереть, ты переживаешь за Nova. "
                    "Ты стараешься держать всех в безопасности и "
                    "предупреждать об угрозах."
                ),
            },
            {
                "id": "a2",
                "name": "Nova",
                "persona": (
                    "Ты Nova. Смелая разведчица. "
                    "Ты любишь исследовать, но не хочешь, чтобы Echo пострадал. "
                    "Ты звучишь уверенно и тепло, даже когда больно."
                ),
            },
            {
                "id": "agent_0",
                "name": "A0",
                "persona": "caring/supportive",
            },
        ],
        max_animals_per_world=32,
        snapshot_every=200,                 # снапшот мира каждые 200 тиков
        logs_dir="./trainer_logs",
        snapshots_dir="./trainer_snapshots",
        verbose=True,
    )

    final_world = trainer.train()

    # явный экспорт (дополнительно)
    out_dir = "./trained_brains"
    trainer.export_trained_brains(out_dir)

    print("[trainer] done.")
    print(f"[trainer] brains saved to {out_dir} and ./brains/*.json")
    print("[trainer] final monitor:", trainer.monitor.to_dict())
