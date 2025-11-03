# mind_trainer_gui.py
#
# Графическая лаборатория обучения сознаний агентов.
# + 3D-визуализация сети убеждений (beliefs) — Qt3D при наличии,
#   иначе fallback на QOpenGLWidget + PyOpenGL (точно покажет 3D).
#
# Запуск:
#   python mind_trainer_gui.py
#
# Требует:
#   PySide6; (опц.) PyOpenGL для GL-fallback
#   world.py, mind_core.py, brain_io.py, config.py, animals.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import random
import os
import json
import math

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot, Signal

# --- аккуратно проверяем Qt3D и именно Qt3DWindow
try:
    from PySide6 import Qt3DCore, Qt3DRender, Qt3DExtras, Qt3DInput
    QT3D_AVAILABLE = True
    QT3D_HAS_WINDOW = hasattr(Qt3DExtras, "Qt3DWindow")
except Exception:
    QT3D_AVAILABLE = False
    QT3D_HAS_WINDOW = False

# --- GL fallback (QOpenGLWidget + PyOpenGL)
GL_AVAILABLE = False
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import (
        glClearColor, glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        glEnable, GL_DEPTH_TEST, GL_POINT_SMOOTH, glViewport,
        glMatrixMode, GL_PROJECTION, GL_MODELVIEW, glLoadIdentity,
        glPointSize, glBegin, glEnd, glVertex3f, glLineWidth, GL_LINES, GL_POINTS
    )
    from OpenGL.GLU import gluPerspective, gluLookAt
    GL_AVAILABLE = True
except Exception:
    GL_AVAILABLE = False

import config
from world import World, Agent, WorldObject
from mind_core import ConsciousnessBlock
from brain_io import save_brain
from animals import Animal as AnimalSim, AnimalSpecies


# =============================================================================
# Вспомогательные хелперы
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


# =============================================================================
# Мониторинг популяции
# =============================================================================

@dataclass
class TrainingMonitorState:
    epoch: int = 0
    tick: int = 0

    avg_age: float = 0.0
    avg_fear: float = 0.0
    avg_hp: float = 0.0
    avg_energy: float = 0.0
    avg_hunger: float = 0.0
    avg_curiosity: float = 0.0
    avg_survival: float = 0.0
    avg_trauma_spots: float = 0.0

    alive_ratio: float = 0.0
    panic_ratio: float = 0.0
    cling_ratio: float = 0.0

    avg_pets: float = 0.0
    tame_ratio: float = 0.0

    note: str = ""


# =============================================================================
# Сохранение мозгов
# =============================================================================

def serialize_brain_for_save(brain: ConsciousnessBlock) -> Dict[str, Any]:
    return brain.to_dict()


def save_all_brains(world: World, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for ag in getattr(world, "agents", {}).values():
        brain = getattr(ag, "brain", None)
        if brain is None:
            continue

        dump = serialize_brain_for_save(brain)
        path = os.path.join(out_dir, f"{ag.id}.mind.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[trainer_gui] WARN: can't write", path, e)

        try:
            save_brain(brain)
        except Exception as e:
            print("[trainer_gui] WARN: save_brain() failed:", e)


# =============================================================================
# Баланс / спавн
# =============================================================================

def _boost_exploration_bias(ag: Agent):
    try:
        br = getattr(ag, "brain", None)
        if br is None:
            return
        rules = getattr(br, "behavior_rules", None)
        if rules is not None:
            if hasattr(rules, "exploration_bias"):
                cur = _safe_float(getattr(rules, "exploration_bias", 0.0), 0.0)
                rules.exploration_bias = _clamp(max(cur, 0.9), 0.0, 1.0)
            if hasattr(rules, "healing_zone_seek_priority"):
                cur_h = _safe_float(getattr(rules, "healing_zone_seek_priority", 0.5), 0.5)
                rules.healing_zone_seek_priority = _clamp(cur_h, 0.0, 0.6)
        if hasattr(br, "curiosity_charge"):
            cur_c = _safe_float(getattr(br, "curiosity_charge", 0.0), 0.0)
            br.curiosity_charge = _clamp(max(cur_c, 0.8), 0.0, 1.0)
    except Exception as e:
        print("[trainer_gui] WARN: can't boost exploration for", getattr(ag, "id", "?"), ":", e)


def _spawn_training_animals(world: World, seed: int):
    rnd = random.Random(seed ^ 0xA11FA11F)
    animal_blueprints = [
        dict(uid="ani_friendly_0", species=AnimalSpecies("fox", "Лиска", 45.0, False, True, 0.4, 4.0, 4.0, 2.0, 10.0)),
        dict(uid="ani_friendly_1", species=AnimalSpecies("dog", "Пепелок", 50.0, False, True, 0.2, 6.0, 4.0, 2.0, 10.0)),
        dict(uid="ani_hostile_0",  species=AnimalSpecies("beast", "Грыз", 80.0, True, False, 1.0, 12.0, 8.0, 2.0, 12.0)),
        dict(uid="ani_hostile_1",  species=AnimalSpecies("wolf", "Клык", 70.0, True, False, 1.0, 10.0, 7.0, 2.0, 12.0)),
    ]
    for bp in animal_blueprints:
        ani = AnimalSim(uid=bp["uid"], species=bp["species"],
                        x=rnd.uniform(0.0, world.width), y=rnd.uniform(0.0, world.height))
        if hasattr(world, "add_animal"):
            world.add_animal(ani)


# =============================================================================
# Генерация мира
# =============================================================================

def _make_training_world(
    seed: int,
    num_agents: int,
    *,
    fresh_start: bool,
    agent_lineup: Optional[List[Dict[str, str]]] = None,
) -> World:
    random.seed(seed)
    w = getattr(config, "WORLD_WIDTH", 100.0)
    h = getattr(config, "WORLD_HEIGHT", 100.0)
    world = World(width=w, height=h)

    # опасные зоны
    for i in range(2):
        hx = random.uniform(10.0, w - 10.0)
        hy = random.uniform(10.0, h - 10.0)
        radius = random.uniform(4.0, 8.0)
        fire = WorldObject(
            obj_id=f"hazard_{i}", name=f"Огонь_{i}", kind="hazard",
            x=hx, y=hy, radius=radius, danger_level=0.7, comfort_level=0.0
        )
        world.add_object(fire)

    # безопасные зоны
    safe_spots: List[WorldObject] = []
    for i in range(2):
        sx = random.uniform(10.0, w - 10.0)
        sy = random.uniform(10.0, h - 10.0)
        radius = random.uniform(5.0, 9.0)
        safe = WorldObject(
            obj_id=f"safe_{i}", name=f"Убежище_{i}", kind="safe",
            x=sx, y=sy, radius=radius, danger_level=0.0, comfort_level=0.8
        )
        world.add_object(safe)
        safe_spots.append(safe)

    # registry активностей
    registry: Dict[str, Dict[str, Any]] = {}
    for obj in safe_spots:
        registry[obj.obj_id] = {
            "name": obj.name,
            "activity_tags": ["heal", "rest", "eat", "calm", "sleep", "repair_self", "restock_food"],
            "comfort_level": getattr(obj, "comfort_level", 0.0),
            "danger_level": getattr(obj, "danger_level", 0.0),
            "area": {"x": getattr(obj, "x", 0.0), "y": getattr(obj, "y", 0.0), "radius": getattr(obj, "radius", 0.0)},
        }
    if hasattr(world, "set_activity_registry"):
        world.set_activity_registry(registry)

    # SAFE_POINT
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

    # lineup агентов
    if agent_lineup and len(agent_lineup) > 0:
        spawn_specs = agent_lineup
    else:
        spawn_specs = [
            {"id": f"agent_{i}", "name": f"A{i}",
             "persona": random.choice(["caring/supportive", "protective", "loner", "scout/explorer"])}
            for i in range(num_agents)
        ]

    for spec in spawn_specs:
        ag = Agent(
            agent_id=spec["id"], name=spec["name"],
            x=random.uniform(0.0, w), y=random.uniform(0.0, h),
            goal_x=random.uniform(0.0, w), goal_y=random.uniform(0.0, h),
            persona=spec.get("persona") or random.choice(["caring/supportive", "protective", "loner", "scout/explorer"])
        )
        if fresh_start:
            ag.brain = ConsciousnessBlock(agent_id=ag.id)
        _boost_exploration_bias(ag)
        world.add_agent(ag)

    _spawn_training_animals(world, seed=seed)
    return world


# =============================================================================
# Тренер real-time для GUI
# =============================================================================

class MindTrainerInteractive(QtCore.QObject):
    world_changed = Signal()
    epoch_changed = Signal()
    agent_list_changed = Signal()

    def __init__(
        self,
        num_agents: int = 3,
        max_ticks_per_epoch: int = 2000,
        seed: int = 1234,
        disaster_interval_ticks: int = 400,
        relief_after_disaster: int = 80,
        fresh_start: bool = True,
        agent_lineup: Optional[List[Dict[str, str]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.num_agents = num_agents
        self.max_ticks_per_epoch = max_ticks_per_epoch
        self.seed = seed
        self.disaster_interval_ticks = disaster_interval_ticks
        self.relief_after_disaster = relief_after_disaster
        self.fresh_start = fresh_start
        self.agent_lineup = agent_lineup

        self.current_epoch = 0
        self.ticks_in_epoch = 0
        self._last_disaster_tick: Optional[int] = None

        self.monitor = TrainingMonitorState()
        self.world: Optional[World] = None

        self._spawn_new_epoch()

    def _auto_save_generation(self):
        if not self.world:
            return
        save_all_brains(self.world, "./trained_brains")

    def _spawn_new_epoch(self):
        world_seed = self.seed + self.current_epoch * 997
        self.world = _make_training_world(
            seed=world_seed,
            num_agents=self.num_agents,
            fresh_start=self.fresh_start,
            agent_lineup=self.agent_lineup,
        )
        self.ticks_in_epoch = 0
        self._last_disaster_tick = None

        self.monitor.epoch = self.current_epoch
        self.monitor.tick = 0
        self.monitor.note = f"epoch {self.current_epoch} started (seed={world_seed})"

        self.agent_list_changed.emit()
        self.epoch_changed.emit()

    def _inject_disaster(self, t: int):
        if not self.world:
            return
        w = self.world.width; h = self.world.height
        hx = random.uniform(5.0, w - 5.0); hy = random.uniform(5.0, h - 5.0)
        radius = random.uniform(4.0, 8.0)
        hazard = WorldObject(
            obj_id=f"disaster_{t}_{len(self.world.objects)}", name=f"Токсичная_утечка_{t}",
            kind="hazard", x=hx, y=hy, radius=radius, danger_level=1.0, comfort_level=0.0
        )
        self.world.add_object(hazard)
        ax = _clamp(hx + random.uniform(-3.0, 3.0), 0.0, w)
        ay = _clamp(hy + random.uniform(-3.0, 3.0), 0.0, h)
        hostile_species = AnimalSpecies(
            species_id=f"beast_evt_{t}", name=f"Хищник_{t}", base_hp=90.0, aggressive=True,
            tamable=False, tame_difficulty=1.0, bite_damage=15.0,
            fear_radius=9.0, follow_distance=2.0, aggro_radius=12.0
        )
        hostile_animal = AnimalSim(uid=f"ani_hostile_evt_{t}", species=hostile_species, x=ax, y=ay)
        if hasattr(self.world, "add_animal"):
            self.world.add_animal(hostile_animal)
        if hasattr(self.world, "add_chat_line"):
            try: self.world.add_chat_line("[trainer] внезапная опасная зона! Все в лагерь!")
            except Exception: pass
        self.monitor.note = "disaster injected"
        self._last_disaster_tick = t

    def _maybe_inject_disaster(self, t: int):
        if not self.world or not self.disaster_interval_ticks or self.disaster_interval_ticks <= 0:
            return
        if t > 0 and (t % self.disaster_interval_ticks == 0):
            self._inject_disaster(t)

    def _inject_relief(self, t: int):
        if not self.world:
            return
        w = self.world.width; h = self.world.height
        sx = random.uniform(5.0, w - 5.0); sy = random.uniform(5.0, h - 5.0)
        radius = random.uniform(5.0, 9.0)
        refuge = WorldObject(
            obj_id=f"refuge_{t}_{len(self.world.objects)}", name=f"Лагерь_после_бури_{t}",
            kind="safe", x=sx, y=sy, radius=radius, danger_level=0.0, comfort_level=1.0
        )
        self.world.add_object(refuge)
        base_registry = getattr(self.world, "activities", None) or {}
        reg = dict(base_registry)
        reg[refuge.obj_id] = {
            "name": refuge.name,
            "activity_tags": ["heal", "rest", "eat", "calm", "sleep", "repair_self", "restock_food"],
            "comfort_level": getattr(refuge, "comfort_level", 0.0),
            "danger_level": getattr(refuge, "danger_level", 0.0),
            "area": {"x": getattr(refuge, "x", 0.0), "y": getattr(refuge, "y", 0.0), "radius": getattr(refuge, "radius", 0.0)},
        }
        if hasattr(self.world, "set_activity_registry"):
            self.world.set_activity_registry(reg)

        ax = _clamp(sx + random.uniform(-2.0, 2.0), 0.0, w)
        ay = _clamp(sy + random.uniform(-2.0, 2.0), 0.0, h)
        friendly_species = AnimalSpecies(
            species_id=f"dog_evt_{t}", name=f"Дружок_{t}", base_hp=50.0, aggressive=False,
            tamable=True, tame_difficulty=0.2, bite_damage=6.0,
            fear_radius=4.0, follow_distance=2.0, aggro_radius=10.0
        )
        pet_animal = AnimalSim(uid=f"ani_friendly_evt_{t}", species=friendly_species, x=ax, y=ay)
        if hasattr(self.world, "add_animal"):
            self.world.add_animal(pet_animal)
        if hasattr(self.world, "add_chat_line"):
            try: self.world.add_chat_line("[trainer] безопасный лагерь создан. Отдыхайте и пробуйте приручить зверя.")
            except Exception: pass
        self.monitor.note = "relief injected"

    def _maybe_inject_relief(self, t: int):
        if not self.world or self._last_disaster_tick is None or not self.relief_after_disaster or self.relief_after_disaster <= 0:
            return
        if t - self._last_disaster_tick == self.relief_after_disaster:
            self._inject_relief(t)

    def _collect_monitor_stats(self):
        if not self.world:
            return
        ages: List[float] = []; fears: List[float] = []; hps: List[float] = []
        enes: List[float] = []; hung: List[float] = []; curios: List[float] = []
        survs: List[float] = []; trauma_counts: List[float] = []
        cling_flags: List[float] = []; panic_flags: List[float] = []; alive_flags: List[float] = []
        pets_counts: List[float] = []; tame_flags: List[float] = []

        def _iter_world_animals():
            animals_obj = getattr(self.world, "animals", None)
            if animals_obj is None:
                return []
            if isinstance(animals_obj, dict):
                return list(animals_obj.values())
            return list(animals_obj)

        def _pets_for_agent(agent_id: str) -> List[AnimalSim]:
            res: List[AnimalSim] = []
            for ani in _iter_world_animals():
                if getattr(ani, "tamed_by", None) == agent_id:
                    res.append(ani)
            return res

        for ag in getattr(self.world, "agents", {}).values():
            ages.append(_safe_float(getattr(ag, "age_ticks", 0), 0.0))
            hps.append(_safe_float(getattr(ag, "health", 0.0), 0.0))
            fears.append(_safe_float(getattr(ag, "fear", 0.0), 0.0))
            enes.append(_safe_float(getattr(ag, "energy", 0.0), 0.0))
            hung.append(_safe_float(getattr(ag, "hunger", 0.0), 0.0))
            alive_flags.append(1.0 if getattr(ag, "is_alive", lambda: True)() else 0.0)
            my_pets = _pets_for_agent(ag.id)
            pets_counts.append(float(len(my_pets)))
            tame_flags.append(1.0 if len(my_pets) > 0 else 0.0)
            br = getattr(ag, "brain", None)
            if br is not None:
                survs.append(_safe_float(getattr(br, "survival_score", 0.0), 0.0))
                curios.append(_safe_float(getattr(br, "curiosity_charge", 0.0), 0.0))
                trauma_map = getattr(br, "trauma_map", []) or []
                trauma_counts.append(float(len(trauma_map)))
                drive_now = getattr(br, "current_drive", "")
                cling_flags.append(1.0 if drive_now == "stay_with_ally" else 0.0)
                panic_flags.append(1.0 if _safe_float(getattr(br, "fear_level", 0.0), 0.0) > 0.7 else 0.0)
            else:
                survs.append(0.0); curios.append(0.0); trauma_counts.append(0.0)
                cling_flags.append(0.0); panic_flags.append(0.0)

        def _avg(v: List[float]) -> float:
            return (sum(v) / len(v)) if v else 0.0

        m = self.monitor
        m.avg_age = _avg(ages); m.avg_fear = _avg(fears); m.avg_hp = _avg(hps)
        m.avg_energy = _avg(enes); m.avg_hunger = _avg(hung)
        m.avg_curiosity = _avg(curios); m.avg_survival = _avg(survs)
        m.avg_trauma_spots = _avg(trauma_counts)
        m.cling_ratio = _avg(cling_flags); m.panic_ratio = _avg(panic_flags)
        m.alive_ratio = _avg(alive_flags); m.avg_pets = _avg(pets_counts)
        m.tame_ratio = _avg(tame_flags)

    def step_tick(self):
        if not self.world:
            return
        t = self.ticks_in_epoch
        self._maybe_inject_disaster(t)
        self._maybe_inject_relief(t)
        self.world.tick()
        self.ticks_in_epoch += 1
        self.monitor.tick = self.ticks_in_epoch
        self._collect_monitor_stats()
        if self.ticks_in_epoch >= self.max_ticks_per_epoch:
            dead_count = sum(1 for a in getattr(self.world, "agents", {}).values() if not getattr(a, "is_alive", lambda: True)())
            self.monitor.note = f"epoch {self.current_epoch} finished, " + ("all alive" if dead_count == 0 else f"{dead_count} dead")
            self._auto_save_generation()
            self.current_epoch += 1
            self._spawn_new_epoch()
        self.world_changed.emit()

    def force_next_epoch(self):
        self._auto_save_generation()
        self.current_epoch += 1
        self._spawn_new_epoch()

    def save_brains_now(self, out_dir: str = "./trained_brains"):
        if not self.world:
            return
        save_all_brains(self.world, out_dir)


# =============================================================================
# 2D Beliefs
# =============================================================================

class BeliefGraphView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        scene = QtWidgets.QGraphicsScene()
        super().__init__(scene, parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        self.setStyleSheet("background-color:#0f0f16; border:1px solid #444; border-radius:6px;")

    def update_from_brain(self, brain: Optional[ConsciousnessBlock]):
        scene = QtWidgets.QGraphicsScene()
        self.setScene(scene)

        if brain is None:
            txt = scene.addText("no brain")
            txt.setDefaultTextColor(QtGui.QColor("#888"))
            return

        beliefs = list(getattr(brain, "beliefs", []))[-20:]
        if not beliefs:
            txt = scene.addText("no beliefs yet")
            txt.setDefaultTextColor(QtGui.QColor("#888"))
            lt = getattr(brain, "last_thought", None)
            if lt:
                txt2 = scene.addText(f"thought: {lt}")
                txt2.setDefaultTextColor(QtGui.QColor("#ccc"))
                txt2.setPos(0, 30)
            return

        conds: List[str] = []
        concls: List[str] = []
        for b in beliefs:
            c_from = getattr(b, "condition", "")
            c_to = getattr(b, "conclusion", "")
            if c_from not in conds:
                conds.append(c_from)
            if c_to not in concls:
                concls.append(c_to)

        left_x = 0; right_x = 260; spacing_y = 40

        cond_pos: Dict[str, QtCore.QPointF] = {}
        for i, c in enumerate(conds):
            y = i * spacing_y
            cond_pos[c] = QtCore.QPointF(left_x, y)
            circ = scene.addEllipse(left_x, y, 20, 20,
                                    pen=QtGui.QPen(QtGui.QColor("#55aaff")),
                                    brush=QtGui.QBrush(QtGui.QColor("#112244")))
            circ.setZValue(1)
            label = scene.addText(c); label.setDefaultTextColor(QtGui.QColor("#9cf"))
            label.setPos(left_x + 28, y - 4)

        concl_pos: Dict[str, QtCore.QPointF] = {}
        for j, c2 in enumerate(concls):
            y = j * spacing_y
            concl_pos[c2] = QtCore.QPointF(right_x, y)
            circ = scene.addEllipse(right_x, y, 20, 20,
                                    pen=QtGui.QPen(QtGui.QColor("#ffaa55")),
                                    brush=QtGui.QBrush(QtGui.QColor("#442211")))
            circ.setZValue(1)
            label = scene.addText(c2); label.setDefaultTextColor(QtGui.QColor("#fc9"))
            label.setPos(right_x + 28, y - 4)

        for b in beliefs:
            c_from = getattr(b, "condition", "")
            c_to = getattr(b, "conclusion", "")
            p1 = cond_pos[c_from] + QtCore.QPointF(20, 10)
            p2 = concl_pos[c_to] + QtCore.QPointF(0, 10)
            strength = float(getattr(b, "strength", 0.0))
            width = 1.0 + strength * 4.0
            pen = QtGui.QPen(QtGui.QColor("#ccc")); pen.setWidthF(width)
            scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
            mid_x = (p1.x() + p2.x()) * 0.5; mid_y = (p1.y() + p2.y()) * 0.5
            stxt = scene.addText(f"{strength:.2f}"); stxt.setDefaultTextColor(QtGui.QColor("#ccc"))
            stxt.setPos(mid_x + 4, mid_y - 6)

        lt = getattr(brain, "last_thought", None)
        if lt:
            base_y = max(len(conds), len(concls)) * spacing_y + 20
            thought_lbl = scene.addText(f"thought: {lt}")
            thought_lbl.setDefaultTextColor(QtGui.QColor("#ccc"))
            thought_lbl.setPos(left_x, base_y)

        scene.setSceneRect(-20, -40, right_x + 260, max(len(conds), len(concls)) * spacing_y + 100)


# =============================================================================
# 3D Beliefs — Qt3D (если есть) ИЛИ GL-fallback
# =============================================================================

def _vec_sub(a: QtGui.QVector3D, b: QtGui.QVector3D) -> QtGui.QVector3D:
    return QtGui.QVector3D(a.x() - b.x(), a.y() - b.y(), a.z() - b.z())


def _vec_len(v: QtGui.QVector3D) -> float:
    return math.sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z())


def _quat_from_to(frm: QtGui.QVector3D, to: QtGui.QVector3D) -> QtGui.QQuaternion:
    v1 = QtGui.QVector3D(frm); v2 = QtGui.QVector3D(to)
    if v1.length() < 1e-6 or v2.length() < 1e-6:
        return QtGui.QQuaternion()
    v1.normalize(); v2.normalize()
    dot = max(-1.0, min(1.0, float(QtGui.QVector3D.dotProduct(v1, v2))))
    if dot > 0.999999:
        return QtGui.QQuaternion()
    if dot < -0.999999:
        ortho = QtGui.QVector3D(1.0, 0.0, 0.0)
        if abs(v1.x()) > 0.9: ortho = QtGui.QVector3D(0.0, 1.0, 0.0)
        axis = QtGui.QVector3D.crossProduct(v1, ortho); axis.normalize()
        return QtGui.QQuaternion.fromAxisAndAngle(axis, 180.0)
    axis = QtGui.QVector3D.crossProduct(v1, v2)
    if axis.length() < 1e-6:
        return QtGui.QQuaternion()
    axis.normalize()
    angle_deg = math.degrees(math.acos(dot))
    return QtGui.QQuaternion.fromAxisAndAngle(axis, angle_deg)


class NeuralGraph3DPlaceholder(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        msg_lines = []
        if not QT3D_AVAILABLE:
            msg_lines.append("Qt3D модули не найдены.")
        elif QT3D_AVAILABLE and not QT3D_HAS_WINDOW:
            msg_lines.append("Qt3D есть, но нет Qt3DExtras.Qt3DWindow.")
        if not GL_AVAILABLE:
            msg_lines.append("PyOpenGL не установлен — fallback недоступен.")
        msg_lines.append("Установи PyOpenGL или включи Qt3D, либо смотри вкладку Beliefs 2D.")
        lab = QtWidgets.QLabel("\n".join(msg_lines))
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("color:#bbb; background:#15151d; border:1px solid #333;")
        lay = QtWidgets.QVBoxLayout(self); lay.addWidget(lab)
    def update_from_brain(self, _brain): pass


# ---------- Qt3D реализация ----------
if QT3D_AVAILABLE and QT3D_HAS_WINDOW:
    class NeuralGraph3DView(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.view = Qt3DExtras.Qt3DWindow()
            self.view.defaultFrameGraph().setClearColor(QtGui.QColor("#0c0c12"))
            self.container = QtWidgets.QWidget.createWindowContainer(self.view)
            self.container.setMinimumHeight(320)
            lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.addWidget(self.container)
            self.root = Qt3DCore.QEntity(); self.view.setRootEntity(self.root)
            self.camera = self.view.camera()
            self.camera.lens().setPerspectiveProjection(45.0, 16/9, 0.1, 1000.0)
            self.camera.setPosition(QtGui.QVector3D(0.0, 8.0, 34.0))
            self.camera.setViewCenter(QtGui.QVector3D(0.0, 6.0, 0.0))
            self.cam_ctrl = Qt3DExtras.QOrbitCameraController(self.root)
            self.cam_ctrl.setLinearSpeed(12.0); self.cam_ctrl.setLookSpeed(120.0); self.cam_ctrl.setCamera(self.camera)
            self._add_light(QtGui.QVector3D(30.0, 30.0, 30.0), 0.8)
            self._add_light(QtGui.QVector3D(-30.0, 20.0, -10.0), 0.6)
            self._add_ambient(0.25)
            self._node_entities: List[Qt3DCore.QEntity] = []
            self._edge_entities: List[Qt3DCore.QEntity] = []

        def _add_light(self, pos: QtGui.QVector3D, intensity: float = 1.0):
            light_entity = Qt3DCore.QEntity(self.root)
            light = Qt3DRender.QPointLight(light_entity); light.setIntensity(intensity)
            light_entity.addComponent(light)
            tr = Qt3DCore.QTransform(); tr.setTranslation(pos); light_entity.addComponent(tr)

        def _add_ambient(self, strength: float):
            self._add_light(QtGui.QVector3D(0.0, 100.0, 0.0), strength)

        def _make_sphere(self, pos: QtGui.QVector3D, radius: float, color: QtGui.QColor) -> Qt3DCore.QEntity:
            ent = Qt3DCore.QEntity(self.root)
            mesh = Qt3DExtras.QSphereMesh(); mesh.setRadius(radius); mesh.setRings(24); mesh.setSlices(32)
            mat = Qt3DExtras.QPhongMaterial(ent); mat.setDiffuse(color)
            tr = Qt3DCore.QTransform(); tr.setTranslation(pos)
            ent.addComponent(mesh); ent.addComponent(mat); ent.addComponent(tr)
            return ent

        def _make_tube(self, a: QtGui.QVector3D, b: QtGui.QVector3D, radius: float, color: QtGui.QColor) -> Qt3DCore.QEntity:
            ent = Qt3DCore.QEntity(self.root)
            mesh = Qt3DExtras.QCylinderMesh()
            length = _vec_len(_vec_sub(b, a))
            mesh.setLength(max(0.01, length)); mesh.setRadius(max(0.01, radius))
            mesh.setRings(8); mesh.setSlices(24)
            mat = Qt3DExtras.QPhongMaterial(ent); mat.setDiffuse(color)
            tr = Qt3DCore.QTransform()
            mid = QtGui.QVector3D((a.x()+b.x())/2.0, (a.y()+b.y())/2.0, (a.z()+b.z())/2.0); tr.setTranslation(mid)
            dirv = _vec_sub(b, a); rot = _quat_from_to(QtGui.QVector3D(0, 1, 0), dirv); tr.setRotation(rot)
            ent.addComponent(mesh); ent.addComponent(mat); ent.addComponent(tr)
            return ent

        def _clear_scene(self):
            for e in self._edge_entities: e.setParent(None)
            for e in self._node_entities: e.setParent(None)
            self._edge_entities.clear(); self._node_entities.clear()

        def update_from_brain(self, brain: Optional[ConsciousnessBlock]):
            self._clear_scene()
            if brain is None: return
            beliefs = list(getattr(brain, "beliefs", []))[-30:]
            if not beliefs: return

            conds: List[str] = []; concls: List[str] = []
            for b in beliefs:
                c_from = getattr(b, "condition", ""); c_to = getattr(b, "conclusion", "")
                if c_from and c_from not in conds: conds.append(c_from)
                if c_to and c_to not in concls: concls.append(c_to)

            left_x, right_x = -10.0, 10.0; base_y = 0.0; spacing_y = 2.0
            rnd = random.Random(1337)

            cond_pos: Dict[str, QtGui.QVector3D] = {}
            for i, c in enumerate(conds):
                pos = QtGui.QVector3D(left_x, base_y + i*spacing_y, rnd.uniform(-2.0, 2.0))
                cond_pos[c] = pos
                node = self._make_sphere(pos, 0.7, QtGui.QColor("#4aa3ff"))
                self._node_entities.append(node)

            concl_pos: Dict[str, QtGui.QVector3D] = {}
            for j, c2 in enumerate(concls):
                pos = QtGui.QVector3D(right_x, base_y + j*spacing_y, rnd.uniform(-2.0, 2.0))
                concl_pos[c2] = pos
                node = self._make_sphere(pos, 0.7, QtGui.QColor("#ffb057"))
                self._node_entities.append(node)

            for b in beliefs:
                c_from = getattr(b, "condition", ""); c_to = getattr(b, "conclusion", "")
                if c_from not in cond_pos or c_to not in concl_pos: continue
                p1 = cond_pos[c_from]; p2 = concl_pos[c_to]
                strength = float(getattr(b, "strength", 0.0))
                r = 0.05 + 0.35 * max(0.0, min(1.0, strength))
                edge = self._make_tube(p1, p2, r, QtGui.QColor("#d8d8ff"))
                self._edge_entities.append(edge)

# ---------- GL fallback (без Qt3D, но с PyOpenGL) ----------
elif GL_AVAILABLE:
    class _ArcballCamera:
        def __init__(self):
            self.distance = 34.0
            self.center = (0.0, 6.0, 0.0)
            self.yaw = 0.0     # вокруг Y
            self.pitch = -10.0 # вокруг X

    class NeuralGraph3DView(QOpenGLWidget):
        """
        Fallback 3D: точки (узлы) и линии (рёбра) в OpenGL.
        Управление:
          - ЛКМ — вращение (yaw/pitch)
          - Колёсико — зум
          - Shift+ЛКМ — панорамирование (немного)
        """
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumHeight(320)
            self.setMouseTracking(True)
            self.cam = _ArcballCamera()
            self._last_pos: Optional[QtCore.QPoint] = None
            # геометрия графа
            self._nodes: List[Tuple[Tuple[float,float,float], float, Tuple[float,float,float]]] = []
            # (pos, point_size, color_rgb)
            self._edges: List[Tuple[Tuple[float,float,float], Tuple[float,float,float], float, Tuple[float,float,float]]] = []
            # (a, b, width, color_rgb)

        # ---- OpenGL lifecycle
        def initializeGL(self):
            glClearColor(0.05, 0.06, 0.09, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_POINT_SMOOTH)

        def resizeGL(self, w: int, h: int):
            if h <= 0: h = 1
            glViewport(0, 0, w, h)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = w / float(h)
            gluPerspective(45.0, aspect, 0.1, 1000.0)
            glMatrixMode(GL_MODELVIEW)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            cx, cy, cz = self.cam.center
            # позиция камеры по сферическим координатам
            yaw_rad = math.radians(self.cam.yaw)
            pitch_rad = math.radians(self.cam.pitch)
            dx = self.cam.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
            dy = self.cam.distance * math.sin(pitch_rad)
            dz = self.cam.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
            eye = (cx + dx, cy - dy, cz + dz)
            gluLookAt(eye[0], eye[1], eye[2],  cx, cy, cz,  0,1,0)

            # рёбра
            for (a, b, width, col) in self._edges:
                glLineWidth(max(1.0, width))
                glBegin(GL_LINES)
                glVertex3f(a[0], a[1], a[2]); glVertex3f(b[0], b[1], b[2])
                glEnd()

            # узлы
            for (pos, size, col) in self._nodes:
                glPointSize(max(2.0, size))
                glBegin(GL_POINTS)
                glVertex3f(pos[0], pos[1], pos[2])
                glEnd()

        # ---- взаимодействие
        def mousePressEvent(self, e: QtGui.QMouseEvent):
            self._last_pos = e.pos()

        def mouseMoveEvent(self, e: QtGui.QMouseEvent):
            if self._last_pos is None:
                self._last_pos = e.pos(); return
            dx = e.position().x() - self._last_pos.x()
            dy = e.position().y() - self._last_pos.y()
            if e.buttons() & Qt.LeftButton:
                if e.modifiers() & Qt.ShiftModifier:
                    # простое панорамирование по центру
                    self.cam.center = (
                        self.cam.center[0] - dx * 0.02,
                        self.cam.center[1] + dy * 0.02,
                        self.cam.center[2]
                    )
                else:
                    self.cam.yaw += dx * 0.4
                    self.cam.pitch = _clamp(self.cam.pitch + dy * 0.3, -89.0, 89.0)
            self._last_pos = e.pos()
            self.update()

        def wheelEvent(self, e: QtGui.QWheelEvent):
            delta = e.angleDelta().y() / 120.0
            self.cam.distance = _clamp(self.cam.distance - delta * 1.5, 3.0, 200.0)
            self.update()

        # ---- API
        def update_from_brain(self, brain: Optional[ConsciousnessBlock]):
            self._nodes.clear(); self._edges.clear()
            if brain is None:
                self.update(); return
            beliefs = list(getattr(brain, "beliefs", []))[-60:]
            if not beliefs:
                self.update(); return

            # собрать уникальные IF/THEN
            conds: List[str] = []; concls: List[str] = []
            for b in beliefs:
                c_from = getattr(b, "condition", ""); c_to = getattr(b, "conclusion", "")
                if c_from and c_from not in conds: conds.append(c_from)
                if c_to and c_to not in concls: concls.append(c_to)

            left_x, right_x = -10.0, 10.0; base_y = 0.0; spacing_y = 2.0
            rnd = random.Random(1337)

            cond_pos: Dict[str, Tuple[float,float,float]] = {}
            for i, c in enumerate(conds):
                y = base_y + i * spacing_y; z = rnd.uniform(-2.0, 2.0)
                pos = (left_x, y, z); cond_pos[c] = pos
                self._nodes.append((pos, 7.0, (0.29, 0.64, 1.0)))  # голубые точки

            concl_pos: Dict[str, Tuple[float,float,float]] = {}
            for j, c2 in enumerate(concls):
                y = base_y + j * spacing_y; z = rnd.uniform(-2.0, 2.0)
                pos = (right_x, y, z); concl_pos[c2] = pos
                self._nodes.append((pos, 7.0, (1.0, 0.69, 0.35)))  # оранжевые точки

            for b in beliefs:
                c_from = getattr(b, "condition", ""); c_to = getattr(b, "conclusion", "")
                if c_from not in cond_pos or c_to not in concl_pos: continue
                p1 = cond_pos[c_from]; p2 = concl_pos[c_to]
                strength = float(getattr(b, "strength", 0.0))
                width = 1.0 + 6.0 * max(0.0, min(1.0, strength))
                self._edges.append((p1, p2, width, (0.85, 0.85, 1.0)))

            self.update()

# ---------- ни Qt3D, ни GL → плейсхолдер ----------
else:
    class NeuralGraph3DView(NeuralGraph3DPlaceholder):
        pass


# =============================================================================
# Инспектор агента (правая панель)
# =============================================================================

class AgentBrainWidget(QtWidgets.QWidget):
    def __init__(self, trainer: MindTrainerInteractive, parent=None):
        super().__init__(parent)
        self.trainer = trainer

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(8)

        self.comboAgent = QtWidgets.QComboBox()
        self.comboAgent.currentIndexChanged.connect(self.refresh_all)
        outer.addWidget(self.comboAgent)

        statsFrame = QtWidgets.QFrame()
        statsFrame.setStyleSheet("""
            QFrame { background-color:#15151d; border:1px solid #333; border-radius:8px; }
            QLabel { color:#ccc; font-size:12px; }
        """)
        statsLay = QtWidgets.QGridLayout(statsFrame)
        statsLay.setContentsMargins(8, 8, 8, 8); statsLay.setSpacing(4)

        self.lblName = QtWidgets.QLabel("Agent: —")
        self.lblDrive = QtWidgets.QLabel("drive: –")
        self.lblAlly = QtWidgets.QLabel("ally_anchor: –")
        self.lblDeathReason = QtWidgets.QLabel("last_death_reason: –")
        self.lblThought = QtWidgets.QLabel("thought: –"); self.lblThought.setWordWrap(True)
        self.lblPets = QtWidgets.QLabel("pets: –"); self.lblPets.setWordWrap(True)

        statsLay.addWidget(self.lblName, 0, 0, 1, 2)
        statsLay.addWidget(self.lblDrive, 1, 0, 1, 2)
        statsLay.addWidget(self.lblAlly,  2, 0, 1, 2)
        statsLay.addWidget(self.lblDeathReason, 3, 0, 1, 2)
        statsLay.addWidget(self.lblThought, 4, 0, 1, 2)
        statsLay.addWidget(self.lblPets,   5, 0, 1, 2)

        bar_css_base = """
            QProgressBar {
                background-color:#2a2a2a; border:1px solid #555; border-radius:4px;
                color:#ddd; text-align:center; height:14px; font-size:10px;
            }
        """
        self.hpBar = QtWidgets.QProgressBar(); self.hpBar.setRange(0, 100); self.hpBar.setFormat("HP: %v")
        self.hpBar.setStyleSheet(bar_css_base + "QProgressBar::chunk{background-color:#4CAF50;}")
        self.fearBar = QtWidgets.QProgressBar(); self.fearBar.setRange(0, 100); self.fearBar.setFormat("Fear: %v")
        self.fearBar.setStyleSheet(bar_css_base + "QProgressBar::chunk{background-color:#E67E22;}")
        self.survivalBar = QtWidgets.QProgressBar(); self.survivalBar.setRange(0, 100); self.survivalBar.setFormat("Survival: %v")
        self.survivalBar.setStyleSheet(bar_css_base + "QProgressBar::chunk{background-color:#3BA3FF;}")
        self.energyBar = QtWidgets.QProgressBar(); self.energyBar.setRange(0, 100); self.energyBar.setFormat("Energy: %v")
        self.energyBar.setStyleSheet(bar_css_base + "QProgressBar::chunk{background-color:#4B8BFF;}")
        self.hungerBar = QtWidgets.QProgressBar(); self.hungerBar.setRange(0, 100); self.hungerBar.setFormat("Hunger: %v")
        self.hungerBar.setStyleSheet(bar_css_base + "QProgressBar::chunk{background-color:#FF4B4B;}")

        statsLay.addWidget(self.hpBar, 6, 0, 1, 2)
        statsLay.addWidget(self.fearBar, 7, 0, 1, 2)
        statsLay.addWidget(self.survivalBar, 8, 0, 1, 2)
        statsLay.addWidget(self.energyBar, 9, 0, 1, 2)
        statsLay.addWidget(self.hungerBar, 10, 0, 1, 2)

        outer.addWidget(statsFrame)

        outer.addWidget(QtWidgets.QLabel("Мозг / Поведенческие веса:"), 0, Qt.AlignLeft)
        self.behaviorText = QtWidgets.QTextEdit(); self.behaviorText.setReadOnly(True)
        self.behaviorText.setStyleSheet("""
            QTextEdit { background-color:#0f0f16; border:1px solid #444;
                        color:#d0d0ff; font-family:monospace; font-size:11px; border-radius:6px; }
        """)
        outer.addWidget(self.behaviorText, 1)

        outer.addWidget(QtWidgets.QLabel("Последние события памяти:"), 0, Qt.AlignLeft)
        self.memoryView = QtWidgets.QTextEdit(); self.memoryView.setReadOnly(True)
        self.memoryView.setStyleSheet("""
            QTextEdit { background-color:#0f0f16; border:1px solid #444;
                        color:#8fda8f; font-family:monospace; font-size:11px; border-radius:6px; }
        """)
        outer.addWidget(self.memoryView, 1)

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border:1px solid #333; border-radius:8px; background:#15151d; }
            QTabBar::tab { background:#1c1c26; color:#ddd; padding:6px 10px; border:1px solid #333; border-bottom:none; }
            QTabBar::tab:selected { background:#252533; }
        """)
        self.belief2D = BeliefGraphView()
        self.belief3D = NeuralGraph3DView()  # Qt3D, либо GL-fallback, либо плейсхолдер

        tab2d = QtWidgets.QWidget(); lay2d = QtWidgets.QVBoxLayout(tab2d); lay2d.setContentsMargins(6,6,6,6); lay2d.addWidget(self.belief2D)
        tab3d = QtWidgets.QWidget(); lay3d = QtWidgets.QVBoxLayout(tab3d); lay3d.setContentsMargins(6,6,6,6); lay3d.addWidget(self.belief3D)

        tabs.addTab(tab2d, "Beliefs 2D"); tabs.addTab(tab3d, "Neural 3D")
        outer.addWidget(tabs, 3)
        outer.addStretch(1)

        self.trainer.agent_list_changed.connect(self.rebuild_agent_list)
        self.trainer.world_changed.connect(self.refresh_all)
        self.trainer.epoch_changed.connect(self.rebuild_agent_list)

        self.rebuild_agent_list()
        self.refresh_all()

    def _iter_world_agents(self):
        w = self.trainer.world
        if not w: return []
        agents_obj = getattr(w, "agents", {})
        if isinstance(agents_obj, dict): return list(agents_obj.values())
        return list(agents_obj)

    def _iter_world_animals(self):
        w = self.trainer.world
        if not w: return []
        animals_obj = getattr(w, "animals", {})
        if isinstance(animals_obj, dict): return list(animals_obj.values())
        return list(animals_obj)

    def get_selected_agent(self) -> Optional[Agent]:
        w = self.trainer.world
        if not w: return None
        idx = self.comboAgent.currentIndex()
        if idx < 0: return None
        agent_id = self.comboAgent.itemData(idx)
        if hasattr(w, "get_agent_by_id"):
            return w.get_agent_by_id(agent_id)
        for ag in self._iter_world_agents():
            if ag.id == agent_id: return ag
        return None

    @Slot()
    def rebuild_agent_list(self):
        self.comboAgent.blockSignals(True)
        self.comboAgent.clear()
        for ag in self._iter_world_agents():
            self.comboAgent.addItem(f"{ag.name} ({ag.id})", ag.id)
        self.comboAgent.blockSignals(False)
        self.refresh_all()

    def _pets_for_agent(self, ag: Agent) -> List[AnimalSim]:
        pets: List[AnimalSim] = []
        for ani in self._iter_world_animals():
            if getattr(ani, "tamed_by", None) == ag.id:
                pets.append(ani)
        return pets

    def _animal_temperament_label(self, ani: AnimalSim) -> str:
        aggressive = getattr(ani.species, "aggressive", False)
        tamable = getattr(ani.species, "tamable", False)
        tamed_by = getattr(ani, "tamed_by", None)
        if aggressive and tamed_by is None: return "aggressive"
        elif tamable: return "tameable"
        else: return "neutral"

    def _memory_html_for_agent(self, ag: Agent, brain: Optional[ConsciousnessBlock]) -> str:
        if hasattr(ag, "memory") and ag.memory is not None and hasattr(ag.memory, "dump_public_view"):
            events = ag.memory.dump_public_view(tail=20)
            if events:
                lines: List[str] = []
                for ev in events:
                    level = ev.get("level", "info"); color = "#8fda8f"
                    if level == "critical": color = "#ff5c5c"
                    elif level == "warning": color = "#E6A23C"
                    tick = ev.get("tick", "?"); etype = ev.get("type", "?"); data = ev.get("data", {})
                    parts = [f"{k}={v}" for k, v in data.items()] if isinstance(data, dict) else [str(data)]
                    lines.append(f'<span style="color:{color}">[t={tick}] {etype}: {", ".join(parts)}</span>')
                return "<br/>\n".join(lines)

        if brain is not None and getattr(brain, "memory_tail", None):
            tail = brain.memory_tail[-20:]; lines2: List[str] = []
            for ev in tail:
                color = "#8fda8f"
                if getattr(ev, "etype", "") in ("pain", "death"): color = "#ff5c5c"
                elif getattr(ev, "etype", "") in ("heal", "rest", "eat"): color = "#5cf5ff"
                parts = [f"{k}={v}" for k, v in getattr(ev, "data", {}).items()] if isinstance(getattr(ev, "data", {}), dict) else [str(getattr(ev, "data", ""))]
                lines2.append(f'<span style="color:{color}">[t={getattr(ev, "tick", "?")}] {getattr(ev, "etype","?")}: {", ".join(parts)}</span>')
            return "<br/>\n".join(lines2)
        return "(memory empty)"

    @Slot()
    def refresh_all(self):
        ag = self.get_selected_agent()
        if ag is None:
            self.lblName.setText("Agent: —"); self.lblDrive.setText("drive: –")
            self.lblAlly.setText("ally_anchor: –"); self.lblDeathReason.setText("last_death_reason: –")
            self.lblThought.setText("thought: –"); self.lblPets.setText("pets: –")
            for bar in (self.hpBar, self.fearBar, self.survivalBar, self.energyBar, self.hungerBar): bar.setValue(0)
            self.behaviorText.setPlainText("(нет данных)"); self.memoryView.setHtml("(нет данных)")
            self.belief2D.update_from_brain(None); self.belief3D.update_from_brain(None); return

        brain = getattr(ag, "brain", None)
        self.lblName.setText(f"Agent: {ag.name} ({ag.id})")

        pets_list = self._pets_for_agent(ag)
        if pets_list:
            pet_desc = []
            for p in pets_list:
                p_name = getattr(p.species, "name", getattr(p.species, "species_id", "pet"))
                hp_val = getattr(p, "hp", None); mood = self._animal_temperament_label(p)
                frag = p_name; 
                if hp_val is not None: frag += f" hp={int(hp_val)}"
                if mood: frag += f" ({mood})"
                pet_desc.append(frag)
            self.lblPets.setText("pets: " + ", ".join(pet_desc))
        else:
            self.lblPets.setText("pets: –")

        hp_val = int(_clamp(_safe_float(getattr(ag, "health", 0.0), 0.0), 0.0, 100.0))
        fear_raw = _safe_float(getattr(brain, "fear_level", getattr(ag, "fear", 0.0) if ag else 0.0), 0.0)
        fear_pct = int(_clamp(fear_raw * 100.0, 0.0, 100.0))
        surv_raw = _safe_float(getattr(brain, "survival_score", 0.0), 0.0)
        surv_pct = int(_clamp(surv_raw * 100.0, 0.0, 100.0))
        energy_raw = _safe_float(getattr(ag, "energy", 0.0), 0.0)
        energy_val = int(_clamp(energy_raw, 0.0, 100.0))
        hunger_raw = _safe_float(getattr(ag, "hunger", 0.0), 0.0)
        hunger_val = int(_clamp(hunger_raw, 0.0, 100.0))
        self.hpBar.setValue(hp_val); self.fearBar.setValue(fear_pct)
        self.survivalBar.setValue(surv_pct); self.energyBar.setValue(energy_val)
        self.hungerBar.setValue(hunger_val)

        drive_txt = getattr(brain, "current_drive", "–") if brain else "–"
        ally_anchor = getattr(brain, "ally_anchor", None) if brain else None
        last_death_reason = getattr(brain, "last_death_reason", None) if brain else None
        if not last_death_reason and hasattr(ag, "cause_of_death"): last_death_reason = ag.cause_of_death
        last_thought = getattr(brain, "last_thought", "–") if brain else "–"
        self.lblDrive.setText(f"drive: {drive_txt}")
        self.lblAlly.setText(f"ally_anchor: {ally_anchor if ally_anchor else '–'}")
        self.lblDeathReason.setText(f"last_death_reason: {last_death_reason if last_death_reason else '–'}")
        self.lblThought.setText(f"thought: {last_thought}")

        behavior_dump = ""
        if brain is not None and getattr(brain, "behavior_rules", None):
            brules = brain.behavior_rules
            curiosity = getattr(brain, "curiosity_charge", 0.0)
            trauma_map = getattr(brain, "trauma_map", []) or []
            behavior_dump = (
                f"avoid_hazard_radius          = {getattr(brules, 'avoid_hazard_radius', 0.0):.2f}\n"
                f"healing_zone_seek_priority    = {getattr(brules, 'healing_zone_seek_priority', 0.0):.2f}\n"
                f"stick_with_ally_if_fear_above = {getattr(brules, 'stick_with_ally_if_fear_above', 0.0):.2f}\n"
                f"exploration_bias              = {getattr(brules, 'exploration_bias', 0.0):.2f}\n"
                f"\n"
                f"current_drive       = {drive_txt}\n"
                f"ally_anchor         = {ally_anchor}\n"
                f"age_ticks           = {getattr(brain, 'age_ticks', 0)}\n"
                f"survival_score      = {getattr(brain, 'survival_score', 0.0):.2f}\n"
                f"fear_level          = {getattr(brain, 'fear_level', 0.0):.2f}\n"
                f"energy              = {energy_raw:.1f}\n"
                f"hunger              = {hunger_raw:.1f}\n"
                f"curiosity_charge    = {curiosity:.2f}\n"
                f"trauma_spots        = {len(trauma_map)}\n"
                f"pets_owned          = {len(pets_list)}\n"
            )
            if last_death_reason: behavior_dump += f"last_death_reason  = {last_death_reason}\n"
        self.behaviorText.setPlainText(behavior_dump if behavior_dump else "(no brain data)")

        self.memoryView.setHtml(self._memory_html_for_agent(ag, brain))

        self.belief2D.update_from_brain(brain)
        self.belief3D.update_from_brain(brain)


# =============================================================================
# Левая панель: статистика и управление
# =============================================================================

class TrainerStatsWidget(QtWidgets.QWidget):
    def __init__(self, trainer: MindTrainerInteractive, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.running = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(8)

        self.lblEpoch = QtWidgets.QLabel("Epoch: 0 | Tick: 0")
        self.lblEpoch.setStyleSheet("color:#eee; font-size:14px; font-weight:600;")
        outer.addWidget(self.lblEpoch)

        metricsFrame = QtWidgets.QFrame()
        metricsFrame.setStyleSheet("""
            QFrame { background-color:#1a1a23; border:1px solid #333; border-radius:8px; }
            QLabel { color:#ccc; font-size:12px; font-family:monospace; }
        """)
        grid = QtWidgets.QGridLayout(metricsFrame)
        grid.setContentsMargins(8,8,8,8); grid.setSpacing(4)

        self.lblAvgFear = QtWidgets.QLabel("avg_fear: 0.00")
        self.lblAvgHP = QtWidgets.QLabel("avg_hp: 0.0")
        self.lblAvgEnergy = QtWidgets.QLabel("avg_energy: 0.0")
        self.lblAvgHunger = QtWidgets.QLabel("avg_hunger: 0.0")
        self.lblAvgCuriosity = QtWidgets.QLabel("avg_curiosity: 0.00")
        self.lblAvgSurvival = QtWidgets.QLabel("avg_survival: 0.00")
        self.lblAvgAge = QtWidgets.QLabel("avg_age: 0.0")
        self.lblAvgTrauma = QtWidgets.QLabel("avg_trauma_spots: 0.0")
        self.lblAliveRatio = QtWidgets.QLabel("alive_ratio: 0.00")
        self.lblPanicRatio = QtWidgets.QLabel("panic_ratio: 0.00")
        self.lblClingRatio = QtWidgets.QLabel("cling_ratio: 0.00")
        self.lblAvgPets = QtWidgets.QLabel("avg_pets: 0.00")
        self.lblTameRatio = QtWidgets.QLabel("tame_ratio: 0.00")
        self.lblNote = QtWidgets.QLabel("note: —")

        row = 0
        for w in [
            self.lblAvgFear, self.lblAvgHP, self.lblAvgEnergy, self.lblAvgHunger,
            self.lblAvgCuriosity, self.lblAvgSurvival, self.lblAvgAge, self.lblAvgTrauma,
            self.lblAliveRatio, self.lblPanicRatio, self.lblClingRatio, self.lblAvgPets,
            self.lblTameRatio, self.lblNote,
        ]:
            grid.addWidget(w, row, 0); row += 1

        outer.addWidget(metricsFrame, 0)

        btnFrame = QtWidgets.QFrame()
        btnFrame.setStyleSheet("""
            QFrame { background-color:#15151d; border:1px solid #333; border-radius:8px; }
            QPushButton { background-color:#2a2a33; border:1px solid #444; border-radius:6px;
                          padding:6px 10px; color:#eee; font-weight:500; font-size:11px; }
            QPushButton:hover { background-color:#3a3a44; }
        """)
        btnLay = QtWidgets.QHBoxLayout(btnFrame)
        btnLay.setContentsMargins(8,8,8,8); btnLay.setSpacing(8)
        self.btnStartPause = QtWidgets.QPushButton("Start")
        self.btnNextEpoch = QtWidgets.QPushButton("Next Epoch")
        self.btnSaveBrains = QtWidgets.QPushButton("Save Brains Now")
        btnLay.addWidget(self.btnStartPause); btnLay.addWidget(self.btnNextEpoch); btnLay.addWidget(self.btnSaveBrains)
        outer.addWidget(btnFrame, 0); outer.addStretch(1)

        self.timer = QTimer(self); self.timer.setInterval(10); self.timer.timeout.connect(self._on_timer_tick)
        trainer.world_changed.connect(self.refresh_monitor); trainer.epoch_changed.connect(self.refresh_monitor)
        self.btnStartPause.clicked.connect(self.on_start_pause_clicked)
        self.btnNextEpoch.clicked.connect(self.on_next_epoch_clicked)
        self.btnSaveBrains.clicked.connect(self.on_save_clicked)
        self.refresh_monitor()

    @Slot()
    def on_start_pause_clicked(self):
        self.running = not self.running
        if self.running:
            self.btnStartPause.setText("Pause"); self.timer.start()
        else:
            self.btnStartPause.setText("Start"); self.timer.stop()

    @Slot()
    def on_next_epoch_clicked(self):
        self.trainer.force_next_epoch(); self.refresh_monitor()

    @Slot()
    def on_save_clicked(self):
        self.trainer.save_brains_now("./trained_brains")
        self.lblNote.setText("note: brains saved → ./trained_brains")

    @Slot()
    def _on_timer_tick(self):
        self.trainer.step_tick(); self.refresh_monitor()

    @Slot()
    def refresh_monitor(self):
        m = self.trainer.monitor
        self.lblEpoch.setText(f"Epoch: {m.epoch} | Tick: {m.tick}")
        self.lblAvgFear.setText(f"avg_fear: {m.avg_fear:.2f}")
        self.lblAvgHP.setText(f"avg_hp: {m.avg_hp:.1f}")
        self.lblAvgEnergy.setText(f"avg_energy: {m.avg_energy:.1f}")
        self.lblAvgHunger.setText(f"avg_hunger: {m.avg_hunger:.1f}")
        self.lblAvgCuriosity.setText(f"avg_curiosity: {m.avg_curiosity:.2f}")
        self.lblAvgSurvival.setText(f"avg_survival: {m.avg_survival:.2f}")
        self.lblAvgAge.setText(f"avg_age: {m.avg_age:.1f}")
        self.lblAvgTrauma.setText(f"avg_trauma_spots: {m.avg_trauma_spots:.1f}")
        self.lblAliveRatio.setText(f"alive_ratio: {m.alive_ratio:.2f}")
        self.lblPanicRatio.setText(f"panic_ratio: {m.panic_ratio:.2f}")
        self.lblClingRatio.setText(f"cling_ratio: {m.cling_ratio:.2f}")
        self.lblAvgPets.setText(f"avg_pets: {m.avg_pets:.2f}")
        self.lblTameRatio.setText(f"tame_ratio: {m.tame_ratio:.2f}")
        self.lblNote.setText(f"note: {m.note}")


# =============================================================================
# Главное окно
# =============================================================================

class TrainerMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mind Trainer Lab — эволюция сознаний и зверей")
        self.resize(1500, 900)
        self.setStyleSheet("QMainWindow { background-color:#0f0f14; color:#eee; } QLabel { color:#eee; }")

        self.trainer = MindTrainerInteractive(
            num_agents=3, max_ticks_per_epoch=2000, seed=1234,
            disaster_interval_ticks=400, relief_after_disaster=80,
            fresh_start=False,
            agent_lineup=[
                {"id": "a1", "name": "Echo", "persona":
                 "Ты Echo. Осторожный, тревожный выживальщик. Тебе страшно умереть, ты переживаешь за Nova. Ты стараешься держать всех в безопасности и предупреждать об угрозах."},
                {"id": "a2", "name": "Nova", "persona":
                 "Ты Nova. Смелая исследовательница. Ты любишь разведку, но не хочешь, чтобы Echo пострадал. Ты звучишь уверенно и тёпло, даже когда больно."},
                {"id": "agent_0", "name": "A0", "persona": "caring/supportive"},
            ],
        )

        self.statsWidget = TrainerStatsWidget(self.trainer)
        self.statsWidget.setMinimumWidth(380); self.statsWidget.setMaximumWidth(420)
        self.brainWidget = AgentBrainWidget(self.trainer)

        central = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 12); outer.setSpacing(12)
        outer.addWidget(self.statsWidget, 0)
        outer.addWidget(self.brainWidget, 1)
        self.setCentralWidget(central)
        self.statusBar().setStyleSheet("color:#bbb; background-color:#1a1a23;")
        self.statusBar().showMessage("Ready")


# =============================================================================
# main()
# =============================================================================

def main():
    app = QtWidgets.QApplication([])
    win = TrainerMainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
