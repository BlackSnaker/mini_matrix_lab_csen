from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import math
import random
import sys

from animals import Animal as AnimalSim, AnimalSpecies
from brain_io import export_room_brain
from ollama_coach import (
    CoachAdvice,
    OllamaCoach,
    apply_advice_to_agent,
    apply_goal_feedback,
    build_training_snapshot,
    infer_operator_goal,
)
from lab_subject_selector import (
    format_candidate_summary,
    load_candidate_brain,
    make_lab_subject_lineup_spec,
    select_best_trained_agent,
)
from world import Agent, WorldObject

if __name__ == "__main__":
    sys.modules.setdefault("training_room", sys.modules[__name__])


LAB_AGENT_TAG = "lab_subject"
TRAINING_ROOM_TAG = "training_room"
MORPHEUS_MENTOR_TAG = "morpheus_mentor"
ROOM_BRAINS_DIR = "room_brains"
DEFAULT_ROOM_AGENT_ID = "agent_1"
DEFAULT_MORPHEUS_AGENT_ID = "morpheus_mentor"
DEFAULT_TRAINING_WOLF_ID = "training_wolf"
LESSON_IDLE = "idle"
LESSON_SPARRING = "sparring"
LESSON_WOLF = "wolf_drill"


def _room_only_agent_lineup(agent_id: Optional[str]) -> list[dict[str, str]]:
    target_id = str(agent_id or DEFAULT_ROOM_AGENT_ID).strip() or DEFAULT_ROOM_AGENT_ID
    primary_spec = {
        "id": target_id,
        "name": "Agent 1" if target_id == "agent_1" else target_id,
        "persona": (
            "Ты Agent 1. Первый агент лаборатории CSEN. "
            "Комната Морфеуса для тебя безопасная учебная среда. "
            "Ты внимательно воспринимаешь указания оператора и используешь их для обучения."
        ),
    }
    fallback_specs = [
        {"id": "a2", "name": "Nova", "persona": "Ты Nova. Смелая исследовательница. Действуешь решительно, но не рискуешь зря."},
        {"id": "agent_0", "name": "A0", "persona": "scout/explorer"},
    ]
    lineup = [primary_spec]
    for spec in fallback_specs:
        if str(spec.get("id", "")) != target_id:
            lineup.append(spec)
    return lineup


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _iter_agents(world: Any) -> Iterable[Any]:
    agents = getattr(world, "agents", {}) or {}
    if isinstance(agents, dict):
        return list(agents.values())
    if isinstance(agents, (list, tuple, set)):
        return list(agents)
    return []


def _iter_animals(world: Any) -> Iterable[Any]:
    animals = getattr(world, "animals", {}) or {}
    if isinstance(animals, dict):
        return list(animals.values())
    if isinstance(animals, (list, tuple, set)):
        return list(animals)
    return []


def _is_alive_agent(agent: Any) -> bool:
    if agent is None:
        return False
    try:
        alive_fn = getattr(agent, "is_alive", None)
        if callable(alive_fn):
            return bool(alive_fn())
    except Exception:
        pass
    try:
        return bool(getattr(agent, "alive", True)) and float(getattr(agent, "health", 1.0)) > 0.0
    except Exception:
        return True


def _is_alive_animal(animal: Any) -> bool:
    if animal is None:
        return False
    try:
        alive_fn = getattr(animal, "is_alive", None)
        if callable(alive_fn):
            return bool(alive_fn())
    except Exception:
        pass
    try:
        return float(getattr(animal, "hp", 1.0)) > 0.0
    except Exception:
        return True


def _pct_add(value: Any, delta_pct: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    if raw <= 1.5:
        return _clamp(raw + float(delta_pct) / 100.0, 0.0, 1.0)
    return _clamp(raw + float(delta_pct), 0.0, 100.0)


def _pct_sub(value: Any, delta_pct: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    if raw <= 1.5:
        return _clamp(raw - float(delta_pct) / 100.0, 0.0, 1.0)
    return _clamp(raw - float(delta_pct), 0.0, 100.0)


@dataclass(frozen=True)
class RoomBounds:
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

    def contains_point(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        return (
            float(x) >= self.left - margin
            and float(x) <= self.right + margin
            and float(y) >= self.top - margin
            and float(y) <= self.bottom + margin
        )

    def contains_circle(self, x: float, y: float, radius: float, *, margin: float = 0.0) -> bool:
        return self.contains_point(float(x), float(y), margin=float(radius) + float(margin))

    def clamp_point(self, x: float, y: float, *, padding: float = 1.5) -> Tuple[float, float]:
        px = _clamp(float(x), self.left + padding, self.right - padding)
        py = _clamp(float(y), self.top + padding, self.bottom - padding)
        return (px, py)


class TrainingRoomManager:
    """
    Держит один безопасный учебный отсек внутри общего мира.
    Агент с тегом LAB_AGENT_TAG остаётся выделенным "лабораторным" агентом,
    а TRAINING_ROOM_TAG означает, что он прямо сейчас заперт в комнате.
    """

    def __init__(
        self,
        *,
        room_id: str = "training_room",
        label: str = "Учебная_комната",
        seed: int = 20260404,
        room_brains_dir: str = ROOM_BRAINS_DIR,
        autosave_ticks: int = 180,
    ) -> None:
        self.room_id = str(room_id)
        self.label = str(label)
        self.object_id = f"{self.room_id}_zone"
        self.agent_id: Optional[str] = None
        self.released: bool = False
        self._rng = random.Random(int(seed))
        self.room_brains_dir = str(room_brains_dir or ROOM_BRAINS_DIR)
        self.autosave_ticks = max(1, int(autosave_ticks))
        self._last_room_brain_save_tick: int = -10**9
        self._last_room_brain_path: Optional[str] = None
        self.combat_lessons_enabled: bool = False
        self.lesson_mode: str = LESSON_IDLE
        self.mentor_agent_id: str = DEFAULT_MORPHEUS_AGENT_ID
        self.training_wolf_id: str = DEFAULT_TRAINING_WOLF_ID
        self._last_lesson_tick: int = -10**9
        self._wolf_taming_prev_allow: Optional[bool] = None

    def is_lab_agent(self, agent_id: Optional[str]) -> bool:
        return bool(agent_id) and str(agent_id) == str(self.agent_id or "")

    def is_mentor_agent(self, agent_id: Optional[str]) -> bool:
        return bool(agent_id) and str(agent_id) == str(self.mentor_agent_id or "")

    def is_confined(self, agent_id: Optional[str]) -> bool:
        return self.is_lab_agent(agent_id) and not self.released

    def bounds_for(self, world: Any) -> RoomBounds:
        world_w = max(24.0, float(getattr(world, "width", 100.0)))
        world_h = max(24.0, float(getattr(world, "height", 100.0)))
        room_w = _clamp(world_w * 0.16, 22.0, 30.0)
        room_h = _clamp(world_h * 0.16, 22.0, 30.0)
        left = _clamp(world_w * 0.08, 4.0, max(4.0, world_w - room_w - 4.0))
        top = _clamp(world_h * 0.08, 4.0, max(4.0, world_h - room_h - 4.0))
        return RoomBounds(
            left=float(left),
            top=float(top),
            right=float(min(world_w - 4.0, left + room_w)),
            bottom=float(min(world_h - 4.0, top + room_h)),
        )

    def room_center(self, world: Any) -> Tuple[float, float]:
        return self.bounds_for(world).center

    def last_room_brain_path(self) -> Optional[str]:
        return self._last_room_brain_path

    def enable_combat_lessons(self, enabled: bool = True) -> None:
        self.combat_lessons_enabled = bool(enabled)
        if not self.combat_lessons_enabled:
            self.lesson_mode = LESSON_IDLE

    def start_sparring(self, world: Any, *, announce: bool = True) -> str:
        self.enable_combat_lessons(True)
        self.lesson_mode = LESSON_SPARRING
        self._ensure_combat_mentor(world, announce=announce)
        self._remove_training_wolf(world)
        self._position_combat_entities(world, sparring=True)
        if announce:
            self._announce(world, "[dojo] Морфеус начинает спарринг-урок. Учись держать дистанцию и отвечать на удар.")
        return self.lesson_mode

    def start_wolf_drill(self, world: Any, *, announce: bool = True) -> str:
        self.enable_combat_lessons(True)
        self.lesson_mode = LESSON_WOLF
        self._ensure_combat_mentor(world, announce=False)
        self._ensure_training_wolf(world, announce=announce)
        self._position_combat_entities(world, sparring=False)
        if announce:
            self._announce(world, "[dojo] Морфеус выпускает тренировочного волка. Покажи ответ на атаку и удержи бой.")
        return self.lesson_mode

    def stop_combat_lesson(self, world: Any, *, announce: bool = True) -> str:
        self.lesson_mode = LESSON_IDLE
        self._remove_training_wolf(world)
        self._position_combat_entities(world, sparring=False, reset_only=True)
        self._apply_wolf_taming_policy(world)
        if announce:
            self._announce(world, "[dojo] Боевой урок остановлен. Комната вернулась в спокойный режим.")
        return self.lesson_mode

    def combat_status(self, world: Any) -> Dict[str, Any]:
        lab = self._get_agent(world, self.agent_id)
        mentor = self._get_agent(world, self.mentor_agent_id)
        wolf = self._get_training_wolf(world)
        return {
            "enabled": bool(self.combat_lessons_enabled),
            "lesson_mode": str(self.lesson_mode or LESSON_IDLE),
            "lab_agent_id": str(self.agent_id or ""),
            "mentor_present": bool(mentor is not None and _is_alive_agent(mentor)),
            "wolf_present": bool(wolf is not None and _is_alive_animal(wolf)),
            "lab_combat_skill": round(float(getattr(lab, "combat_skill", 0.0) or 0.0), 3) if lab is not None else 0.0,
            "lab_health": round(float(getattr(lab, "health", 0.0) or 0.0), 1) if lab is not None else 0.0,
            "mentor_health": round(float(getattr(mentor, "health", 0.0) or 0.0), 1) if mentor is not None else 0.0,
            "wolf_hp": round(float(getattr(wolf, "hp", 0.0) or 0.0), 1) if wolf is not None else 0.0,
        }

    def release_point(self, world: Any) -> Tuple[float, float]:
        bounds = self.bounds_for(world)
        world_w = max(8.0, float(getattr(world, "width", 100.0)))
        world_h = max(8.0, float(getattr(world, "height", 100.0)))
        x = _clamp(bounds.right + 9.0, 4.0, world_w - 4.0)
        y = _clamp(bounds.top + bounds.height * 0.55, 4.0, world_h - 4.0)
        return (x, y)

    def clamp_point_for_agent(self, world: Any, agent_id: Optional[str], x: float, y: float) -> Tuple[float, float]:
        if not self.is_confined(agent_id):
            return (float(x), float(y))
        return self.bounds_for(world).clamp_point(float(x), float(y))

    def attach_world(
        self,
        world: Any,
        *,
        preferred_agent_id: Optional[str] = None,
        announce: bool = False,
    ) -> Optional[str]:
        if world is None:
            return None
        if preferred_agent_id:
            self.agent_id = str(preferred_agent_id)
        elif self.agent_id and self._get_agent(world, self.agent_id) is None:
            self.agent_id = None
        if self.agent_id is None:
            picked = self._pick_default_agent_id(world)
            if picked:
                self.agent_id = picked
        self._ensure_room_object(world)
        self._sync_tags(world)
        if self.agent_id and not self.released:
            self._place_agent_in_room(world, self.agent_id, announce=announce)
        self.maintain_world(world)
        self._save_room_brain(world, reason="room_attach", force=True)
        return self.agent_id

    def assign_agent(self, world: Any, agent_id: Optional[str], *, announce: bool = True) -> Optional[str]:
        if world is None:
            return None
        picked = str(agent_id or self._pick_default_agent_id(world) or "")
        if not picked:
            return None
        self.agent_id = picked
        self.released = False
        self._sync_tags(world)
        self._place_agent_in_room(world, picked, announce=announce)
        self.maintain_world(world)
        self._save_room_brain(world, reason="room_assign", force=True)
        return picked

    def release_agent(self, world: Any, *, announce: bool = True) -> Optional[str]:
        if world is None or not self.agent_id:
            return None
        self.stop_combat_lesson(world, announce=False)
        agent = self._get_agent(world, self.agent_id)
        self.released = True
        self._sync_tags(world)
        if agent is not None:
            rx, ry = self.release_point(world)
            self._move_entity(agent, rx, ry, set_goal=True)
            self._save_room_brain(world, reason="room_release", force=True)
            if hasattr(world, "add_chat_line") and announce:
                try:
                    world.add_chat_line(f"[lab] {getattr(agent, 'name', self.agent_id)} выпущен(а) из учебной комнаты")
                    if self._last_room_brain_path:
                        world.add_chat_line(f"[lab] room brain saved: {self._last_room_brain_path}")
                except Exception:
                    pass
            if hasattr(world, "add_event") and announce:
                try:
                    world.add_event({
                        "type": "training_room_release",
                        "who": getattr(agent, "id", self.agent_id),
                        "name": getattr(agent, "name", self.agent_id),
                    })
                except Exception:
                    pass
        return self.agent_id

    def maintain_world(self, world: Any) -> None:
        if world is None:
            return
        self._ensure_room_object(world)
        if self.combat_lessons_enabled:
            self._ensure_combat_mentor(world, announce=False)
        self._sync_tags(world)
        self._expel_hazards(world)
        self._expel_animals(world)
        self._keep_other_agents_out(world)
        self._apply_wolf_taming_policy(world)
        self._stabilize_room_agent(world)
        self._maintain_combat_lesson(world)
        self._save_room_brain(world, reason="room_autosave", force=False)

    def _pick_default_agent_id(self, world: Any) -> Optional[str]:
        for agent in _iter_agents(world):
            if agent is None:
                continue
            agent_id = getattr(agent, "id", getattr(agent, "agent_id", None))
            if agent_id and str(agent_id) != str(self.mentor_agent_id):
                return str(agent_id)
        return None

    def _get_agent(self, world: Any, agent_id: Optional[str]) -> Optional[Any]:
        if world is None or not agent_id:
            return None
        if hasattr(world, "get_agent_by_id"):
            try:
                agent = world.get_agent_by_id(agent_id)
                if agent is not None:
                    return agent
            except Exception:
                pass
        for agent in _iter_agents(world):
            if agent is None:
                continue
            if str(getattr(agent, "id", getattr(agent, "agent_id", ""))) == str(agent_id):
                return agent
        return None

    def _get_training_wolf(self, world: Any) -> Optional[Any]:
        if world is None:
            return None
        animals = getattr(world, "animals", {}) or {}
        if isinstance(animals, dict):
            wolf = animals.get(self.training_wolf_id)
            if wolf is not None:
                return wolf
        for animal in _iter_animals(world):
            if animal is None:
                continue
            if str(getattr(animal, "uid", "")) == str(self.training_wolf_id):
                return animal
        return None

    def _announce(self, world: Any, text: str) -> None:
        if world is None:
            return
        if hasattr(world, "add_chat_line"):
            try:
                world.add_chat_line(str(text))
            except Exception:
                pass

    def _combat_ring_positions(self, world: Any) -> Dict[str, Tuple[float, float]]:
        bounds = self.bounds_for(world)
        cx, cy = bounds.center
        spar_offset = _clamp(bounds.width * 0.09, 1.3, 2.1)
        wolf_offset = _clamp(bounds.width * 0.15, 2.2, 3.6)
        mentor_idle_offset = _clamp(bounds.width * 0.22, 3.0, 5.2)
        return {
            "lab": (cx - spar_offset, cy + 0.20),
            "mentor": (cx + spar_offset, cy - 0.20),
            "wolf": (cx + wolf_offset * 0.20, cy + wolf_offset),
            "mentor_idle": (cx + mentor_idle_offset, cy - max(2.4, bounds.height * 0.18)),
        }

    def _ensure_combat_mentor(self, world: Any, *, announce: bool) -> Optional[Any]:
        if world is None or not self.combat_lessons_enabled:
            return None
        mentor = self._get_agent(world, self.mentor_agent_id)
        if mentor is None:
            spawn = self._combat_ring_positions(world)["mentor_idle"]
            mentor = Agent(
                agent_id=self.mentor_agent_id,
                name="Morpheus",
                x=float(spawn[0]),
                y=float(spawn[1]),
                goal_x=float(spawn[0]),
                goal_y=float(spawn[1]),
                persona=(
                    "Ты Морфеус. Хладнокровный боевой наставник. "
                    "Ты учишь агента держать стойку, отвечать на удар и не паниковать перед волком."
                ),
            )
            try:
                mentor.attack_power = 6.0
                mentor.attack_range = max(1.7, float(getattr(mentor, "attack_range", 1.6)))
                mentor.attack_cooldown = max(8, min(14, int(getattr(mentor, "attack_cooldown", 12))))
                mentor.combat_skill = max(4.0, float(getattr(mentor, "combat_skill", 0.0)))
                mentor.fear = 0.02
                mentor.energy = _pct_add(getattr(mentor, "energy", 100.0), 0.0)
            except Exception:
                pass
            try:
                world.add_agent(mentor)
            except Exception:
                agents = getattr(world, "agents", None)
                if isinstance(agents, dict):
                    agents[self.mentor_agent_id] = mentor
            if announce:
                self._announce(world, "[dojo] Морфеус вошёл в комнату. Начинаем урок боя.")
        return mentor

    def _remove_training_wolf(self, world: Any) -> None:
        if world is None:
            return
        animals = getattr(world, "animals", None)
        if isinstance(animals, dict):
            animals.pop(self.training_wolf_id, None)
            return
        if isinstance(animals, list):
            world.animals = [a for a in animals if str(getattr(a, "uid", "")) != str(self.training_wolf_id)]

    def _ensure_training_wolf(self, world: Any, *, announce: bool) -> Optional[Any]:
        if world is None or not self.combat_lessons_enabled:
            return None
        wolf = self._get_training_wolf(world)
        if wolf is None:
            pos = self._combat_ring_positions(world)["wolf"]
            species = AnimalSpecies(
                "wolf",
                "Тренировочный_волк",
                42.0,
                True,
                False,
                1.0,
                5.0,
                6.0,
                2.0,
                10.0,
            )
            wolf = AnimalSim(uid=self.training_wolf_id, species=species, x=float(pos[0]), y=float(pos[1]))
            try:
                wolf.vx = 0.0
                wolf.vy = 0.0
                wolf.last_action = "await_drill"
            except Exception:
                pass
            try:
                world.add_animal(wolf)
            except Exception:
                animals = getattr(world, "animals", None)
                if isinstance(animals, dict):
                    animals[self.training_wolf_id] = wolf
            if announce:
                self._announce(world, "[dojo] В комнату впущен тренировочный волк.")
        return wolf

    def _position_combat_entities(self, world: Any, *, sparring: bool, reset_only: bool = False) -> None:
        if world is None:
            return
        spots = self._combat_ring_positions(world)
        lab = self._get_agent(world, self.agent_id)
        mentor = self._get_agent(world, self.mentor_agent_id)
        wolf = self._get_training_wolf(world)
        if lab is not None and not reset_only:
            self._move_entity(lab, *spots["lab"], set_goal=True)
        if mentor is not None:
            key = "mentor" if sparring and not reset_only else "mentor_idle"
            self._move_entity(mentor, *spots[key], set_goal=True)
        if wolf is not None and not reset_only:
            self._move_entity(wolf, *spots["wolf"], set_goal=False)

    def _mentor_influence(self, world: Any, lab: Any, mentor: Any) -> None:
        tick = self._world_tick(world)
        if mentor is None or lab is None:
            return
        lx, ly = float(getattr(lab, "x", 0.0)), float(getattr(lab, "y", 0.0))
        mx, my = float(getattr(mentor, "x", 0.0)), float(getattr(mentor, "y", 0.0))
        dx, dy = mx - lx, my - ly
        dist = math.hypot(dx, dy)
        if dist > 2.2:
            self._move_entity(mentor, lx - dx / max(dist, 1e-6) * 1.2, ly - dy / max(dist, 1e-6) * 1.2, set_goal=True)
            try:
                lab.set_goal(mx, my, world_size=(float(getattr(world, "width", 0.0)), float(getattr(world, "height", 0.0))), reason="dojo_spar", tick=tick)
            except Exception:
                pass

    def _combat_feedback(self, agent: Any, *, hp_delta: float, dist_delta: float, done: bool = False) -> None:
        brain = getattr(agent, "brain", None)
        if brain is not None and hasattr(brain, "note_combat_feedback"):
            try:
                brain.note_combat_feedback(hp_delta=float(hp_delta), dist_delta=float(dist_delta), done=bool(done))
            except Exception:
                pass

    def _apply_training_hit(self, world: Any, attacker: Any, defender: Any, *, damage: float, lesson: str) -> bool:
        if attacker is None or defender is None:
            return False
        now = self._world_tick(world)
        can_attack = getattr(attacker, "can_attack", None)
        if callable(can_attack):
            try:
                if not bool(can_attack(now)):
                    return False
            except Exception:
                pass
        before_hp = float(getattr(defender, "health", 100.0))
        dmg = float(max(0.5, damage))
        hp_floor = 22.0 if self.lesson_mode == LESSON_SPARRING else 30.0
        max_dmg = max(0.0, before_hp - hp_floor)
        if max_dmg <= 0.0:
            return False
        dmg = min(dmg, max_dmg)
        if dmg <= 0.0:
            return False
        try:
            defender.receive_damage(dmg, attacker_id=str(getattr(attacker, "id", getattr(attacker, "uid", "dojo"))), world_tick=now)
        except Exception:
            try:
                defender.health = max(hp_floor, before_hp - dmg)
            except Exception:
                return False
        mark_attack = getattr(attacker, "mark_attack", None)
        if callable(mark_attack):
            try:
                mark_attack(now)
            except Exception:
                pass
        try:
            attacker.combat_skill = min(5.0, float(getattr(attacker, "combat_skill", 0.0)) + 0.03)
        except Exception:
            pass
        self._combat_feedback(attacker, hp_delta=0.0, dist_delta=-0.6, done=False)
        self._combat_feedback(defender, hp_delta=-dmg, dist_delta=0.0, done=False)
        if hasattr(world, "add_event"):
            try:
                world.add_event({
                    "type": "dojo_hit",
                    "lesson": lesson,
                    "tick": now,
                    "who": getattr(attacker, "name", getattr(attacker, "id", "mentor")),
                    "target": getattr(defender, "name", getattr(defender, "id", "agent")),
                    "damage": round(dmg, 2),
                })
            except Exception:
                pass
        return True

    def _run_sparring_lesson(self, world: Any, lab: Any, mentor: Any) -> None:
        if lab is None or mentor is None:
            return
        self._mentor_influence(world, lab, mentor)
        lx, ly = float(getattr(lab, "x", 0.0)), float(getattr(lab, "y", 0.0))
        mx, my = float(getattr(mentor, "x", 0.0)), float(getattr(mentor, "y", 0.0))
        dist = math.hypot(mx - lx, my - ly)
        if dist > 2.4:
            self._position_combat_entities(world, sparring=True)
            lx, ly = float(getattr(lab, "x", 0.0)), float(getattr(lab, "y", 0.0))
            mx, my = float(getattr(mentor, "x", 0.0)), float(getattr(mentor, "y", 0.0))
            dist = math.hypot(mx - lx, my - ly)
        if dist > 1.9:
            return
        mentor_hit = self._apply_training_hit(world, mentor, lab, damage=3.6, lesson=LESSON_SPARRING)
        if mentor_hit:
            try:
                lab.combat_skill = min(5.0, float(getattr(lab, "combat_skill", 0.0)) + 0.025)
            except Exception:
                pass
        confidence = max(0.0, min(1.0, float(getattr(lab, "combat_skill", 0.0)) / 5.0))
        if confidence >= 0.10:
            hit_back = self._apply_training_hit(world, lab, mentor, damage=2.8 + confidence * 1.8, lesson=LESSON_SPARRING)
            if hit_back:
                try:
                    lab.fear = _clamp(float(getattr(lab, "fear", 0.0)) - 0.035, 0.0, 1.0)
                except Exception:
                    pass

    def _run_wolf_lesson(self, world: Any, lab: Any, mentor: Any) -> None:
        wolf = self._ensure_training_wolf(world, announce=False)
        if lab is None or wolf is None:
            return
        if mentor is not None:
            self._move_entity(mentor, *self._combat_ring_positions(world)["mentor_idle"], set_goal=True)
        try:
            wolf.aggression_target = str(getattr(lab, "id", self.agent_id))
            wolf.last_action = f"drill_target:{getattr(lab, 'id', self.agent_id)}"
        except Exception:
            pass
        wx = float(getattr(wolf, "x", 0.0))
        wy = float(getattr(wolf, "y", 0.0))
        try:
            lab.set_goal(wx, wy, world_size=(float(getattr(world, "width", 0.0)), float(getattr(world, "height", 0.0))), reason="wolf_drill", tick=self._world_tick(world))
        except Exception:
            pass
        if not _is_alive_animal(wolf):
            self._announce(world, "[dojo] Волк повержен. Морфеус запускает новый заход.")
            self._remove_training_wolf(world)
            self._ensure_training_wolf(world, announce=False)

    def _stabilize_combat_actor(self, actor: Any, *, min_health: float, target_fear: float) -> None:
        if actor is None:
            return
        try:
            actor.health = max(float(min_health), float(getattr(actor, "health", 100.0)))
        except Exception:
            pass
        try:
            actor.energy = _pct_add(getattr(actor, "energy", 100.0), 0.35)
        except Exception:
            pass
        try:
            actor.fear = _clamp(min(float(getattr(actor, "fear", 0.0)), float(target_fear)), 0.0, 1.0)
        except Exception:
            pass

    def _apply_wolf_taming_policy(self, world: Any) -> None:
        if world is None:
            return
        taming = getattr(world, "wolf_taming", None)
        if taming is None or not hasattr(taming, "allow_tame_wolves"):
            return
        if self.combat_lessons_enabled and self.lesson_mode == LESSON_WOLF and not self.released:
            if self._wolf_taming_prev_allow is None:
                try:
                    self._wolf_taming_prev_allow = bool(getattr(taming, "allow_tame_wolves", True))
                except Exception:
                    self._wolf_taming_prev_allow = True
            try:
                taming.allow_tame_wolves = False
            except Exception:
                pass
            return
        if self._wolf_taming_prev_allow is not None:
            try:
                taming.allow_tame_wolves = bool(self._wolf_taming_prev_allow)
            except Exception:
                pass
            self._wolf_taming_prev_allow = None

    def _maintain_combat_lesson(self, world: Any) -> None:
        if world is None or not self.combat_lessons_enabled or self.released:
            return
        tick = self._world_tick(world)
        if tick == self._last_lesson_tick:
            return
        self._last_lesson_tick = tick

        mentor = self._ensure_combat_mentor(world, announce=False)
        lab = self._get_agent(world, self.agent_id)
        if lab is None or mentor is None:
            return

        self._stabilize_combat_actor(lab, min_health=34.0, target_fear=0.42 if self.lesson_mode == LESSON_WOLF else 0.32)
        self._stabilize_combat_actor(mentor, min_health=48.0, target_fear=0.12)
        if self.lesson_mode == LESSON_SPARRING:
            self._remove_training_wolf(world)
            self._run_sparring_lesson(world, lab, mentor)
        elif self.lesson_mode == LESSON_WOLF:
            self._run_wolf_lesson(world, lab, mentor)

    def _set_tags(self, agent: Any, *, is_lab: bool, confined: bool) -> None:
        tags = [
            str(t)
            for t in list(getattr(agent, "tags", []) or [])
            if str(t) not in (LAB_AGENT_TAG, TRAINING_ROOM_TAG, MORPHEUS_MENTOR_TAG)
        ]
        if is_lab:
            tags.append(LAB_AGENT_TAG)
        if self.is_mentor_agent(getattr(agent, "id", getattr(agent, "agent_id", None))):
            tags.append(MORPHEUS_MENTOR_TAG)
        if confined:
            tags.append(TRAINING_ROOM_TAG)
        try:
            agent.tags = tags
        except Exception:
            pass

    def _sync_tags(self, world: Any) -> None:
        for agent in _iter_agents(world):
            if agent is None:
                continue
            agent_id = str(getattr(agent, "id", getattr(agent, "agent_id", "")) or "")
            self._set_tags(
                agent,
                is_lab=(agent_id == str(self.agent_id or "")),
                confined=(agent_id == str(self.agent_id or "") and not self.released),
            )

    def _ensure_room_object(self, world: Any) -> None:
        bounds = self.bounds_for(world)
        cx, cy = bounds.center
        room_obj = None
        deduped = []
        for obj in list(getattr(world, "objects", []) or []):
            if str(getattr(obj, "obj_id", "")) == self.object_id:
                if room_obj is None:
                    room_obj = obj
                    deduped.append(obj)
                continue
            deduped.append(obj)
        if hasattr(world, "objects"):
            try:
                world.objects = deduped
            except Exception:
                pass
        if room_obj is None:
            room_obj = WorldObject(
                obj_id=self.object_id,
                name=self.label,
                kind="safe",
                x=cx,
                y=cy,
                radius=bounds.safe_radius,
                danger_level=0.0,
                comfort_level=1.0,
            )
            if hasattr(world, "add_object"):
                world.add_object(room_obj)
        else:
            try:
                room_obj.name = self.label
                room_obj.kind = "safe"
                room_obj.x = cx
                room_obj.y = cy
                room_obj.radius = bounds.safe_radius
                room_obj.danger_level = 0.0
                room_obj.comfort_level = 1.0
            except Exception:
                pass

    def _random_point_in_room(self, world: Any) -> Tuple[float, float]:
        bounds = self.bounds_for(world)
        return (
            self._rng.uniform(bounds.left + 2.2, bounds.right - 2.2),
            self._rng.uniform(bounds.top + 2.2, bounds.bottom - 2.2),
        )

    def _place_agent_in_room(self, world: Any, agent_id: str, *, announce: bool) -> None:
        agent = self._get_agent(world, agent_id)
        if agent is None:
            return
        rx, ry = self._random_point_in_room(world)
        self._move_entity(agent, rx, ry, set_goal=True)
        try:
            agent.last_attacker_id = None
        except Exception:
            pass
        if hasattr(world, "add_chat_line") and announce:
            try:
                world.add_chat_line(f"[lab] {getattr(agent, 'name', agent_id)} переведён(а) в учебную комнату")
            except Exception:
                pass
        if hasattr(world, "add_event") and announce:
            try:
                world.add_event({
                    "type": "training_room_assign",
                    "who": getattr(agent, "id", agent_id),
                    "name": getattr(agent, "name", agent_id),
                })
            except Exception:
                pass

    def _move_entity(self, ent: Any, x: float, y: float, *, set_goal: bool) -> None:
        old_x = float(getattr(ent, "x", x))
        old_y = float(getattr(ent, "y", y))
        try:
            ent.x = float(x)
            ent.y = float(y)
        except Exception:
            pass
        if set_goal:
            try:
                ent.goal_x = float(x)
                ent.goal_y = float(y)
            except Exception:
                pass
        try:
            ent.vx = 0.0
            ent.vy = 0.0
        except Exception:
            pass
        apply_manual = getattr(ent, "apply_manual_control", None)
        if callable(apply_manual):
            try:
                apply_manual(
                    float(x),
                    float(y),
                    dt=max(1e-3, math.hypot(float(x) - old_x, float(y) - old_y)),
                    world_size=None,
                    facing=(0.0, 1.0),
                    tick=None,
                    hold_ticks=2,
                    source="training_room",
                )
            except Exception:
                pass

    def _outside_point(self, world: Any, slot: int) -> Tuple[float, float]:
        bounds = self.bounds_for(world)
        world_w = max(8.0, float(getattr(world, "width", 100.0)))
        world_h = max(8.0, float(getattr(world, "height", 100.0)))
        cx, cy = bounds.center
        ring_r = max(bounds.width, bounds.height) * 0.72 + 8.0 + (slot // 6) * 3.0
        angle_deg = 18.0 + (slot % 6) * 56.0
        ang = math.radians(angle_deg)
        px = _clamp(cx + math.cos(ang) * ring_r, 4.0, world_w - 4.0)
        py = _clamp(cy + math.sin(ang) * ring_r, 4.0, world_h - 4.0)
        if bounds.contains_point(px, py, margin=2.0):
            px = _clamp(bounds.right + 8.0, 4.0, world_w - 4.0)
            py = _clamp(bounds.bottom + 8.0, 4.0, world_h - 4.0)
        return (px, py)

    def _expel_hazards(self, world: Any) -> None:
        bounds = self.bounds_for(world)
        slot = 0
        for obj in list(getattr(world, "objects", []) or []):
            if obj is None or str(getattr(obj, "obj_id", "")) == self.object_id:
                continue
            if str(getattr(obj, "kind", "")).lower() != "hazard":
                continue
            ox = float(getattr(obj, "x", 0.0))
            oy = float(getattr(obj, "y", 0.0))
            radius = float(getattr(obj, "radius", 0.0))
            if not bounds.contains_circle(ox, oy, radius, margin=4.0):
                continue
            nx, ny = self._outside_point(world, slot)
            slot += 1
            try:
                obj.x = nx
                obj.y = ny
            except Exception:
                pass

    def _expel_animals(self, world: Any) -> None:
        bounds = self.bounds_for(world)
        slot = 10
        for ani in _iter_animals(world):
            if ani is None:
                continue
            if self.lesson_mode == LESSON_WOLF and str(getattr(ani, "uid", "")) == str(self.training_wolf_id):
                continue
            ax = float(getattr(ani, "x", 0.0))
            ay = float(getattr(ani, "y", 0.0))
            if not bounds.contains_point(ax, ay, margin=2.5):
                continue
            nx, ny = self._outside_point(world, slot)
            slot += 1
            self._move_entity(ani, nx, ny, set_goal=False)
            try:
                ani.atk_cd = max(float(getattr(ani, "atk_cd", 0.0)), 1.0)
            except Exception:
                pass

    def _keep_other_agents_out(self, world: Any) -> None:
        bounds = self.bounds_for(world)
        slot = 30
        for agent in _iter_agents(world):
            if agent is None:
                continue
            agent_id = str(getattr(agent, "id", getattr(agent, "agent_id", "")) or "")
            if self.is_confined(agent_id):
                continue
            if self.is_mentor_agent(agent_id):
                continue
            ax = float(getattr(agent, "x", 0.0))
            ay = float(getattr(agent, "y", 0.0))
            if not bounds.contains_point(ax, ay, margin=1.5):
                continue
            nx, ny = self._outside_point(world, slot)
            slot += 1
            self._move_entity(agent, nx, ny, set_goal=True)

    def _stabilize_room_agent(self, world: Any) -> None:
        if not self.agent_id or self.released:
            return
        agent = self._get_agent(world, self.agent_id)
        if agent is None:
            return
        bounds = self.bounds_for(world)
        ax = float(getattr(agent, "x", bounds.center[0]))
        ay = float(getattr(agent, "y", bounds.center[1]))
        gx = float(getattr(agent, "goal_x", ax))
        gy = float(getattr(agent, "goal_y", ay))
        if not bounds.contains_point(ax, ay):
            nx, ny = bounds.clamp_point(ax, ay)
            self._move_entity(agent, nx, ny, set_goal=True)
        elif not bounds.contains_point(gx, gy):
            rx, ry = self._random_point_in_room(world)
            try:
                agent.goal_x = rx
                agent.goal_y = ry
            except Exception:
                pass
        try:
            heal_rate = 0.04 if self.lesson_mode in (LESSON_SPARRING, LESSON_WOLF) else 0.08
            agent.health = _clamp(float(getattr(agent, "health", 100.0)) + heal_rate, 0.0, 100.0)
        except Exception:
            pass
        try:
            agent.energy = _pct_add(getattr(agent, "energy", 100.0), 0.6)
        except Exception:
            pass
        try:
            agent.hunger = _pct_sub(getattr(agent, "hunger", 0.0), 0.8)
        except Exception:
            pass
        try:
            fear_delta = 0.004 if self.lesson_mode in (LESSON_SPARRING, LESSON_WOLF) else 0.012
            agent.fear = _clamp(float(getattr(agent, "fear", 0.0)) - fear_delta, 0.0, 1.0)
        except Exception:
            pass
        if self.lesson_mode == LESSON_IDLE:
            try:
                agent.last_attacker_id = None
            except Exception:
                pass

    def _world_tick(self, world: Any) -> int:
        for attr in ("tick_count", "ticks", "time", "tick"):
            raw = getattr(world, attr, None)
            if callable(raw):
                continue
            try:
                return int(raw)
            except Exception:
                continue
        return 0

    def _room_brain_export_id(self, world: Any, agent: Any) -> str:
        agent_id = str(getattr(agent, "id", getattr(agent, "agent_id", "lab_agent")) or "lab_agent")
        return f"{agent_id}.room_ollama"

    def _room_brain_export_meta(self, world: Any, agent: Any, *, reason: str) -> Dict[str, Any]:
        brain = getattr(agent, "brain", None)
        ollama_model = None
        if brain is not None:
            ollama_model = getattr(brain, "ollama_last_model", None) or getattr(brain, "ollama_model", None)
        return {
            "source": "training_room",
            "room_id": self.room_id,
            "room_label": self.label,
            "reason": str(reason or "room_export"),
            "tick": int(self._world_tick(world)),
            "agent_id": str(getattr(agent, "id", getattr(agent, "agent_id", "lab_agent"))),
            "agent_name": str(getattr(agent, "name", getattr(agent, "id", "lab_agent"))),
            "lab_agent": True,
            "training_room_confined": bool(self.is_confined(getattr(agent, "id", None))),
            "ollama_embedded": bool(getattr(brain, "ollama_enabled", False) or getattr(brain, "ollama_last_model", None) or getattr(brain, "ollama_dialogue_tail", None)),
            "ollama_model": str(ollama_model) if ollama_model else None,
        }

    def _save_room_brain(self, world: Any, *, reason: str, force: bool) -> Optional[str]:
        if world is None or not self.agent_id:
            return None
        agent = self._get_agent(world, self.agent_id)
        if agent is None:
            return None
        brain = getattr(agent, "brain", None)
        if brain is None:
            return None
        tick = self._world_tick(world)
        if not force and (tick - int(self._last_room_brain_save_tick)) < int(self.autosave_ticks):
            return self._last_room_brain_path
        export_meta = self._room_brain_export_meta(world, agent, reason=reason)
        export_id = self._room_brain_export_id(world, agent)
        try:
            path = export_room_brain(
                brain,
                export_id=export_id,
                export_meta=export_meta,
                out_dir=self.room_brains_dir,
            )
        except Exception:
            return self._last_room_brain_path
        self._last_room_brain_save_tick = int(tick)
        self._last_room_brain_path = str(path)
        return self._last_room_brain_path


def _smoke_demo() -> int:
    from world import World, Agent

    world = World(width=120.0, height=120.0)
    world.add_agent(Agent(agent_id="lab_0", name="LabSubject", x=60.0, y=60.0, goal_x=60.0, goal_y=60.0))
    world.add_agent(Agent(agent_id="peer_0", name="Peer", x=24.0, y=24.0, goal_x=24.0, goal_y=24.0))

    room = TrainingRoomManager()
    agent_id = room.attach_world(world)
    center = room.room_center(world)
    print("training_room smoke ok")
    print(f"lab_agent={agent_id}")
    print(f"room_center=({center[0]:.1f}, {center[1]:.1f})")
    return 0


def _parse_cli_options(args: Iterable[str]) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "smoke": False,
        "ollama_smoke": False,
        "no_ollama": False,
        "auto_ollama": False,
        "best_agent": False,
        "white_room": False,
        "conversation_only": False,
        "start_sparring": True,
        "room_agent_id": None,
        "ollama_model": None,
        "ollama_host": None,
        "ollama_interval_ms": 3600,
    }
    for raw in list(args or []):
        arg = str(raw or "").strip()
        if not arg:
            continue
        if arg == "--smoke":
            options["smoke"] = True
        elif arg == "--ollama-smoke":
            options["ollama_smoke"] = True
        elif arg == "--no-ollama":
            options["no_ollama"] = True
        elif arg == "--auto-ollama":
            options["auto_ollama"] = True
        elif arg == "--best-agent":
            options["best_agent"] = True
        elif arg in ("--white-room", "--blank-room"):
            options["white_room"] = True
        elif arg in ("--conversation-only", "--no-sparring"):
            options["conversation_only"] = True
            options["start_sparring"] = False
        elif arg == "--start-sparring":
            options["conversation_only"] = False
            options["start_sparring"] = True
        elif arg.startswith("--room-agent-id=") or arg.startswith("--agent-id="):
            value = arg.split("=", 1)[1].strip()
            options["room_agent_id"] = value or None
        elif arg.startswith("--ollama-model="):
            value = arg.split("=", 1)[1].strip()
            options["ollama_model"] = value or None
        elif arg.startswith("--ollama-host="):
            value = arg.split("=", 1)[1].strip()
            options["ollama_host"] = value or None
        elif arg.startswith("--ollama-interval-ms="):
            try:
                options["ollama_interval_ms"] = max(1000, int(arg.split("=", 1)[1]))
            except Exception:
                pass
    return options


def _ollama_smoke(options: Dict[str, Any]) -> int:
    from world import World, Agent

    world = World(width=30.0, height=30.0)
    agent = Agent(agent_id="lab_0", name="LabSubject", x=15.0, y=15.0, goal_x=15.0, goal_y=15.0)
    world.add_agent(agent)

    room = TrainingRoomManager()
    room.attach_world(world, preferred_agent_id="lab_0")

    coach = OllamaCoach(
        model=options.get("ollama_model"),
        host=options.get("ollama_host"),
    )
    snap = build_training_snapshot(agent, world, room)
    advice = coach.request_advice(snap)
    applied = apply_advice_to_agent(agent, world, room, advice)
    print("ollama smoke ok")
    print(f"model={advice.model}")
    print(f"lesson={applied.get('lesson')}")
    goal = applied.get("goal")
    if goal is not None:
        print(f"goal=({goal[0]:.1f}, {goal[1]:.1f})")
    return 0


def _build_room_only_window(runtime_options: Optional[Dict[str, Any]] = None):
    import config
    from combined_app import CombinedMainWindow
    from PySide6 import QtCore, QtWidgets, QtGui

    class _OllamaLessonWorker(QtCore.QObject):
        finished = QtCore.Signal(object, object)

        def __init__(self, coach: OllamaCoach, snapshot: Dict[str, Any]):
            super().__init__()
            self._coach = coach
            self._snapshot = dict(snapshot or {})

        @QtCore.Slot()
        def run(self):
            try:
                advice = self._coach.request_advice(self._snapshot)
            except Exception as exc:
                self.finished.emit(None, str(exc))
                return
            self.finished.emit(advice, None)

    class MorpheusRoomWindow(CombinedMainWindow):
        def __init__(self):
            self._runtime_options = dict(runtime_options or {})
            self._room_only_world_size = (30.0, 30.0)
            self._white_room = bool(self._runtime_options.get("white_room"))
            self._conversation_only = bool(self._runtime_options.get("conversation_only"))
            self._start_sparring_on_open = bool(self._runtime_options.get("start_sparring", True)) and not self._conversation_only
            self._lab_subject_candidate = None
            if bool(self._runtime_options.get("best_agent")):
                self._lab_subject_candidate = select_best_trained_agent()
            selected_candidate_id = getattr(self._lab_subject_candidate, "agent_id", None)
            self._room_only_agent_id = str(
                self._runtime_options.get("room_agent_id")
                or selected_candidate_id
                or DEFAULT_ROOM_AGENT_ID
                or ""
            ).strip() or DEFAULT_ROOM_AGENT_ID
            self._ollama_enabled = bool(self._runtime_options.get("auto_ollama")) and not bool(self._runtime_options.get("no_ollama"))
            self._ollama_interval_ms = max(1000, int(self._runtime_options.get("ollama_interval_ms") or 3600))
            self._ollama_interval_ticks = max(12, int(round(float(self._ollama_interval_ms) / 16.0)))
            self._ollama_model_override = self._runtime_options.get("ollama_model")
            self._ollama_host_override = self._runtime_options.get("ollama_host")
            self._ollama_status_label: Optional[QtWidgets.QLabel] = None
            self._combat_status_label: Optional[QtWidgets.QLabel] = None
            self._ollama_console_dock: Optional[QtWidgets.QDockWidget] = None
            self._ollama_console_view: Optional[QtWidgets.QPlainTextEdit] = None
            self._ollama_input: Optional[QtWidgets.QPlainTextEdit] = None
            self._ollama_input_viewport: Optional[QtWidgets.QWidget] = None
            self._ollama_ui_timer: Optional[QtCore.QTimer] = None
            self._ollama_last_seen_log_seq: int = 0
            self._ollama_online_announced = False
            self._ollama_text_entry_locked = False
            self._ollama_locked_actions: list[tuple[Any, bool]] = []
            super().__init__()
            self.setWindowTitle("White Room Lab" if self._white_room else "Morpheus Room Lab")
            self.trainer.disaster_interval_ticks = 0
            self.trainer.relief_after_disaster = 0
            self._configure_room_only_manager()
            self._apply_room_only_world(announce=False)
            if not self._conversation_only:
                self._enable_room_combat_lessons(announce=False)
            if hasattr(self, "_combat_timer"):
                self._combat_timer.stop()
            self._combat_paused = True
            if hasattr(self, "act_play"):
                self.act_play.setEnabled(False)
                self.act_play.setToolTip("Room-only mode: combat disabled")
            if hasattr(self, "minimap"):
                self.minimap.hide()
            if hasattr(self, "world_map_overlay"):
                self.world_map_overlay.hide()
            self.bridge._push_snapshot()
            self._ensure_training_room_selection()
            self._setup_ollama_console()
            self._setup_ollama_training()
            if self._start_sparring_on_open:
                self._start_morpheus_sparring(announce=False)
            else:
                self._stop_room_combat_lesson(announce=False)
            if self._lab_subject_candidate is not None and hasattr(self.trainer.world, "add_chat_line"):
                try:
                    self.trainer.world.add_chat_line(f"[lab] selected {format_candidate_summary(self._lab_subject_candidate)}")
                except Exception:
                    pass
            self.statusBar().showMessage("White room ready" if self._white_room else "Morpheus room ready", 1800)

        def _initial_agent_lineup(self) -> list[dict[str, str]]:
            candidate = getattr(self, "_lab_subject_candidate", None)
            if candidate is not None:
                return [make_lab_subject_lineup_spec(candidate)]
            return _room_only_agent_lineup(getattr(self, "_room_only_agent_id", DEFAULT_ROOM_AGENT_ID))

        def _help_text_for_current_mode(self) -> str:
            room_name = "Белая комната" if getattr(self, "_white_room", False) else "Комната Морфеуса"
            if hasattr(self, "view3d") and self.view3d.is_first_person_mode():
                return f"{room_name}: мышь — обзор • WASD/ЦФЫВ — вести агента по комнате • Shift — быстрее • Ctrl — точный шаг • Ctrl+Shift+M — спарринг • Ctrl+Shift+W — wolf drill • Ctrl+Shift+K — стоп урока • Ctrl+Shift+O — автокоуч • Ctrl+Shift+I — фокус ввода Ollama • V/М или Esc — выход"
            return f"{room_name}: ЛКМ — выбрать • ПКМ — цель в комнате • RMB — орбита • колесо — зум • F/А — фокус • R/К — сброс • Ctrl+Shift+M — спарринг • Ctrl+Shift+W — wolf drill • Ctrl+Shift+K — стоп урока • Ctrl+Shift+O — автокоуч • Ctrl+Shift+I — панель указаний • V/М — first person"

        def _make_toolbar(self):
            super()._make_toolbar()
            tb = getattr(self, "_main_toolbar", None)
            if tb is None:
                return
            tb.addSeparator()

            self.act_ollama_toggle = QtGui.QAction("Auto Coach (Ctrl+Shift+O)", self, checkable=True)
            self.act_ollama_toggle.setShortcut("Ctrl+Shift+O")
            self.act_ollama_toggle.setChecked(bool(self._ollama_enabled))
            self.act_ollama_toggle.toggled.connect(self._toggle_ollama_coach)
            self.addAction(self.act_ollama_toggle)
            tb.addAction(self.act_ollama_toggle)

            self.act_ollama_now = QtGui.QAction("Coach Now (Ctrl+Shift+L)", self)
            self.act_ollama_now.setShortcut("Ctrl+Shift+L")
            self.act_ollama_now.triggered.connect(self._request_ollama_lesson_now)
            self.addAction(self.act_ollama_now)
            tb.addAction(self.act_ollama_now)

            self.act_ollama_focus = QtGui.QAction("Prompt Ollama (Ctrl+Shift+I)", self)
            self.act_ollama_focus.setShortcut("Ctrl+Shift+I")
            self.act_ollama_focus.triggered.connect(self._focus_ollama_input)
            self.addAction(self.act_ollama_focus)
            tb.addAction(self.act_ollama_focus)

            tb.addSeparator()

            self.act_sparring = QtGui.QAction("Morpheus Sparring (Ctrl+Shift+M)", self)
            self.act_sparring.setShortcut("Ctrl+Shift+M")
            self.act_sparring.triggered.connect(self._start_morpheus_sparring)
            self.addAction(self.act_sparring)
            tb.addAction(self.act_sparring)

            self.act_wolf_drill = QtGui.QAction("Wolf Drill (Ctrl+Shift+W)", self)
            self.act_wolf_drill.setShortcut("Ctrl+Shift+W")
            self.act_wolf_drill.triggered.connect(self._start_wolf_drill)
            self.addAction(self.act_wolf_drill)
            tb.addAction(self.act_wolf_drill)

            self.act_stop_lesson = QtGui.QAction("Stop Lesson (Ctrl+Shift+K)", self)
            self.act_stop_lesson.setShortcut("Ctrl+Shift+K")
            self.act_stop_lesson.triggered.connect(self._stop_room_combat_lesson)
            self.addAction(self.act_stop_lesson)
            tb.addAction(self.act_stop_lesson)

            self._combat_status_label = QtWidgets.QLabel("Combat: idle")
            self._combat_status_label.setStyleSheet("QLabel { color:#f0d98a; }")
            tb.addWidget(self._combat_status_label)

            self._ollama_status_label = QtWidgets.QLabel("Ollama: pending")
            self._ollama_status_label.setStyleSheet("QLabel { color:#8be08b; }")
            tb.addWidget(self._ollama_status_label)

        def _toggle_world_map_overlay(self):
            return

        def _toggle_combat(self):
            self.statusBar().showMessage("Room-only mode: combat disabled", 1400)

        def _spawn_wolves(self):
            self._toast("Room-only mode: wolves disabled")

        def _release_training_room_agent(self):
            self._toast("Room-only mode: only Morpheus room is available")

        def _configure_room_only_manager(self) -> None:
            room = self._training_room()
            if room is None:
                return
            if getattr(self, "_white_room", False):
                room.room_id = "white_room"
                room.label = "Белая_комната"
                room.object_id = "white_room_zone"

        def _apply_selected_lab_subject_brain(self, agent: Any) -> None:
            candidate = getattr(self, "_lab_subject_candidate", None)
            if candidate is None or agent is None:
                return
            if str(getattr(agent, "_lab_subject_source_path", "")) == str(candidate.path):
                return
            brain = load_candidate_brain(candidate, agent_id=getattr(agent, "id", candidate.agent_id))
            if brain is None:
                return
            try:
                if hasattr(brain, "set_persona"):
                    brain.set_persona(getattr(agent, "persona", ""))
                elif hasattr(brain, "persona"):
                    setattr(brain, "persona", getattr(agent, "persona", ""))
            except Exception:
                pass
            agent.brain = brain
            try:
                agent.health = float(max(1.0, min(100.0, getattr(brain, "health", getattr(agent, "health", 100.0)))))
                energy = float(getattr(brain, "energy", getattr(agent, "energy", 1.0)))
                hunger = float(getattr(brain, "hunger", getattr(agent, "hunger", 0.0)))
                agent.energy = max(0.0, min(1.0, energy / 100.0 if energy > 1.5 else energy))
                agent.hunger = max(0.0, min(1.0, hunger / 100.0 if hunger > 1.5 else hunger))
                agent.fear = max(0.0, min(1.0, float(getattr(brain, "fear_level", getattr(agent, "fear", 0.0)))))
                agent.age_ticks = int(getattr(brain, "age_ticks", getattr(agent, "age_ticks", 0)) or 0)
                agent.alive = bool(getattr(brain, "alive", True)) and agent.health > 0.0
            except Exception:
                pass
            try:
                sync_memory = getattr(agent, "_sync_memory_from_brain", None)
                if callable(sync_memory):
                    sync_memory()
            except Exception:
                pass
            try:
                setattr(agent, "_lab_subject_source_path", str(candidate.path))
            except Exception:
                pass

        def _add_showcase_safe_havens(self, world) -> int:
            return 0

        def _rebuild_environment_for_world(self, world):
            if world is None:
                return
            self.engine.world.width = float(getattr(world, "width", self._room_only_world_size[0]))
            self.engine.world.height = float(getattr(world, "height", self._room_only_world_size[1]))
            if hasattr(self.engine, "load_static_environment"):
                self.engine.load_static_environment([])

        def _prepare_showcase_world(self, *, announce: bool, rebuild_environment: bool):
            self._showcase_world_w = float(self._room_only_world_size[0])
            self._showcase_world_h = float(self._room_only_world_size[1])
            config.WORLD_WIDTH = self._showcase_world_w
            config.WORLD_HEIGHT = self._showcase_world_h
            self._apply_room_only_world(announce=announce)
            if rebuild_environment:
                self._rebuild_environment_for_world(getattr(self.trainer, "world", None))

        def _apply_room_only_world(self, *, announce: bool) -> None:
            world = getattr(self.trainer, "world", None)
            room = self._training_room()
            if world is None or room is None:
                return

            target_w = float(self._room_only_world_size[0])
            target_h = float(self._room_only_world_size[1])
            config.WORLD_WIDTH = target_w
            config.WORLD_HEIGHT = target_h
            self._resize_world_if_needed(world, target_w, target_h)

            preferred_id = self._training_room_agent_id() or getattr(self, "_room_only_agent_id", None)
            keep_id = room.attach_world(world, preferred_agent_id=preferred_id, announce=announce)
            if not keep_id:
                return

            keep_agent = room._get_agent(world, keep_id)
            if keep_agent is None:
                return
            self._apply_selected_lab_subject_brain(keep_agent)

            lineup = list(getattr(self.trainer, "agent_lineup", []) or [])
            selected_spec = None
            for spec in lineup:
                if str(spec.get("id", "")) == str(keep_id):
                    selected_spec = dict(spec)
                    break
            if selected_spec is None:
                selected_spec = {
                    "id": str(getattr(keep_agent, "id", keep_id)),
                    "name": str(getattr(keep_agent, "name", keep_id)),
                    "persona": str(getattr(keep_agent, "persona", "caring/supportive")),
                }
            self.trainer.agent_lineup = [selected_spec]
            self.trainer.num_agents = 1

            if isinstance(getattr(world, "agents", None), dict):
                world.agents = {str(keep_id): keep_agent}
            else:
                world.agents = [keep_agent]

            world.animals = {}
            world.objects = [obj for obj in list(getattr(world, "objects", []) or []) if str(getattr(obj, "obj_id", "")) == room.object_id]
            room.maintain_world(world)

            center_x, center_y = room.room_center(world)
            config.SAFE_POINT = (center_x, center_y)
            world.activities = {
                room.object_id: {
                    "name": room.label,
                    "activity_tags": ["heal", "rest", "calm", "sleep", "repair_self"],
                    "comfort_level": 1.0,
                    "danger_level": 0.0,
                    "area": {"x": center_x, "y": center_y, "radius": room.bounds_for(world).safe_radius},
                }
            }

            self.engine.world.width = float(world.width)
            self.engine.world.height = float(world.height)
            if hasattr(self.engine, "load_static_environment"):
                self.engine.load_static_environment([])

            if hasattr(self, "view3d"):
                self.view3d.center_x = center_x
                self.view3d.center_z = center_y
                self.view3d.distance = max(30.0, max(float(world.width), float(world.height)) * 1.00)
                self.view3d.yaw_deg = -135.0
                self.view3d.pitch_deg = 28.0
                self.view3d._fp_x = center_x
                self.view3d._fp_z = center_y + 3.6
                self.view3d._fp_yaw_deg = 180.0
                self.view3d._fp_target_yaw_deg = 180.0
                self.view3d._fp_pitch_deg = -4.0
                self.view3d._fp_target_pitch_deg = -4.0

            if hasattr(self, "shared"):
                self.shared.set_selected_agent(str(keep_id))

        def _enable_room_combat_lessons(self, *, announce: bool) -> None:
            room = self._training_room()
            world = getattr(self.trainer, "world", None)
            if room is None or world is None:
                return
            room.enable_combat_lessons(True)
            room.maintain_world(world)
            if announce:
                self.statusBar().showMessage("Combat lessons enabled in Morpheus room", 1800)

        def _refresh_room_combat_ui(self) -> None:
            room = self._training_room()
            world = getattr(self.trainer, "world", None)
            label = self._combat_status_label
            if room is None or world is None or label is None:
                return
            state = room.combat_status(world)
            mode = str(state.get("lesson_mode") or LESSON_IDLE)
            skill = float(state.get("lab_combat_skill", 0.0) or 0.0)
            hp = float(state.get("lab_health", 0.0) or 0.0)
            mentor = bool(state.get("mentor_present"))
            wolf = bool(state.get("wolf_present"))
            suffix = "mentor" if mentor else "no mentor"
            if wolf:
                suffix += " + wolf"
            color = "#f0d98a"
            if mode == LESSON_WOLF:
                color = "#ffb06c"
            elif mode == LESSON_IDLE:
                color = "#a7b0c7"
            label.setText(f"Combat: {mode} | skill {skill:.2f} | hp {hp:.0f} | {suffix}")
            label.setStyleSheet(f"QLabel {{ color:{color}; }}")

        @QtCore.Slot()
        def _start_morpheus_sparring(self, *, announce: bool = True) -> None:
            room = self._training_room()
            world = getattr(self.trainer, "world", None)
            if room is None or world is None:
                return
            room.enable_combat_lessons(True)
            room.start_sparring(world, announce=announce)
            self.bridge._push_snapshot()
            self._refresh_room_combat_ui()
            if announce:
                self._toast("Morpheus sparring started")

        @QtCore.Slot()
        def _start_wolf_drill(self, *, announce: bool = True) -> None:
            room = self._training_room()
            world = getattr(self.trainer, "world", None)
            if room is None or world is None:
                return
            room.enable_combat_lessons(True)
            room.start_wolf_drill(world, announce=announce)
            self.bridge._push_snapshot()
            self._refresh_room_combat_ui()
            if announce:
                self._toast("Wolf drill started")

        @QtCore.Slot()
        def _stop_room_combat_lesson(self, *, announce: bool = True) -> None:
            room = self._training_room()
            world = getattr(self.trainer, "world", None)
            if room is None or world is None:
                return
            room.stop_combat_lesson(world, announce=announce)
            self.bridge._push_snapshot()
            self._refresh_room_combat_ui()
            if announce:
                self._toast("Combat lesson stopped")

        def _ollama_service(self):
            return getattr(self.trainer, "ollama_brain_service", None)

        def _focus_is_within_widget(self, root: Optional[QtWidgets.QWidget]) -> bool:
            if root is None:
                return False
            current = self.focusWidget()
            while current is not None:
                if current is root:
                    return True
                current = current.parentWidget()
            return False

        def _ollama_input_has_focus(self) -> bool:
            return self._focus_is_within_widget(self._ollama_input)

        def _iter_ollama_lock_actions(self) -> Iterable[Any]:
            seen: set[int] = set()
            toolbar = getattr(self, "_main_toolbar", None)
            for action in list(getattr(toolbar, "actions", lambda: [])() or []):
                if action is None or id(action) in seen:
                    continue
                seen.add(id(action))
                yield action

        def _set_ollama_text_entry_lock(self, active: bool) -> None:
            active = bool(active)
            if active == self._ollama_text_entry_locked:
                return
            self._ollama_text_entry_locked = active
            view3d = getattr(self, "view3d", None)
            if active:
                if view3d is not None:
                    try:
                        view3d.suspend_first_person_capture()
                    except Exception:
                        pass
                    try:
                        view3d._pressed_keys.clear()
                    except Exception:
                        pass
                self._ollama_locked_actions = []
                for action in self._iter_ollama_lock_actions():
                    try:
                        was_enabled = bool(action.isEnabled())
                    except Exception:
                        continue
                    self._ollama_locked_actions.append((action, was_enabled))
                    try:
                        action.setEnabled(False)
                    except Exception:
                        pass
                self.statusBar().showMessage("Typing to Ollama: interface controls locked", 1600)
                return
            for action, was_enabled in list(self._ollama_locked_actions):
                try:
                    action.setEnabled(bool(was_enabled))
                except Exception:
                    pass
            self._ollama_locked_actions = []
            self.statusBar().showMessage("Ollama input unlocked", 1000)

        def _sync_ollama_text_entry_lock(self) -> None:
            self._set_ollama_text_entry_lock(self._ollama_input_has_focus())

        def eventFilter(self, obj, event):
            etype = event.type()
            if obj in (self._ollama_input, self._ollama_input_viewport):
                if etype in (QtCore.QEvent.Type.FocusIn, QtCore.QEvent.Type.FocusOut):
                    QtCore.QTimer.singleShot(0, self._sync_ollama_text_entry_lock)
                elif etype in (QtCore.QEvent.Type.Hide, QtCore.QEvent.Type.Close):
                    self._set_ollama_text_entry_lock(False)
            if self._ollama_text_entry_locked:
                blocked_mouse_events = {
                    QtCore.QEvent.Type.MouseButtonPress,
                    QtCore.QEvent.Type.MouseButtonRelease,
                    QtCore.QEvent.Type.MouseButtonDblClick,
                    QtCore.QEvent.Type.MouseMove,
                    QtCore.QEvent.Type.Wheel,
                }
                if obj is getattr(self, "view3d", None) and etype in blocked_mouse_events:
                    event.accept()
                    return True
                if obj is getattr(self, "_main_toolbar", None) and etype in blocked_mouse_events:
                    event.accept()
                    return True
            return super().eventFilter(obj, event)

        def _forward_key_event_to_view3d(self, e: QtGui.QKeyEvent, *, source=None) -> bool:
            if self._ollama_text_entry_locked or self._ollama_input_has_focus():
                return False
            return super()._forward_key_event_to_view3d(e, source=source or self.focusWidget())

        def _ollama_brain(self) -> Optional[Any]:
            agent = self._training_room_agent_object()
            return getattr(agent, "brain", None) if agent is not None else None

        def _training_room_agent_object(self) -> Optional[Any]:
            world = getattr(self.trainer, "world", None)
            room = self._training_room()
            if world is None or room is None:
                return None
            agent_id = self._training_room_agent_id()
            if not agent_id:
                return None
            return room._get_agent(world, agent_id)

        def _current_world_tick(self) -> int:
            world = getattr(self.trainer, "world", None)
            if world is None:
                return 0
            for attr in ("tick_count", "ticks", "time", "tick"):
                raw = getattr(world, attr, None)
                if callable(raw):
                    continue
                try:
                    return int(raw)
                except Exception:
                    continue
            return 0

        def _set_ollama_status(self, text: str, *, color: str = "#8be08b") -> None:
            if self._ollama_status_label is not None:
                self._ollama_status_label.setText(str(text))
                self._ollama_status_label.setStyleSheet(f"QLabel {{ color:{color}; }}")

        def _brain_ollama_state(self) -> Dict[str, Any]:
            brain = self._ollama_brain()
            if brain is None:
                return {}
            if hasattr(brain, "export_ollama_state_for_ui"):
                try:
                    return dict(brain.export_ollama_state_for_ui() or {})
                except Exception:
                    return {}
            return {}

        def _append_ollama_console(self, role: str, text: str) -> None:
            view = self._ollama_console_view
            if view is None:
                return
            line = str(text or "").strip()
            if not line:
                return
            if role == "raw":
                view.appendPlainText(line)
                sb = view.verticalScrollBar()
                if sb is not None:
                    sb.setValue(sb.maximum())
                return
            tick = self._current_world_tick()
            prefix = str(role or "log").upper()
            view.appendPlainText(f"[t={tick:04d}] {prefix}: {line}")
            sb = view.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

        def _setup_ollama_console(self) -> None:
            dock = QtWidgets.QDockWidget("Ollama Console", self)
            dock.setObjectName("ollama_console_dock")
            dock.setAllowedAreas(
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea
                | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
                | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            )
            root = QtWidgets.QWidget(dock)
            lay = QtWidgets.QVBoxLayout(root)
            lay.setContentsMargins(10, 10, 10, 10)
            lay.setSpacing(8)

            if getattr(self, "_white_room", False):
                title_text = "Разговаривай с агентом прямо через Ollama."
                hint_text = "Пример: «расскажи, что ты помнишь о выживании и как чувствуешь себя в белой комнате»"
            else:
                title_text = "Пиши указания агенту прямо через Ollama."
                hint_text = "Пример: «посади агента в левое кресло и заставь запомнить маршрут к столику»"

            title = QtWidgets.QLabel(title_text)
            title.setStyleSheet("QLabel { color:#d8ddf7; font-weight:600; }")
            title.setWordWrap(True)
            lay.addWidget(title)

            hint = QtWidgets.QLabel(hint_text)
            hint.setStyleSheet("QLabel { color:#98a3c7; font-size:11px; }")
            hint.setWordWrap(True)
            lay.addWidget(hint)

            view = QtWidgets.QPlainTextEdit(root)
            view.setReadOnly(True)
            view.setMaximumBlockCount(300)
            view.setStyleSheet(
                "QPlainTextEdit { background:#0f131b; color:#dfe6ff; border:1px solid #2a3142; border-radius:10px; padding:8px; font-family:monospace; }"
            )
            lay.addWidget(view, 1)

            input_box = QtWidgets.QPlainTextEdit(root)
            input_box.setPlaceholderText("Напиши задачу для агента. Ctrl+Enter — отправить в Ollama.")
            input_box.setFixedHeight(96)
            input_box.setStyleSheet(
                "QPlainTextEdit { background:#131923; color:#eef2ff; border:1px solid #38435a; border-radius:10px; padding:8px; }"
            )
            lay.addWidget(input_box)

            btn_row = QtWidgets.QHBoxLayout()
            btn_row.setSpacing(8)
            btn_send = QtWidgets.QPushButton("Send To Agent")
            btn_send.clicked.connect(self._send_ollama_operator_instruction)
            btn_clear = QtWidgets.QPushButton("Clear Log")
            btn_clear.clicked.connect(view.clear)
            btn_row.addWidget(btn_send)
            btn_row.addWidget(btn_clear)
            btn_row.addStretch(1)
            lay.addLayout(btn_row)

            dock.setWidget(root)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.resize(380, 520)

            self._ollama_console_dock = dock
            self._ollama_console_view = view
            self._ollama_input = input_box
            self._ollama_input_viewport = input_box.viewport()
            self._ollama_send_shortcut_return = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), input_box)
            self._ollama_send_shortcut_return.activated.connect(self._send_ollama_operator_instruction)
            self._ollama_send_shortcut_enter = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Enter"), input_box)
            self._ollama_send_shortcut_enter.activated.connect(self._send_ollama_operator_instruction)
            input_box.installEventFilter(self)
            if self._ollama_input_viewport is not None:
                self._ollama_input_viewport.installEventFilter(self)
            if hasattr(self, "view3d") and self.view3d is not None:
                self.view3d.installEventFilter(self)
            if getattr(self, "_main_toolbar", None) is not None:
                self._main_toolbar.installEventFilter(self)
            self._append_ollama_console("system", "Панель готова. Можно отправлять инструкции агенту.")

        @QtCore.Slot()
        def _focus_ollama_input(self) -> None:
            if self._ollama_console_dock is not None:
                self._ollama_console_dock.show()
                self._ollama_console_dock.raise_()
            if self._ollama_input is not None:
                self._ollama_input.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

        def _clear_ollama_goal(self) -> None:
            brain = self._ollama_brain()
            if brain is not None and hasattr(brain, "clear_ollama_goal"):
                brain.clear_ollama_goal()

        def _active_lesson_payload(self) -> Optional[Dict[str, Any]]:
            state = self._brain_ollama_state()
            lesson = str(state.get("active_lesson") or "").strip()
            goal = state.get("active_goal")
            if not lesson and not isinstance(goal, dict):
                return None
            payload: Dict[str, Any] = {
                "lesson": lesson,
                "model": str(state.get("last_model") or ""),
                "started_tick": state.get("goal_started_tick"),
            }
            if isinstance(goal, dict) and ("x" in goal) and ("y" in goal):
                payload["goal"] = {
                    "x": round(float(goal.get("x", 0.0)), 2),
                    "y": round(float(goal.get("y", 0.0)), 2),
                }
            return payload

        def _format_ollama_log_entry(self, row: Dict[str, Any]) -> str:
            tick = int(row.get("tick", self._current_world_tick()) or 0)
            role = str(row.get("role") or "log").upper()
            text = str(row.get("text") or "").strip()
            return f"[t={tick:04d}] {role}: {text}"

        def _refresh_ollama_ui(self) -> None:
            self._refresh_room_combat_ui()
            state = self._brain_ollama_state()
            if not state:
                self._set_ollama_status("Ollama: no brain", color="#a7b0c7")
                return
            status = str(state.get("status") or "off")
            model = str(state.get("last_model") or state.get("model") or "local")
            color = "#8be08b"
            if status in ("error", "offline", "handler_error"):
                color = "#ff9f8b"
            elif status in ("queued", "thinking"):
                color = "#9bd1ff"
            elif status in ("off", "manual"):
                color = "#a7b0c7"
            self._set_ollama_status(f"Ollama: {model} [{status}]", color=color)

            seq_now = int(state.get("log_seq", 0) or 0)
            if seq_now < self._ollama_last_seen_log_seq:
                self._ollama_last_seen_log_seq = 0
            for row in list(state.get("dialogue_tail", []) or []):
                if not isinstance(row, dict):
                    continue
                seq = int(row.get("seq", 0) or 0)
                if seq <= self._ollama_last_seen_log_seq:
                    continue
                self._append_ollama_console("raw", self._format_ollama_log_entry(row))
                self._ollama_last_seen_log_seq = seq

        def _configure_ollama_brain(self, *, announce: bool) -> None:
            service = self._ollama_service()
            brain = self._ollama_brain()
            if service is None or brain is None:
                self._set_ollama_status("Ollama: unavailable", color="#ff9f8b")
                return
            service.configure_brain(
                brain,
                enabled=True,
                auto_mode=self._ollama_enabled,
                model=self._ollama_model_override,
                host=self._ollama_host_override,
                interval_ticks=self._ollama_interval_ticks,
                tick=self._current_world_tick(),
            )
            if announce and not self._ollama_online_announced:
                self._append_ollama_console("system", "Ollama привязан прямо к мозгу агента.")
                self._ollama_online_announced = True
            self._refresh_ollama_ui()

        def _setup_ollama_training(self) -> None:
            self._configure_ollama_brain(announce=True)
            if self._ollama_ui_timer is None:
                self._ollama_ui_timer = QtCore.QTimer(self)
                self._ollama_ui_timer.setInterval(250)
                self._ollama_ui_timer.timeout.connect(self._refresh_ollama_ui)
                self._ollama_ui_timer.start()
            if self._ollama_enabled:
                self._announce_ollama_online()
                QtCore.QTimer.singleShot(300, self._request_ollama_lesson_now)
            else:
                self._append_ollama_console("system", "Автокоуч выключен по умолчанию. Ручные команды доступны сразу.")

        def _announce_ollama_online(self) -> None:
            if self._ollama_online_announced:
                return
            world = getattr(self.trainer, "world", None)
            brain = self._ollama_brain()
            state = self._brain_ollama_state()
            model = str(state.get("model") or getattr(brain, "ollama_model", "") or "ollama")
            if world is not None and hasattr(world, "add_chat_line"):
                try:
                    world.add_chat_line(f"[ollama] brain online: {model}")
                except Exception:
                    pass
            self._ollama_online_announced = True

        def _toggle_ollama_coach(self, enabled: bool) -> None:
            self._ollama_enabled = bool(enabled)
            service = self._ollama_service()
            brain = self._ollama_brain()
            if service is None or brain is None:
                return
            service.configure_brain(
                brain,
                enabled=True,
                auto_mode=self._ollama_enabled,
                model=self._ollama_model_override,
                host=self._ollama_host_override,
                interval_ticks=self._ollama_interval_ticks,
                tick=self._current_world_tick(),
            )
            if not self._ollama_enabled:
                self._append_ollama_console("system", "Автокоуч выключен. Мозг Ollama остался доступен для ручных команд.")
                self.statusBar().showMessage("Auto coach paused; brain prompt remains active", 2200)
                self._refresh_ollama_ui()
                return
            self._announce_ollama_online()
            service.force_request(brain)
            self._append_ollama_console("system", "Автокоуч включен на уровне мозга.")
            self.statusBar().showMessage("Ollama brain auto mode active", 1800)
            self._refresh_ollama_ui()

        def _maybe_complete_ollama_goal(self, agent: Any) -> bool:
            return False

        def _start_ollama_request(self, snapshot: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> bool:
            return False

        def _apply_coach_result(self, result: Dict[str, Any]) -> None:
            self._refresh_ollama_ui()

        def _request_ollama_lesson(
            self,
            *,
            force: bool,
            operator_instruction: Optional[str] = None,
            source: str = "auto",
        ) -> None:
            service = self._ollama_service()
            brain = self._ollama_brain()
            if service is None or brain is None:
                return
            manual_instruction = str(operator_instruction or "").strip()
            if manual_instruction:
                service.queue_instruction(brain, manual_instruction, tick=self._current_world_tick())
            elif force or source == "manual":
                service.force_request(brain)
            self._refresh_ollama_ui()

        @QtCore.Slot()
        def _request_ollama_lesson_now(self) -> None:
            service = self._ollama_service()
            brain = self._ollama_brain()
            if service is None or brain is None:
                return
            service.force_request(brain)
            self._refresh_ollama_ui()

        @QtCore.Slot()
        def _on_ollama_timer(self) -> None:
            self._refresh_ollama_ui()

        @QtCore.Slot()
        def _send_ollama_operator_instruction(self) -> None:
            box = self._ollama_input
            if box is None:
                return
            text = str(box.toPlainText() or "").strip()
            if not text:
                self.statusBar().showMessage("Введите указание для агента", 1800)
                return
            service = self._ollama_service()
            brain = self._ollama_brain()
            if service is None or brain is None:
                self._append_ollama_console("error", "Мозг агента не готов к Ollama.")
                self.statusBar().showMessage("Ollama brain is not available", 2200)
                return
            world = getattr(self.trainer, "world", None)
            if world is not None and hasattr(world, "add_chat_line"):
                try:
                    world.add_chat_line(f"[operator] {text}")
                except Exception:
                    pass
            box.clear()
            service.queue_instruction(brain, text, tick=self._current_world_tick())
            self.statusBar().showMessage("Команда передана в мозг агента", 2000)
            self._refresh_ollama_ui()

        @QtCore.Slot(object, object)
        def _on_ollama_result(self, advice: Any, error_text: Any) -> None:
            return

        def _on_trainer_epoch_changed(self):
            super()._on_trainer_epoch_changed()
            self._ollama_last_seen_log_seq = 0
            self._ollama_online_announced = False
            self._configure_room_only_manager()
            if getattr(self, "_conversation_only", False):
                self._stop_room_combat_lesson(announce=False)
            else:
                self._enable_room_combat_lessons(announce=False)
            self._configure_ollama_brain(announce=False)
            self._refresh_room_combat_ui()
            if self._ollama_enabled:
                QtCore.QTimer.singleShot(500, self._request_ollama_lesson_now)

        def closeEvent(self, event):
            if self._ollama_ui_timer is not None:
                self._ollama_ui_timer.stop()
            world = getattr(self.trainer, "world", None)
            room = self._training_room()
            if world is not None and room is not None:
                try:
                    room._save_room_brain(world, reason="room_window_close", force=True)
                except Exception:
                    pass
            super().closeEvent(event)

    return MorpheusRoomWindow()


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    options = _parse_cli_options(args)
    if options.get("smoke"):
        return _smoke_demo()
    if options.get("ollama_smoke"):
        return _ollama_smoke(options)

    try:
        from PySide6 import QtWidgets, QtGui
        from combined_app import APP_FONT, APP_QSS, COL_BG, COL_TEXT
    except Exception as exc:
        print(f"[training_room] failed to import GUI entry point: {exc}", file=sys.stderr)
        print("[training_room] run `python training_room.py --smoke` for a console self-check.", file=sys.stderr)
        return 1

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont(APP_FONT, 10))
    app.setStyleSheet(APP_QSS)

    pal = app.palette()
    CR = QtGui.QPalette.ColorRole
    pal.setColor(CR.Window, QtGui.QColor(COL_BG))
    pal.setColor(CR.WindowText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Base, QtGui.QColor(12, 14, 20))
    pal.setColor(CR.AlternateBase, QtGui.QColor(18, 20, 28))
    pal.setColor(CR.ToolTipBase, QtGui.QColor(22, 24, 32))
    pal.setColor(CR.ToolTipText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Text, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Button, QtGui.QColor(18, 20, 28))
    pal.setColor(CR.ButtonText, QtGui.QColor(COL_TEXT))
    pal.setColor(CR.Highlight, QtGui.QColor(90, 130, 255))
    pal.setColor(CR.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)

    win = _build_room_only_window(options)
    win.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
