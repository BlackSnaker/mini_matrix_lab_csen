from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import random

import config


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


class WolfTamingSystem:
    """
    Отдельный модуль для:
      1) обучения агентов приручать волков;
      2) помощи прирученных волков в защите хозяина.

    Встроен мягко: работает через getattr/setattr и не требует менять базовые
    классы Agent/Animal.
    """

    def __init__(self, world: Any) -> None:
        self.world = world
        self.random = random.Random(1337)

        self.tame_radius = float(getattr(config, "WOLF_TAME_RADIUS", getattr(config, "TAME_RADIUS", 2.5)))
        self.tame_cooldown_ticks = int(getattr(config, "WOLF_TAME_ATTEMPT_COOLDOWN", 6))
        self.allow_tame_wolves = bool(getattr(config, "AGENT_CAN_TAME_WOLVES", True))
        # Социальное обучение: союзники рядом перенимают часть опыта приручения.
        self.share_enabled = bool(getattr(config, "WOLF_TAME_SHARE_ENABLED", True))
        self.share_radius = float(getattr(config, "WOLF_TAME_SHARE_RADIUS", 8.0))
        self.share_ratio = float(getattr(config, "WOLF_TAME_SHARE_RATIO", 0.35))
        self.share_min_gain = float(getattr(config, "WOLF_TAME_SHARE_MIN_GAIN", 0.004))
        self.share_max_gain = float(getattr(config, "WOLF_TAME_SHARE_MAX_GAIN", 0.040))

        self.defend_radius = float(getattr(config, "WOLF_DEFEND_RADIUS", 7.0))
        self.defend_bite_range = float(getattr(config, "WOLF_DEFEND_BITE_RANGE", 1.4))
        self.defend_damage = float(getattr(config, "WOLF_DEFEND_DAMAGE", 7.0))
        self.defend_cooldown_ticks = int(getattr(config, "WOLF_DEFEND_COOLDOWN", 12))

        self._last_tame_attempt: Dict[str, int] = {}
        self._last_defend_strike: Dict[str, int] = {}

    def step(self) -> None:
        if not self.world:
            return
        self._taming_step()
        self._defense_step()

    # ------------------------------------------------------------------
    # Приручение
    # ------------------------------------------------------------------

    def _taming_step(self) -> None:
        if not self.allow_tame_wolves:
            return
        now = int(getattr(self.world, "tick_count", 0))
        agents = self._agents()
        wolves = [w for w in self._wolves() if self._is_alive_animal(w) and getattr(w, "tamed_by", None) is None]

        if not agents or not wolves:
            return

        for ag in agents:
            if not self._is_alive_agent(ag):
                continue
            self._ensure_tame_skill(ag)

            aid = str(getattr(ag, "id", getattr(ag, "agent_id", "")))
            if not aid:
                continue
            if (now - self._last_tame_attempt.get(aid, -10**9)) < max(1, self.tame_cooldown_ticks):
                continue

            fear = _clamp(float(getattr(ag, "fear", 0.0)), 0.0, 1.0)

            target = self._nearest_untamed_wolf(ag, wolves, self.tame_radius)
            if target is None:
                continue

            self._last_tame_attempt[aid] = now
            self._attempt_tame(ag, target)

    def _attempt_tame(self, ag: Any, wolf: Any) -> None:
        aid = str(getattr(ag, "id", getattr(ag, "agent_id", "?")))
        name = str(getattr(ag, "name", aid))
        skill = _clamp(float(getattr(ag, "wolf_tame_skill", 0.0)), 0.0, 1.0)
        fear = _clamp(float(getattr(ag, "fear", 0.0)), 0.0, 1.0)
        hp = _clamp(float(getattr(ag, "health", 100.0)) / 100.0, 0.0, 1.0)

        allies = 0
        if hasattr(self.world, "_count_allies_near"):
            try:
                allies = int(self.world._count_allies_near(ag, 4.0))
            except Exception:
                allies = 0
        allies = max(0, min(allies, 3))

        chance = 0.35 + 0.45 * skill + 0.20 * hp + 0.08 * allies - 0.15 * fear
        chance = _clamp(chance, 0.15, 0.95)
        ok = bool(self.random.random() < chance)

        if ok:
            setattr(wolf, "tamed_by", aid)
            setattr(wolf, "aggression_target", None)
            setattr(wolf, "last_action", f"tamed_by:{aid}")

            new_skill = _clamp(skill + 0.08, 0.0, 1.0)
            setattr(ag, "wolf_tame_skill", new_skill)
            setattr(ag, "wolf_tame_successes", int(getattr(ag, "wolf_tame_successes", 0)) + 1)
            setattr(ag, "fear", _clamp(fear - 0.08, 0.0, 1.0))

            self._chat(f"[tame] {name} приручил(а) волка {getattr(wolf, 'uid', '?')}")
            self._event({
                "type": "tame_success",
                "who": aid,
                "name": name,
                "animal_id": getattr(wolf, "uid", None),
                "species": "wolf",
                "chance": round(chance, 3),
                "skill": round(float(getattr(ag, "wolf_tame_skill", 0.0)), 3),
            })
            self._brain_memory(ag, {
                "type": "tame_success",
                "tick": int(getattr(self.world, "tick_count", 0)),
                "actor": getattr(wolf, "uid", "wolf"),
                "data": {"species": "wolf", "chance": round(chance, 3)},
            })
            self._brain_combat_feedback(ag, hp_delta=+1.0, dist_delta=0.0, done=True)
            self._share_taming_experience(
                teacher=ag,
                source_wolf=wolf,
                teacher_gain=(new_skill - skill),
                outcome="success",
                chance=chance,
            )
            return

        # Даже неудачные попытки дают небольшой прогресс обучения.
        progress = _clamp(skill + (0.08 + 0.04 * (1.0 - fear)), 0.0, 1.0)
        setattr(ag, "wolf_tame_skill", progress)
        self._event({
            "type": "tame_progress",
            "who": aid,
            "name": name,
            "animal_id": getattr(wolf, "uid", None),
            "species": "wolf",
            "chance": round(chance, 3),
            "progress": round(progress, 3),
        })
        self._brain_memory(ag, {
            "type": "tame_progress",
            "tick": int(getattr(self.world, "tick_count", 0)),
            "actor": getattr(wolf, "uid", "wolf"),
            "data": {"species": "wolf", "progress": round(progress, 3), "chance": round(chance, 3)},
        })
        self._share_taming_experience(
            teacher=ag,
            source_wolf=wolf,
            teacher_gain=(progress - skill),
            outcome="progress",
            chance=chance,
        )

    # ------------------------------------------------------------------
    # Защита хозяина прирученными волками
    # ------------------------------------------------------------------

    def _defense_step(self) -> None:
        now = int(getattr(self.world, "tick_count", 0))
        for wolf in self._wolves():
            if not self._is_alive_animal(wolf):
                continue

            owner_id = getattr(wolf, "tamed_by", None)
            if not owner_id:
                continue
            owner = self._owner(owner_id)
            if owner is None or not self._is_alive_agent(owner):
                continue

            target = self._pick_threat_for_owner(owner, wolf)
            if target is None:
                setattr(wolf, "last_action", f"guard {owner_id}")
                continue

            wx, wy = float(getattr(wolf, "x", 0.0)), float(getattr(wolf, "y", 0.0))
            tx, ty = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
            d2 = _dist2(wx, wy, tx, ty)

            if d2 <= (self.defend_bite_range * self.defend_bite_range):
                wid = str(getattr(wolf, "uid", "wolf"))
                if (now - self._last_defend_strike.get(wid, -10**9)) < max(1, self.defend_cooldown_ticks):
                    continue
                self._last_defend_strike[wid] = now
                self._wolf_bite_target(wolf, owner, target)
            else:
                self._wolf_chase_target(wolf, target)

    def _pick_threat_for_owner(self, owner: Any, pet_wolf: Any) -> Optional[Any]:
        # 1) Приоритет: последний обидчик хозяина.
        last = getattr(owner, "last_attacker_id", None)
        if last and isinstance(getattr(self.world, "animals", None), dict):
            cand = self.world.animals.get(last)
            if cand is not None and cand is not pet_wolf and self._is_hostile_to_owner(cand, owner):
                return cand

        # 2) Иначе — ближайший агрессивный дикий зверь у хозяина.
        ox, oy = float(getattr(owner, "x", 0.0)), float(getattr(owner, "y", 0.0))
        best = None
        best_d2 = self.defend_radius * self.defend_radius
        for ani in self._animals():
            if ani is pet_wolf:
                continue
            if not self._is_hostile_to_owner(ani, owner):
                continue
            d2 = _dist2(ox, oy, float(getattr(ani, "x", 0.0)), float(getattr(ani, "y", 0.0)))
            if d2 <= best_d2:
                best_d2 = d2
                best = ani
        return best

    def _is_hostile_to_owner(self, ani: Any, owner: Any) -> bool:
        if not self._is_alive_animal(ani):
            return False
        if getattr(ani, "tamed_by", None):
            return False
        try:
            if not bool(getattr(getattr(ani, "species", None), "aggressive", False)):
                return False
        except Exception:
            return False
        return True

    def _wolf_bite_target(self, wolf: Any, owner: Any, target: Any) -> None:
        dmg = max(1.0, self.defend_damage * (0.85 + 0.3 * self.random.random()))
        hp_before = float(getattr(target, "hp", 0.0))
        hp_after = max(0.0, hp_before - dmg)
        setattr(target, "hp", hp_after)
        setattr(wolf, "last_action", f"defend_bite {getattr(target, 'uid', '?')}")

        oid = str(getattr(owner, "id", getattr(owner, "agent_id", "?")))
        oname = str(getattr(owner, "name", oid))
        self._event({
            "type": "pet_defend",
            "owner_id": oid,
            "owner_name": oname,
            "pet_id": getattr(wolf, "uid", None),
            "target_id": getattr(target, "uid", None),
            "damage": round(dmg, 2),
            "target_hp": round(hp_after, 2),
        })
        self._brain_memory(owner, {
            "type": "pet_defend",
            "tick": int(getattr(self.world, "tick_count", 0)),
            "actor": getattr(wolf, "uid", "wolf_pet"),
            "data": {
                "target": getattr(target, "uid", "?"),
                "damage": round(dmg, 2),
            },
        })
        self._brain_combat_feedback(owner, hp_delta=0.0, dist_delta=-0.5, done=(hp_after <= 0.0))

        if hp_after <= 0.0:
            self._chat(
                f"[pet] волк {getattr(wolf, 'uid', '?')} защитил {oname} и добил "
                f"{getattr(target, 'uid', '?')}"
            )
            # Дополнительный маленький рост боевого навыка хозяина через опыт команды.
            cur = float(getattr(owner, "combat_skill", 0.0))
            setattr(owner, "combat_skill", _clamp(cur + 0.04, 0.0, 5.0))

    def _wolf_chase_target(self, wolf: Any, target: Any) -> None:
        wx, wy = float(getattr(wolf, "x", 0.0)), float(getattr(wolf, "y", 0.0))
        tx, ty = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
        dx, dy = tx - wx, ty - wy
        dist = math.hypot(dx, dy) + 1e-6
        speed = max(0.7, float(getattr(wolf, "speed", 0.9)))
        nx = wx + (dx / dist) * speed
        ny = wy + (dy / dist) * speed
        nx = _clamp(nx, 0.0, float(getattr(self.world, "width", 100.0)))
        ny = _clamp(ny, 0.0, float(getattr(self.world, "height", 100.0)))
        setattr(wolf, "x", nx)
        setattr(wolf, "y", ny)
        setattr(wolf, "last_action", f"defend_chase {getattr(target, 'uid', '?')}")

    def _share_taming_experience(
        self,
        teacher: Any,
        source_wolf: Any,
        teacher_gain: float,
        outcome: str,
        chance: float,
    ) -> None:
        """
        Передаёт часть опыта приручения союзникам рядом с «учителем».
        Работает как при успехе, так и при неудачной попытке (progress).
        """
        if not self.share_enabled:
            return
        if teacher_gain <= 1e-6:
            return
        radius = max(0.1, float(self.share_radius))
        recipients = self._allies_near(teacher, radius)
        if not recipients:
            return

        now = int(getattr(self.world, "tick_count", 0))
        teacher_id = str(getattr(teacher, "id", getattr(teacher, "agent_id", "?")))
        teacher_name = str(getattr(teacher, "name", teacher_id))
        wolf_id = getattr(source_wolf, "uid", "wolf")
        shared_lines: List[str] = []

        tx, ty = float(getattr(teacher, "x", 0.0)), float(getattr(teacher, "y", 0.0))
        for ally in recipients:
            if not self._is_alive_agent(ally):
                continue
            self._ensure_tame_skill(ally)

            ox, oy = float(getattr(ally, "x", 0.0)), float(getattr(ally, "y", 0.0))
            dist = math.sqrt(_dist2(tx, ty, ox, oy))
            proximity = _clamp(1.0 - (dist / radius), 0.0, 1.0)

            gain_raw = teacher_gain * self.share_ratio * (0.5 + 0.5 * proximity)
            gain = _clamp(gain_raw, self.share_min_gain, self.share_max_gain)

            old_skill = _clamp(float(getattr(ally, "wolf_tame_skill", 0.0)), 0.0, 1.0)
            new_skill = _clamp(old_skill + gain, 0.0, 1.0)
            real_gain = new_skill - old_skill
            if real_gain <= 1e-6:
                continue
            setattr(ally, "wolf_tame_skill", new_skill)

            ally_id = str(getattr(ally, "id", getattr(ally, "agent_id", "?")))
            ally_name = str(getattr(ally, "name", ally_id))
            shared_lines.append(f"{ally_name} +{real_gain:.3f}")

            self._event({
                "type": "tame_peer_share",
                "teacher_id": teacher_id,
                "teacher_name": teacher_name,
                "student_id": ally_id,
                "student_name": ally_name,
                "animal_id": wolf_id,
                "outcome": outcome,
                "source_gain": round(float(teacher_gain), 3),
                "shared_gain": round(float(real_gain), 3),
                "chance": round(float(chance), 3),
            })
            self._brain_memory(ally, {
                "type": "tame_peer_share",
                "tick": now,
                "actor": teacher_id,
                "data": {
                    "teacher": teacher_name,
                    "animal_id": wolf_id,
                    "outcome": outcome,
                    "shared_gain": round(float(real_gain), 3),
                    "chance": round(float(chance), 3),
                },
            })

        if shared_lines:
            suffix = ", ".join(shared_lines[:3])
            if len(shared_lines) > 3:
                suffix += ", ..."
            self._chat(f"[tame] {teacher_name} передал(а) опыт приручения: {suffix}")

    # ------------------------------------------------------------------
    # Вспомогалки
    # ------------------------------------------------------------------

    def _agents(self) -> List[Any]:
        a = getattr(self.world, "agents", {})
        if isinstance(a, dict):
            return list(a.values())
        if isinstance(a, list):
            return a
        return []

    def _animals(self) -> List[Any]:
        if hasattr(self.world, "iter_animals"):
            try:
                return list(self.world.iter_animals())
            except Exception:
                pass
        z = getattr(self.world, "animals", {})
        if isinstance(z, dict):
            return list(z.values())
        if isinstance(z, list):
            return z
        return []

    def _wolves(self) -> List[Any]:
        out: List[Any] = []
        for ani in self._animals():
            sp = getattr(ani, "species", None)
            sid = str(getattr(sp, "species_id", "")).lower() if sp is not None else ""
            if sid == "wolf":
                out.append(ani)
        return out

    def _is_alive_agent(self, ag: Any) -> bool:
        try:
            alive = ag.is_alive() if hasattr(ag, "is_alive") else bool(getattr(ag, "alive", True))
        except Exception:
            alive = True
        return bool(alive) and float(getattr(ag, "health", 0.0)) > 0.0

    def _is_alive_animal(self, ani: Any) -> bool:
        try:
            alive = ani.is_alive() if hasattr(ani, "is_alive") else (float(getattr(ani, "hp", 0.0)) > 0.0)
        except Exception:
            alive = True
        return bool(alive) and float(getattr(ani, "hp", 0.0)) > 0.0

    def _owner(self, owner_id: str) -> Optional[Any]:
        if hasattr(self.world, "get_agent_by_id"):
            try:
                return self.world.get_agent_by_id(owner_id)
            except Exception:
                return None
        return None

    def _nearest_untamed_wolf(self, ag: Any, wolves: List[Any], radius: float) -> Optional[Any]:
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        best = None
        best_d2 = radius * radius
        for w in wolves:
            d2 = _dist2(ax, ay, float(getattr(w, "x", 0.0)), float(getattr(w, "y", 0.0)))
            if d2 <= best_d2:
                best_d2 = d2
                best = w
        return best

    def _allies_near(self, ag: Any, radius: float) -> List[Any]:
        aid = str(getattr(ag, "id", getattr(ag, "agent_id", "")))
        if not aid:
            return []
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        r2 = radius * radius
        out: List[Any] = []
        for other in self._agents():
            oid = str(getattr(other, "id", getattr(other, "agent_id", "")))
            if not oid or oid == aid:
                continue
            if not self._is_alive_agent(other):
                continue
            d2 = _dist2(ax, ay, float(getattr(other, "x", 0.0)), float(getattr(other, "y", 0.0)))
            if d2 <= r2:
                out.append(other)
        return out

    def _ensure_tame_skill(self, ag: Any) -> None:
        if not hasattr(ag, "wolf_tame_skill"):
            setattr(ag, "wolf_tame_skill", 0.0)
        if not hasattr(ag, "wolf_tame_successes"):
            setattr(ag, "wolf_tame_successes", 0)

    def _brain_memory(self, ag: Any, ev: Dict[str, Any]) -> None:
        try:
            br = getattr(ag, "brain", None)
            if br and hasattr(br, "add_memory"):
                br.add_memory(ev)
        except Exception:
            pass

    def _brain_combat_feedback(self, ag: Any, hp_delta: float, dist_delta: float, done: bool) -> None:
        try:
            br = getattr(ag, "brain", None)
            if br and hasattr(br, "note_combat_feedback"):
                br.note_combat_feedback(hp_delta=hp_delta, dist_delta=dist_delta, done=done)
        except Exception:
            pass

    def _event(self, data: Dict[str, Any]) -> None:
        if hasattr(self.world, "add_event"):
            try:
                self.world.add_event(data)
            except Exception:
                pass

    def _chat(self, line: str) -> None:
        if hasattr(self.world, "add_chat_line"):
            try:
                self.world.add_chat_line(str(line))
            except Exception:
                pass
