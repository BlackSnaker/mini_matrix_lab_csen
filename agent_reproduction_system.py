from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import random

import config
from agent import Agent
from mind_core import Belief, ConsciousnessBlock


def _safe_float(v: Any, fallback: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return fallback


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _as_percent(v: Any, fallback: float = 0.0) -> float:
    """
    Нормализация шкал:
    - если метрика хранится как 0..1 -> переводим в 0..100
    - если уже 0..100 -> оставляем как есть
    """
    x = _safe_float(v, fallback)
    if x <= 1.5:
        return x * 100.0
    return x


class AgentReproductionSystem:
    """
    Отдельный модуль размножения агентов:
      1) поиск партнёра у взрослых и достаточно «ресурсных» агентов;
      2) рождение нового агента при близости пары;
      3) передача части опыта/убеждений новому поколению.

    Встроен мягко и работает через duck-typing (getattr/setattr), чтобы не
    ломать существующие классы.
    """

    def __init__(self, world: Any) -> None:
        self.world = world
        self.random = random.Random(2026)

        self.enabled = bool(getattr(config, "AGENT_REPRO_ENABLED", True))
        self.max_population = int(getattr(config, "AGENT_REPRO_MAX_POPULATION", 10))

        self.maturity_ticks = int(getattr(config, "AGENT_REPRO_MATURITY_TICKS", 180))
        self.cooldown_ticks = int(getattr(config, "AGENT_REPRO_COOLDOWN_TICKS", 260))

        self.search_radius = float(getattr(config, "AGENT_REPRO_SEARCH_RADIUS", 24.0))
        self.mate_radius = float(getattr(config, "AGENT_REPRO_MATE_RADIUS", 2.2))
        self.search_interval = int(getattr(config, "AGENT_REPRO_SEARCH_INTERVAL", 18))

        self.min_health = float(getattr(config, "AGENT_REPRO_HEALTH_MIN", 55.0))
        self.min_energy = float(getattr(config, "AGENT_REPRO_ENERGY_MIN", 40.0))
        self.max_hunger = float(getattr(config, "AGENT_REPRO_HUNGER_MAX", 70.0))
        self.max_fear = float(getattr(config, "AGENT_REPRO_FEAR_MAX", 0.55))

        self.belief_keep = int(getattr(config, "AGENT_REPRO_INHERIT_BELIEFS", 18))
        self.rule_noise = float(getattr(config, "AGENT_REPRO_RULE_NOISE", 0.05))
        self.skill_noise = float(getattr(config, "AGENT_REPRO_SKILL_NOISE", 0.06))
        self.role_mutation_prob = float(getattr(config, "AGENT_REPRO_ROLE_MUTATION_PROB", 0.12))
        self.roles: Tuple[str, ...] = ("balanced", "scout", "protector", "tamer", "medic")

        self._last_repro_tick: Dict[str, int] = {}
        self._last_seek_tick: Dict[str, int] = {}
        self._child_counter = 0

    def step(self) -> None:
        if not self.enabled or not self.world:
            return

        agents = [a for a in self._agents() if self._is_alive_agent(a)]
        if len(agents) < 2:
            return

        # Наследуемым полям даём дефолт для «старых» агентов.
        for ag in agents:
            if not hasattr(ag, "generation"):
                setattr(ag, "generation", 0)
            if not hasattr(ag, "lineage_role"):
                setattr(ag, "lineage_role", self._derive_role(ag))

        now = int(getattr(self.world, "tick_count", 0))
        eligible = [a for a in agents if self._can_reproduce(a, now)]

        # Сначала партнёр-поиск, чтобы агенты сближались.
        self._seek_partners(eligible, now)

        if len(eligible) < 2:
            return
        if len(self._agents()) >= max(2, self.max_population):
            return

        used: set[str] = set()
        for ag in eligible:
            aid = self._agent_id(ag)
            if not aid or aid in used:
                continue

            partner = self._nearest_partner(ag, eligible, blocked_ids=used, within=self.mate_radius)
            if partner is None:
                continue
            bid = self._agent_id(partner)
            if not bid or bid in used:
                continue

            child = self._spawn_child(ag, partner, now)
            if child is None:
                continue

            used.add(aid)
            used.add(bid)
            self._last_repro_tick[aid] = now
            self._last_repro_tick[bid] = now
            cid = self._agent_id(child)
            if cid:
                self._last_repro_tick[cid] = now

            if len(self._agents()) >= max(2, self.max_population):
                break

    # ------------------------------------------------------------------
    # Партнёр-поиск
    # ------------------------------------------------------------------

    def _seek_partners(self, eligible: List[Any], now: int) -> None:
        if not eligible:
            return
        if self.search_radius <= 0.0:
            return

        for ag in eligible:
            aid = self._agent_id(ag)
            if not aid:
                continue
            last = self._last_seek_tick.get(aid, -10**9)
            if (now - last) < max(1, self.search_interval):
                continue

            partner = self._nearest_partner(ag, eligible, blocked_ids={aid}, within=self.search_radius)
            if partner is None:
                continue

            ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
            px, py = float(getattr(partner, "x", 0.0)), float(getattr(partner, "y", 0.0))
            d2 = _dist2(ax, ay, px, py)
            if d2 <= self.mate_radius * self.mate_radius:
                self._last_seek_tick[aid] = now
                continue

            jitter = 0.8
            gx = _clamp(px + self.random.uniform(-jitter, jitter), 0.0, float(getattr(self.world, "width", 100.0)))
            gy = _clamp(py + self.random.uniform(-jitter, jitter), 0.0, float(getattr(self.world, "height", 100.0)))
            setattr(ag, "goal_x", gx)
            setattr(ag, "goal_y", gy)
            self._last_seek_tick[aid] = now

            self._event({
                "type": "seek_partner",
                "who": aid,
                "name": str(getattr(ag, "name", aid)),
                "target": self._agent_id(partner),
            })
            self._brain_memory(ag, {
                "type": "seek_partner",
                "tick": now,
                "actor": self._agent_id(partner),
                "data": {"goal": (round(gx, 2), round(gy, 2))},
            })

    # ------------------------------------------------------------------
    # Рождение / наследование
    # ------------------------------------------------------------------

    def _spawn_child(self, parent_a: Any, parent_b: Any, now: int) -> Optional[Agent]:
        aid = self._agent_id(parent_a) or "pA"
        bid = self._agent_id(parent_b) or "pB"

        child_id = self._make_child_id(now)
        gen = max(int(getattr(parent_a, "generation", 0)), int(getattr(parent_b, "generation", 0))) + 1

        ax, ay = float(getattr(parent_a, "x", 0.0)), float(getattr(parent_a, "y", 0.0))
        bx, by = float(getattr(parent_b, "x", 0.0)), float(getattr(parent_b, "y", 0.0))
        cx = _clamp((ax + bx) * 0.5 + self.random.uniform(-0.6, 0.6), 0.0, float(getattr(self.world, "width", 100.0)))
        cy = _clamp((ay + by) * 0.5 + self.random.uniform(-0.6, 0.6), 0.0, float(getattr(self.world, "height", 100.0)))

        # Наследование «прикладных» навыков.
        combat_seed = (
            (_safe_float(getattr(parent_a, "combat_skill", 0.0), 0.0) +
             _safe_float(getattr(parent_b, "combat_skill", 0.0), 0.0)) * 0.5
        )
        tame_seed = (
            (_safe_float(getattr(parent_a, "wolf_tame_skill", 0.0), 0.0) +
             _safe_float(getattr(parent_b, "wolf_tame_skill", 0.0), 0.0)) * 0.5
        )
        role = self._choose_child_role(parent_a, parent_b, combat_seed, tame_seed)
        persona = self._persona_for_role(role)
        name = f"Gen{gen}_{self._child_counter}"
        brain = self._build_child_brain(child_id, parent_a, parent_b)

        child = Agent(
            agent_id=child_id,
            name=name,
            persona=persona,
            x=cx,
            y=cy,
            goal_x=cx,
            goal_y=cy,
            brain=brain,
        )
        child.combat_skill = _clamp(combat_seed * 0.75 + self.random.uniform(-self.skill_noise, self.skill_noise), 0.0, 5.0)
        child.wolf_tame_skill = _clamp(tame_seed * 0.80 + self.random.uniform(-0.02, 0.05), 0.0, 1.0)
        child.wolf_tame_successes = int(
            max(0, round(
                (_safe_float(getattr(parent_a, "wolf_tame_successes", 0), 0.0) +
                 _safe_float(getattr(parent_b, "wolf_tame_successes", 0), 0.0)) * 0.10
            ))
        )

        child.fear = _clamp(
            (
                _safe_float(getattr(parent_a, "fear", 0.0), 0.0) +
                _safe_float(getattr(parent_b, "fear", 0.0), 0.0)
            ) * 0.3,
            0.0,
            1.0,
        )
        child.health = 100.0
        child.energy = 1.0
        child.hunger = 0.0
        child.age_ticks = 0
        self._apply_role_traits(child, role)

        setattr(child, "generation", gen)
        setattr(child, "lineage_role", role)
        setattr(child, "parents", [aid, bid])
        setattr(child, "born_tick", now)
        child.tags = list({*(list(getattr(child, "tags", []) or [])), f"role:{role}"})

        self.world.add_agent(child)

        # Стоимость для родителей, чтобы размножение не было «бесплатным».
        self._apply_parent_cost(parent_a)
        self._apply_parent_cost(parent_b)

        aname = str(getattr(parent_a, "name", aid))
        bname = str(getattr(parent_b, "name", bid))
        self._chat(f"[gen] {aname} + {bname} -> {name} (поколение {gen})")
        self._event({
            "type": "agent_birth",
            "child_id": child_id,
            "child_name": name,
            "generation": gen,
            "lineage_role": role,
            "parent_roles": [
                self._normalize_role(getattr(parent_a, "lineage_role", self._derive_role(parent_a))),
                self._normalize_role(getattr(parent_b, "lineage_role", self._derive_role(parent_b))),
            ],
            "parents": [aid, bid],
            "pos": (round(cx, 2), round(cy, 2)),
            "combat_seed": round(float(child.combat_skill), 3),
            "tame_seed": round(float(child.wolf_tame_skill), 3),
        })

        self._brain_memory(parent_a, {
            "type": "offspring_born",
            "tick": now,
            "actor": child_id,
            "data": {"partner": bid, "child": child_id, "generation": gen, "lineage_role": role},
        })
        self._brain_memory(parent_b, {
            "type": "offspring_born",
            "tick": now,
            "actor": child_id,
            "data": {"partner": aid, "child": child_id, "generation": gen, "lineage_role": role},
        })
        self._brain_memory(child, {
            "type": "lineage_inherit",
            "tick": now,
            "actor": "lineage",
            "data": {
                "parents": [aid, bid],
                "generation": gen,
                "lineage_role": role,
                "combat_seed": round(float(child.combat_skill), 3),
                "tame_seed": round(float(child.wolf_tame_skill), 3),
            },
        })
        return child

    def _build_child_brain(self, child_id: str, parent_a: Any, parent_b: Any) -> ConsciousnessBlock:
        br_a = getattr(parent_a, "brain", None)
        br_b = getattr(parent_b, "brain", None)
        donor = self._pick_donor_brain(br_a, br_b)

        child_brain: Optional[ConsciousnessBlock] = None
        if donor is not None and hasattr(donor, "to_dict"):
            try:
                data = donor.to_dict()
                if isinstance(data, dict):
                    data["agent_id"] = child_id
                    data["memory_tail"] = []
                    data["age_ticks"] = 0
                    data["alive"] = True
                    data["health"] = 100.0
                    data["energy"] = 100.0
                    data["hunger"] = 0.0
                    data["fear_level"] = 0.0
                    data["current_drive"] = "idle"
                    data["ally_anchor"] = None
                    data["last_death_reason"] = None
                    data["trauma_map"] = []
                    child_brain = ConsciousnessBlock.from_dict(data)
            except Exception:
                child_brain = None

        if child_brain is None:
            child_brain = ConsciousnessBlock(agent_id=child_id)

        # Смешиваем поведенческие правила родителей.
        self._blend_behavior_rules(child_brain, br_a, br_b)
        # Переносим полезные убеждения (с затуханием силы).
        self._blend_beliefs(child_brain, br_a, br_b)
        return child_brain

    def _blend_behavior_rules(self, child_brain: Any, br_a: Any, br_b: Any) -> None:
        r_child = getattr(child_brain, "behavior_rules", None)
        if r_child is None:
            return

        r_a = getattr(br_a, "behavior_rules", None) if br_a is not None else None
        r_b = getattr(br_b, "behavior_rules", None) if br_b is not None else None

        def _avg_rule(key: str, fallback: float) -> float:
            vals: List[float] = []
            if r_a is not None and hasattr(r_a, key):
                vals.append(_safe_float(getattr(r_a, key, fallback), fallback))
            if r_b is not None and hasattr(r_b, key):
                vals.append(_safe_float(getattr(r_b, key, fallback), fallback))
            if not vals:
                vals.append(_safe_float(getattr(r_child, key, fallback), fallback))
            return sum(vals) / max(1, len(vals))

        avoid = _avg_rule("avoid_hazard_radius", _safe_float(getattr(r_child, "avoid_hazard_radius", 6.0), 6.0))
        heal = _avg_rule("healing_zone_seek_priority", _safe_float(getattr(r_child, "healing_zone_seek_priority", 0.5), 0.5))
        ally = _avg_rule("stick_with_ally_if_fear_above", _safe_float(getattr(r_child, "stick_with_ally_if_fear_above", 0.7), 0.7))
        explore = _avg_rule("exploration_bias", _safe_float(getattr(r_child, "exploration_bias", 0.2), 0.2))

        setattr(r_child, "avoid_hazard_radius", _clamp(avoid + self.random.uniform(-0.6, 0.6), 1.0, 20.0))
        setattr(r_child, "healing_zone_seek_priority", _clamp(heal + self.random.uniform(-self.rule_noise, self.rule_noise), 0.0, 1.0))
        setattr(r_child, "stick_with_ally_if_fear_above", _clamp(ally + self.random.uniform(-self.rule_noise, self.rule_noise), 0.0, 1.0))
        setattr(r_child, "exploration_bias", _clamp(explore + self.random.uniform(-self.rule_noise, self.rule_noise), 0.0, 1.0))

    def _blend_beliefs(self, child_brain: Any, br_a: Any, br_b: Any) -> None:
        bag: Dict[Tuple[str, str], float] = {}

        for br in (br_a, br_b):
            for cond, concl, strength in self._iter_beliefs(br):
                key = (cond, concl)
                old = bag.get(key, 0.0)
                bag[key] = max(old, _clamp(strength, 0.0, 1.0))

        if not bag:
            return

        ranked = sorted(bag.items(), key=lambda kv: kv[1], reverse=True)
        keep_n = max(4, int(self.belief_keep))
        child_beliefs: List[Belief] = []
        for (cond, concl), s in ranked[:keep_n]:
            inherited = _clamp(s * 0.75, 0.0, 1.0)
            child_beliefs.append(Belief(condition=cond, conclusion=concl, strength=inherited))

        if child_beliefs:
            setattr(child_brain, "beliefs", child_beliefs)

    def _normalize_role(self, role: Any) -> str:
        r = str(role or "").strip().lower()
        if r in self.roles:
            return r
        return "balanced"

    def _derive_role(self, ag: Any) -> str:
        role = self._normalize_role(getattr(ag, "lineage_role", ""))
        if role != "balanced":
            return role

        combat = _safe_float(getattr(ag, "combat_skill", 0.0), 0.0)
        tame = _safe_float(getattr(ag, "wolf_tame_skill", 0.0), 0.0)
        fear = _clamp(_safe_float(getattr(ag, "fear", 0.0), 0.0), 0.0, 1.0)

        br = getattr(ag, "brain", None)
        heal_prio = 0.5
        if br is not None and getattr(br, "behavior_rules", None) is not None:
            heal_prio = _safe_float(getattr(br.behavior_rules, "healing_zone_seek_priority", 0.5), 0.5)

        if tame >= 0.55:
            return "tamer"
        if combat >= 1.6:
            return "protector"
        if heal_prio >= 0.72 or fear >= 0.62:
            return "medic"
        if fear <= 0.18:
            return "scout"
        return "balanced"

    def _choose_child_role(self, parent_a: Any, parent_b: Any, combat_seed: float, tame_seed: float) -> str:
        ra = self._normalize_role(getattr(parent_a, "lineage_role", self._derive_role(parent_a)))
        rb = self._normalize_role(getattr(parent_b, "lineage_role", self._derive_role(parent_b)))

        if ra == rb:
            base = ra
        else:
            # Мягкая доминантность: выбираем роль, которая лучше объясняет профиль пары.
            avg_fear = 0.5 * (
                _clamp(_safe_float(getattr(parent_a, "fear", 0.0), 0.0), 0.0, 1.0) +
                _clamp(_safe_float(getattr(parent_b, "fear", 0.0), 0.0), 0.0, 1.0)
            )
            if tame_seed >= max(0.48, combat_seed * 0.75):
                base = "tamer"
            elif combat_seed >= max(1.0, tame_seed * 2.1):
                base = "protector"
            elif avg_fear >= 0.45:
                base = "medic"
            elif avg_fear <= 0.20:
                base = "scout"
            else:
                base = self.random.choice([ra, rb, "balanced"])

        # Редкая мутация роли, чтобы популяция не застревала.
        if self.random.random() < _clamp(self.role_mutation_prob, 0.0, 1.0):
            candidates = [r for r in self.roles if r != base]
            if candidates:
                base = self.random.choice(candidates)
        return self._normalize_role(base)

    def _persona_for_role(self, role: str) -> str:
        role = self._normalize_role(role)
        mapping = {
            "scout": "lineage/scout-explorer",
            "protector": "lineage/protector-guardian",
            "tamer": "lineage/tamer-keeper",
            "medic": "lineage/medic-support",
            "balanced": "lineage/balanced-survivor",
        }
        return mapping.get(role, mapping["balanced"])

    def _apply_role_traits(self, child: Any, role: str) -> None:
        role = self._normalize_role(role)

        # Лёгкие стартовые сдвиги по телу/поведению, без жёсткой специализации.
        if role == "scout":
            child.move_speed = _safe_float(getattr(child, "move_speed", 3.0), 3.0) * 1.12
            child.fear = _clamp(_safe_float(getattr(child, "fear", 0.0), 0.0) - 0.06, 0.0, 1.0)
        elif role == "protector":
            child.attack_power = _safe_float(getattr(child, "attack_power", 8.0), 8.0) * 1.12
            child.attack_cooldown = max(8, int(_safe_float(getattr(child, "attack_cooldown", 18), 18.0) * 0.90))
            child.fear = _clamp(_safe_float(getattr(child, "fear", 0.0), 0.0) - 0.03, 0.0, 1.0)
        elif role == "tamer":
            child.wolf_tame_skill = _clamp(_safe_float(getattr(child, "wolf_tame_skill", 0.0), 0.0) + 0.08, 0.0, 1.0)
        elif role == "medic":
            child.fear = _clamp(_safe_float(getattr(child, "fear", 0.0), 0.0) - 0.02, 0.0, 1.0)

        br = getattr(child, "brain", None)
        rules = getattr(br, "behavior_rules", None) if br is not None else None
        if rules is None:
            return
        if role == "scout":
            rules.exploration_bias = _clamp(_safe_float(getattr(rules, "exploration_bias", 0.2), 0.2) + 0.10, 0.0, 1.0)
        elif role == "protector":
            rules.stick_with_ally_if_fear_above = _clamp(
                _safe_float(getattr(rules, "stick_with_ally_if_fear_above", 0.7), 0.7) - 0.05,
                0.0,
                1.0,
            )
        elif role == "tamer":
            rules.exploration_bias = _clamp(_safe_float(getattr(rules, "exploration_bias", 0.2), 0.2) + 0.04, 0.0, 1.0)
        elif role == "medic":
            rules.healing_zone_seek_priority = _clamp(
                _safe_float(getattr(rules, "healing_zone_seek_priority", 0.5), 0.5) + 0.14,
                0.0,
                1.0,
            )

    def _iter_beliefs(self, brain: Any) -> List[Tuple[str, str, float]]:
        if brain is None:
            return []
        out: List[Tuple[str, str, float]] = []
        for b in list(getattr(brain, "beliefs", []) or []):
            cond = ""
            concl = ""
            strength = 0.0
            try:
                if hasattr(b, "condition"):
                    cond = str(getattr(b, "condition", ""))
                    concl = str(getattr(b, "conclusion", ""))
                    strength = _safe_float(getattr(b, "strength", 0.0), 0.0)
                elif isinstance(b, dict):
                    cond = str(b.get("if") or b.get("condition") or "")
                    concl = str(b.get("then") or b.get("conclusion") or "")
                    strength = _safe_float(b.get("strength", 0.0), 0.0)
            except Exception:
                cond = ""
                concl = ""
                strength = 0.0
            if cond and concl:
                out.append((cond, concl, _clamp(strength, 0.0, 1.0)))
        return out

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

    def _agent_id(self, ag: Any) -> str:
        return str(getattr(ag, "id", getattr(ag, "agent_id", "")) or "")

    def _is_alive_agent(self, ag: Any) -> bool:
        try:
            alive = ag.is_alive() if hasattr(ag, "is_alive") else bool(getattr(ag, "alive", True))
        except Exception:
            alive = True
        health = _safe_float(getattr(ag, "health", 0.0), 0.0)
        return bool(alive) and health > 0.0

    def _can_reproduce(self, ag: Any, now_tick: int) -> bool:
        if not self._is_alive_agent(ag):
            return False
        if len(self._agents()) >= max(2, self.max_population):
            return False

        aid = self._agent_id(ag)
        if not aid:
            return False
        last = self._last_repro_tick.get(aid, -10**9)
        if (now_tick - last) < max(1, self.cooldown_ticks):
            return False

        age = int(getattr(ag, "age_ticks", 0))
        if age < max(1, self.maturity_ticks):
            return False

        health = _safe_float(getattr(ag, "health", 0.0), 0.0)
        energy = _as_percent(getattr(ag, "energy", 0.0), 0.0)
        hunger = _as_percent(getattr(ag, "hunger", 0.0), 0.0)
        fear = _clamp(_safe_float(getattr(ag, "fear", 0.0), 0.0), 0.0, 1.0)

        if health < self.min_health:
            return False
        if energy < self.min_energy:
            return False
        if hunger > self.max_hunger:
            return False
        if fear > self.max_fear:
            return False
        return True

    def _nearest_partner(self, me: Any, pool: List[Any], blocked_ids: set[str], within: float) -> Optional[Any]:
        mid = self._agent_id(me)
        if not mid:
            return None
        mx, my = _safe_float(getattr(me, "x", 0.0), 0.0), _safe_float(getattr(me, "y", 0.0), 0.0)
        lim2 = max(0.0, within) * max(0.0, within)

        best = None
        best_d2 = lim2
        for other in pool:
            oid = self._agent_id(other)
            if not oid or oid == mid or oid in blocked_ids:
                continue
            ox, oy = _safe_float(getattr(other, "x", 0.0), 0.0), _safe_float(getattr(other, "y", 0.0), 0.0)
            d2 = _dist2(mx, my, ox, oy)
            if d2 <= best_d2:
                best_d2 = d2
                best = other
        return best

    def _make_child_id(self, now_tick: int) -> str:
        while True:
            self._child_counter += 1
            cid = f"gen_{now_tick}_{self._child_counter}"
            if cid not in getattr(self.world, "agents", {}):
                return cid

    def _pick_donor_brain(self, br_a: Any, br_b: Any) -> Optional[Any]:
        def _score(br: Any) -> float:
            if br is None:
                return -1.0
            surv = _safe_float(getattr(br, "survival_score", 0.0), 0.0)
            steps = _safe_float(getattr(br, "gc_steps", 0.0), 0.0)
            return surv * 1000.0 + steps

        return br_a if _score(br_a) >= _score(br_b) else br_b

    def _apply_parent_cost(self, ag: Any) -> None:
        # здоровье
        hp = _safe_float(getattr(ag, "health", 100.0), 100.0)
        setattr(ag, "health", _clamp(hp - self.random.uniform(2.0, 5.0), 1.0, 100.0))

        # энергия: поддерживаем обе шкалы (0..1 и 0..100)
        energy = _safe_float(getattr(ag, "energy", 0.0), 0.0)
        if energy <= 1.5:
            setattr(ag, "energy", _clamp(energy - self.random.uniform(0.08, 0.16), 0.0, 1.0))
        else:
            setattr(ag, "energy", _clamp(energy - self.random.uniform(8.0, 16.0), 0.0, 100.0))

        hunger = _safe_float(getattr(ag, "hunger", 0.0), 0.0)
        if hunger <= 1.5:
            setattr(ag, "hunger", _clamp(hunger + self.random.uniform(0.06, 0.15), 0.0, 1.0))
        else:
            setattr(ag, "hunger", _clamp(hunger + self.random.uniform(6.0, 15.0), 0.0, 100.0))

        fear = _safe_float(getattr(ag, "fear", 0.0), 0.0)
        setattr(ag, "fear", _clamp(fear + self.random.uniform(0.02, 0.06), 0.0, 1.0))

    def _brain_memory(self, ag: Any, ev: Dict[str, Any]) -> None:
        try:
            br = getattr(ag, "brain", None)
            if br is not None and hasattr(br, "add_memory"):
                br.add_memory(ev)
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
