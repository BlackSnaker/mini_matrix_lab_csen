from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BeliefKey = Tuple[str, str]


def _safe_float(v: Any, fallback: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return fallback
    if x != x:
        return fallback
    if x in (float("inf"), float("-inf")):
        return fallback
    return x


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class AgentLearningSnapshot:
    tick: int = 0
    combat_skill: float = 0.0
    wolf_tame_skill: float = 0.0
    wolf_tame_successes: int = 0
    tame_success_events: int = 0
    tame_progress_events: int = 0
    tame_peer_share_events: int = 0
    last_tame_peer_hint: str = ""
    seek_partner_events: int = 0
    offspring_born_events: int = 0
    lineage_inherit_events: int = 0
    gc_steps: int = 0
    goals: Tuple[str, ...] = field(default_factory=tuple)
    beliefs: Dict[BeliefKey, float] = field(default_factory=dict)
    rules: Dict[str, float] = field(default_factory=dict)


@dataclass
class LearningDelta:
    learned: List[str] = field(default_factory=list)
    strengthened_links: List[str] = field(default_factory=list)
    changed_beliefs: List[str] = field(default_factory=list)
    peer_tame_lessons: int = 0

    def has_signal(self) -> bool:
        return bool(self.learned or self.strengthened_links or self.changed_beliefs)


def capture_learning_snapshot(agent: Any, tick: int) -> AgentLearningSnapshot:
    br = getattr(agent, "brain", None)

    goals: List[str] = []
    gc_steps = 0
    tame_success_events = 0
    tame_progress_events = 0
    tame_peer_share_events = 0
    last_tame_peer_hint = ""
    seek_partner_events = 0
    offspring_born_events = 0
    lineage_inherit_events = 0
    beliefs: Dict[BeliefKey, float] = {}
    rules: Dict[str, float] = {}

    if br is not None:
        try:
            gc_steps = int(getattr(br, "gc_steps", 0))
        except Exception:
            gc_steps = 0

        try:
            gv = getattr(br, "gc_goal_vocab", {}) or {}
            if isinstance(gv, dict):
                goals = sorted(str(k) for k in gv.keys())
        except Exception:
            goals = []

        # Считаем в памяти явные события приручения/передачи опыта.
        # Это нужно, чтобы маленькие peer-share приросты (<0.01) тоже были видны в трейнерах.
        mem_tail = list(getattr(br, "memory_tail", []) or [])
        for ev in mem_tail:
            etype = ""
            data: Dict[str, Any] = {}
            try:
                if hasattr(ev, "etype"):
                    etype = str(getattr(ev, "etype", ""))
                    raw_data = getattr(ev, "data", {})
                    if isinstance(raw_data, dict):
                        data = raw_data
                elif isinstance(ev, dict):
                    etype = str(ev.get("etype") or ev.get("type") or "")
                    raw_data = ev.get("data", {})
                    if isinstance(raw_data, dict):
                        data = raw_data
            except Exception:
                etype = ""
                data = {}

            if etype == "tame_success":
                tame_success_events += 1
            elif etype == "tame_progress":
                tame_progress_events += 1
            elif etype == "tame_peer_share":
                tame_peer_share_events += 1
                teacher = str(data.get("teacher", "ally"))
                gain = _safe_float(data.get("shared_gain", 0.0), 0.0)
                outcome = str(data.get("outcome", "share"))
                last_tame_peer_hint = f"{teacher} ({outcome}, +{gain:.3f})"
            elif etype == "seek_partner":
                seek_partner_events += 1
            elif etype == "offspring_born":
                offspring_born_events += 1
            elif etype == "lineage_inherit":
                lineage_inherit_events += 1

        for b in list(getattr(br, "beliefs", []) or []):
            cond = str(getattr(b, "condition", ""))
            concl = str(getattr(b, "conclusion", ""))
            if not cond or not concl:
                continue
            beliefs[(cond, concl)] = _clamp(_safe_float(getattr(b, "strength", 0.0), 0.0), 0.0, 1.0)

        r = getattr(br, "behavior_rules", None)
        if r is not None:
            for key in (
                "avoid_hazard_radius",
                "healing_zone_seek_priority",
                "stick_with_ally_if_fear_above",
                "exploration_bias",
            ):
                rules[key] = _safe_float(getattr(r, key, 0.0), 0.0)

    return AgentLearningSnapshot(
        tick=int(tick),
        combat_skill=_safe_float(getattr(agent, "combat_skill", 0.0), 0.0),
        wolf_tame_skill=_safe_float(getattr(agent, "wolf_tame_skill", 0.0), 0.0),
        wolf_tame_successes=int(getattr(agent, "wolf_tame_successes", 0)),
        tame_success_events=int(tame_success_events),
        tame_progress_events=int(tame_progress_events),
        tame_peer_share_events=int(tame_peer_share_events),
        last_tame_peer_hint=last_tame_peer_hint,
        seek_partner_events=int(seek_partner_events),
        offspring_born_events=int(offspring_born_events),
        lineage_inherit_events=int(lineage_inherit_events),
        gc_steps=int(gc_steps),
        goals=tuple(goals),
        beliefs=beliefs,
        rules=rules,
    )


def diff_learning(prev: AgentLearningSnapshot, curr: AgentLearningSnapshot) -> LearningDelta:
    out = LearningDelta()

    d_combat = curr.combat_skill - prev.combat_skill
    if d_combat >= 0.01:
        out.learned.append(f"combat_skill +{d_combat:.2f} -> {curr.combat_skill:.2f}")

    d_tame = curr.wolf_tame_skill - prev.wolf_tame_skill
    if d_tame >= 0.01:
        out.learned.append(f"wolf_tame_skill +{d_tame:.2f} -> {curr.wolf_tame_skill:.2f}")

    d_tame_ok = curr.wolf_tame_successes - prev.wolf_tame_successes
    if d_tame_ok > 0:
        out.learned.append(f"wolf_tame_successes +{d_tame_ok} -> {curr.wolf_tame_successes}")

    d_tame_success_events = curr.tame_success_events - prev.tame_success_events
    if d_tame_success_events > 0:
        out.learned.append(f"tame_success_events +{d_tame_success_events}")

    d_tame_progress_events = curr.tame_progress_events - prev.tame_progress_events
    if d_tame_progress_events > 0:
        out.learned.append(f"tame_progress_events +{d_tame_progress_events}")

    d_tame_peer_share_events = curr.tame_peer_share_events - prev.tame_peer_share_events
    if d_tame_peer_share_events > 0:
        out.peer_tame_lessons = int(d_tame_peer_share_events)
        msg = f"peer_tame_lessons +{d_tame_peer_share_events}"
        if curr.last_tame_peer_hint:
            msg += f" from {curr.last_tame_peer_hint}"
        out.learned.append(msg)

    d_seek_partner = curr.seek_partner_events - prev.seek_partner_events
    if d_seek_partner > 0:
        out.learned.append(f"partner_search_events +{d_seek_partner}")

    d_offspring = curr.offspring_born_events - prev.offspring_born_events
    if d_offspring > 0:
        out.learned.append(f"offspring_born_events +{d_offspring}")

    d_lineage = curr.lineage_inherit_events - prev.lineage_inherit_events
    if d_lineage > 0:
        out.learned.append(f"lineage_inherit_events +{d_lineage}")

    d_gc = curr.gc_steps - prev.gc_steps
    if d_gc > 0:
        out.learned.append(f"policy_steps +{d_gc} -> {curr.gc_steps}")

    prev_goals = set(prev.goals)
    curr_goals = set(curr.goals)
    new_goals = sorted(curr_goals - prev_goals)
    if new_goals:
        out.learned.append("new_goals: " + ", ".join(new_goals))

    strengthened: List[Tuple[float, str]] = []
    weakened: List[Tuple[float, str]] = []
    for key, prev_s in prev.beliefs.items():
        if key not in curr.beliefs:
            continue
        curr_s = curr.beliefs[key]
        d = curr_s - prev_s
        if d >= 0.04:
            strengthened.append((d, f"{key[0]} -> {key[1]} (+{d:.2f})"))
        elif d <= -0.04:
            weakened.append((-d, f"{key[0]} -> {key[1]} (-{-d:.2f})"))

    strengthened.sort(key=lambda x: x[0], reverse=True)
    weakened.sort(key=lambda x: x[0], reverse=True)
    out.strengthened_links.extend(s for _, s in strengthened[:6])
    out.changed_beliefs.extend(s for _, s in weakened[:6])

    new_beliefs = sorted(set(curr.beliefs.keys()) - set(prev.beliefs.keys()))
    for cond, concl in new_beliefs[:6]:
        s = curr.beliefs.get((cond, concl), 0.0)
        out.changed_beliefs.append(f"new: {cond} -> {concl} (s={s:.2f})")

    removed_beliefs = sorted(set(prev.beliefs.keys()) - set(curr.beliefs.keys()))
    for cond, concl in removed_beliefs[:4]:
        out.changed_beliefs.append(f"removed: {cond} -> {concl}")

    for rk, curr_v in curr.rules.items():
        prev_v = prev.rules.get(rk, curr_v)
        d = curr_v - prev_v
        if d >= 0.05:
            out.strengthened_links.append(f"rule {rk}: +{d:.2f} -> {curr_v:.2f}")
        elif abs(d) >= 0.05:
            out.changed_beliefs.append(f"rule {rk}: {prev_v:.2f} -> {curr_v:.2f}")

    return out
