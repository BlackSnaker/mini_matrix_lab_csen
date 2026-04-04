# mind_core.py — ConsciousnessBlock с переносом навыков из тренера и мягким дообучением в мире

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import math
import random

# Попробуем подключить NumPy для компактной математики; если нет — дадим понятную ошибку
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError("mind_core.py требует numpy; установи 'pip install numpy'.")

# =============================================================================
# Параметры психики / поведения
# =============================================================================

# насколько страх союзника "заражает" нас (0..1, где 1 = полностью перенимаем)
SOCIAL_PANIC_TRANSFER = 0.5

# насколько наличие союзников само по себе снижает страх
SOCIAL_SAFETY_BONUS = 0.10

# карта травм
MAX_TRAUMA_SPOTS = 32
TRAUMA_DECAY_PER_TICK = 0.01  # как быстро угасает интенсивность травмы за тик

# любопытство (топливо для исследования)
CURIOUS_RECOVERY_RATE = 0.02      # реген любопытства за тик, если всё ок
CURIOUS_DRAIN_EXPLORE = 0.10      # сколько тратим за тик активного исследования
CURIOUS_MIN_FOR_EXPLORE = 0.15    # ниже этого не даём себе "explore"

# =============================================================================
# Мелкие хелперы безопасности значений
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

def _round_tuple_xy(pos: Tuple[float, float]) -> Tuple[float, float]:
    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return (0.0, 0.0)
    x = _safe_float(pos[0], 0.0)
    y = _safe_float(pos[1], 0.0
    )
    return (x, y)

# --- Безопасные хелперы для извлечения времени и окружения из мира (для agent_ref/world_ref) ---

def _world_time_int(world) -> int:
    candidates = ("get_time", "time", "ticks", "tick_count", "current_tick", "frame", "step", "t", "tick")
    for name in candidates:
        v = getattr(world, name, None)
        if v is None:
            continue
        if callable(v) and name == "tick":
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

def _extract_env(world, agent, ally_radius: float = 12.0) -> Tuple[
    List[Tuple[float, float, float]],  # hazards_near
    List[Tuple[float, float, float]],  # healzones_near
    List[str],                         # allies_near (ids)
    List[Tuple[str, float, float]]     # allies_near_details: (ally_id, ally_fear, ally_health)
]:
    ax = _safe_float(getattr(agent, "x", 0.0), 0.0)
    ay = _safe_float(getattr(agent, "y", 0.0), 0.0)

    hazards: List[Tuple[float, float, float]] = []
    heals:   List[Tuple[float, float, float]] = []

    # объекты мира
    for obj in (getattr(world, "objects", []) or []):
        kind = str(getattr(obj, "kind", "")).lower()
        x = _safe_float(getattr(obj, "x", 0.0), 0.0)
        y = _safe_float(getattr(obj, "y", 0.0), 0.0)
        r = _safe_float(getattr(obj, "radius", 2.0), 2.0)
        if "safe" in kind or "heal" in kind:
            heals.append((x, y, r))
        elif "hazard" in kind or "danger" in kind or "threat" in kind:
            hazards.append((x, y, r))

    # агрессивные животные как опасность
    animals = getattr(world, "animals", {}) or {}
    for ani in animals.values():
        try:
            species = getattr(ani, "species", None)
            aggressive = bool(getattr(species, "aggressive", True))
            if aggressive:
                x = _safe_float(getattr(ani, "x", 0.0), 0.0)
                y = _safe_float(getattr(ani, "y", 0.0), 0.0)
                hazards.append((x, y, 2.0))
        except Exception:
            continue

    # союзники поблизости
    allies_ids: List[str] = []
    allies_details: List[Tuple[str, float, float]] = []

    raw_agents = getattr(world, "agents", None)
    agent_items = []
    if isinstance(raw_agents, dict):
        agent_items = list(raw_agents.values())
    elif isinstance(raw_agents, list):
        agent_items = raw_agents

    my_id = getattr(agent, "id", getattr(agent, "agent_id", None))
    for ag in agent_items or []:
        try:
            aid = getattr(ag, "id", getattr(ag, "agent_id", None))
            if aid is None or aid == my_id:
                continue
            x = _safe_float(getattr(ag, "x", 0.0), 0.0)
            y = _safe_float(getattr(ag, "y", 0.0), 0.0)
            d = math.hypot(x - ax, y - ay)
            if d <= ally_radius:
                allies_ids.append(str(aid))
                afear = _clamp(_safe_float(getattr(ag, "fear", 0.0), 0.0), 0.0, 1.0)
                ahealth = _clamp(_safe_float(getattr(ag, "health", 100.0), 100.0), 0.0, 100.0)
                allies_details.append((str(aid), afear, ahealth))
        except Exception:
            continue

    return hazards, heals, allies_ids, allies_details

# =============================================================================
# Базовые структуры памяти/убеждений сознания
# =============================================================================

@dataclass
class MemoryEvent:
    tick: int
    etype: str
    data: Dict[str, Any]

@dataclass
class Belief:
    condition: str
    conclusion: str
    strength: float = 1.0

# =============================================================================
# Правила поведения, которые эволюционируют
# =============================================================================

@dataclass
class BehaviorRules:
    avoid_hazard_radius: float = 6.0
    healing_zone_seek_priority: float = 0.5
    stick_with_ally_if_fear_above: float = 0.7
    exploration_bias: float = 0.2

    def mutate_from_experience(
        self,
        died_recently: bool,
        took_damage: bool,
        healed: bool,
        survival_score_snapshot: float,
    ):
        if died_recently:
            self.avoid_hazard_radius = min(self.avoid_hazard_radius + 1.5, 20.0)
            self.exploration_bias = max(0.0, self.exploration_bias - 0.05)
            self.stick_with_ally_if_fear_above = max(0.0, self.stick_with_ally_if_fear_above - 0.05)
        if took_damage:
            self.avoid_hazard_radius = min(self.avoid_hazard_radius + 0.5, 20.0)
            self.stick_with_ally_if_fear_above = max(0.0, self.stick_with_ally_if_fear_above - 0.02)
        if healed:
            self.healing_zone_seek_priority = min(self.healing_zone_seek_priority + 0.1, 1.0)
            self.stick_with_ally_if_fear_above = min(1.0, self.stick_with_ally_if_fear_above + 0.01)
        if survival_score_snapshot < 0.3:
            self.exploration_bias = max(0.0, self.exploration_bias - 0.02)
        elif survival_score_snapshot > 0.7:
            self.exploration_bias = min(1.0, self.exploration_bias + 0.02)
        self.exploration_bias = min(1.0, self.exploration_bias + 0.01)

# =============================================================================
# Лёгкий цель-условный policy/critic (навыки), реплей и дообучение
# =============================================================================

@dataclass
class Transition:
    obs: np.ndarray
    goal_id: int
    action: np.ndarray  # (dx,dz)
    reward: float
    next_obs: np.ndarray
    done: bool

@dataclass
class Replay:
    cap: int
    buf: List[Transition] = field(default_factory=list)
    def add(self, tr: Transition):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(tr)
    def sample(self, n: int) -> List[Transition]:
        n = min(n, len(self.buf))
        if n <= 0:
            return []
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        return [self.buf[int(i)] for i in idx]

def _init_linear(in_dim: int, out_dim: int, scale: float = 0.01) -> Dict[str, np.ndarray]:
    return {"W": (np.random.randn(in_dim, out_dim) * scale).astype(np.float32),
            "b": np.zeros((out_dim,), dtype=np.float32)}

def _forward_linear(x: np.ndarray, layer: Dict[str, np.ndarray]) -> np.ndarray:
    return x @ layer["W"] + layer["b"]

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def _softsign(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.abs(x))

# =============================================================================
# Основной мозг агента (ConsciousnessBlock)
# =============================================================================

@dataclass
class ConsciousnessBlock:
    agent_id: str

    # --- долговременные куски, которые переживут смерть -----------------
    memory_tail: List[MemoryEvent] = field(default_factory=list)
    beliefs: List[Belief] = field(default_factory=list)
    behavior_rules: BehaviorRules = field(default_factory=BehaviorRules)
    trauma_map: List[Dict[str, Any]] = field(default_factory=list)

    # --- физиология / эмоции в рантайме ---------------------------------
    fear_level: float = 0.0         # 0..1
    health: float = 100.0           # 0..100
    energy: float = 100.0           # 0..100
    hunger: float = 0.0             # 0..100
    alive: bool = True
    age_ticks: int = 0

    survival_score: float = 1.0
    current_drive: str = "idle"
    ally_anchor: Optional[str] = None
    last_death_reason: Optional[str] = None

    curiosity_charge: float = 0.5   # 0..1
    last_thought: str = "…"

    # --- Ollama brain integration --------------------------------------
    ollama_enabled: bool = False
    ollama_auto_mode: bool = False
    ollama_interval_ticks: int = 180
    ollama_model: Optional[str] = None
    ollama_host: Optional[str] = None
    ollama_status: str = "off"
    ollama_last_error: Optional[str] = None
    ollama_last_model: Optional[str] = None
    ollama_last_thought: str = ""
    ollama_active_lesson: str = ""
    ollama_active_goal: Optional[Tuple[float, float]] = None
    ollama_goal_started_tick: Optional[int] = None
    ollama_active_reward_hint: float = 0.12
    ollama_pending_instruction: Optional[str] = None
    ollama_last_operator_instruction: Optional[str] = None
    ollama_force_request: bool = False
    ollama_request_inflight: bool = False
    ollama_last_request_tick: int = -10**9
    ollama_last_response_tick: int = -10**9
    ollama_dialogue_tail: List[Dict[str, Any]] = field(default_factory=list)
    ollama_log_seq: int = 0
    ollama_command_examples: List[Dict[str, Any]] = field(default_factory=list)
    ollama_command_example_seq: int = 0
    ollama_authoritative_commands: bool = True
    ollama_operator_override_until_tick: int = -10**9
    ollama_active_command_source: Optional[str] = None

    # --- временные флаги / контекст последнего тика ---------------------
    _took_damage_recently: bool = False
    _healed_recently: bool = False
    _died_last_tick: bool = False

    _allies_near_recent: List[str] = field(default_factory=list)

    _gc_last_obs: Optional[np.ndarray] = field(default=None, repr=False)
    _gc_last_goal_tag: Optional[str] = field(default=None, repr=False)
    _gc_last_action: Optional[np.ndarray] = field(default=None, repr=False)

    # --- НАВЫКИ / ПОЛИТИКА + КРИТИК -------------------------------------
    gc_version: str = "0"  # "0" = нет навыков; "2" = актуальная версия ниже
    gc_obs_dim: int = 16
    gc_hidden_dim: int = 64
    gc_goal_vocab: Dict[str, int] = field(default_factory=dict)

    gc_actor: Dict[str, np.ndarray] = field(default_factory=dict)
    gc_critic: Dict[str, np.ndarray] = field(default_factory=dict)

    gc_obs_mean: np.ndarray = field(default_factory=lambda: np.zeros((16,), dtype=np.float32))
    gc_obs_std:  np.ndarray = field(default_factory=lambda: np.ones((16,), dtype=np.float32))

    gc_anchor_actor: Dict[str, np.ndarray] = field(default_factory=dict)
    gc_anchor_critic: Dict[str, np.ndarray] = field(default_factory=dict)

    gc_hps: Dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-3,
        "ewc_lambda": 0.01,
        "online_batch": 64,
        "mix_rehearsal_ratio": 0.5,
        "gamma": 0.98,
        "combat_r_hp_loss": 0.5,
        "combat_r_hp_gain": 0.1,
        "combat_r_approach": 0.2,
        "combat_done_bonus": 0.0,
    })
    gc_steps: int = 0

    _gc_world_replay: Replay = field(default_factory=lambda: Replay(cap=50_000), repr=False)
    _gc_rehearsal: Replay = field(default_factory=lambda: Replay(cap=50_000), repr=False)

    # --------------------------------------------------------------------
    # Сериализация / десериализация (для brain_io)
    # --------------------------------------------------------------------

    def _ev_triplet(self, ev: Any) -> Dict[str, Any]:
        if hasattr(ev, "tick") and hasattr(ev, "etype") and hasattr(ev, "data"):
            return {"tick": getattr(ev, "tick", None),
                    "etype": getattr(ev, "etype", None),
                    "data": getattr(ev, "data", None)}
        if isinstance(ev, dict):
            etype = ev.get("etype") or ev.get("type") or "event"
            base = {k: v for k, v in ev.items() if k not in ("tick", "etype", "type")}
            return {"tick": ev.get("tick"), "etype": etype, "data": ev.get("data", base)}
        return {"tick": None, "etype": type(ev).__name__, "data": {"value": ev}}

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "agent_id": self.agent_id,

            "memory_tail": [self._ev_triplet(ev) for ev in self.memory_tail[-200:]],

            "beliefs": [
                {"condition": b.condition, "conclusion": b.conclusion,
                 "strength": float(_clamp(b.strength, 0.0, 1.0))}
                for b in self.beliefs
            ],

            "behavior_rules": {
                "avoid_hazard_radius": float(self.behavior_rules.avoid_hazard_radius),
                "healing_zone_seek_priority": float(self.behavior_rules.healing_zone_seek_priority),
                "stick_with_ally_if_fear_above": float(self.behavior_rules.stick_with_ally_if_fear_above),
                "exploration_bias": float(_clamp(self.behavior_rules.exploration_bias, 0.0, 1.0)),
            },

            "fear_level": float(_clamp(self.fear_level, 0.0, 1.0)),
            "health": float(_clamp(self.health, 0.0, 100.0)),
            "energy": float(_clamp(self.energy, 0.0, 100.0)),
            "hunger": float(_clamp(self.hunger, 0.0, 100.0)),
            "alive": bool(self.alive),
            "age_ticks": int(self.age_ticks),

            "survival_score": float(_clamp(self.survival_score, 0.0, 1.0)),
            "current_drive": str(self.current_drive),

            "ally_anchor": self.ally_anchor,
            "last_death_reason": self.last_death_reason,

            "trauma_map": [
                {"pos": tm.get("pos"),
                 "intensity": float(_clamp(_safe_float(tm.get("intensity", 0.0), 0.0), 0.0, 1.0))}
                for tm in self.trauma_map[-MAX_TRAUMA_SPOTS:]
            ],

            "curiosity_charge": float(_clamp(self.curiosity_charge, 0.0, 1.0)),
            "last_thought": self.last_thought,
            "ollama": {
                "enabled": bool(self.ollama_enabled),
                "auto_mode": bool(self.ollama_auto_mode),
                "interval_ticks": int(max(1, self.ollama_interval_ticks)),
                "model": self.ollama_model,
                "host": self.ollama_host,
                "status": str(self.ollama_status or "off"),
                "last_error": self.ollama_last_error,
                "last_model": self.ollama_last_model,
                "last_thought": str(self.ollama_last_thought or ""),
                "active_lesson": str(self.ollama_active_lesson or ""),
                "active_goal": (
                    {"x": float(self.ollama_active_goal[0]), "y": float(self.ollama_active_goal[1])}
                    if isinstance(self.ollama_active_goal, tuple) and len(self.ollama_active_goal) == 2
                    else None
                ),
                "goal_started_tick": self.ollama_goal_started_tick,
                "active_reward_hint": float(self.ollama_active_reward_hint),
                "pending_instruction": self.ollama_pending_instruction,
                "last_operator_instruction": self.ollama_last_operator_instruction,
                "last_request_tick": int(self.ollama_last_request_tick),
                "last_response_tick": int(self.ollama_last_response_tick),
                "dialogue_tail": list(self.ollama_dialogue_tail[-80:]),
                "log_seq": int(self.ollama_log_seq),
                "command_examples": list(self.ollama_command_examples[-80:]),
                "command_example_seq": int(self.ollama_command_example_seq),
                "authoritative_commands": bool(self.ollama_authoritative_commands),
                "operator_override_until_tick": int(self.ollama_operator_override_until_tick),
                "active_command_source": self.ollama_active_command_source,
            },
        }

        if self.gc_version != "0" and self.gc_actor and self.gc_critic:
            out["gc_policy"] = {
                "version": self.gc_version,
                "obs_dim": int(self.gc_obs_dim),
                "hidden_dim": int(self.gc_hidden_dim),
                "goal_vocab": dict(self.gc_goal_vocab),
                "actor": {
                    "l1": {"W": self.gc_actor["l1"]["W"].tolist(), "b": self.gc_actor["l1"]["b"].tolist()},
                    "l2": {"W": self.gc_actor["l2"]["W"].tolist(), "b": self.gc_actor["l2"]["b"].tolist()},
                },
                "critic": {
                    "l1": {"W": self.gc_critic["l1"]["W"].tolist(), "b": self.gc_critic["l1"]["b"].tolist()},
                    "v":  {"W": self.gc_critic["v"]["W"].tolist(),  "b": self.gc_critic["v"]["b"].tolist()},
                },
                "obs_mean": self.gc_obs_mean.tolist(),
                "obs_std":  self.gc_obs_std.tolist(),
                "anchor_actor": {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in self.gc_anchor_actor.items()},
                "anchor_critic": {k: {"W": v["W"].tolist(), "b": v["b"].tolist()} for k, v in self.gc_anchor_critic.items()},
                "hps": dict(self.gc_hps),
                "steps": int(self.gc_steps),
            }
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConsciousnessBlock":
        br_raw = data.get("behavior_rules", {}) or {}
        br = BehaviorRules(
            avoid_hazard_radius=_safe_float(br_raw.get("avoid_hazard_radius", 6.0), 6.0),
            healing_zone_seek_priority=_safe_float(br_raw.get("healing_zone_seek_priority", 0.5), 0.5),
            stick_with_ally_if_fear_above=_safe_float(br_raw.get("stick_with_ally_if_fear_above", 0.7), 0.7),
            exploration_bias=_clamp(_safe_float(br_raw.get("exploration_bias", 0.2), 0.2), 0.0, 1.0),
        )

        mem_tail: List[MemoryEvent] = []
        for ev in data.get("memory_tail", []) or []:
            tick_raw = ev.get("tick", 0)
            try:
                tick_val = int(0 if tick_raw is None else tick_raw)
            except Exception:
                tick_val = 0
            mem_tail.append(MemoryEvent(
                tick=tick_val,
                etype=str(ev.get("etype") or ev.get("type") or ev.get("kind") or "?"),
                data=dict(ev.get("data", {})),
            ))

        beliefs_list: List[Belief] = []
        for b in data.get("beliefs", []) or []:
            beliefs_list.append(Belief(
                condition=str(b.get("condition") or b.get("if") or ""),
                conclusion=str(b.get("conclusion") or b.get("then") or ""),
                strength=_clamp(_safe_float(b.get("strength", 1.0), 1.0), 0.0, 1.0),
            ))

        trauma_map_raw: List[Dict[str, Any]] = []
        for tm in data.get("trauma_map", []) or []:
            trauma_map_raw.append({
                "pos": _round_tuple_xy(tm.get("pos", (0.0, 0.0))),
                "intensity": _clamp(_safe_float(tm.get("intensity", 0.0), 0.0), 0.0, 1.0),
            })

        block = ConsciousnessBlock(
            agent_id=str(data.get("agent_id", "unknown")),
            memory_tail=mem_tail,
            beliefs=beliefs_list,
            behavior_rules=br,
            trauma_map=trauma_map_raw,

            fear_level=_clamp(_safe_float(data.get("fear_level", 0.0), 0.0), 0.0, 1.0),
            health=_clamp(_safe_float(data.get("health", 100.0), 100.0), 0.0, 100.0),
            energy=_clamp(_safe_float(data.get("energy", 100.0), 100.0), 0.0, 100.0),
            hunger=_clamp(_safe_float(data.get("hunger", 0.0), 0.0), 0.0, 100.0),
            alive=bool(data.get("alive", True)),
            age_ticks=int(data.get("age_ticks", 0)),

            survival_score=_clamp(_safe_float(data.get("survival_score", 1.0), 1.0), 0.0, 1.0),
            current_drive=str(data.get("current_drive", "idle")),

            ally_anchor=data.get("ally_anchor", None),
            last_death_reason=data.get("last_death_reason", None),

            curiosity_charge=_clamp(_safe_float(data.get("curiosity_charge", 0.5), 0.5), 0.0, 1.0),
            last_thought=str(data.get("last_thought", "…")),
        )

        ollama_raw = data.get("ollama", {}) or {}
        if isinstance(ollama_raw, dict):
            goal_raw = ollama_raw.get("active_goal")
            goal_val = None
            if isinstance(goal_raw, dict) and ("x" in goal_raw) and ("y" in goal_raw):
                goal_val = (
                    _safe_float(goal_raw.get("x"), 0.0),
                    _safe_float(goal_raw.get("y"), 0.0),
                )
            block.ollama_enabled = bool(ollama_raw.get("enabled", False))
            block.ollama_auto_mode = bool(ollama_raw.get("auto_mode", False))
            block.ollama_interval_ticks = max(1, int(_safe_float(ollama_raw.get("interval_ticks", 180), 180.0)))
            block.ollama_model = ollama_raw.get("model")
            block.ollama_host = ollama_raw.get("host")
            block.ollama_status = str(ollama_raw.get("status", "off"))
            block.ollama_last_error = ollama_raw.get("last_error")
            block.ollama_last_model = ollama_raw.get("last_model")
            block.ollama_last_thought = str(ollama_raw.get("last_thought", ""))
            block.ollama_active_lesson = str(ollama_raw.get("active_lesson", ""))
            block.ollama_active_goal = goal_val
            gst = ollama_raw.get("goal_started_tick")
            block.ollama_goal_started_tick = int(gst) if isinstance(gst, (int, float)) else None
            block.ollama_active_reward_hint = _clamp(_safe_float(ollama_raw.get("active_reward_hint", 0.12), 0.12), -1.0, 1.0)
            block.ollama_pending_instruction = ollama_raw.get("pending_instruction")
            block.ollama_last_operator_instruction = ollama_raw.get("last_operator_instruction")
            block.ollama_force_request = False
            block.ollama_request_inflight = False
            block.ollama_last_request_tick = int(_safe_float(ollama_raw.get("last_request_tick", -10**9), -10**9))
            block.ollama_last_response_tick = int(_safe_float(ollama_raw.get("last_response_tick", -10**9), -10**9))
            dialog = ollama_raw.get("dialogue_tail", []) or []
            if isinstance(dialog, list):
                block.ollama_dialogue_tail = [dict(row) for row in dialog if isinstance(row, dict)][-80:]
            block.ollama_log_seq = int(_safe_float(ollama_raw.get("log_seq", len(block.ollama_dialogue_tail)), len(block.ollama_dialogue_tail)))
            examples = ollama_raw.get("command_examples", []) or []
            if isinstance(examples, list):
                block.ollama_command_examples = [dict(row) for row in examples if isinstance(row, dict)][-80:]
            block.ollama_command_example_seq = int(_safe_float(ollama_raw.get("command_example_seq", len(block.ollama_command_examples)), len(block.ollama_command_examples)))
            block.ollama_authoritative_commands = bool(ollama_raw.get("authoritative_commands", True))
            block.ollama_operator_override_until_tick = int(_safe_float(ollama_raw.get("operator_override_until_tick", -10**9), -10**9))
            src = ollama_raw.get("active_command_source")
            block.ollama_active_command_source = str(src).strip() if src is not None and str(src).strip() else None

        gc_raw = data.get("gc_policy")
        if gc_raw:
            try:
                block.gc_version = str(gc_raw.get("version", "2"))
                block.gc_obs_dim = int(gc_raw.get("obs_dim", 16))
                block.gc_hidden_dim = int(gc_raw.get("hidden_dim", 64))
                block.gc_goal_vocab = dict(gc_raw.get("goal_vocab", {}))

                actor = gc_raw.get("actor", {})
                critic = gc_raw.get("critic", {})
                block.gc_actor = {
                    "l1": {"W": np.array(actor["l1"]["W"], dtype=np.float32), "b": np.array(actor["l1"]["b"], dtype=np.float32)},
                    "l2": {"W": np.array(actor["l2"]["W"], dtype=np.float32), "b": np.array(actor["l2"]["b"], dtype=np.float32)},
                }
                block.gc_critic = {
                    "l1": {"W": np.array(critic["l1"]["W"], dtype=np.float32), "b": np.array(critic["l1"]["b"], dtype=np.float32)},
                    "v":  {"W": np.array(critic["v"]["W"],  dtype=np.float32), "b": np.array(critic["v"]["b"],  dtype=np.float32)},
                }

                block.gc_obs_mean = np.array(gc_raw.get("obs_mean", [0.0]*block.gc_obs_dim), dtype=np.float32)
                block.gc_obs_std  = np.array(gc_raw.get("obs_std",  [1.0]*block.gc_obs_dim), dtype=np.float32)

                block.gc_anchor_actor = {k: {"W": np.array(v["W"], dtype=np.float32), "b": np.array(v["b"], dtype=np.float32)}
                                         for k, v in (gc_raw.get("anchor_actor", {}) or {}).items()}
                block.gc_anchor_critic = {k: {"W": np.array(v["W"], dtype=np.float32), "b": np.array(v["b"], dtype=np.float32)}
                                          for k, v in (gc_raw.get("anchor_critic", {}) or {}).items()}

                block.gc_hps = dict(gc_raw.get("hps", block.gc_hps))
                block.gc_steps = int(gc_raw.get("steps", 0))
            except Exception:
                block.gc_version = "0"
                block.gc_actor = {}
                block.gc_critic = {}

        return block

    # --------------------------------------------------------------------
    # Внутренние методы навыков (policy/critic)
    # --------------------------------------------------------------------

    def _gc_init_if_needed(self, obs_dim: Optional[int] = None, hidden_dim: Optional[int] = None,
                            goal_vocab: Optional[Dict[str, int]] = None):
        if self.gc_version != "0" and self.gc_actor and self.gc_critic:
            return
        if obs_dim is not None:
            self.gc_obs_dim = int(obs_dim)
        if hidden_dim is not None:
            self.gc_hidden_dim = int(hidden_dim)
        if goal_vocab is None:
            goal_vocab = {"explore": 0}
        self.gc_goal_vocab = dict(goal_vocab)

        gdim = max(1, len(self.gc_goal_vocab))
        in_dim = self.gc_obs_dim + gdim
        hid = self.gc_hidden_dim
        self.gc_actor = {"l1": _init_linear(in_dim, hid), "l2": _init_linear(hid, 2)}
        self.gc_critic = {"l1": _init_linear(in_dim, hid), "v": _init_linear(hid, 1)}
        self.gc_obs_mean = np.zeros((self.gc_obs_dim,), dtype=np.float32)
        self.gc_obs_std  = np.ones((self.gc_obs_dim,), dtype=np.float32)

        self.gc_anchor_actor = {k: {"W": v["W"].copy(), "b": v["b"].copy()} for k, v in self.gc_actor.items()}
        self.gc_anchor_critic= {k: {"W": v["W"].copy(), "b": v["b"].copy()} for k, v in self.gc_critic.items()}

        self.gc_version = "2"
        self.gc_steps = 0

    def _gc_goal_onehot(self, goal_tag: str) -> np.ndarray:
        if goal_tag not in self.gc_goal_vocab:
            new_id = len(self.gc_goal_vocab)
            self.gc_goal_vocab[goal_tag] = new_id
            self._gc_expand_input_dim_for_new_goal()
        g = np.zeros((len(self.gc_goal_vocab),), dtype=np.float32)
        g[self.gc_goal_vocab[goal_tag]] = 1.0
        return g

    def gc_goal_id(self, goal_tag: str) -> int:
        self._gc_goal_onehot(goal_tag)
        return int(self.gc_goal_vocab[goal_tag])

    def _gc_expand_input_dim_for_new_goal(self):
        for net in (self.gc_actor, self.gc_critic):
            if not net:
                continue
            W = net["l1"]["W"]; b = net["l1"]["b"]
            in_dim, out_dim = W.shape
            new_row = (np.random.randn(1, out_dim) * 0.01).astype(np.float32)
            net["l1"]["W"] = np.vstack([W, new_row])
            net["l1"]["b"] = b
        for net in (self.gc_anchor_actor, self.gc_anchor_critic):
            if not net:
                continue
            W = net["l1"]["W"]; b = net["l1"]["b"]
            _, out_dim = W.shape
            new_row = np.zeros((1, out_dim), dtype=np.float32)
            net["l1"]["W"] = np.vstack([W, new_row])
            net["l1"]["b"] = b

    def _gc_norm_obs(self, obs: np.ndarray) -> np.ndarray:
        s = np.maximum(self.gc_obs_std, 1e-6)
        return (obs - self.gc_obs_mean) / s

    def _gc_actor_forward(self, z: np.ndarray) -> np.ndarray:
        a1 = _tanh(_forward_linear(z, self.gc_actor["l1"]))
        out = _softsign(_forward_linear(a1, self.gc_actor["l2"]))
        return out

    def _gc_critic_forward(self, z: np.ndarray) -> np.ndarray:
        c1 = _tanh(_forward_linear(z, self.gc_critic["l1"]))
        v  = _forward_linear(c1, self.gc_critic["v"])
        return v.squeeze(-1)

    # --------------------------------------------------------------------
    # Публичный API навыков
    # --------------------------------------------------------------------

    def policy_act(self, obs: np.ndarray, goal_tag: str, rng: Optional[random.Random] = None) -> Dict[str, float]:
        self._gc_init_if_needed()
        if rng is None:
            rng = random.Random()
        g = self._gc_goal_onehot(goal_tag)
        obs = obs.astype(np.float32)
        z = np.concatenate([self._gc_norm_obs(obs), g], axis=0)[None, ...]
        a = self._gc_actor_forward(z)[0]
        eps = ((rng.random() * 2.0) - 1.0) * 0.05
        action = np.array([a[0] + eps, a[1] - eps], dtype=np.float32)

        self._gc_last_obs = obs
        self._gc_last_goal_tag = goal_tag
        self._gc_last_action = action
        return {"dx": float(action[0]), "dz": float(action[1])}

    def policy_remember(self, tr: Transition, to_rehearsal: bool = False):
        (self._gc_rehearsal if to_rehearsal else self._gc_world_replay).add(tr)

    def policy_preload_rehearsal(self, transitions: List[Transition]):
        for t in transitions:
            self._gc_rehearsal.add(t)

    def policy_learn_online(self, steps: int = 1):
        if self.gc_version == "0" or not self.gc_actor or not self._gc_world_replay.buf:
            return
        lr = float(self.gc_hps.get("lr", 1e-3))
        lam = float(self.gc_hps.get("ewc_lambda", 0.01))
        bs = int(self.gc_hps.get("online_batch", 64))
        mix = float(self.gc_hps.get("mix_rehearsal_ratio", 0.5))
        gamma = float(self.gc_hps.get("gamma", 0.98))

        for _ in range(steps):
            n_world = max(1, int(bs * (1.0 - mix)))
            n_reh = max(0, bs - n_world)
            batch: List[Transition] = self._gc_world_replay.sample(n_world) + self._gc_rehearsal.sample(n_reh)
            if not batch:
                return

            obs = np.stack([t.obs for t in batch], axis=0).astype(np.float32)

            # восстановим one-hot целей по id с авторасширением словаря
            goals = []
            for t in batch:
                tag = next((tag for tag, gid in self.gc_goal_vocab.items() if gid == t.goal_id), "explore")
                self._gc_goal_onehot(tag)
                gid = self.gc_goal_vocab[tag]
                goals.append(self._goal_onehot_id(gid))
            goals = np.stack(goals, axis=0).astype(np.float32)

            z = np.concatenate([self._gc_norm_obs(obs), goals], axis=1)

            a_taken = np.stack([t.action for t in batch], axis=0).astype(np.float32)
            r = np.array([t.reward for t in batch], dtype=np.float32)
            nxt = np.stack([t.next_obs for t in batch], axis=0).astype(np.float32)
            dn = np.array([t.done for t in batch], dtype=np.float32)

            z2 = np.concatenate([self._gc_norm_obs(nxt), goals], axis=1)

            # КРИТИК
            v = self._gc_critic_forward(z)
            v2 = self._gc_critic_forward(z2)
            target = r + (1.0 - dn) * gamma * v2
            td = target - v

            c1 = _tanh(_forward_linear(z, self.gc_critic["l1"]))
            grad_v_out = (-td)[:, None] / max(1, len(z))
            vW = c1.T @ grad_v_out
            vb = grad_v_out.sum(axis=0)
            dc1 = grad_v_out @ self.gc_critic["v"]["W"].T
            dc1 *= (1.0 - c1 ** 2)
            l1W = z.T @ dc1 / max(1, len(z))
            l1b = dc1.mean(axis=0)

            self._gc_sgd(self.gc_critic["v"],  vW,  vb,  lr, lam, self.gc_anchor_critic, key="v")
            self._gc_sgd(self.gc_critic["l1"], l1W, l1b, lr, lam, self.gc_anchor_critic, key="l1")

            # АКТЁР
            a1 = _tanh(_forward_linear(z, self.gc_actor["l1"]))
            out = _softsign(_forward_linear(a1, self.gc_actor["l2"]))
            d_out = (out - a_taken) * (-td[:, None]) / max(1, len(z))
            l2W = a1.T @ d_out
            l2b = d_out.sum(axis=0)
            da1 = d_out @ self.gc_actor["l2"]["W"].T
            da1 *= (1.0 - a1 ** 2)
            l1W = z.T @ da1 / max(1, len(z))
            l1b = da1.mean(axis=0)

            self._gc_sgd(self.gc_actor["l2"], l2W, l2b, lr, lam, self.gc_anchor_actor, key="l2")
            self._gc_sgd(self.gc_actor["l1"], l1W, l1b, lr, lam, self.gc_anchor_actor, key="l1")

            self.gc_steps += 1

    def _gc_sgd(self, layer: Dict[str, np.ndarray], gradW: np.ndarray, gradb: np.ndarray,
                lr: float, lam: float, anchor: Optional[Dict[str, np.ndarray]] = None, key: Optional[str] = None):
        if anchor and key and key in anchor:
            layer["W"] -= lr * (gradW + lam * (layer["W"] - anchor[key]["W"]))
            layer["b"] -= lr * (gradb + lam * (layer["b"] - anchor[key]["b"]))
        else:
            layer["W"] -= lr * gradW
            layer["b"] -= lr * gradb

    def _goal_onehot_id(self, gid: int) -> np.ndarray:
        v = np.zeros((len(self.gc_goal_vocab),), dtype=np.float32)
        if 0 <= gid < len(v):
            v[gid] = 1.0
        return v

    # --- Shaping-награда и мостик боёвки --------------------------------

    def _shape_reward_combat(self, hp_delta: float, dist_delta: float, done: bool) -> float:
        hp_loss_w = float(self.gc_hps.get("combat_r_hp_loss", 0.5))
        hp_gain_w = float(self.gc_hps.get("combat_r_hp_gain", 0.1))
        approach_w = float(self.gc_hps.get("combat_r_approach", 0.2))
        done_bonus = float(self.gc_hps.get("combat_done_bonus", 0.0))

        r = 0.0
        if hp_delta < 0:
            r -= (-hp_delta) * hp_loss_w
        elif hp_delta > 0:
            r += (hp_delta) * hp_gain_w
        r += (-dist_delta) * approach_w
        if done:
            r += done_bonus
        return float(r)

    def note_combat_feedback(
        self,
        action: Optional[Tuple[float, float]] = None,
        hp_delta: float = 0.0,
        dist_delta: float = 0.0,
        next_obs: Optional[np.ndarray] = None,
        done: bool = False,
        to_rehearsal: bool = False,
    ):
        self._gc_init_if_needed()
        if self._gc_last_obs is None:
            return

        gid = self.gc_goal_id("combat")
        if action is not None:
            a = np.array([_safe_float(action[0], 0.0), _safe_float(action[1], 0.0)], dtype=np.float32)
        elif self._gc_last_action is not None:
            a = self._gc_last_action.astype(np.float32)
        else:
            return

        nxt = np.asarray(next_obs, dtype=np.float32) if next_obs is not None else self._gc_last_obs.copy()
        rew = self._shape_reward_combat(_safe_float(hp_delta, 0.0), _safe_float(dist_delta, 0.0), bool(done))

        tr = Transition(obs=self._gc_last_obs.copy(), goal_id=gid, action=a.copy(), reward=rew, next_obs=nxt, done=bool(done))
        self.policy_remember(tr, to_rehearsal=to_rehearsal)

    def build_obs(
        self,
        pos: Tuple[float, float],
        goal_target: Tuple[float, float],
        hazard_nearest: Optional[Tuple[float, float, float]] = None,
        food_nearest: Optional[Tuple[float, float, float]] = None,
        vel: Tuple[float, float] = (0.0, 0.0),
        yaw_rad: float = 0.0,
    ) -> np.ndarray:
        px, py = _round_tuple_xy(pos)
        gx, gy = _round_tuple_xy(goal_target)
        dx, dy = gx - px, gy - py
        dist = math.hypot(dx, dy)
        if hazard_nearest:
            hx, hy, hdist = hazard_nearest
        else:
            hx, hy, hdist = (px, py, 1e6)
        if food_nearest:
            fx, fy, fdist = food_nearest
        else:
            fx, fy, fdist = (px, py, 1e6)
        vdx, vdy = vel
        obs = np.array([
            dx/20.0, dy/20.0, dist/50.0,
            (hx-px)/20.0, (hy-py)/20.0, hdist/30.0,
            (fx-px)/20.0, (fy-py)/20.0, fdist/30.0,
            vdx/5.0, vdy/5.0,
            self.fear_level, self.health/100.0, self.energy/100.0,
            math.cos(yaw_rad), math.sin(yaw_rad),
        ], dtype=np.float32)
        if self.gc_obs_mean.shape[0] != obs.shape[0]:
            self.gc_obs_dim = int(obs.shape[0])
            self.gc_obs_mean = np.zeros((self.gc_obs_dim,), dtype=np.float32)
            self.gc_obs_std = np.ones((self.gc_obs_dim,), dtype=np.float32)
        return obs

    # --------------------------------------------------------------------
    # Вспомогательные внутренние методы (травма / убеждения / соц. якорь)
    # --------------------------------------------------------------------

    def _remember_trauma_spot(self, pos_xy: Tuple[float, float], intensity: float):
        px, py = _round_tuple_xy(pos_xy)
        intensity = _clamp(_safe_float(intensity, 0.0), 0.0, 1.0)
        for tm in self.trauma_map:
            tx, ty = tm["pos"]
            dist = math.hypot(tx - px, ty - py)
            if dist < 2.0:
                tm["intensity"] = _clamp(tm["intensity"] + intensity, 0.0, 1.0)
                break
        else:
            self.trauma_map.append({"pos": (px, py), "intensity": intensity})

        if len(self.trauma_map) > MAX_TRAUMA_SPOTS:
            self.trauma_map.sort(key=lambda t: t["intensity"], reverse=True)
            self.trauma_map = self.trauma_map[:MAX_TRAUMA_SPOTS]

    def _remember_safe_spot(self, pos_xy: Tuple[float, float], relief: float = 0.2):
        px, py = _round_tuple_xy(pos_xy)
        relief = _clamp(_safe_float(relief, 0.2), 0.0, 1.0)
        new_map = []
        for tm in self.trauma_map:
            tx, ty = tm["pos"]
            dist = math.hypot(tx - px, ty - py)
            if dist < 2.0:
                new_int = tm["intensity"] - relief
                if new_int > 0.01:
                    new_map.append({"pos": tm["pos"], "intensity": new_int})
            else:
                new_map.append(tm)
        self.trauma_map = new_map

    def _decay_trauma_map(self):
        new_map = []
        for tm in self.trauma_map:
            new_int = tm["intensity"] - TRAUMA_DECAY_PER_TICK
            if new_int > 0.01:
                new_map.append({"pos": tm["pos"], "intensity": new_int})
        self.trauma_map = new_map

    def _add_or_strengthen_belief(self, condition: str, conclusion: str, boost: float):
        for b in self.beliefs:
            if b.condition == condition and b.conclusion == conclusion:
                b.strength = _clamp(b.strength + boost, 0.0, 1.0)
                return
        self.beliefs.append(Belief(condition=condition, conclusion=conclusion, strength=_clamp(boost, 0.0, 1.0)))

    def integrate_world_observation(
        self,
        tick: int,
        pos: Tuple[float, float],
        hazards_near: List[Tuple[float, float, float]],
        healzones_near: List[Tuple[float, float, float]],
        allies_near: List[str],
    ):
        px, py = _round_tuple_xy(pos)
        for (hx, hy, rad) in hazards_near:
            dist = math.hypot(hx - px, hy - py)
            if dist < rad + 2.0:
                self._add_or_strengthen_belief(
                    condition=f"saw_hazard@({round(hx,1)},{round(hy,1)})",
                    conclusion="stay_away",
                    boost=0.2,
                )
        for (sx, sy, rad) in healzones_near:
            dist = math.hypot(sx - px, sy - py)
            if dist < rad + 2.0:
                self._add_or_strengthen_belief(
                    condition=f"healzone@({round(sx,1)},{round(sy,1)})",
                    conclusion="safe_place",
                    boost=0.15,
                )
        for ally_id in allies_near:
            self._add_or_strengthen_belief(
                condition=f"near({ally_id})",
                conclusion="safety++",
                boost=0.1,
            )
        self._allies_near_recent = list(allies_near)

    def _infect_fear_from_allies(self, allies_near_details: List[Tuple[str, float, float]]):
        if not allies_near_details:
            return
        self.fear_level = _clamp(self.fear_level - SOCIAL_SAFETY_BONUS, 0.0, 1.0)
        for (_aid, afear, _ahealth) in allies_near_details:
            afear = _clamp(_safe_float(afear, 0.0), 0.0, 1.0)
            if afear > self.fear_level:
                self.fear_level = _clamp(self.fear_level + (afear - self.fear_level) * SOCIAL_PANIC_TRANSFER, 0.0, 1.0)

    def _update_social_anchor(self):
        self.ally_anchor = None
        if not self._allies_near_recent:
            return
        candidate = self._allies_near_recent[0]
        if self.fear_level >= self.behavior_rules.stick_with_ally_if_fear_above:
            self.ally_anchor = candidate

    # --------------------------------------------------------------------
    # Внутренние вычисления драйва / метрик выживания / любопытства
    # --------------------------------------------------------------------

    def choose_drive(self):
        if not self.alive:
            self.current_drive = "dead"
            return
        if self.health < 30.0:
            self.current_drive = "heal"; return
        if self.hunger > 80.0:
            self.current_drive = "find_food"; return
        if self.energy < 20.0:
            self.current_drive = "rest"; return
        if self.ally_anchor and self.fear_level > 0.7:
            self.current_drive = "stay_with_ally"; return
        if self.fear_level > 0.7:
            self.current_drive = "seek_safety"; return
        if (self.behavior_rules.exploration_bias > 0.1 and self.curiosity_charge >= CURIOUS_MIN_FOR_EXPLORE):
            self.current_drive = "explore"; return
        self.current_drive = "idle"

    def update_survival_score(self):
        if not self.alive:
            self.survival_score = 0.0
            return
        health_term = _clamp(self.health / 100.0, 0.0, 1.0)
        fear_term = _clamp(1.0 - self.fear_level, 0.0, 1.0)
        self.survival_score = _clamp(0.5 * health_term + 0.5 * fear_term, 0.0, 1.0)

    def _update_curiosity(self):
        safe_enough = (self.fear_level < 0.4 and self.health > 50.0 and self.energy > 40.0 and self.hunger < 60.0)
        if safe_enough:
            self.curiosity_charge = _clamp(self.curiosity_charge + CURIOUS_RECOVERY_RATE, 0.0, 1.0)
        if self.current_drive == "explore":
            self.curiosity_charge = _clamp(self.curiosity_charge - CURIOUS_DRAIN_EXPLORE, 0.0, 1.0)

    def _update_last_thought(self):
        if not self.alive:
            self.last_thought = "..."
            return
        if self.ollama_last_thought:
            recent_window = max(8, int(max(1, self.ollama_interval_ticks)) // 3)
            if (int(self.age_ticks) - int(self.ollama_last_response_tick)) <= recent_window:
                self.last_thought = str(self.ollama_last_thought)
                return
        drive = self.current_drive
        if drive == "heal":
            self.last_thought = "Мне больно. Мне нужно восстановиться."
        elif drive == "find_food":
            self.last_thought = "Я голоден. Ищу еду."
        elif drive == "rest":
            self.last_thought = "Я вымотан. Хочу отдохнуть."
        elif drive == "stay_with_ally":
            self.last_thought = "Мне страшно. Я держусь рядом."
        elif drive == "seek_safety":
            self.last_thought = "Опасно здесь. Надо уйти в безопасное место."
        elif drive == "explore":
            self.last_thought = "Вроде всё спокойно. Посмотрю, что есть вокруг."
        elif drive == "idle":
            self.last_thought = "Я дышу. Я наблюдаю."
        else:
            self.last_thought = "…"

    def _append_ollama_log(self, role: str, text: str, *, tick: Optional[int] = None) -> None:
        line = str(text or "").strip()
        if not line:
            return
        try:
            tick_val = int(self.age_ticks if tick is None else tick)
        except Exception:
            tick_val = int(self.age_ticks)
        self.ollama_log_seq = int(self.ollama_log_seq) + 1
        self.ollama_dialogue_tail.append({
            "seq": int(self.ollama_log_seq),
            "tick": tick_val,
            "role": str(role or "log"),
            "text": line,
        })
        if len(self.ollama_dialogue_tail) > 120:
            self.ollama_dialogue_tail = self.ollama_dialogue_tail[-120:]

    def append_ollama_command_example(self, row: Dict[str, Any], *, tick: Optional[int] = None) -> None:
        if not isinstance(row, dict):
            return
        command = str(row.get("command") or "").strip()
        parsed = str(row.get("parsed") or "").strip()
        if not command or not parsed:
            return
        try:
            tick_val = int(self.age_ticks if tick is None else tick)
        except Exception:
            tick_val = int(self.age_ticks)
        payload = dict(row)
        payload["command"] = command[:160]
        payload["parsed"] = parsed[:200]
        payload["tick"] = tick_val
        self.ollama_command_example_seq = int(self.ollama_command_example_seq) + 1
        payload["seq"] = int(self.ollama_command_example_seq)
        dedup_key = (
            payload["command"],
            payload["parsed"],
            str(payload.get("outcome") or "")[:160],
            bool(payload.get("success")),
        )
        kept: List[Dict[str, Any]] = []
        for existing in list(self.ollama_command_examples or []):
            if not isinstance(existing, dict):
                continue
            existing_key = (
                str(existing.get("command") or ""),
                str(existing.get("parsed") or ""),
                str(existing.get("outcome") or "")[:160],
                bool(existing.get("success")),
            )
            if existing_key == dedup_key:
                continue
            kept.append(dict(existing))
        kept.append(payload)
        self.ollama_command_examples = kept[-80:]

    def configure_ollama(
        self,
        *,
        enabled: Optional[bool] = None,
        auto_mode: Optional[bool] = None,
        model: Optional[str] = None,
        host: Optional[str] = None,
        interval_ticks: Optional[int] = None,
        tick: Optional[int] = None,
    ) -> None:
        if enabled is not None:
            self.ollama_enabled = bool(enabled)
        if auto_mode is not None:
            self.ollama_auto_mode = bool(auto_mode)
        if model is not None:
            self.ollama_model = str(model).strip() or None
        if host is not None:
            self.ollama_host = str(host).strip() or None
        if interval_ticks is not None:
            self.ollama_interval_ticks = max(1, int(interval_ticks))
        self.ollama_status = "ready" if self.ollama_enabled else "off"
        self._append_ollama_log(
            "system",
            "Ollama integrated into brain" if self.ollama_enabled else "Ollama integration disabled",
            tick=tick,
        )

    def queue_ollama_instruction(self, text: str, *, tick: Optional[int] = None) -> None:
        line = str(text or "").strip()
        if not line:
            return
        self.ollama_enabled = True
        self.ollama_pending_instruction = line
        self.ollama_last_operator_instruction = line
        self.ollama_force_request = True
        self.ollama_status = "queued"
        self.mark_ollama_operator_override(tick=tick, hold_ticks=max(180, int(self.ollama_interval_ticks) * 2))
        self._append_ollama_log("you", line, tick=tick)

    def clear_ollama_goal(self) -> None:
        self.ollama_active_goal = None
        self.ollama_goal_started_tick = None
        self.ollama_active_lesson = ""
        self.ollama_active_reward_hint = 0.12
        self.ollama_active_command_source = None

    def mark_ollama_operator_override(self, *, tick: Optional[int] = None, hold_ticks: int = 240) -> None:
        base_tick = int(self.age_ticks if tick is None else tick)
        self.ollama_operator_override_until_tick = max(
            int(self.ollama_operator_override_until_tick),
            int(base_tick) + max(1, int(hold_ticks)),
        )

    def clear_ollama_operator_override(self) -> None:
        self.ollama_operator_override_until_tick = -10**9

    def is_ollama_operator_override_active(self, *, tick: Optional[int] = None) -> bool:
        now = int(self.age_ticks if tick is None else tick)
        return bool(self.ollama_authoritative_commands) and int(now) <= int(self.ollama_operator_override_until_tick)

    def export_ollama_state_for_ui(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.ollama_enabled),
            "auto_mode": bool(self.ollama_auto_mode),
            "interval_ticks": int(max(1, self.ollama_interval_ticks)),
            "model": self.ollama_model,
            "host": self.ollama_host,
            "status": str(self.ollama_status or "off"),
            "last_error": self.ollama_last_error,
            "last_model": self.ollama_last_model,
            "last_thought": str(self.ollama_last_thought or ""),
            "active_lesson": str(self.ollama_active_lesson or ""),
            "active_goal": (
                {"x": float(self.ollama_active_goal[0]), "y": float(self.ollama_active_goal[1])}
                if isinstance(self.ollama_active_goal, tuple) and len(self.ollama_active_goal) == 2
                else None
            ),
            "goal_started_tick": self.ollama_goal_started_tick,
            "active_reward_hint": float(self.ollama_active_reward_hint),
            "pending_instruction": self.ollama_pending_instruction,
            "last_operator_instruction": self.ollama_last_operator_instruction,
            "force_request": bool(self.ollama_force_request),
            "request_inflight": bool(self.ollama_request_inflight),
            "last_request_tick": int(self.ollama_last_request_tick),
            "last_response_tick": int(self.ollama_last_response_tick),
            "authoritative_commands": bool(self.ollama_authoritative_commands),
            "operator_override_until_tick": int(self.ollama_operator_override_until_tick),
            "active_command_source": self.ollama_active_command_source,
            "dialogue_tail": list(self.ollama_dialogue_tail[-60:]),
            "log_seq": int(self.ollama_log_seq),
            "command_examples": list(self.ollama_command_examples[-24:]),
            "command_example_seq": int(self.ollama_command_example_seq),
        }

    # --------------------------------------------------------------------
    # Интеграция с AgentMemory.summarize_recent()
    # --------------------------------------------------------------------

    def absorb_recent_memory_summary(self, recent: Dict[str, Any]):
        if not recent:
            return
        if recent.get("took_damage"):
            self._took_damage_recently = True
        if recent.get("healed"):
            self._healed_recently = True
        if recent.get("dead_flag"):
            self._died_last_tick = True
        for spot in recent.get("hazard_spots", []) or []:
            self._remember_trauma_spot(_round_tuple_xy(spot), intensity=0.4)
        for spot in recent.get("safe_spots", []) or []:
            self._remember_safe_spot(_round_tuple_xy(spot), relief=0.2)
        if recent.get("saw_panic"):
            self.fear_level = _clamp(self.fear_level + 0.1, 0.0, 1.0)

    # --------------------------------------------------------------------
    # Жизненные события (ручные записи боли/лечения/смерти)
    # --------------------------------------------------------------------

    def add_memory(self, ev: Any):
        """Унифицированное добавление события памяти (dict или MemoryEvent)."""
        if isinstance(ev, MemoryEvent):
            me = ev
        elif isinstance(ev, dict):
            etype = str(ev.get("etype") or ev.get("type") or ev.get("event") or "event")
            tick_raw = ev.get("tick", self.age_ticks)
            try:
                tick = int(self.age_ticks if tick_raw is None else tick_raw)
            except Exception:
                tick = int(self.age_ticks)
            data = ev.get("data")
            if not isinstance(data, dict):
                data = {k: v for k, v in ev.items() if k not in ("tick", "etype", "type", "event")}
            me = MemoryEvent(tick=tick, etype=etype, data=dict(data))
        else:
            me = MemoryEvent(tick=self.age_ticks, etype=type(ev).__name__, data={"value": ev})
        self.memory_tail.append(me)
        if len(self.memory_tail) > 256:
            self.memory_tail = self.memory_tail[-256:]

    def record_event(self, tick: int, etype: str, data: Dict[str, Any]):
        self.add_memory(MemoryEvent(tick=tick, etype=etype, data=data))
        if etype == "pain":
            self._took_damage_recently = True
            if "pos" in data:
                self._remember_trauma_spot(_round_tuple_xy(data["pos"]), intensity=0.4)
        if etype == "heal":
            self._healed_recently = True
            if "pos" in data:
                self._remember_safe_spot(_round_tuple_xy(data["pos"]), relief=0.2)
        if etype == "death":
            self._died_last_tick = True
            if "pos" in data:
                self._remember_trauma_spot(_round_tuple_xy(data["pos"]), intensity=1.0)
            reason = data.get("reason") or data.get("cause") or data.get("why")
            if isinstance(reason, str):
                self.last_death_reason = reason

    def evolve_after_tick(self) -> None:
        self.behavior_rules.mutate_from_experience(
            died_recently=bool(self._died_last_tick),
            took_damage=bool(self._took_damage_recently),
            healed=bool(self._healed_recently),
            survival_score_snapshot=float(_clamp(self.survival_score, 0.0, 1.0)),
        )
        self._took_damage_recently = False
        self._healed_recently = False
        self._died_last_tick = False

    def on_pain(
        self,
        source_id: Optional[str] = None,
        amount: float = 0.0,
        pos: Optional[Tuple[float, float]] = None,
        tick: Optional[int] = None,
    ) -> None:
        dmg = max(0.0, _safe_float(amount, 0.0))
        self.fear_level = _clamp(self.fear_level + min(0.35, dmg / 120.0), 0.0, 1.0)
        data: Dict[str, Any] = {"amount": dmg}
        if source_id:
            data["source"] = source_id
        if pos is not None:
            data["pos"] = _round_tuple_xy(pos)
        ev_tick = int(self.age_ticks if tick is None else tick)
        self.record_event(ev_tick, "pain", data)

    def on_threat(
        self,
        kind: str = "threat",
        attacker_id: Optional[str] = None,
        species: Optional[str] = None,
        damage: float = 0.0,
        pos: Optional[Tuple[float, float]] = None,
        tick: Optional[int] = None,
    ) -> None:
        self.fear_level = _clamp(self.fear_level + 0.08, 0.0, 1.0)
        data: Dict[str, Any] = {
            "kind": kind,
            "damage": max(0.0, _safe_float(damage, 0.0)),
        }
        if attacker_id:
            data["attacker"] = attacker_id
        if species:
            data["species"] = species
        if pos is not None:
            data["pos"] = _round_tuple_xy(pos)
        ev_tick = int(self.age_ticks if tick is None else tick)
        self.record_event(ev_tick, "animal_attack" if kind == "animal" else "threat", data)

    def on_step(self, state: Dict[str, Any], reward: float) -> None:
        tick_raw = state.get("tick", self.age_ticks)
        try:
            tick = int(self.age_ticks if tick_raw is None else tick_raw)
        except Exception:
            tick = int(self.age_ticks)
        self.add_memory({
            "type": "step",
            "tick": tick,
            "data": {"reward": float(_safe_float(reward, 0.0)), **dict(state or {})},
        })

    # --------------------------------------------------------------------
    # Beliefs API (для UI-шимов)
    # --------------------------------------------------------------------

    def get_beliefs(self) -> List[Dict[str, Any]]:
        return [{
            "if": b.condition,
            "then": b.conclusion,
            "strength": float(_clamp(b.strength, 0.0, 1.0)),
        } for b in self.beliefs]

    def add_belief(self, b: Dict[str, Any]) -> None:
        cond = str(b.get("if") or b.get("condition") or "")
        concl = str(b.get("then") or b.get("conclusion") or "")
        strength = _clamp(_safe_float(b.get("strength", 0.5), 0.5), 0.0, 1.0)
        if not cond or not concl:
            return
        self._add_or_strengthen_belief(cond, concl, strength)

    # --------------------------------------------------------------------
    # Главный апдейт за тик (вызов мира)
    # --------------------------------------------------------------------

    def _tick_update_explicit(
        self,
        tick: int,
        pos: Tuple[float, float],
        health: float,
        fear_level: float,
        alive: bool,
        hazards_near: List[Tuple[float, float, float]],
        healzones_near: List[Tuple[float, float, float]],
        allies_near: List[str],
        energy: Optional[float] = None,
        hunger: Optional[float] = None,
        allies_near_details: Optional[List[Tuple[str, float, float]]] = None,
    ):
        self.age_ticks += 1

        self.health = _clamp(_safe_float(health, self.health), 0.0, 100.0)
        self.alive = bool(alive)
        if energy is not None:
            self.energy = _clamp(_safe_float(energy, self.energy), 0.0, 100.0)
        if hunger is not None:
            self.hunger = _clamp(_safe_float(hunger, self.hunger), 0.0, 100.0)
        self.fear_level = _clamp(_safe_float(fear_level, self.fear_level), 0.0, 1.0)

        self.integrate_world_observation(tick=tick, pos=pos,
                                         hazards_near=hazards_near,
                                         healzones_near=healzones_near,
                                         allies_near=allies_near)

        if allies_near_details:
            self._infect_fear_from_allies(allies_near_details)

        self._decay_trauma_map()
        self._update_social_anchor()
        self.choose_drive()
        self.update_survival_score()
        self.evolve_after_tick()
        self._update_curiosity()
        self._update_last_thought()

    def tick_update(self, *args, **kwargs):
        """
        Унифицированный вход:
        - старый подробный вызов: tick_update(tick, pos, health, fear_level, alive, hazards_near, healzones_near, allies_near, ...)
        - новый удобный вызов:    tick_update(agent_ref=agent, world_ref=world)
        """
        if "agent_ref" in kwargs and "world_ref" in kwargs:
            agent = kwargs.get("agent_ref")
            world = kwargs.get("world_ref")

            tick = _world_time_int(world)
            pos = (_safe_float(getattr(agent, "x", 0.0), 0.0), _safe_float(getattr(agent, "y", 0.0), 0.0))
            health = _safe_float(getattr(agent, "health", 100.0), 100.0)
            fear_level = _clamp(_safe_float(getattr(agent, "fear", 0.0), 0.0), 0.0, 1.0)
            alive = bool(getattr(agent, "is_alive", lambda: health > 0.0)())

            energy = getattr(agent, "energy", None)
            if energy is not None and energy <= 1.5:
                energy = _safe_float(energy * 100.0, 100.0)
            elif energy is not None:
                energy = _safe_float(energy, 100.0)

            hunger = getattr(agent, "hunger", None)
            if hunger is not None and hunger <= 1.5:
                hunger = _safe_float(hunger * 100.0, 0.0)
            elif hunger is not None:
                hunger = _safe_float(hunger, 0.0)

            hazards, heals, allies, allies_details = _extract_env(world, agent)
            return self._tick_update_explicit(
                tick=tick, pos=pos, health=health, fear_level=fear_level, alive=alive,
                hazards_near=hazards, healzones_near=heals, allies_near=allies,
                energy=energy, hunger=hunger, allies_near_details=allies_details
            )
        else:
            return self._tick_update_explicit(*args, **kwargs)

    # --------------------------------------------------------------------
    # Экспорт состояния мозга наружу (для UI Mind Tab)
    # --------------------------------------------------------------------

    def export_public_state_for_ui(self) -> Dict[str, Any]:
        return {
            "age_ticks": int(self.age_ticks),
            "fear_level": float(_clamp(self.fear_level, 0.0, 1.0)),
            "health": float(_clamp(self.health, 0.0, 100.0)),
            "energy": float(_clamp(self.energy, 0.0, 100.0)),
            "hunger": float(_clamp(self.hunger, 0.0, 100.0)),
            "alive": bool(self.alive),
            "survival_score": float(_clamp(self.survival_score, 0.0, 1.0)),
            "current_drive": self.current_drive,
            "ally_anchor": self.ally_anchor,
            "last_death_reason": self.last_death_reason,
            "curiosity_charge": round(_clamp(self.curiosity_charge, 0.0, 1.0), 2),
            "last_thought": self.last_thought,
            "behavior_rules": {
                "avoid_hazard_radius": float(self.behavior_rules.avoid_hazard_radius),
                "healing_zone_seek_priority": float(self.behavior_rules.healing_zone_seek_priority),
                "stick_with_ally_if_fear_above": float(self.behavior_rules.stick_with_ally_if_fear_above),
                "exploration_bias": float(_clamp(self.behavior_rules.exploration_bias, 0.0, 1.0)),
            },
            "ollama": self.export_ollama_state_for_ui(),
            "beliefs": [{"if": b.condition, "then": b.conclusion,
                         "strength": round(_clamp(b.strength, 0.0, 1.0), 2)} for b in self.beliefs[-20:]],
            "memory_tail": [{"tick": getattr(ev, "tick", None),
                             "type": getattr(ev, "etype", getattr(ev, "type", "event")),
                             "data": getattr(ev, "data", {})} for ev in self.memory_tail[-20:]],
            "trauma_map": [{"pos": tm["pos"], "intensity": round(_clamp(tm["intensity"], 0.0, 1.0), 2)} for tm in self.trauma_map],
            "skills": {"version": self.gc_version, "steps": int(self.gc_steps),
                       "goals": list(self.gc_goal_vocab.keys())} if (self.gc_version != "0") else None,
        }
