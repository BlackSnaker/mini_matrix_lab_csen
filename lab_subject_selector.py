from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from mind_core import ConsciousnessBlock


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LabSubjectCandidate:
    agent_id: str
    source: str
    path: str
    score: float
    best_survival_score: float
    survival_score: float
    total_age_ticks: int
    age_ticks: int
    skills_steps: int
    memory_events: int
    belief_count: int
    deaths: int
    alive: bool


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _brain_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    brain = raw.get("brain")
    return brain if isinstance(brain, dict) else raw


def _belief_count(brain: Dict[str, Any]) -> int:
    beliefs = brain.get("beliefs")
    if isinstance(beliefs, dict):
        return len(beliefs)
    if isinstance(beliefs, list):
        return len(beliefs)
    return 0


def _memory_count(brain: Dict[str, Any]) -> int:
    for key in ("memory_tail", "memory", "memories"):
        value = brain.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _candidate_from_path(path: Path, source: str) -> Optional[LabSubjectCandidate]:
    raw = _read_json(path)
    if raw is None:
        return None
    brain = _brain_payload(raw)
    lineage = raw.get("_lineage") if isinstance(raw.get("_lineage"), dict) else {}
    runtime = raw.get("_runtime") if isinstance(raw.get("_runtime"), dict) else {}

    fallback_id = path.name
    if fallback_id.endswith(".mind.json"):
        fallback_id = fallback_id[:-10]
    elif fallback_id.endswith(".json"):
        fallback_id = fallback_id[:-5]

    agent_id = str(brain.get("agent_id") or lineage.get("lineage_id") or fallback_id).strip()
    if not agent_id:
        return None

    survival = _safe_float(runtime.get("survival_score", brain.get("survival_score", 0.0)))
    best_survival = _safe_float(lineage.get("best_survival_score", survival), survival)
    age_ticks = _safe_int(brain.get("age_ticks", runtime.get("age_ticks", 0)))
    total_age = max(age_ticks, _safe_int(lineage.get("total_age_ticks", 0)))
    skills_steps = _safe_int(runtime.get("skills_steps", brain.get("gc_steps", 0)))
    memory_events = _memory_count(brain)
    belief_count = _belief_count(brain)
    deaths = _safe_int(lineage.get("deaths", 0))
    alive = bool(runtime.get("alive_now", brain.get("alive", True)))

    # The score prefers agents that both survived well and accumulated a large,
    # structured history. Caps keep a huge belief list from being the only signal.
    score = (
        max(0.0, min(1.0, best_survival)) * 100_000.0
        + max(0.0, min(1.0, survival)) * 30_000.0
        + math.log1p(max(0, total_age)) * 2_500.0
        + min(max(0, skills_steps), 100_000) * 0.25
        + min(max(0, memory_events), 500) * 15.0
        + min(max(0, belief_count), 10_000) * 20.0
        - max(0, deaths) * 1_000.0
        + (1_000.0 if alive else -10_000.0)
    )

    return LabSubjectCandidate(
        agent_id=agent_id,
        source=source,
        path=str(path),
        score=float(score),
        best_survival_score=float(best_survival),
        survival_score=float(survival),
        total_age_ticks=int(total_age),
        age_ticks=int(age_ticks),
        skills_steps=int(skills_steps),
        memory_events=int(memory_events),
        belief_count=int(belief_count),
        deaths=int(deaths),
        alive=bool(alive),
    )


def iter_lab_subject_candidates(
    *,
    base_dir: Path | str = PROJECT_ROOT,
    include_room_brains: bool = False,
) -> Iterable[LabSubjectCandidate]:
    root = Path(base_dir)
    sources = (
        ("brains", root / "brains", "*.json"),
        ("trained_brains", root / "trained_brains", "*.mind.json"),
    )
    if include_room_brains:
        sources = sources + (("room_brains", root / "room_brains", "*.json"),)
    for source, folder, pattern in sources:
        if not folder.exists():
            continue
        for path in sorted(folder.glob(pattern)):
            candidate = _candidate_from_path(path, source)
            if candidate is not None:
                yield candidate


def select_best_trained_agent(
    *,
    base_dir: Path | str = PROJECT_ROOT,
    include_room_brains: bool = False,
) -> Optional[LabSubjectCandidate]:
    candidates = list(iter_lab_subject_candidates(base_dir=base_dir, include_room_brains=include_room_brains))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            c.score,
            c.best_survival_score,
            c.survival_score,
            c.total_age_ticks,
            c.belief_count,
            c.agent_id,
        ),
    )


def load_candidate_brain(candidate: LabSubjectCandidate, *, agent_id: Optional[str] = None) -> Optional[ConsciousnessBlock]:
    raw = _read_json(Path(candidate.path))
    if raw is None:
        return None
    try:
        block = ConsciousnessBlock.from_dict(_brain_payload(raw))
        block.agent_id = str(agent_id or candidate.agent_id)
        return block
    except Exception:
        return None


def make_lab_subject_lineup_spec(candidate: LabSubjectCandidate) -> Dict[str, str]:
    return {
        "id": candidate.agent_id,
        "name": f"Subject {candidate.agent_id}",
        "persona": (
            f"Ты лабораторный агент {candidate.agent_id}. "
            "Ты уже прошёл длительное обучение и находишься в пустой белой комнате для разговорной подготовки. "
            "Отвечай оператору от первого лица: коротко, осмысленно, спокойно и с опорой на свою память. "
            "Не выдумывай интерьер комнаты: здесь только белые стены, белый пол и открытое пространство."
        ),
    }


def format_candidate_summary(candidate: LabSubjectCandidate) -> str:
    return (
        f"{candidate.agent_id} from {candidate.source}: "
        f"score={candidate.score:.1f}, best_survival={candidate.best_survival_score:.3f}, "
        f"survival={candidate.survival_score:.3f}, age={candidate.total_age_ticks}, "
        f"beliefs={candidate.belief_count}, memories={candidate.memory_events}"
    )
