from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import re


_BRAIN_DIRS = ("room_brains", "brains")
_MAX_ARCHIVE_FILES = 96
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")

_CACHE_SIGNATURE: Optional[Tuple[Tuple[str, int, int], ...]] = None
_CACHE_PRIORS: List[Dict[str, Any]] = []
_CACHE_DEMOS: List[Dict[str, Any]] = []


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    if math.isnan(out) or math.isinf(out):
        return fallback
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold().replace("ё", "е")).strip()


def _tokenize(text: Any) -> List[str]:
    raw = _normalize_text(text)
    if not raw:
        return []
    return [tok for tok in _TOKEN_RE.findall(raw) if len(tok) >= 3]


def _pct(value: Any, fallback: float = 0.0) -> float:
    out = _safe_float(value, fallback)
    if out <= 1.5:
        out *= 100.0
    return _clamp(out, 0.0, 100.0)


def _iter_brain_files() -> List[Path]:
    files: List[Path] = []
    for root in _BRAIN_DIRS:
        base = Path(root)
        if not base.exists():
            continue
        for path in base.glob("*.json"):
            if not path.is_file():
                continue
            files.append(path)
    files.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    return files[:_MAX_ARCHIVE_FILES]


def _extract_behaviors(brain: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    rules = brain.get("behavior_rules", {}) if isinstance(brain, dict) else {}
    beliefs = list(brain.get("beliefs", []) or []) if isinstance(brain, dict) else []
    drive = str(runtime.get("current_drive") or brain.get("current_drive") or "idle")
    fear = _clamp(_safe_float(runtime.get("fear_level", brain.get("fear_level", 0.0)), 0.0), 0.0, 1.0)
    survival = _clamp(_safe_float(runtime.get("survival_score", brain.get("survival_score", 1.0)), 1.0), 0.0, 1.0)
    curiosity = _clamp(_safe_float(runtime.get("curiosity_charge", brain.get("curiosity_charge", 0.5)), 0.5), 0.0, 1.0)
    energy = _pct(runtime.get("energy", brain.get("energy", 100.0)), 100.0)
    hunger = _pct(runtime.get("hunger", brain.get("hunger", 0.0)), 0.0)
    health = _safe_float(runtime.get("health", brain.get("health", 100.0)), 100.0)
    avoid_radius = _safe_float(rules.get("avoid_hazard_radius", 6.0), 6.0)
    heal_priority = _clamp(_safe_float(rules.get("healing_zone_seek_priority", 0.5), 0.5), 0.0, 1.0)
    explore_bias = _clamp(_safe_float(rules.get("exploration_bias", 0.2), 0.2), 0.0, 1.0)
    ally_bias = _clamp(_safe_float(rules.get("stick_with_ally_if_fear_above", 0.7), 0.7), 0.0, 1.0)

    traits: List[str] = []
    if fear <= 0.18:
        traits.append("calm")
    elif fear >= 0.55:
        traits.append("tense")
    if curiosity >= 0.68:
        traits.append("curious")
    elif curiosity <= 0.28:
        traits.append("reserved")
    if survival >= 0.82:
        traits.append("resilient")
    if health <= 35.0:
        traits.append("fragile")
    if hunger >= 70.0:
        traits.append("food_driven")
    if energy <= 30.0:
        traits.append("tired")
    if drive in {"explore", "find_food", "heal", "rest", "stay_with_ally", "seek_safety"}:
        traits.append(f"drive:{drive}")
    if avoid_radius >= 7.8:
        traits.append("hazard_aware")
    if heal_priority >= 0.62 or drive == "heal":
        traits.append("self_preserving")
    if explore_bias >= 0.32 or drive == "explore":
        traits.append("explorer")
    if ally_bias <= 0.45:
        traits.append("independent")
    elif ally_bias >= 0.78:
        traits.append("social")
    if any(str(row.get("conclusion") or "").strip() == "stay_away" for row in beliefs if isinstance(row, dict)):
        traits.append("learned_avoidance")

    traits = list(dict.fromkeys(traits))[:8]
    beliefs_tail: List[str] = []
    for row in beliefs[-4:]:
        if not isinstance(row, dict):
            continue
        cond = str(row.get("condition") or row.get("if") or "").strip()
        concl = str(row.get("conclusion") or row.get("then") or "").strip()
        if cond and concl:
            beliefs_tail.append(f"{cond} -> {concl}")

    summary_parts = [
        f"drive={drive}",
        f"fear={fear:.2f}",
        f"curiosity={curiosity:.2f}",
        f"survival={survival:.2f}",
    ]
    if traits:
        summary_parts.append("traits=" + ",".join(traits))
    return {
        "drive": drive,
        "fear": round(fear, 3),
        "survival": round(survival, 3),
        "curiosity": round(curiosity, 3),
        "energy": round(energy, 1),
        "hunger": round(hunger, 1),
        "health": round(health, 1),
        "rules": {
            "avoid_hazard_radius": round(avoid_radius, 2),
            "healing_zone_seek_priority": round(heal_priority, 2),
            "exploration_bias": round(explore_bias, 2),
            "stick_with_ally_if_fear_above": round(ally_bias, 2),
        },
        "traits": traits,
        "belief_hints": beliefs_tail[:3],
        "summary": "; ".join(summary_parts),
    }


def _dialogue_examples(dialogue_tail: Iterable[Dict[str, Any]], *, source: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in list(dialogue_tail or []):
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip().casefold()
        text = str(row.get("text") or "").strip()
        tick = int(_safe_float(row.get("tick", 0), 0.0))
        if role == "you":
            if current and current.get("success") is not None:
                examples.append(current)
            current = {
                "source": source,
                "tick": tick,
                "command": text,
                "parsed": "",
                "plan": "",
                "lesson": "",
                "thought": "",
                "outcome": "",
                "success": None,
                "brain_traits": list(profile.get("traits", []) or []),
                "drive": str(profile.get("drive") or "idle"),
            }
            continue
        if current is None:
            continue
        if role == "parsed" and not current["parsed"]:
            current["parsed"] = text
        elif role == "plan" and not current["plan"]:
            current["plan"] = text
        elif role == "ollama" and not current["lesson"]:
            current["lesson"] = text
        elif role == "agent" and not current["thought"]:
            current["thought"] = text
        elif role == "system":
            low = _normalize_text(text)
            if (
                "команда движения" in low
                or "выполнен локально" in low
                or "изменена локально" in low
                or "цель достигнута" in low
                or "удерживаю позицию" in low
                or "сохранена в память локально" in low
                or "команда ожидания выполнена" in low
            ):
                current["success"] = True
                current["outcome"] = text
            elif "не выполнена" in low:
                current["success"] = False
                current["outcome"] = text
        elif role == "error":
            current["success"] = False
            current["outcome"] = text
    if current and current.get("success") is not None:
        examples.append(current)
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for row in reversed(examples):
        key = (
            str(row.get("command") or ""),
            str(row.get("parsed") or ""),
            str(row.get("outcome") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.reverse()
    return out[-24:]


def _structured_examples(rows: Iterable[Dict[str, Any]], *, source: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        command = str(row.get("command") or "").strip()
        parsed = str(row.get("parsed") or "").strip()
        if not command or not parsed:
            continue
        out.append({
            "source": source,
            "tick": int(_safe_float(row.get("tick", 0), 0.0)),
            "command": command[:160],
            "parsed": parsed[:200],
            "plan": str(row.get("plan") or "").strip()[:200],
            "lesson": str(row.get("lesson") or "").strip()[:180],
            "thought": str(row.get("thought") or "").strip()[:160],
            "outcome": str(row.get("outcome") or "").strip()[:180],
            "success": bool(row.get("success")),
            "brain_traits": list(row.get("traits", row.get("brain_traits", profile.get("traits", []))) or [])[:8],
            "drive": str(row.get("drive") or profile.get("drive") or "idle"),
        })
    return out[-40:]


def _brain_file_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_archive_cache() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    global _CACHE_SIGNATURE, _CACHE_PRIORS, _CACHE_DEMOS
    files = _iter_brain_files()
    signature = tuple((str(path), int(path.stat().st_mtime_ns), int(path.stat().st_size)) for path in files)
    if signature == _CACHE_SIGNATURE:
        return list(_CACHE_PRIORS), list(_CACHE_DEMOS)

    priors: List[Dict[str, Any]] = []
    demos: List[Dict[str, Any]] = []
    for path in files:
        payload = _brain_file_payload(path)
        if not isinstance(payload, dict):
            continue
        brain = payload.get("brain", payload)
        if not isinstance(brain, dict):
            continue
        runtime = payload.get("_runtime", {}) or {}
        if not isinstance(runtime, dict):
            runtime = {}
        profile = _extract_behaviors(brain, runtime)
        priors.append({
            "source": str(path),
            "profile": profile,
            "last_thought": str(runtime.get("last_thought") or brain.get("last_thought") or "").strip()[:120],
            "last_death_reason": str(runtime.get("last_death_reason") or "")[:80],
        })
        ollama = brain.get("ollama", {}) if isinstance(brain.get("ollama"), dict) else {}
        structured = list(ollama.get("command_examples", []) or [])
        if structured:
            demos.extend(_structured_examples(structured, source=str(path), profile=profile))
        dialogue_tail = ollama.get("dialogue_tail", []) or []
        if dialogue_tail and not structured:
            demos.extend(_dialogue_examples(dialogue_tail, source=str(path), profile=profile))

    _CACHE_SIGNATURE = signature
    _CACHE_PRIORS = priors
    _CACHE_DEMOS = demos
    return list(priors), list(demos)


def _score_prior(current: Dict[str, Any], prior: Dict[str, Any]) -> float:
    profile = prior.get("profile", {}) if isinstance(prior.get("profile"), dict) else {}
    score = 0.0
    if str(profile.get("drive") or "") == str(current.get("drive") or ""):
        score += 2.4
    current_traits = set(str(x) for x in list(current.get("traits", []) or []))
    profile_traits = set(str(x) for x in list(profile.get("traits", []) or []))
    score += 0.7 * len(current_traits & profile_traits)
    score -= abs(_safe_float(current.get("fear"), 0.0) - _safe_float(profile.get("fear"), 0.0)) * 1.7
    score -= abs(_safe_float(current.get("curiosity"), 0.0) - _safe_float(profile.get("curiosity"), 0.0)) * 1.4
    score -= abs(_safe_float(current.get("survival"), 0.0) - _safe_float(profile.get("survival"), 0.0)) * 1.2
    return score


def _action_family(text: Any) -> str:
    raw = _normalize_text(text)
    if not raw:
        return "unknown"
    head = raw.split("->", 1)[0].strip()
    return head.split("(", 1)[0].strip() or raw.split(" ", 1)[0]


def _score_demo(current_profile: Dict[str, Any], operator_instruction: str, demo: Dict[str, Any]) -> float:
    score = 0.0
    cmd_tokens = set(_tokenize(operator_instruction))
    demo_tokens = set(_tokenize(str(demo.get("command") or ""))) | set(_tokenize(str(demo.get("parsed") or "")))
    score += float(len(cmd_tokens & demo_tokens)) * 1.8
    if _action_family(demo.get("parsed") or demo.get("plan") or demo.get("lesson")) in _normalize_text(operator_instruction):
        score += 1.2
    demo_traits = set(str(x) for x in list(demo.get("brain_traits", []) or []))
    current_traits = set(str(x) for x in list(current_profile.get("traits", []) or []))
    score += 0.55 * len(current_traits & demo_traits)
    if str(demo.get("drive") or "") == str(current_profile.get("drive") or ""):
        score += 0.8
    source = str(demo.get("source") or "")
    if "live_examples" in source:
        score += 1.0
    elif "room_brains" in source and "dialog" not in source:
        score += 0.4
    score += 1.0 if bool(demo.get("success")) else -1.4
    return score


def build_behavior_context(
    snapshot: Dict[str, Any],
    *,
    live_dialogue: Optional[Iterable[Dict[str, Any]]] = None,
    live_examples: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    agent = snapshot.get("agent", {}) or {}
    brain = snapshot.get("brain", {}) or {}
    current_profile = _extract_behaviors(
        {
            "behavior_rules": dict(brain.get("behavior_rules", {}) or {}),
            "beliefs": list(brain.get("beliefs", []) or []),
            "fear_level": brain.get("fear_level", brain.get("fear", 0.0)),
            "survival_score": brain.get("survival_score", 1.0),
            "curiosity_charge": brain.get("curiosity_charge", 0.5),
            "current_drive": brain.get("current_drive", "idle"),
            "last_thought": brain.get("last_thought", ""),
        },
        {
            "current_drive": brain.get("current_drive", "idle"),
            "fear_level": brain.get("fear", 0.0),
            "survival_score": brain.get("survival_score", 1.0),
            "curiosity_charge": brain.get("curiosity_charge", 0.5),
            "energy": agent.get("energy", 100.0),
            "hunger": agent.get("hunger", 0.0),
            "health": agent.get("health", 100.0),
            "last_thought": brain.get("last_thought", ""),
        },
    )

    archive_priors, archive_demos = _ensure_archive_cache()
    demos: List[Dict[str, Any]] = []
    if live_examples:
        demos.extend(_structured_examples(live_examples, source="live_examples", profile=current_profile))
    live_rows = list(live_dialogue or [])
    if live_rows:
        demos.extend(_dialogue_examples(live_rows, source="live_brain", profile=current_profile))
    demos.extend(archive_demos)

    operator_instruction = str(snapshot.get("operator_instruction") or "").strip()
    ranked_demos = sorted(
        demos,
        key=lambda row: _score_demo(current_profile, operator_instruction, row),
        reverse=True,
    )
    selected_demos: List[Dict[str, Any]] = []
    seen_demo_keys: set[Tuple[str, str]] = set()
    for row in ranked_demos:
        key = (str(row.get("command") or ""), str(row.get("parsed") or row.get("lesson") or ""))
        if key in seen_demo_keys:
            continue
        seen_demo_keys.add(key)
        selected_demos.append({
            "source": str(row.get("source") or ""),
            "command": str(row.get("command") or "")[:120],
            "parsed": str(row.get("parsed") or row.get("plan") or "")[:120],
            "lesson": str(row.get("lesson") or "")[:140],
            "outcome": str(row.get("outcome") or "")[:140],
            "success": bool(row.get("success")),
            "traits": list(row.get("brain_traits", []) or [])[:6],
        })
        if len(selected_demos) >= 4:
            break

    ranked_priors = sorted(archive_priors, key=lambda row: _score_prior(current_profile, row), reverse=True)
    selected_priors: List[Dict[str, Any]] = []
    seen_sources: set[str] = set()
    for row in ranked_priors:
        source = str(row.get("source") or "")
        if source in seen_sources:
            continue
        seen_sources.add(source)
        profile = row.get("profile", {}) if isinstance(row.get("profile"), dict) else {}
        selected_priors.append({
            "source": source,
            "summary": str(profile.get("summary") or "")[:180],
            "traits": list(profile.get("traits", []) or [])[:6],
            "drive": str(profile.get("drive") or "idle"),
            "last_thought": str(row.get("last_thought") or "")[:100],
        })
        if len(selected_priors) >= 4:
            break

    summary = (
        f"profile={current_profile.get('summary', '')}; "
        f"examples={len(selected_demos)}; priors={len(selected_priors)}"
    )
    return {
        "current_profile": current_profile,
        "command_examples": selected_demos,
        "style_priors": selected_priors,
        "summary": summary[:220],
    }
