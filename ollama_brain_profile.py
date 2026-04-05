from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import math
import time


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


def _pct(value: Any, fallback: float = 0.0) -> float:
    out = _safe_float(value, fallback)
    if out <= 1.5:
        out *= 100.0
    return _clamp(out, 0.0, 100.0)


def _truncate(text: Any, limit: int) -> str:
    line = str(text or "").strip()
    if len(line) <= int(limit):
        return line
    return line[: max(0, int(limit) - 3)].rstrip() + "..."


def _normalize_memory_rows(memory_tail: Iterable[Any], *, limit: int = 200) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(memory_tail or [])[-max(1, int(limit)):]:
        if isinstance(item, dict):
            tick = int(_safe_float(item.get("tick", 0), 0.0))
            etype = str(item.get("etype") or item.get("type") or "event")
            data = item.get("data", {})
        else:
            tick = int(_safe_float(getattr(item, "tick", 0), 0.0))
            etype = str(getattr(item, "etype", getattr(item, "type", "event")))
            data = getattr(item, "data", {})
        payload = dict(data) if isinstance(data, dict) else {"value": _truncate(data, 120)}
        rows.append({
            "tick": tick,
            "type": etype[:48],
            "data": payload,
            "summary": _truncate(f"{etype}: {payload}", 180),
        })
    return rows


def _normalize_belief_rows(beliefs: Iterable[Any], *, limit: int = 160) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in list(beliefs or [])[-max(1, int(limit)):]:
        if isinstance(item, dict):
            condition = str(item.get("condition") or item.get("if") or "").strip()
            conclusion = str(item.get("conclusion") or item.get("then") or "").strip()
            strength = round(_clamp(_safe_float(item.get("strength", 0.5), 0.5), 0.0, 1.0), 3)
        else:
            condition = str(getattr(item, "condition", "") or "").strip()
            conclusion = str(getattr(item, "conclusion", "") or "").strip()
            strength = round(_clamp(_safe_float(getattr(item, "strength", 0.5), 0.5), 0.0, 1.0), 3)
        if not condition or not conclusion:
            continue
        rows.append({
            "if": condition[:140],
            "then": conclusion[:140],
            "strength": strength,
        })
    return rows


def _normalize_command_examples(rows: Iterable[Dict[str, Any]], *, limit: int = 120) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or [])[-max(1, int(limit)):]:
        if not isinstance(row, dict):
            continue
        command = str(row.get("command") or "").strip()
        parsed = str(row.get("parsed") or "").strip()
        if not command or not parsed:
            continue
        out.append({
            "tick": int(_safe_float(row.get("tick", 0), 0.0)),
            "command": command[:160],
            "parsed": parsed[:200],
            "lesson": _truncate(row.get("lesson"), 180),
            "thought": _truncate(row.get("thought"), 160),
            "outcome": _truncate(row.get("outcome"), 180),
            "success": bool(row.get("success")),
            "traits": [str(x) for x in list(row.get("traits", row.get("brain_traits", [])) or [])[:8]],
            "drive": str(row.get("drive") or "idle"),
        })
    return out


def _normalize_dialogue_rows(rows: Iterable[Dict[str, Any]], *, limit: int = 160) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in list(rows or [])[-max(1, int(limit)):]:
        if not isinstance(row, dict):
            continue
        role = str(row.get("role") or "").strip()
        text = str(row.get("text") or "").strip()
        if not role or not text:
            continue
        out.append({
            "tick": int(_safe_float(row.get("tick", 0), 0.0)),
            "role": role[:32],
            "text": _truncate(text, 220),
        })
    return out


def _derive_emotion_baseline(block: Any, runtime: Dict[str, Any]) -> Dict[str, Any]:
    current_drive = str(
        runtime.get("current_drive")
        or getattr(block, "current_drive", None)
        or "idle"
    )
    fear = round(_clamp(_safe_float(runtime.get("fear_level", getattr(block, "fear_level", 0.0)), 0.0), 0.0, 1.0), 3)
    curiosity = round(_clamp(_safe_float(runtime.get("curiosity_charge", getattr(block, "curiosity_charge", 0.5)), 0.5), 0.0, 1.0), 3)
    survival = round(_clamp(_safe_float(runtime.get("survival_score", getattr(block, "survival_score", 1.0)), 1.0), 0.0, 1.0), 3)
    energy = round(_pct(runtime.get("energy", getattr(block, "energy", 100.0)), getattr(block, "energy", 100.0)), 2)
    hunger = round(_pct(runtime.get("hunger", getattr(block, "hunger", 0.0)), getattr(block, "hunger", 0.0)), 2)
    health = round(_safe_float(runtime.get("health", getattr(block, "health", 100.0)), getattr(block, "health", 100.0)), 2)
    tone: List[str] = []
    if fear <= 0.18:
        tone.append("calm")
    elif fear >= 0.55:
        tone.append("tense")
    if curiosity >= 0.65:
        tone.append("curious")
    elif curiosity <= 0.28:
        tone.append("reserved")
    if energy <= 32.0:
        tone.append("tired")
    elif energy >= 72.0:
        tone.append("vigorous")
    if survival >= 0.82:
        tone.append("resilient")
    if not tone:
        tone.append("balanced")
    return {
        "drive": current_drive,
        "fear": fear,
        "curiosity": curiosity,
        "survival": survival,
        "energy": energy,
        "hunger": hunger,
        "health": health,
        "tone": tone,
    }


def build_full_brain_study_payload(
    block: Any,
    *,
    lineage: Optional[Dict[str, Any]] = None,
    runtime: Optional[Dict[str, Any]] = None,
    snapshots: Optional[List[Dict[str, Any]]] = None,
    export_meta: Optional[Dict[str, Any]] = None,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    runtime_meta = dict(runtime or {})
    lineage_meta = dict(lineage or {})
    export_meta_val = dict(export_meta or {})
    snapshot_rows = [dict(row) for row in list(snapshots or [])[-12:] if isinstance(row, dict)]

    beliefs = _normalize_belief_rows(getattr(block, "beliefs", []) or [])
    memory_rows = _normalize_memory_rows(getattr(block, "memory_tail", []) or [])
    command_examples = _normalize_command_examples(getattr(block, "ollama_command_examples", []) or [])
    dialogue_tail = _normalize_dialogue_rows(getattr(block, "ollama_dialogue_tail", []) or [])
    trauma_map = [dict(row) for row in list(getattr(block, "trauma_map", []) or [])[-48:] if isinstance(row, dict)]
    behavior_rules = {
        "avoid_hazard_radius": round(_safe_float(getattr(getattr(block, "behavior_rules", None), "avoid_hazard_radius", 6.0), 6.0), 3),
        "healing_zone_seek_priority": round(_clamp(_safe_float(getattr(getattr(block, "behavior_rules", None), "healing_zone_seek_priority", 0.5), 0.5), 0.0, 1.0), 3),
        "stick_with_ally_if_fear_above": round(_clamp(_safe_float(getattr(getattr(block, "behavior_rules", None), "stick_with_ally_if_fear_above", 0.7), 0.7), 0.0, 1.0), 3),
        "exploration_bias": round(_clamp(_safe_float(getattr(getattr(block, "behavior_rules", None), "exploration_bias", 0.2), 0.2), 0.0, 1.0), 3),
    }
    emotion_baseline = _derive_emotion_baseline(block, runtime_meta)
    gc_vocab = getattr(block, "gc_goal_vocab", {}) or {}
    if not isinstance(gc_vocab, dict):
        gc_vocab = {}

    command_family_hits: Dict[str, int] = {}
    for row in command_examples:
        family = str(row.get("parsed") or "").split("(", 1)[0].strip() or "unknown"
        command_family_hits[family] = int(command_family_hits.get(family, 0)) + 1

    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_path": str(source_path or ""),
        "agent": {
            "id": str(getattr(block, "agent_id", "agent")),
            "current_drive": str(getattr(block, "current_drive", "idle")),
            "last_thought": _truncate(getattr(block, "last_thought", ""), 180),
            "last_death_reason": _truncate(getattr(block, "last_death_reason", ""), 120),
            "age_ticks": int(_safe_float(getattr(block, "age_ticks", 0), 0.0)),
            "alive": bool(getattr(block, "alive", True)),
        },
        "emotion_baseline": emotion_baseline,
        "behavior_rules": behavior_rules,
        "beliefs": beliefs,
        "memory_tail": memory_rows,
        "trauma_map": trauma_map,
        "runtime": runtime_meta,
        "lineage": lineage_meta,
        "snapshots": snapshot_rows,
        "export_meta": export_meta_val,
        "ollama": {
            "model": getattr(block, "ollama_last_model", None) or getattr(block, "ollama_model", None),
            "status": str(getattr(block, "ollama_status", "off") or "off"),
            "command_examples": command_examples,
            "dialogue_tail": dialogue_tail,
            "command_families": command_family_hits,
        },
        "skills": {
            "gc_version": str(getattr(block, "gc_version", "0")),
            "gc_steps": int(_safe_float(getattr(block, "gc_steps", 0), 0.0)),
            "gc_goal_vocab_size": len(gc_vocab),
            "gc_goal_vocab_sample": [str(k) for k in list(gc_vocab.keys())[:24]],
        },
        "totals": {
            "beliefs": len(beliefs),
            "memories": len(memory_rows),
            "trauma_spots": len(trauma_map),
            "dialogue_entries": len(dialogue_tail),
            "command_examples": len(command_examples),
        },
    }


def _chunk_rows(rows: List[Dict[str, Any]], *, chunk_size: int) -> List[List[Dict[str, Any]]]:
    step = max(1, int(chunk_size))
    return [rows[idx: idx + step] for idx in range(0, len(rows), step)]


def _stage_resource(
    *,
    timeout_scale: float,
    num_predict: int,
    retries: int,
    synthesis_num_predict: int,
) -> Dict[str, Any]:
    return {
        "timeout_scale": round(max(0.35, float(timeout_scale)), 3),
        "num_predict": max(120, int(num_predict)),
        "retries": max(1, int(retries)),
        "synthesis_num_predict": max(120, int(synthesis_num_predict)),
    }


def _compact_runtime(runtime: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(runtime, dict):
        return {}
    keys = (
        "current_drive",
        "fear_level",
        "curiosity_charge",
        "survival_score",
        "energy",
        "hunger",
        "health",
        "age_ticks",
        "last_thought",
        "last_death_reason",
        "skills_version",
        "skills_steps",
        "skills_goals",
    )
    return {key: runtime.get(key) for key in keys if key in runtime}


def _compact_lineage(lineage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(lineage, dict):
        return {}
    keys = (
        "brain_id",
        "origin",
        "generation",
        "parent_ids",
        "created_at",
        "note",
    )
    return {key: lineage.get(key) for key in keys if key in lineage}


def build_brain_study_stages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages: List[Dict[str, Any]] = []
    agent = payload.get("agent", {}) if isinstance(payload.get("agent"), dict) else {}
    stages.append({
        "id": "identity",
        "layer": "identity",
        "title": "Identity And Baseline",
        "focus": "Выяви базовый стиль поведения, драйвы, ограничения и общую стратегию агента.",
        "resource": _stage_resource(timeout_scale=0.75, num_predict=240, retries=2, synthesis_num_predict=220),
        "data": {
            "agent": agent,
            "emotion_baseline": payload.get("emotion_baseline", {}),
            "behavior_rules": payload.get("behavior_rules", {}),
            "runtime": _compact_runtime(payload.get("runtime", {})),
            "lineage": _compact_lineage(payload.get("lineage", {})),
            "skills": payload.get("skills", {}),
            "totals": payload.get("totals", {}),
        },
    })

    beliefs = list(payload.get("beliefs", []) or [])
    for index, chunk in enumerate(_chunk_rows(beliefs, chunk_size=16), start=1):
        stages.append({
            "id": f"beliefs_{index}",
            "layer": "beliefs",
            "title": f"Beliefs Chunk {index}",
            "focus": "Изучи убеждения агента, выведи устойчивые правила, запреты и приоритеты.",
            "resource": _stage_resource(timeout_scale=0.6, num_predict=210, retries=2, synthesis_num_predict=220),
            "data": {"beliefs": chunk},
        })

    memories = list(payload.get("memory_tail", []) or [])
    for index, chunk in enumerate(_chunk_rows(memories, chunk_size=20), start=1):
        stages.append({
            "id": f"memory_{index}",
            "layer": "memory",
            "title": f"Memory Chunk {index}",
            "focus": "Изучи эпизодическую память, повторяющиеся события, полезные реакции и триггеры.",
            "resource": _stage_resource(timeout_scale=0.62, num_predict=220, retries=2, synthesis_num_predict=220),
            "data": {"memory_tail": chunk},
        })

    command_examples = list(((payload.get("ollama") or {}).get("command_examples", []) if isinstance(payload.get("ollama"), dict) else []) or [])
    if command_examples:
        for index, chunk in enumerate(_chunk_rows(command_examples, chunk_size=12), start=1):
            stages.append({
                "id": f"commands_{index}",
                "layer": "commands",
                "title": f"Command Examples {index}",
                "focus": "Изучи, как агент реагирует на команды, какие формулировки успешны и как лучше интерпретировать произвольные запросы оператора.",
                "resource": _stage_resource(timeout_scale=0.7, num_predict=230, retries=2, synthesis_num_predict=240),
                "data": {"command_examples": chunk},
            })

    dialogue_tail = list(((payload.get("ollama") or {}).get("dialogue_tail", []) if isinstance(payload.get("ollama"), dict) else []) or [])
    if dialogue_tail:
        for index, chunk in enumerate(_chunk_rows(dialogue_tail, chunk_size=16), start=1):
            stages.append({
                "id": f"dialogue_{index}",
                "layer": "dialogue",
                "title": f"Dialogue Chunk {index}",
                "focus": "Изучи диалоговый стиль и то, как агент формулирует свои мысли, реакции и внутреннее состояние.",
                "resource": _stage_resource(timeout_scale=0.72, num_predict=230, retries=2, synthesis_num_predict=240),
                "data": {"dialogue_tail": chunk},
            })

    stages.append({
        "id": "emotion_policy",
        "layer": "emotion",
        "title": "Emotion Policy",
        "focus": "Собери карту эмоционального управления: как успокаивать, фокусировать, вдохновлять и переключать агентa.",
        "resource": _stage_resource(timeout_scale=0.8, num_predict=250, retries=3, synthesis_num_predict=260),
        "data": {
            "emotion_baseline": payload.get("emotion_baseline", {}),
            "behavior_rules": payload.get("behavior_rules", {}),
            "trauma_map": payload.get("trauma_map", []),
            "snapshots": payload.get("snapshots", []),
        },
    })
    return stages


def summarize_brain_profile(profile: Any) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    identity = profile.get("identity", {}) if isinstance(profile.get("identity"), dict) else {}
    emotions = profile.get("emotional_profile", {}) if isinstance(profile.get("emotional_profile"), dict) else {}
    command_policy = profile.get("command_policy", {}) if isinstance(profile.get("command_policy"), dict) else {}
    brain_map = profile.get("brain_map", {}) if isinstance(profile.get("brain_map"), dict) else {}
    stage_summaries = list(profile.get("stage_summaries", []) or [])
    short_stages: List[Dict[str, Any]] = []
    for row in stage_summaries[-8:]:
        if not isinstance(row, dict):
            continue
        short_stages.append({
            "stage": str(row.get("stage") or "")[:48],
            "summary": _truncate(row.get("summary"), 140),
        })
    return {
        "trained_at": profile.get("trained_at"),
        "brain_signature": profile.get("brain_signature", {}),
        "identity": {
            "style": _truncate(identity.get("style"), 120),
            "summary": _truncate(identity.get("summary"), 180),
            "core_drives": [str(x) for x in list(identity.get("core_drives", []) or [])[:6]],
            "command_tone": _truncate(identity.get("command_tone"), 120),
        },
        "emotional_profile": {
            "baseline": emotions.get("baseline", {}),
            "calm_triggers": [str(x) for x in list(emotions.get("calm_triggers", []) or [])[:6]],
            "focus_triggers": [str(x) for x in list(emotions.get("focus_triggers", []) or [])[:6]],
            "energize_triggers": [str(x) for x in list(emotions.get("energize_triggers", []) or [])[:6]],
            "drive_map": emotions.get("drive_map", {}),
        },
        "command_policy": {
            "operator_rules": [str(x) for x in list(command_policy.get("operator_rules", []) or [])[:8]],
            "command_aliases": list(command_policy.get("command_aliases", []) or [])[:12],
            "allowed_action_families": [str(x) for x in list(command_policy.get("allowed_action_families", []) or [])[:10]],
        },
        "brain_map": {
            "layers_completed": [str(x) for x in list(brain_map.get("layers_completed", []) or [])[:8]],
            "identity_traits": [str(x) for x in list(brain_map.get("identity_traits", []) or [])[:8]],
            "command_rules": [str(x) for x in list(brain_map.get("command_rules", []) or [])[:8]],
            "emotion_rules": [str(x) for x in list(brain_map.get("emotion_rules", []) or [])[:8]],
            "allowed_action_families": [str(x) for x in list(brain_map.get("allowed_action_families", []) or [])[:8]],
        },
        "stage_summaries": short_stages,
    }
