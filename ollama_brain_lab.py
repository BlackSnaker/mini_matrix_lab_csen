from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import socket
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error

from brain_io import load_brain_with_meta
from mind_core import ConsciousnessBlock
from ollama_brain_profile import (
    build_brain_study_stages,
    build_full_brain_study_payload,
    summarize_brain_profile,
)
from ollama_coach import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_TIMEOUT,
    _extract_json_block,
    _post_json,
    pick_default_model,
)


LAB_DEFAULT_TIMEOUT = max(180.0, float(DEFAULT_OLLAMA_TIMEOUT))
STAGE_NUM_PREDICT = 320
FINAL_NUM_PREDICT = 420
_INTERRUPT_CONTEXT: Dict[str, Any] = {}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    if out != out or out in (float("inf"), float("-inf")):
        return fallback
    return out


def _truncate(text: Any, limit: int) -> str:
    line = str(text or "").strip()
    if len(line) <= int(limit):
        return line
    return line[: max(0, int(limit) - 3)].rstrip() + "..."


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


class BrainLabRequestError(RuntimeError):
    pass


class TrainingInterrupted(RuntimeError):
    def __init__(self, current_stage: Optional[Dict[str, Any]], reason: str = "Interrupted by operator") -> None:
        super().__init__(reason)
        self.current_stage = dict(current_stage or {}) if isinstance(current_stage, dict) else {}
        self.reason = str(reason or "Interrupted by operator")


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _println(text: str = "") -> None:
    print(text, flush=True)


def _color(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _rule() -> str:
    return "-" * 78


def _progress(done: int, total: int, *, width: int = 26) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    fill = int(round((done / float(total)) * width))
    return "[" + "#" * fill + "." * max(0, width - fill) + f"] {done:02d}/{total:02d}"


def _print_block(title: str, rows: List[str], *, color_code: str = "36") -> None:
    _println(_rule())
    _println(_color(title, color_code))
    _println(_rule())
    for row in rows:
        for line in textwrap.wrap(str(row), width=78) or [""]:
            _println(line)


def _load_source(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any], ConsciousnessBlock, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    if args.brain_file:
        path = Path(args.brain_file).expanduser().resolve()
        raw = _read_json(path)
        if isinstance(raw, dict) and raw.get("_schema_version") == 3:
            brain_dict = raw.get("brain", {}) or {}
            block = ConsciousnessBlock.from_dict(brain_dict)
            lineage = raw.get("_lineage", {}) if isinstance(raw.get("_lineage"), dict) else {}
            runtime = raw.get("_runtime", {}) if isinstance(raw.get("_runtime"), dict) else {}
            snapshots = [dict(row) for row in list(raw.get("_snapshots", []) or []) if isinstance(row, dict)]
            export_meta = raw.get("_export_meta", {}) if isinstance(raw.get("_export_meta"), dict) else {}
            return path, raw, block, lineage, runtime, snapshots, export_meta
        if isinstance(raw, dict) and isinstance(raw.get("brain"), dict):
            block = ConsciousnessBlock.from_dict(raw.get("brain", {}) or {})
            return path, raw, block, {}, {}, [], {}
        block = ConsciousnessBlock.from_dict(raw if isinstance(raw, dict) else {})
        return path, raw if isinstance(raw, dict) else {"brain": block.to_dict()}, block, {}, {}, [], {}

    if not args.brain_id:
        raise RuntimeError("Нужно указать --brain-id или --brain-file")
    meta = load_brain_with_meta(args.brain_id)
    if not isinstance(meta, dict):
        raise RuntimeError(f"Не удалось загрузить brain по id: {args.brain_id}")
    path = Path("brains") / f"{args.brain_id}.json"
    raw = _read_json(path) if path.exists() else {"brain": meta["brain"].to_dict()}
    return (
        path.resolve(),
        raw,
        meta["brain"],
        dict(meta.get("lineage", {}) or {}),
        dict(meta.get("runtime", {}) or {}),
        [dict(row) for row in list(meta.get("snapshots", []) or []) if isinstance(row, dict)],
        dict(meta.get("export_meta", {}) or {}),
    )


def _stage_prompt(
    stage: Dict[str, Any],
    payload: Dict[str, Any],
    stage_notes: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
) -> str:
    recent_notes = _compact_stage_digest(stage_notes, limit=4)
    schema = {
        "stage": str(stage.get("id") or ""),
        "summary": "короткое резюме этапа на русском, 1-2 предложения",
        "identity_hints": ["до 3 коротких характеристик"],
        "command_rules": ["до 3 правил интерпретации команд"],
        "emotion_rules": ["до 3 правил эмоционального управления"],
        "aliases": ["до 4 коротких пар: фраза -> действие"],
        "risks": ["до 2 коротких риска"],
        "next_focus": ["1-2 коротких фокуса на следующий этап"],
    }
    return (
        "Ты проводишь поэтапное изучение полного мозга цифрового агента.\n"
        "Твоя задача: на каждом этапе извлечь устойчивые свойства поведения, правила интерпретации команд и эмоциональные рычаги.\n"
        "Нельзя писать общие фразы без опоры на данные.\n"
        "Нужно отвечать только JSON без Markdown.\n"
        "Будь очень кратким: не раздувай списки, не повторяй обзор мозга, не пиши лишних пояснений.\n"
        "Изучи stage.data, учти brain-wide overview и краткие выводы предыдущих этапов.\n"
        "Сконцентрируйся на том, как потом интерпретировать любые команды оператора и как управлять эмоциональным состоянием агента.\n"
        "Учитывай уже собранную карту мозга и не дублируй выводы без нужды.\n"
        "Строгая схема ответа:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        "Глобальный обзор мозга:\n"
        f"{json.dumps({'agent': payload.get('agent', {}), 'emotion_baseline': payload.get('emotion_baseline', {}), 'behavior_rules': payload.get('behavior_rules', {}), 'totals': payload.get('totals', {}), 'skills': payload.get('skills', {})}, ensure_ascii=False, sort_keys=True)}\n"
        "Уже собранная карта мозга:\n"
        f"{json.dumps(_summarize_brain_map(brain_map), ensure_ascii=False, sort_keys=True)}\n"
        "Последние выводы предыдущих этапов:\n"
        f"{json.dumps(recent_notes, ensure_ascii=False, sort_keys=True)}\n"
        "Текущий этап:\n"
        f"{json.dumps(stage, ensure_ascii=False, sort_keys=True)}\n"
    )


def _final_prompt(
    payload: Dict[str, Any],
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
) -> str:
    schema = {
        "schema_version": 1,
        "identity": {
            "style": "основной поведенческий стиль",
            "summary": "краткий профиль мозга",
            "core_drives": ["до 4 drive"],
            "command_tone": "как лучше говорить с агентом",
            "nonnegotiables": ["до 4 ограничений"],
        },
        "command_policy": {
            "operator_rules": ["до 6 коротких правил"],
            "allowed_action_families": ["до 8 action family"],
            "command_aliases": [
                {"intent": "calm_down", "examples": ["успокойся", "расслабься"]},
                {"intent": "move_to_chair", "examples": ["подойди к креслу"]},
            ],
            "emotion_command_hints": ["до 5 коротких подсказок"],
        },
        "emotional_profile": {
            "baseline": payload.get("emotion_baseline", {}),
            "calm_triggers": ["до 5 триггеров"],
            "focus_triggers": ["до 5 триггеров"],
            "energize_triggers": ["до 5 триггеров"],
            "drive_map": {"calm": "idle", "curious": "explore"},
            "emotion_aliases": [
                {"intent": "focus", "examples": ["соберись", "сосредоточься"]},
            ],
        },
        "stage_summaries": [{"stage": "identity", "summary": "..."}],
    }
    return (
        "Ты завершил многоэтапное изучение полного мозга цифрового агента.\n"
        "Теперь собери единый brain_profile, который будет использоваться при дальнейшем управлении агентом через Ollama.\n"
        "Профиль должен помогать интерпретировать свободные команды оператора и менять эмоциональное состояние агента.\n"
        "Нельзя уходить в абстракции. Опирайся на layered brain map, layer reports и brain-wide overview.\n"
        "Ответь только JSON без Markdown.\n"
        "Будь компактным: короткие правила и краткие примеры, без длинных объяснений.\n"
        "Строгая схема ответа:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        "Global brain payload:\n"
        f"{json.dumps({'agent': payload.get('agent', {}), 'emotion_baseline': payload.get('emotion_baseline', {}), 'behavior_rules': payload.get('behavior_rules', {}), 'totals': payload.get('totals', {}), 'ollama': payload.get('ollama', {}), 'skills': payload.get('skills', {})}, ensure_ascii=False, sort_keys=True)}\n"
        "Layered brain map:\n"
        f"{json.dumps(_summarize_brain_map(brain_map), ensure_ascii=False, sort_keys=True)}\n"
        "Layer reports:\n"
        f"{json.dumps(_compact_stage_digest(layer_reports, limit=8), ensure_ascii=False, sort_keys=True)}\n"
        "Stage digest:\n"
        f"{json.dumps(_compact_stage_digest(stage_results, limit=12), ensure_ascii=False, sort_keys=True)}\n"
    )


def _unique_texts(values: List[Any], *, limit: int) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        text = str(item or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _merge_texts(left: List[Any], right: List[Any], *, limit: int) -> List[str]:
    return _unique_texts([*list(left or []), *list(right or [])], limit=limit)


def _normalize_drive_map(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in list(raw.items())[:12]:
        src = str(key or "").strip()[:40]
        dst = str(value or "").strip()[:40]
        if src and dst:
            out[src] = dst
    return out


def _alias_hints_to_policy_rows(hints: List[Any], *, prefix: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, hint in enumerate(_unique_texts(list(hints or []), limit=8), start=1):
        rows.append({
            "intent": f"{prefix}_{idx}",
            "examples": [str(hint)],
        })
    return rows


def _compact_stage_digest(rows: List[Dict[str, Any]], *, limit: int = 18) -> List[Dict[str, Any]]:
    digest: List[Dict[str, Any]] = []
    for row in list(rows or [])[-max(1, int(limit)):]:
        if not isinstance(row, dict):
            continue
        digest.append({
            "stage": str(row.get("stage") or ""),
            "title": str(row.get("title") or ""),
            "summary": _truncate(row.get("summary"), 160),
            "command_rules": [str(x) for x in list(row.get("command_rules", []) or [])[:3]],
            "emotion_rules": [str(x) for x in list(row.get("emotion_rules", []) or [])[:3]],
            "aliases": [str(x) for x in list(row.get("aliases", []) or [])[:4]],
            "source": str(row.get("source") or "model"),
        })
    return digest


def _empty_brain_map(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "baseline": dict(payload.get("emotion_baseline", {}) or {}),
        "layers_completed": [],
        "identity_traits": [],
        "style_hypotheses": [],
        "command_rules": [],
        "emotion_rules": [],
        "alias_hints": [],
        "allowed_action_families": [],
        "emotion_alias_hints": [],
        "drive_map_hints": {},
        "memory_patterns": [],
        "risk_flags": [],
        "layer_summaries": [],
    }


def _summarize_brain_map(brain_map: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(brain_map, dict):
        return {}
    return {
        "baseline": dict(brain_map.get("baseline", {}) or {}),
        "layers_completed": [str(x) for x in list(brain_map.get("layers_completed", []) or [])[:12]],
        "identity_traits": [str(x) for x in list(brain_map.get("identity_traits", []) or [])[:10]],
        "style_hypotheses": [str(x) for x in list(brain_map.get("style_hypotheses", []) or [])[:8]],
        "command_rules": [str(x) for x in list(brain_map.get("command_rules", []) or [])[:10]],
        "emotion_rules": [str(x) for x in list(brain_map.get("emotion_rules", []) or [])[:10]],
        "alias_hints": [str(x) for x in list(brain_map.get("alias_hints", []) or [])[:10]],
        "allowed_action_families": [str(x) for x in list(brain_map.get("allowed_action_families", []) or [])[:10]],
        "emotion_alias_hints": [str(x) for x in list(brain_map.get("emotion_alias_hints", []) or [])[:8]],
        "drive_map_hints": dict(brain_map.get("drive_map_hints", {}) or {}),
        "memory_patterns": [str(x) for x in list(brain_map.get("memory_patterns", []) or [])[:10]],
        "risk_flags": [str(x) for x in list(brain_map.get("risk_flags", []) or [])[:8]],
        "layer_summaries": [
            {
                "layer": str(row.get("layer") or ""),
                "summary": _truncate(row.get("summary"), 120),
            }
            for row in list(brain_map.get("layer_summaries", []) or [])[-8:]
            if isinstance(row, dict)
        ],
    }


def _infer_action_families(values: List[Any]) -> List[str]:
    family_map = {
        "move_to_landmark": ("подойди", "иди", "подойти", "move", "landmark", "кресл", "стол", "ламп"),
        "face_landmark": ("посмотри", "повернись к", "face_landmark", "look at"),
        "face_direction": ("налево", "направо", "вперед", "назад", "face_direction"),
        "wait": ("стой", "жди", "подожди", "wait"),
        "stop": ("останов", "stop", "замри"),
        "remember_note": ("запомни", "remember"),
        "tune_emotion": ("успокой", "сосредоточь", "смелее", "взбодр", "любопыт", "emotion", "calm", "focus"),
    }
    hits: List[str] = []
    text_blob = " | ".join(str(x or "").lower() for x in list(values or []))
    for family, markers in family_map.items():
        if any(marker in text_blob for marker in markers):
            hits.append(family)
    return hits


def _merge_brain_map_patch(
    brain_map: Dict[str, Any],
    patch: Dict[str, Any],
    *,
    mark_layer_complete: bool = False,
) -> Dict[str, Any]:
    updated = dict(brain_map or {})
    updated["updated_at"] = _utc_now()
    updated["identity_traits"] = _merge_texts(updated.get("identity_traits", []), patch.get("identity_traits", []), limit=18)
    updated["style_hypotheses"] = _merge_texts(updated.get("style_hypotheses", []), patch.get("style_hypotheses", []), limit=12)
    updated["command_rules"] = _merge_texts(updated.get("command_rules", []), patch.get("command_rules", []), limit=18)
    updated["emotion_rules"] = _merge_texts(updated.get("emotion_rules", []), patch.get("emotion_rules", []), limit=18)
    updated["alias_hints"] = _merge_texts(updated.get("alias_hints", []), patch.get("alias_hints", []), limit=18)
    updated["allowed_action_families"] = _merge_texts(updated.get("allowed_action_families", []), patch.get("allowed_action_families", []), limit=12)
    updated["emotion_alias_hints"] = _merge_texts(updated.get("emotion_alias_hints", []), patch.get("emotion_alias_hints", []), limit=12)
    updated["memory_patterns"] = _merge_texts(updated.get("memory_patterns", []), patch.get("memory_patterns", []), limit=14)
    updated["risk_flags"] = _merge_texts(updated.get("risk_flags", []), patch.get("risk_flags", []), limit=12)
    drive_map = dict(updated.get("drive_map_hints", {}) or {})
    drive_map.update(_normalize_drive_map(patch.get("drive_map_hints", {})))
    updated["drive_map_hints"] = drive_map
    layer = str(patch.get("layer") or "").strip()
    summary = _truncate(patch.get("summary"), 180)
    layer_summaries = [
        dict(row)
        for row in list(updated.get("layer_summaries", []) or [])
        if isinstance(row, dict)
    ]
    if layer and summary:
        layer_summaries = [row for row in layer_summaries if str(row.get("layer") or "") != layer]
        layer_summaries.append({"layer": layer, "summary": summary})
    updated["layer_summaries"] = layer_summaries[-12:]
    if mark_layer_complete and layer:
        updated["layers_completed"] = _merge_texts(updated.get("layers_completed", []), [layer], limit=12)
    return updated


def _stage_to_brain_map_patch(stage: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    aliases = [str(x) for x in list(result.get("aliases", []) or [])[:6]]
    emotion_alias_hints = [
        text for text in aliases
        if any(marker in text.lower() for marker in ("успокой", "сосредоточь", "взбодр", "смел", "любопыт", "calm", "focus"))
    ]
    memory_patterns: List[str] = []
    if str(stage.get("layer") or "") == "memory" and result.get("summary"):
        memory_patterns.append(str(result.get("summary") or ""))
    return {
        "layer": str(stage.get("layer") or stage.get("id") or ""),
        "summary": result.get("summary"),
        "identity_traits": [str(x) for x in list(result.get("identity_hints", []) or [])[:4]],
        "style_hypotheses": [str(result.get("summary") or "")] if result.get("summary") else [],
        "command_rules": [str(x) for x in list(result.get("command_rules", []) or [])[:4]],
        "emotion_rules": [str(x) for x in list(result.get("emotion_rules", []) or [])[:4]],
        "alias_hints": aliases,
        "allowed_action_families": _infer_action_families([*aliases, *list(result.get("command_rules", []) or [])]),
        "emotion_alias_hints": emotion_alias_hints,
        "drive_map_hints": {},
        "memory_patterns": memory_patterns,
        "risk_flags": [str(x) for x in list(result.get("risks", []) or [])[:3]],
    }


def _stage_local_policy_reason(
    stage: Dict[str, Any],
    args: argparse.Namespace,
    *,
    force_local_layers: Optional[set[str]] = None,
) -> str:
    layer = str(stage.get("layer") or stage.get("id") or "").strip()
    if layer and layer in set(force_local_layers or set()):
        return "layer_timeout_policy"
    if not getattr(args, "low_resource", False):
        return ""
    # Dense archival layers are processed locally per chunk in low-resource mode.
    # Ollama is still used later on compact layer-synthesis prompts.
    if layer in {"beliefs", "memory", "dialogue"}:
        return "low_resource_dense_layer"
    return ""


def _stage_resource_plan(stage: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    resource = stage.get("resource", {}) if isinstance(stage.get("resource"), dict) else {}
    timeout_scale = _safe_float(resource.get("timeout_scale", 1.0), 1.0)
    effective_timeout = max(30.0, float(args.timeout) * max(0.35, timeout_scale))
    stage_retries = int(resource.get("retries", int(args.retries)))
    predict_scale = max(0.35, _safe_float(getattr(args, "predict_scale", 1.0), 1.0))
    if getattr(args, "low_resource", False) and predict_scale >= 0.99:
        predict_scale = 0.65
    effective_retries = max(1, min(int(args.retries), stage_retries))
    if getattr(args, "low_resource", False):
        effective_retries = 1
    max_request_timeout = _safe_float(getattr(args, "max_request_timeout", 0.0), 0.0)
    if getattr(args, "low_resource", False) and max_request_timeout <= 0.0:
        max_request_timeout = 75.0
    if max_request_timeout > 0.0:
        effective_timeout = min(effective_timeout, max_request_timeout)
    return {
        "timeout": effective_timeout,
        "retries": effective_retries,
        "num_predict": max(120, int(int(resource.get("num_predict", STAGE_NUM_PREDICT)) * predict_scale)),
        "synthesis_num_predict": max(120, int(int(resource.get("synthesis_num_predict", STAGE_NUM_PREDICT)) * predict_scale)),
        "max_request_timeout": max_request_timeout,
    }


def _layer_resource_plan(layer_stages: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    timeouts: List[float] = []
    retries: List[int] = []
    predicts: List[int] = []
    caps: List[float] = []
    for stage in list(layer_stages or []):
        plan = _stage_resource_plan(stage, args)
        timeouts.append(_safe_float(plan.get("timeout", float(args.timeout)), float(args.timeout)))
        retries.append(int(plan.get("retries", int(args.retries))))
        predicts.append(int(plan.get("synthesis_num_predict", STAGE_NUM_PREDICT)))
        caps.append(_safe_float(plan.get("max_request_timeout", 0.0), 0.0))
    timeout = max([max(45.0, float(args.timeout) * 0.7)] + timeouts)
    cap = max([0.0] + caps)
    if cap > 0.0:
        timeout = min(timeout, cap)
    return {
        "timeout": timeout,
        "retries": max(1, min(int(args.retries), max(retries or [2]))),
        "num_predict": max([220] + predicts),
        "max_request_timeout": cap,
    }


def _layer_prompt(
    layer: str,
    payload: Dict[str, Any],
    layer_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
) -> str:
    schema = {
        "layer": layer,
        "summary": "краткий итог слоя, 1-2 предложения",
        "identity_traits": ["до 4 коротких черт"],
        "style_hypotheses": ["до 3 коротких выводов о стиле"],
        "command_rules": ["до 4 правил понимания команд"],
        "emotion_rules": ["до 4 правил управления эмоциями"],
        "alias_hints": ["до 6 коротких пар: фраза -> действие"],
        "allowed_action_families": ["до 6 action family"],
        "emotion_alias_hints": ["до 4 коротких пар: фраза -> emotion"],
        "drive_map_hints": {"calm": "idle"},
        "memory_patterns": ["до 4 паттернов"],
        "risk_flags": ["до 3 рисков"],
    }
    return (
        "Ты собираешь промежуточную карту мозга агента по одному слою данных.\n"
        "Нужно сжать результаты этапов в компактную brain-map patch.\n"
        "Ответь только JSON без Markdown. Будь кратким и практичным.\n"
        "Используй только факты из результатов слоя и уже собранной карты мозга.\n"
        "Строгая схема ответа:\n"
        f"{json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        "Глобальный обзор:\n"
        f"{json.dumps({'agent': payload.get('agent', {}), 'emotion_baseline': payload.get('emotion_baseline', {}), 'behavior_rules': payload.get('behavior_rules', {}), 'totals': payload.get('totals', {})}, ensure_ascii=False, sort_keys=True)}\n"
        "Уже собранная карта мозга:\n"
        f"{json.dumps(_summarize_brain_map(brain_map), ensure_ascii=False, sort_keys=True)}\n"
        "Результаты текущего слоя:\n"
        f"{json.dumps(_compact_stage_digest(layer_results, limit=10), ensure_ascii=False, sort_keys=True)}\n"
    )


def _sanitize_layer_patch(raw: Dict[str, Any], *, layer: str) -> Dict[str, Any]:
    return {
        "layer": layer,
        "summary": _truncate(raw.get("summary"), 220),
        "identity_traits": [str(x) for x in list(raw.get("identity_traits", []) or [])[:6]],
        "style_hypotheses": [str(x) for x in list(raw.get("style_hypotheses", []) or [])[:5]],
        "command_rules": [str(x) for x in list(raw.get("command_rules", []) or [])[:6]],
        "emotion_rules": [str(x) for x in list(raw.get("emotion_rules", []) or [])[:6]],
        "alias_hints": [str(x) for x in list(raw.get("alias_hints", []) or [])[:8]],
        "allowed_action_families": [str(x) for x in list(raw.get("allowed_action_families", []) or [])[:8]],
        "emotion_alias_hints": [str(x) for x in list(raw.get("emotion_alias_hints", []) or [])[:6]],
        "drive_map_hints": _normalize_drive_map(raw.get("drive_map_hints", {})),
        "memory_patterns": [str(x) for x in list(raw.get("memory_patterns", []) or [])[:6]],
        "risk_flags": [str(x) for x in list(raw.get("risk_flags", []) or [])[:6]],
    }


def _local_stage_fallback(
    stage: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    error_text: str,
) -> Dict[str, Any]:
    layer = str(stage.get("layer") or stage.get("id") or "")
    data = stage.get("data", {}) if isinstance(stage.get("data"), dict) else {}
    baseline = payload.get("emotion_baseline", {}) if isinstance(payload.get("emotion_baseline"), dict) else {}
    result: Dict[str, Any] = {
        "stage": str(stage.get("id") or ""),
        "title": str(stage.get("title") or ""),
        "summary": "",
        "identity_hints": [],
        "command_rules": [],
        "emotion_rules": [],
        "aliases": [],
        "risks": [f"local_fallback: {_truncate(error_text, 120)}"],
        "next_focus": [str(stage.get("focus") or "")],
        "source": "fallback",
    }
    if layer == "identity":
        tone = ", ".join(str(x) for x in list(baseline.get("tone", []) or [])[:3]) or "balanced"
        drive = str((data.get("agent", {}) or {}).get("current_drive") or baseline.get("drive") or "idle")
        result["summary"] = f"Локальный fallback: агент ведёт себя как {tone}, базовый drive={drive}."
        result["identity_hints"] = _unique_texts([tone, drive, str((data.get("skills", {}) or {}).get("gc_version") or "learned")], limit=3)
        result["command_rules"] = [
            "Команды лучше давать короткими конкретными фразами.",
            "Навигацию задавать через один ориентир или одно действие за раз.",
        ]
        result["emotion_rules"] = [
            "Эмоциональные команды должны явно менять страх, фокус или энергию.",
            "Смена drive работает лучше при коротких формулировках.",
        ]
    elif layer == "beliefs":
        beliefs = [dict(row) for row in list(data.get("beliefs", []) or []) if isinstance(row, dict)]
        top = [f"{row.get('if')} -> {row.get('then')}" for row in beliefs[:3]]
        result["summary"] = f"Локальный fallback: слой beliefs содержит {len(beliefs)} правил с устойчивыми if->then связями."
        result["identity_hints"] = ["rule_bound", "belief_driven"]
        result["command_rules"] = [
            "Команды не должны конфликтовать с уже выученными запретами.",
            "Лучше интерпретировать запрос через существующие правила if->then.",
        ]
        result["aliases"] = _unique_texts(top, limit=3)
        result["emotion_rules"] = ["Если агент спокоен, можно усиливать explore; при риске лучше снижать страх мягко."]
    elif layer == "memory":
        memories = [dict(row) for row in list(data.get("memory_tail", []) or []) if isinstance(row, dict)]
        counts = Counter(str(row.get("type") or "event") for row in memories)
        frequent = [f"{kind}:{count}" for kind, count in counts.most_common(3)]
        result["summary"] = f"Локальный fallback: память слоя показывает повторяющиеся события {', '.join(frequent) or 'без доминирующего паттерна'}."
        result["identity_hints"] = ["experience_shaped"]
        result["command_rules"] = ["Повторяющиеся успешные сценарии стоит использовать как шаблоны команд."]
        result["emotion_rules"] = ["Эмоциональные команды нужно привязывать к уже знакомым паттернам поведения."]
        result["aliases"] = frequent
    elif layer == "commands":
        examples = [dict(row) for row in list(data.get("command_examples", []) or []) if isinstance(row, dict)]
        example_pairs = [f"{row.get('command')} -> {row.get('parsed')}" for row in examples[:4]]
        families = Counter(str(row.get("parsed") or "").split("(", 1)[0] or "unknown" for row in examples)
        result["summary"] = f"Локальный fallback: слой команд содержит {len(examples)} примеров; ведущие action family: {', '.join(k for k, _v in families.most_common(3)) or 'unknown'}."
        result["command_rules"] = [
            "Лучше использовать глагол действия и один объект команды.",
            "Составные инструкции нужно разбивать на короткие шаги.",
        ]
        result["emotion_rules"] = ["Команды эмоций лучше отделять от навигации."]
        result["aliases"] = _unique_texts(example_pairs, limit=4)
    elif layer == "dialogue":
        dialogue = [dict(row) for row in list(data.get("dialogue_tail", []) or []) if isinstance(row, dict)]
        roles = Counter(str(row.get("role") or "unknown") for row in dialogue)
        result["summary"] = f"Локальный fallback: диалоговый слой содержит роли {', '.join(f'{k}:{v}' for k, v in roles.most_common(3)) or 'n/a'}."
        result["identity_hints"] = ["dialogue_conditioned"]
        result["command_rules"] = ["Операторские запросы стоит давать прямыми фразами без лишнего контекста."]
        result["emotion_rules"] = ["Эмоциональные команды должны звучать как короткая роль или настрой."]
    else:
        trauma = len(list(data.get("trauma_map", []) or []))
        result["summary"] = f"Локальный fallback: эмоциональный слой опирается на baseline fear={baseline.get('fear', 0.0)} и trauma_spots={trauma}."
        result["identity_hints"] = ["emotion_sensitive"]
        result["command_rules"] = ["Перед эмоциональной командой лучше не смешивать несколько целей."]
        result["emotion_rules"] = [
            "Успокаивать через calm/idle.",
            "Фокус усиливать через focus.",
            "Любопытство связывать с explore.",
        ]
        result["aliases"] = ["успокойся -> calm", "сосредоточься -> focus", "прояви любопытство -> explore"]
    result["summary"] = _truncate(result.get("summary"), 220)
    return result


def _build_local_layer_patch(
    layer: str,
    layer_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    *,
    error_text: str = "",
) -> Dict[str, Any]:
    command_rules: List[str] = []
    emotion_rules: List[str] = []
    identity_traits: List[str] = []
    alias_hints: List[str] = []
    risks: List[str] = []
    for row in list(layer_results or []):
        if not isinstance(row, dict):
            continue
        command_rules.extend(list(row.get("command_rules", []) or [])[:3])
        emotion_rules.extend(list(row.get("emotion_rules", []) or [])[:3])
        identity_traits.extend(list(row.get("identity_hints", []) or [])[:3])
        alias_hints.extend(list(row.get("aliases", []) or [])[:3])
        risks.extend(list(row.get("risks", []) or [])[:2])
    summary = f"Локальная свёртка слоя {layer}: {len(layer_results)} stage, {len(command_rules)} command hints, {len(emotion_rules)} emotion hints."
    if error_text:
        risks.append(f"layer_fallback: {_truncate(error_text, 120)}")
    return {
        "layer": layer,
        "summary": _truncate(summary, 220),
        "identity_traits": _unique_texts(identity_traits, limit=6),
        "style_hypotheses": _unique_texts([summary, *list(brain_map.get("style_hypotheses", []) or [])[:1]], limit=3),
        "command_rules": _unique_texts(command_rules, limit=6),
        "emotion_rules": _unique_texts(emotion_rules, limit=6),
        "alias_hints": _unique_texts(alias_hints, limit=8),
        "allowed_action_families": _infer_action_families([*alias_hints, *command_rules]),
        "emotion_alias_hints": [
            text for text in _unique_texts(alias_hints, limit=8)
            if any(marker in text.lower() for marker in ("успокой", "focus", "любопыт", "взбодр", "смел"))
        ][:6],
        "drive_map_hints": dict(brain_map.get("drive_map_hints", {}) or {}),
        "memory_patterns": _unique_texts(
            [str(row.get("summary") or "") for row in list(layer_results or []) if isinstance(row, dict) and row.get("summary")],
            limit=4,
        ),
        "risk_flags": _unique_texts(risks, limit=6),
    }


def _local_profile_from_brain_map(
    payload: Dict[str, Any],
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary_map = _summarize_brain_map(brain_map)
    identity_traits = [str(x) for x in list(summary_map.get("identity_traits", []) or [])[:4]]
    style_hypotheses = [str(x) for x in list(summary_map.get("style_hypotheses", []) or [])[:3]]
    operator_rules = [str(x) for x in list(summary_map.get("command_rules", []) or [])[:8]]
    emotion_rules = [str(x) for x in list(summary_map.get("emotion_rules", []) or [])[:8]]
    alias_hints = [str(x) for x in list(summary_map.get("alias_hints", []) or [])[:8]]
    emotion_alias_hints = [str(x) for x in list(summary_map.get("emotion_alias_hints", []) or [])[:6]]
    baseline = dict(summary_map.get("baseline", {}) or payload.get("emotion_baseline", {}) or {})
    style = identity_traits[0] if identity_traits else "layered_adaptive"
    summary = style_hypotheses[0] if style_hypotheses else "Профиль собран локально из layered brain map."
    command_aliases = [
        {"intent": f"alias_{idx}", "examples": [hint]}
        for idx, hint in enumerate(alias_hints[:6], start=1)
    ]
    emotion_aliases = [
        {"intent": f"emotion_{idx}", "examples": [hint]}
        for idx, hint in enumerate(emotion_alias_hints[:6], start=1)
    ]
    return {
        "schema_version": 1,
        "trained_at": _utc_now(),
        "source_agent_id": str((payload.get("agent", {}) or {}).get("id") or "agent"),
        "brain_signature": {
            "beliefs": int(((payload.get("totals", {}) or {}).get("beliefs") or 0)),
            "memories": int(((payload.get("totals", {}) or {}).get("memories") or 0)),
            "command_examples": int(((payload.get("totals", {}) or {}).get("command_examples") or 0)),
            "dialogue_entries": int(((payload.get("totals", {}) or {}).get("dialogue_entries") or 0)),
        },
        "identity": {
            "style": style,
            "summary": _truncate(summary, 240),
            "core_drives": _unique_texts(
                [
                    str((payload.get("emotion_baseline", {}) or {}).get("drive") or ""),
                    *list(summary_map.get("drive_map_hints", {}).values() if isinstance(summary_map.get("drive_map_hints"), dict) else []),
                ],
                limit=4,
            ),
            "command_tone": "Короткие прямые команды с одним действием.",
            "nonnegotiables": [str(x) for x in list(summary_map.get("risk_flags", []) or [])[:6]],
        },
        "command_policy": {
            "operator_rules": operator_rules or ["Команды лучше давать короткими шагами."],
            "allowed_action_families": [str(x) for x in list(summary_map.get("allowed_action_families", []) or [])[:10]],
            "command_aliases": command_aliases,
            "emotion_command_hints": emotion_rules[:6],
        },
        "emotional_profile": {
            "baseline": baseline,
            "calm_triggers": [rule for rule in emotion_rules if "спокой" in rule.lower() or "calm" in rule.lower()][:5],
            "focus_triggers": [rule for rule in emotion_rules if "фокус" in rule.lower() or "focus" in rule.lower()][:5],
            "energize_triggers": [rule for rule in emotion_rules if "энерг" in rule.lower() or "взбодр" in rule.lower()][:5],
            "drive_map": dict(summary_map.get("drive_map_hints", {}) or {}),
            "emotion_aliases": emotion_aliases,
        },
        "stage_summaries": [
            {"stage": str(row.get("stage") or ""), "summary": _truncate(row.get("summary"), 180)}
            for row in list(stage_results or [])[-24:]
            if isinstance(row, dict)
        ],
        "brain_map": _summarize_brain_map(brain_map),
        "layer_reports": [
            {
                "layer": str(row.get("layer") or ""),
                "summary": _truncate(row.get("summary"), 180),
                "source": str(row.get("source") or "local"),
            }
            for row in list(layer_reports or [])[-12:]
            if isinstance(row, dict)
        ],
        "profile_source": "local_layered_fallback",
    }


def _generate_json(
    *,
    host: str,
    model: str,
    prompt: str,
    timeout: float,
    keep_alive: str,
    label: str,
    retries: int,
    num_predict: int,
    max_request_timeout: float = 0.0,
) -> Dict[str, Any]:
    last_error: Optional[BaseException] = None
    attempts = max(1, int(retries))
    base_timeout = max(30.0, float(timeout))
    timeout_cap = max(0.0, float(max_request_timeout))
    for attempt in range(1, attempts + 1):
        effective_timeout = base_timeout * (1.0 + (attempt - 1) * 0.75)
        if timeout_cap > 0.0:
            effective_timeout = min(effective_timeout, timeout_cap)
        try:
            raw = _post_json(
                f"{host.rstrip('/')}/api/generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "keep_alive": keep_alive,
                    "options": {
                        "temperature": 0.15,
                        "top_p": 0.9,
                        "num_predict": max(120, int(num_predict)),
                    },
                },
                timeout=effective_timeout,
            )
            text = str(raw.get("response") or "").strip()
            data = _extract_json_block(text)
            if not data:
                raise BrainLabRequestError(f"{label}: Ollama returned empty or non-JSON training output")
            return data
        except (TimeoutError, socket.timeout) as exc:
            last_error = exc
        except urllib.error.URLError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            next_timeout = base_timeout * (1.0 + attempt * 0.75)
            if timeout_cap > 0.0:
                next_timeout = min(next_timeout, timeout_cap)
            _println(_color(
                f"Retry {attempt}/{attempts - 1} for {label}: {type(last_error).__name__}. "
                f"Next timeout {next_timeout:0.0f}s",
                "31",
            ))
            time.sleep(min(3.0, 0.6 * attempt))
    err_text = str(last_error or "unknown error").strip() or "unknown error"
    raise BrainLabRequestError(
        f"{label}: запрос к Ollama не завершился успешно после {attempts} попыток. "
        f"Последняя ошибка: {err_text}. Попробуй увеличить --timeout."
    )


def _sanitize_stage_result(stage: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    summary = _truncate(raw.get("summary"), 260)
    identity_hints = [str(x) for x in list(raw.get("identity_hints", []) or [])[:8]]
    command_rules = [str(x) for x in list(raw.get("command_rules", []) or [])[:8]]
    emotion_rules = [str(x) for x in list(raw.get("emotion_rules", []) or [])[:8]]
    aliases = [str(x) for x in list(raw.get("aliases", []) or [])[:10]]
    risks = [str(x) for x in list(raw.get("risks", []) or [])[:8]]
    next_focus = [str(x) for x in list(raw.get("next_focus", []) or [])[:6]]
    return {
        "stage": str(stage.get("id") or ""),
        "title": str(stage.get("title") or ""),
        "summary": summary,
        "identity_hints": identity_hints,
        "command_rules": command_rules,
        "emotion_rules": emotion_rules,
        "aliases": aliases,
        "risks": risks,
        "next_focus": next_focus,
    }


def _normalize_profile(
    raw: Dict[str, Any],
    payload: Dict[str, Any],
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    identity = raw.get("identity", {}) if isinstance(raw.get("identity"), dict) else {}
    command_policy = raw.get("command_policy", {}) if isinstance(raw.get("command_policy"), dict) else {}
    emotional_profile = raw.get("emotional_profile", {}) if isinstance(raw.get("emotional_profile"), dict) else {}
    brain_map_summary = _summarize_brain_map(brain_map)
    command_alias_rows = [dict(row) for row in list(command_policy.get("command_aliases", []) or [])[:18] if isinstance(row, dict)]
    if not command_alias_rows:
        command_alias_rows = _alias_hints_to_policy_rows(brain_map_summary.get("alias_hints", []), prefix="command")
    emotion_alias_rows = [dict(row) for row in list(emotional_profile.get("emotion_aliases", []) or [])[:12] if isinstance(row, dict)]
    if not emotion_alias_rows:
        emotion_alias_rows = _alias_hints_to_policy_rows(brain_map_summary.get("emotion_alias_hints", []), prefix="emotion")
    profile = {
        "schema_version": 1,
        "trained_at": _utc_now(),
        "source_agent_id": str((payload.get("agent", {}) or {}).get("id") or "agent"),
        "brain_signature": {
            "beliefs": int(((payload.get("totals", {}) or {}).get("beliefs") or 0)),
            "memories": int(((payload.get("totals", {}) or {}).get("memories") or 0)),
            "command_examples": int(((payload.get("totals", {}) or {}).get("command_examples") or 0)),
            "dialogue_entries": int(((payload.get("totals", {}) or {}).get("dialogue_entries") or 0)),
        },
        "identity": {
            "style": _truncate(identity.get("style"), 160),
            "summary": _truncate(identity.get("summary"), 240),
            "core_drives": [str(x) for x in list(identity.get("core_drives", []) or [])[:8]],
            "command_tone": _truncate(identity.get("command_tone"), 160),
            "nonnegotiables": [str(x) for x in list(identity.get("nonnegotiables", []) or [])[:8]],
        },
        "command_policy": {
            "operator_rules": _merge_texts(
                [str(x) for x in list(command_policy.get("operator_rules", []) or [])[:12]],
                brain_map_summary.get("command_rules", []),
                limit=12,
            ),
            "allowed_action_families": _merge_texts(
                [str(x) for x in list(command_policy.get("allowed_action_families", []) or [])[:12]],
                brain_map_summary.get("allowed_action_families", []),
                limit=12,
            ),
            "command_aliases": command_alias_rows,
            "emotion_command_hints": _merge_texts(
                [str(x) for x in list(command_policy.get("emotion_command_hints", []) or [])[:10]],
                brain_map_summary.get("emotion_rules", []),
                limit=10,
            ),
        },
        "emotional_profile": {
            "baseline": emotional_profile.get("baseline", payload.get("emotion_baseline", {})),
            "calm_triggers": _merge_texts([str(x) for x in list(emotional_profile.get("calm_triggers", []) or [])[:8]], brain_map_summary.get("emotion_alias_hints", []), limit=8),
            "focus_triggers": [str(x) for x in list(emotional_profile.get("focus_triggers", []) or [])[:8]],
            "energize_triggers": [str(x) for x in list(emotional_profile.get("energize_triggers", []) or [])[:8]],
            "drive_map": dict(emotional_profile.get("drive_map", {}) or brain_map_summary.get("drive_map_hints", {}) or {}),
            "emotion_aliases": emotion_alias_rows,
        },
        "stage_summaries": [
            {"stage": str(row.get("stage") or ""), "summary": _truncate(row.get("summary"), 180)}
            for row in stage_results[-24:]
        ],
        "brain_map": brain_map_summary,
        "layer_reports": [
            {
                "layer": str(row.get("layer") or ""),
                "summary": _truncate(row.get("summary"), 180),
                "source": str(row.get("source") or "model"),
            }
            for row in list(layer_reports or [])[-12:]
            if isinstance(row, dict)
        ],
        "profile_source": str(raw.get("profile_source") or "ollama_layered_synthesis"),
    }
    return profile


def _write_back_profile(
    source_path: Path,
    raw_payload: Dict[str, Any],
    profile: Dict[str, Any],
    stage_results: List[Dict[str, Any]],
) -> None:
    target = dict(raw_payload or {})
    if target.get("_schema_version") == 3 and isinstance(target.get("brain"), dict):
        brain_dict = dict(target.get("brain", {}) or {})
        target["brain"] = brain_dict
    elif isinstance(target.get("brain"), dict):
        brain_dict = dict(target.get("brain", {}) or {})
        target["brain"] = brain_dict
    else:
        brain_dict = target

    ollama = dict(brain_dict.get("ollama", {}) or {})
    ollama["brain_profile"] = dict(profile)
    ollama["brain_profile_updated_at"] = profile.get("trained_at")
    history: List[Dict[str, Any]] = []
    for idx, row in enumerate(stage_results[-48:], start=1):
        if not isinstance(row, dict):
            continue
        history.append({
            "seq": idx,
            "stage": str(row.get("stage") or ""),
            "summary": _truncate(row.get("summary"), 180),
            "tick": int(_safe_float(row.get("tick", idx), idx)),
            "trained_at": profile.get("trained_at"),
        })
    ollama["brain_training_tail"] = history
    ollama["brain_training_seq"] = len(history)
    brain_dict["ollama"] = ollama
    _write_json(source_path, target)


def _report_path(args: argparse.Namespace, agent_id: str) -> Path:
    if args.report_out:
        return Path(args.report_out).expanduser().resolve()
    return Path("ollama_brain_reports").resolve() / f"{agent_id}.brain_profile.json"


def _progress_path(args: argparse.Namespace, agent_id: str) -> Path:
    if args.progress_out:
        return Path(args.progress_out).expanduser().resolve()
    return Path("ollama_brain_reports").resolve() / f"{agent_id}.brain_progress.json"


def _checkpoint_path(args: argparse.Namespace, agent_id: str) -> Path:
    if args.checkpoint_out:
        return Path(args.checkpoint_out).expanduser().resolve()
    return Path("ollama_brain_reports").resolve() / f"{agent_id}.brain_checkpoint.json"


def _write_checkpoint(
    path: Path,
    *,
    status: str,
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_stages: int,
    completed_stages: int,
    current_stage: Optional[Dict[str, Any]],
    stage_results: List[Dict[str, Any]],
    brain_map: Optional[Dict[str, Any]],
    layer_reports: Optional[List[Dict[str, Any]]],
    started_at: str,
    error: Optional[str] = None,
    final_report: Optional[Path] = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpoint_version": 1,
        "status": str(status),
        "updated_at": _utc_now(),
        "started_at": str(started_at),
        "agent_id": str(agent_id),
        "source_path": str(source_path),
        "model": str(model),
        "host": str(host),
        "total_stages": int(total_stages),
        "completed_stages": int(completed_stages),
        "current_stage": dict(current_stage or {}) if isinstance(current_stage, dict) else None,
        "stage_results": [dict(row) for row in list(stage_results or []) if isinstance(row, dict)],
        "brain_map": dict(brain_map or {}) if isinstance(brain_map, dict) else {},
        "layer_reports": [dict(row) for row in list(layer_reports or []) if isinstance(row, dict)],
    }
    if error:
        payload["error"] = str(error)
    if final_report is not None:
        payload["final_report"] = str(final_report)
    _write_json(path, payload)


def _layer_order(stages: List[Dict[str, Any]]) -> List[str]:
    return _unique_texts([str(stage.get("layer") or "") for stage in list(stages or [])], limit=64)


def _layer_stage_defs_map(stages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for stage in list(stages or []):
        layer = str(stage.get("layer") or stage.get("id") or "").strip()
        if not layer:
            continue
        out.setdefault(layer, []).append(stage)
    return out


def _validate_stage_results(stage_results: List[Dict[str, Any]], stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for index, row in enumerate(list(stage_results or [])):
        if index >= len(stages) or not isinstance(row, dict):
            break
        expected = str(stages[index].get("id") or "")
        got = str(row.get("stage") or "")
        if expected != got:
            break
        item = dict(row)
        item["layer"] = str(item.get("layer") or stages[index].get("layer") or "")
        item["source"] = str(item.get("source") or "model")
        valid.append(item)
    return valid


def _eligible_layers_for_stage_count(stage_count: int, layer_stage_defs: Dict[str, List[Dict[str, Any]]], layer_names: List[str]) -> List[str]:
    eligible: List[str] = []
    completed = max(0, int(stage_count))
    consumed = 0
    for layer in list(layer_names or []):
        need = len(layer_stage_defs.get(layer, []) or [])
        consumed += need
        if need and completed >= consumed:
            eligible.append(layer)
    return eligible


def _validate_layer_reports(
    layer_reports: List[Dict[str, Any]],
    *,
    eligible_layers: List[str],
) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for index, row in enumerate(list(layer_reports or [])):
        if index >= len(eligible_layers) or not isinstance(row, dict):
            break
        expected = str(eligible_layers[index] or "")
        got = str(row.get("layer") or "")
        if expected != got:
            break
        item = dict(row)
        item["source"] = str(item.get("source") or "model")
        valid.append(item)
    return valid


def _rebuild_brain_map_from_history(
    payload: Dict[str, Any],
    stages: List[Dict[str, Any]],
    stage_results: List[Dict[str, Any]],
    layer_reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    brain_map = _empty_brain_map(payload)
    stage_by_id = {
        str(stage.get("id") or ""): dict(stage)
        for stage in list(stages or [])
        if isinstance(stage, dict)
    }
    for row in list(stage_results or []):
        if not isinstance(row, dict):
            continue
        stage_id = str(row.get("stage") or "")
        stage = stage_by_id.get(stage_id)
        if stage:
            brain_map = _merge_brain_map_patch(brain_map, _stage_to_brain_map_patch(stage, row))
    for row in list(layer_reports or []):
        if not isinstance(row, dict):
            continue
        aliases = [str(x) for x in list(row.get("aliases", []) or [])[:6]]
        patch = {
            "layer": str(row.get("layer") or ""),
            "summary": row.get("summary"),
            "identity_traits": [],
            "style_hypotheses": [str(row.get("summary") or "")] if row.get("summary") else [],
            "command_rules": [str(x) for x in list(row.get("command_rules", []) or [])[:6]],
            "emotion_rules": [str(x) for x in list(row.get("emotion_rules", []) or [])[:6]],
            "alias_hints": aliases,
            "allowed_action_families": _infer_action_families([*aliases, *list(row.get("command_rules", []) or [])]),
            "emotion_alias_hints": [
                text for text in aliases
                if any(marker in text.lower() for marker in ("успокой", "focus", "любопыт", "взбодр", "смел", "calm"))
            ],
            "drive_map_hints": {},
            "memory_patterns": [str(row.get("summary") or "")] if str(row.get("layer") or "") == "memory" and row.get("summary") else [],
            "risk_flags": [],
        }
        brain_map = _merge_brain_map_patch(brain_map, patch, mark_layer_complete=True)
    return brain_map


def _load_resume_state(
    checkpoint_path: Path,
    *,
    payload: Dict[str, Any],
    stages: List[Dict[str, Any]],
    source_path: Path,
    agent_id: str,
) -> Optional[Dict[str, Any]]:
    if not checkpoint_path.exists():
        return None
    try:
        raw = _read_json(checkpoint_path)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    if str(raw.get("agent_id") or "") != str(agent_id):
        return None
    if str(raw.get("source_path") or "") != str(source_path):
        return None
    layer_names = _layer_order(stages)
    layer_stage_defs = _layer_stage_defs_map(stages)
    stage_results = _validate_stage_results(list(raw.get("stage_results", []) or []), stages)
    eligible_layers = _eligible_layers_for_stage_count(len(stage_results), layer_stage_defs, layer_names)
    layer_reports = _validate_layer_reports(list(raw.get("layer_reports", []) or []), eligible_layers=eligible_layers)
    brain_map = _rebuild_brain_map_from_history(payload, stages, stage_results, layer_reports)
    status = str(raw.get("status") or "running")
    completed_units = len(stage_results) + len(layer_reports)
    return {
        "status": status,
        "started_at": str(raw.get("started_at") or _utc_now()),
        "stage_results": stage_results,
        "layer_reports": layer_reports,
        "brain_map": brain_map,
        "completed_units": completed_units,
        "final_report": str(raw.get("final_report") or ""),
        "current_stage": dict(raw.get("current_stage", {}) or {}) if isinstance(raw.get("current_stage"), dict) else None,
    }


def _write_progress(
    path: Path,
    *,
    status: str,
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_stages: int,
    completed_stages: int,
    current_stage: Optional[Dict[str, Any]],
    stage_results: List[Dict[str, Any]],
    brain_map: Optional[Dict[str, Any]],
    layer_reports: Optional[List[Dict[str, Any]]],
    started_at: str,
    error: Optional[str] = None,
    final_report: Optional[Path] = None,
) -> None:
    payload: Dict[str, Any] = {
        "status": str(status),
        "updated_at": _utc_now(),
        "started_at": str(started_at),
        "agent_id": str(agent_id),
        "source_path": str(source_path),
        "model": str(model),
        "host": str(host),
        "total_stages": int(total_stages),
        "completed_stages": int(completed_stages),
        "current_stage": dict(current_stage or {}) if isinstance(current_stage, dict) else None,
        "stage_results": [
            {
                "stage": str(row.get("stage") or ""),
                "title": str(row.get("title") or ""),
                "summary": _truncate(row.get("summary"), 160),
                "tick": int(_safe_float(row.get("tick", 0), 0.0)),
            }
            for row in stage_results[-24:]
            if isinstance(row, dict)
        ],
        "brain_map_summary": _summarize_brain_map(brain_map or {}),
        "layer_reports": [
            {
                "layer": str(row.get("layer") or ""),
                "summary": _truncate(row.get("summary"), 160),
                "source": str(row.get("source") or "model"),
            }
            for row in list(layer_reports or [])[-12:]
            if isinstance(row, dict)
        ],
    }
    if error:
        payload["error"] = str(error)
    if final_report is not None:
        payload["final_report"] = str(final_report)
    _write_json(path, payload)


def _apply_cooldown(seconds: float, *, reason: str) -> None:
    delay = max(0.0, float(seconds))
    if delay <= 0.0:
        return
    _println(_color(f"Cooldown: {delay:0.1f}s after {reason}", "90"))
    time.sleep(delay)


def _maybe_pause_for_budget(
    *,
    args: argparse.Namespace,
    run_start_units: int,
    completed_units: int,
    total_units: int,
    current_stage: Dict[str, Any],
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    progress_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
    started_at: str,
) -> bool:
    max_units = max(0, int(getattr(args, "max_units_per_run", 0) or 0))
    if max_units <= 0:
        return False
    done_this_run = max(0, int(completed_units) - int(run_start_units))
    if done_this_run < max_units:
        return False
    status = "paused_resource"
    _write_progress(
        progress_path,
        status=status,
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status=status,
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _print_block(
        "TRAINING PAUSED",
        [
            f"Agent:   {agent_id}",
            f"Reason:  resource budget reached ({done_this_run}/{max_units} units this run)",
            f"Stage:   {current_stage.get('title') or current_stage.get('stage') or 'n/a'}",
            f"Checkpoint: {checkpoint_path}",
            "Запусти ту же команду ещё раз, чтобы продолжить с checkpoint.",
        ],
        color_code="33",
    )
    return True


def _interrupt_training(
    *,
    reason: str,
    current_stage: Optional[Dict[str, Any]],
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_units: int,
    completed_units: int,
    progress_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
    started_at: str,
) -> int:
    _write_progress(
        progress_path,
        status="interrupted",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        error=reason,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status="interrupted",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        error=reason,
        final_report=report_path,
    )
    _print_block(
        "TRAINING INTERRUPTED",
        [
            f"Agent:   {agent_id}",
            f"Reason:  {reason}",
            f"Stage:   {(current_stage or {}).get('title') or (current_stage or {}).get('stage') or 'n/a'}",
            f"Checkpoint: {checkpoint_path}",
            "Запусти ту же команду ещё раз, чтобы продолжить с сохранённого места.",
        ],
        color_code="33",
    )
    return 130


def _set_interrupt_context(
    *,
    current_stage: Optional[Dict[str, Any]],
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_units: int,
    completed_units: int,
    progress_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    stage_results: List[Dict[str, Any]],
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
    started_at: str,
) -> None:
    _INTERRUPT_CONTEXT.clear()
    _INTERRUPT_CONTEXT.update({
        "current_stage": dict(current_stage or {}) if isinstance(current_stage, dict) else None,
        "agent_id": str(agent_id),
        "source_path": Path(source_path),
        "model": str(model),
        "host": str(host),
        "total_units": int(total_units),
        "completed_units": int(completed_units),
        "progress_path": Path(progress_path),
        "checkpoint_path": Path(checkpoint_path),
        "report_path": Path(report_path),
        "stage_results": stage_results,
        "brain_map": brain_map,
        "layer_reports": layer_reports,
        "started_at": str(started_at),
    })


def _interrupt_from_context(reason: str) -> int:
    if not _INTERRUPT_CONTEXT:
        _print_block(
            "TRAINING INTERRUPTED",
            [
                f"Reason:  {reason}",
                "Checkpoint context was not initialized.",
            ],
            color_code="33",
        )
        return 130
    return _interrupt_training(
        reason=reason,
        current_stage=_INTERRUPT_CONTEXT.get("current_stage"),
        agent_id=str(_INTERRUPT_CONTEXT.get("agent_id") or "agent"),
        source_path=Path(_INTERRUPT_CONTEXT.get("source_path") or "."),
        model=str(_INTERRUPT_CONTEXT.get("model") or ""),
        host=str(_INTERRUPT_CONTEXT.get("host") or ""),
        total_units=int(_INTERRUPT_CONTEXT.get("total_units") or 0),
        completed_units=int(_INTERRUPT_CONTEXT.get("completed_units") or 0),
        progress_path=Path(_INTERRUPT_CONTEXT.get("progress_path") or "brain_progress.json"),
        checkpoint_path=Path(_INTERRUPT_CONTEXT.get("checkpoint_path") or "brain_checkpoint.json"),
        report_path=Path(_INTERRUPT_CONTEXT.get("report_path") or "brain_profile.json"),
        stage_results=list(_INTERRUPT_CONTEXT.get("stage_results") or []),
        brain_map=dict(_INTERRUPT_CONTEXT.get("brain_map") or {}),
        layer_reports=list(_INTERRUPT_CONTEXT.get("layer_reports") or []),
        started_at=str(_INTERRUPT_CONTEXT.get("started_at") or _utc_now()),
    )


def _finalize_layer(
    *,
    layer: str,
    payload: Dict[str, Any],
    args: argparse.Namespace,
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_units: int,
    completed_units: int,
    started_at: str,
    progress_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
    layer_stage_results: Dict[str, List[Dict[str, Any]]],
    layer_stage_defs: Dict[str, List[Dict[str, Any]]],
    stage_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    if not layer or any(str(row.get("layer") or "") == layer for row in list(layer_reports or []) if isinstance(row, dict)):
        return brain_map, layer_reports, completed_units

    layer_plan = _layer_resource_plan(layer_stage_defs.get(layer, []), args)
    current_stage = {
        "index": completed_units + 1,
        "title": f"Layer Map: {layer}",
        "stage": f"layer_map:{layer}",
        "layer": layer,
        "focus": f"Сборка промежуточной карты слоя {layer}.",
        "resource_plan": {
            "timeout": round(_safe_float(layer_plan.get("timeout", 0.0), 0.0), 1),
            "retries": int(layer_plan.get("retries", int(args.retries))),
            "num_predict": int(layer_plan.get("num_predict", 220)),
        },
    }
    _write_progress(
        progress_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _println(_rule())
    _println(_color(f"{_progress(completed_units, total_units)}  Layer Map: {layer}", "35"))
    try:
        layer_raw = _generate_json(
            host=host,
            model=model,
            prompt=_layer_prompt(layer, payload, layer_stage_results.get(layer, []), brain_map),
            timeout=float(layer_plan.get("timeout", float(args.timeout))),
            keep_alive=str(args.keep_alive),
            label=f"layer:{layer}",
            retries=int(layer_plan.get("retries", int(args.retries))),
            num_predict=int(layer_plan.get("num_predict", 220)),
            max_request_timeout=float(layer_plan.get("max_request_timeout", 0.0)),
        )
        layer_patch = _sanitize_layer_patch(layer_raw, layer=layer)
        layer_patch["source"] = "model"
    except KeyboardInterrupt as exc:
        raise TrainingInterrupted(current_stage, "Interrupted by operator (Ctrl+C)") from exc
    except BrainLabRequestError as exc:
        _println(_color(f"Layer fallback: {exc}", "31"))
        layer_patch = _build_local_layer_patch(layer, layer_stage_results.get(layer, []), brain_map, error_text=str(exc))
        layer_patch["source"] = "fallback"

    brain_map = _merge_brain_map_patch(brain_map, layer_patch, mark_layer_complete=True)
    layer_report = {
        "stage": f"layer_map:{layer}",
        "title": f"Layer Map: {layer}",
        "layer": layer,
        "summary": _truncate(layer_patch.get("summary"), 200),
        "command_rules": [str(x) for x in list(layer_patch.get("command_rules", []) or [])[:4]],
        "emotion_rules": [str(x) for x in list(layer_patch.get("emotion_rules", []) or [])[:4]],
        "aliases": [str(x) for x in list(layer_patch.get("alias_hints", []) or [])[:4]],
        "source": str(layer_patch.get("source") or "model"),
    }
    updated_layer_reports = list(layer_reports or [])
    updated_layer_reports.append(layer_report)
    completed_units += 1

    _println(f"{_color('Layer Summary:', '36')} {layer_patch.get('summary', '')}")
    if layer_patch.get("command_rules"):
        _println(f"{_color('Layer Command:', '32')} {'; '.join(layer_patch['command_rules'][:3])}")
    if layer_patch.get("emotion_rules"):
        _println(f"{_color('Layer Emotion:', '34')} {'; '.join(layer_patch['emotion_rules'][:3])}")
    if str(layer_patch.get("source") or "model") != "model":
        _println(f"{_color('Layer Source:', '31')} {layer_patch.get('source')}")

    current_stage = {
        "index": completed_units,
        "title": f"Layer Map: {layer}",
        "stage": f"layer_map:{layer}",
        "layer": layer,
        "summary": _truncate(layer_patch.get("summary"), 180),
        "source": str(layer_patch.get("source") or "model"),
    }
    _write_progress(
        progress_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=updated_layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=current_stage,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=updated_layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    return brain_map, updated_layer_reports, completed_units


def _finalize_pending_layers(
    *,
    layer_names: List[str],
    layer_stage_defs: Dict[str, List[Dict[str, Any]]],
    layer_stage_results: Dict[str, List[Dict[str, Any]]],
    payload: Dict[str, Any],
    args: argparse.Namespace,
    agent_id: str,
    source_path: Path,
    model: str,
    host: str,
    total_units: int,
    completed_units: int,
    started_at: str,
    progress_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    brain_map: Dict[str, Any],
    layer_reports: List[Dict[str, Any]],
    stage_results: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    updated_brain_map = brain_map
    updated_layer_reports = list(layer_reports or [])
    updated_completed_units = int(completed_units)
    for layer in list(layer_names or []):
        if any(str(row.get("layer") or "") == layer for row in list(updated_layer_reports or []) if isinstance(row, dict)):
            continue
        have = len(layer_stage_results.get(layer, []) or [])
        need = len(layer_stage_defs.get(layer, []) or [])
        if need <= 0 or have < need:
            continue
        updated_brain_map, updated_layer_reports, updated_completed_units = _finalize_layer(
            layer=layer,
            payload=payload,
            args=args,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=updated_completed_units,
            started_at=started_at,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            brain_map=updated_brain_map,
            layer_reports=updated_layer_reports,
            layer_stage_results=layer_stage_results,
            layer_stage_defs=layer_stage_defs,
            stage_results=stage_results,
        )
    return updated_brain_map, updated_layer_reports, updated_completed_units


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Console brain-study trainer for CSEN agents with Ollama.")
    parser.add_argument("--brain-id", help="Agent id from brains/<id>.json")
    parser.add_argument("--brain-file", help="Path to a brain json file, including room_brains/*.json")
    parser.add_argument("--model", default=None, help="Ollama model name")
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama host, default is local runtime")
    parser.add_argument("--timeout", type=float, default=LAB_DEFAULT_TIMEOUT, help="Base request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for each stage and final synthesis")
    parser.add_argument("--keep-alive", default="45m", help="Ollama keep_alive value")
    parser.add_argument("--predict-scale", type=float, default=1.0, help="Scale factor for num_predict budgets")
    parser.add_argument("--max-request-timeout", type=float, default=0.0, help="Hard cap for a single Ollama request in seconds, 0 disables the cap")
    parser.add_argument("--max-units-per-run", type=int, default=0, help="Pause after this many units of work in the current run, 0 disables the limit")
    parser.add_argument("--cooldown-sec", type=float, default=0.0, help="Cooldown between completed units to reduce sustained load")
    parser.add_argument("--low-resource", action="store_true", help="Use smaller generation budgets and shorter model retention")
    parser.add_argument("--report-out", default=None, help="Path for the final training report json")
    parser.add_argument("--progress-out", default=None, help="Path for the live progress json")
    parser.add_argument("--checkpoint-out", default=None, help="Path for the full resume checkpoint json")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists")
    parser.add_argument("--fresh-start", action="store_true", help="Ignore and remove any existing checkpoint before training")
    parser.add_argument("--no-write-back", action="store_true", help="Do not write the trained profile back into the source brain file")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    source_path, raw_payload, block, lineage, runtime, snapshots, export_meta = _load_source(args)
    host = str(args.host or DEFAULT_OLLAMA_HOST).rstrip("/")
    model = pick_default_model(preferred=args.model, host=host)

    payload = build_full_brain_study_payload(
        block,
        lineage=lineage,
        runtime=runtime,
        snapshots=snapshots,
        export_meta=export_meta,
        source_path=str(source_path),
    )
    stages = build_brain_study_stages(payload)
    agent_id = str((payload.get("agent", {}) or {}).get("id") or getattr(block, "agent_id", "agent"))
    progress_path = _progress_path(args, agent_id)
    report_path = _report_path(args, agent_id)
    checkpoint_path = _checkpoint_path(args, agent_id)
    layer_names = _layer_order(stages)
    layer_stage_defs = _layer_stage_defs_map(stages)
    total_units = len(stages) + len(layer_names) + 1
    resumed = False
    resume_note = "fresh"
    if args.fresh_start and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            resume_note = "checkpoint_removed"
        except Exception:
            resume_note = "checkpoint_kept"

    resume_state: Optional[Dict[str, Any]] = None
    if args.resume and not args.fresh_start:
        resume_state = _load_resume_state(
            checkpoint_path,
            payload=payload,
            stages=stages,
            source_path=source_path,
            agent_id=agent_id,
        )
        if resume_state is not None:
            resumed = True
            resume_note = f"resume:{resume_state.get('status', 'running')}"

    started_at = str((resume_state or {}).get("started_at") or _utc_now())
    if args.low_resource and str(args.keep_alive or "45m") == "45m":
        args.keep_alive = "12m"
    stage_results: List[Dict[str, Any]] = [dict(row) for row in list((resume_state or {}).get("stage_results", []) or []) if isinstance(row, dict)]
    brain_map: Dict[str, Any] = dict((resume_state or {}).get("brain_map", {}) or {}) if isinstance((resume_state or {}).get("brain_map", {}), dict) else _empty_brain_map(payload)
    if not brain_map:
        brain_map = _empty_brain_map(payload)
    layer_reports: List[Dict[str, Any]] = [dict(row) for row in list((resume_state or {}).get("layer_reports", []) or []) if isinstance(row, dict)]
    completed_units = int((resume_state or {}).get("completed_units", 0) or 0)
    completed_stage_count = len(stage_results)
    existing_report = Path(str((resume_state or {}).get("final_report") or report_path))
    if resumed and str((resume_state or {}).get("status") or "") in {"completed", "completed_with_fallback"} and existing_report.exists():
        _print_block(
            "RESUME READY",
            [
                f"Agent:   {agent_id}",
                f"Status:  {resume_state.get('status')}",
                f"Report:  {existing_report}",
                f"Checkpoint: {checkpoint_path}",
                "Обучение уже завершено. Используй --fresh-start для полного перезапуска.",
            ],
            color_code="32",
        )
        return 0

    _print_block(
        "CSEN OLLAMA BRAIN LAB",
        [
            f"Agent:   {agent_id}",
            f"Source:  {source_path}",
            f"Model:   {model}",
            f"Host:    {host}",
            f"Stages:  {len(stages)}",
            f"Layers:  {len(layer_names)}",
            f"Units:   {total_units}",
            f"Resume:  {'on' if resumed else 'off'}",
            f"Resume Note: {resume_note}",
            f"Low Resource: {'on' if args.low_resource else 'off'}",
            f"Max Request Timeout: {float(args.max_request_timeout or 0.0):0.1f}s",
            f"Budget Units: {int(args.max_units_per_run or 0)}",
            f"Cooldown: {float(args.cooldown_sec or 0.0):0.1f}s",
            f"Writeback: {'off' if args.no_write_back else 'on'}",
            f"Progress: {progress_path}",
            f"Checkpoint: {checkpoint_path}",
            f"Report:   {report_path}",
        ],
        color_code="35",
    )

    layer_stage_results: Dict[str, List[Dict[str, Any]]] = {}
    force_local_layers: set[str] = set()
    for row in list(stage_results or []):
        if not isinstance(row, dict):
            continue
        layer = str(row.get("layer") or "")
        if layer:
            layer_stage_results.setdefault(layer, []).append(dict(row))
    started = time.time()
    run_start_units = completed_units
    _write_progress(
        progress_path,
        status="resuming" if resumed else "starting",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=(resume_state or {}).get("current_stage"),
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status="resuming" if resumed else "starting",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage=(resume_state or {}).get("current_stage"),
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _set_interrupt_context(
        current_stage=(resume_state or {}).get("current_stage"),
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_units=total_units,
        completed_units=completed_units,
        progress_path=progress_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
    )
    try:
        brain_map, layer_reports, completed_units = _finalize_pending_layers(
            layer_names=layer_names,
            layer_stage_defs=layer_stage_defs,
            layer_stage_results=layer_stage_results,
            payload=payload,
            args=args,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            started_at=started_at,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            brain_map=brain_map,
            layer_reports=layer_reports,
            stage_results=stage_results,
        )
    except TrainingInterrupted as exc:
        return _interrupt_training(
            reason=exc.reason,
            current_stage=exc.current_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        )
    if completed_units > run_start_units and layer_reports:
        current_stage = {
            "index": completed_units,
            "title": str(layer_reports[-1].get("title") or ""),
            "stage": str(layer_reports[-1].get("stage") or ""),
            "layer": str(layer_reports[-1].get("layer") or ""),
            "summary": _truncate(layer_reports[-1].get("summary"), 180),
            "source": str(layer_reports[-1].get("source") or "model"),
        }
        if _maybe_pause_for_budget(
            args=args,
            run_start_units=run_start_units,
            completed_units=completed_units,
            total_units=total_units,
            current_stage=current_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        ):
            return 0
        try:
            _apply_cooldown(float(args.cooldown_sec or 0.0), reason=f"layer:{layer_reports[-1].get('layer') or 'resume'}")
        except KeyboardInterrupt:
            return _interrupt_training(
                reason="Interrupted by operator (Ctrl+C)",
                current_stage=current_stage,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            )

    for index, stage in enumerate(stages, start=1):
        if index <= completed_stage_count:
            continue
        layer = str(stage.get("layer") or stage.get("id") or f"layer_{index}")
        stage_plan = _stage_resource_plan(stage, args)
        current_stage = {
            "index": completed_units + 1,
            "title": str(stage.get("title") or ""),
            "stage": str(stage.get("id") or ""),
            "layer": layer,
            "focus": _truncate(stage.get("focus"), 220),
            "resource_plan": {
                "timeout": round(_safe_float(stage_plan.get("timeout", 0.0), 0.0), 1),
                "retries": int(stage_plan.get("retries", int(args.retries))),
                "num_predict": int(stage_plan.get("num_predict", STAGE_NUM_PREDICT)),
            },
        }
        _set_interrupt_context(
            current_stage=current_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        )
        _write_progress(
            progress_path,
            status="running",
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_stages=total_units,
            completed_stages=completed_units,
            current_stage=current_stage,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
            final_report=report_path,
        )
        _write_checkpoint(
            checkpoint_path,
            status="running",
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_stages=total_units,
            completed_stages=completed_units,
            current_stage=current_stage,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
            final_report=report_path,
        )
        _println(_rule())
        _println(_color(f"{_progress(completed_units, total_units)}  {stage.get('title')}", "33"))
        _println(_truncate(stage.get("focus"), 300))
        local_policy = _stage_local_policy_reason(stage, args, force_local_layers=force_local_layers)
        if local_policy:
            _println(_color(f"Stage policy: local digest mode ({local_policy})", "90"))
            result = _local_stage_fallback(stage, payload, error_text=local_policy)
            result["source"] = "policy_local"
        else:
            try:
                result_raw = _generate_json(
                    host=host,
                    model=model,
                    prompt=_stage_prompt(stage, payload, stage_results, brain_map),
                    timeout=float(stage_plan.get("timeout", float(args.timeout))),
                    keep_alive=str(args.keep_alive),
                    label=f"stage:{stage.get('id')}",
                    retries=int(stage_plan.get("retries", int(args.retries))),
                    num_predict=int(stage_plan.get("num_predict", STAGE_NUM_PREDICT)),
                    max_request_timeout=float(stage_plan.get("max_request_timeout", 0.0)),
                )
                result = _sanitize_stage_result(stage, result_raw)
                result["source"] = "model"
            except KeyboardInterrupt:
                return _interrupt_training(
                    reason="Interrupted by operator (Ctrl+C)",
                    current_stage=current_stage,
                    agent_id=agent_id,
                    source_path=source_path,
                    model=model,
                    host=host,
                    total_units=total_units,
                    completed_units=completed_units,
                    progress_path=progress_path,
                    checkpoint_path=checkpoint_path,
                    report_path=report_path,
                    stage_results=stage_results,
                    brain_map=brain_map,
                    layer_reports=layer_reports,
                    started_at=started_at,
                )
            except BrainLabRequestError as exc:
                if "timed out" in str(exc).lower():
                    force_local_layers.add(layer)
                _println(_color(f"Stage fallback: {exc}", "31"))
                result = _local_stage_fallback(stage, payload, error_text=str(exc))
        result["tick"] = index
        result["layer"] = layer
        stage_results.append(result)
        layer_stage_results.setdefault(layer, []).append(result)
        brain_map = _merge_brain_map_patch(brain_map, _stage_to_brain_map_patch(stage, result))
        completed_units += 1
        _println(f"{_color('Summary:', '36')} {result.get('summary', '')}")
        if result.get("command_rules"):
            _println(f"{_color('Command:', '32')} {'; '.join(result['command_rules'][:3])}")
        if result.get("emotion_rules"):
            _println(f"{_color('Emotion:', '34')} {'; '.join(result['emotion_rules'][:3])}")
        if str(result.get("source") or "model") != "model":
            _println(f"{_color('Source:', '31')} {result.get('source')}")
        _write_progress(
            progress_path,
            status="running",
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_stages=total_units,
            completed_stages=completed_units,
            current_stage={
                "index": completed_units,
                "title": str(stage.get("title") or ""),
                "stage": str(stage.get("id") or ""),
                "layer": layer,
                "summary": _truncate(result.get("summary"), 180),
                "source": str(result.get("source") or "model"),
            },
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
            final_report=report_path,
        )
        _write_checkpoint(
            checkpoint_path,
            status="running",
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_stages=total_units,
            completed_stages=completed_units,
            current_stage={
                "index": completed_units,
                "title": str(stage.get("title") or ""),
                "stage": str(stage.get("id") or ""),
                "layer": layer,
                "summary": _truncate(result.get("summary"), 180),
                "source": str(result.get("source") or "model"),
            },
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
            final_report=report_path,
        )
        done_stage = {
            "index": completed_units,
            "title": str(stage.get("title") or ""),
            "stage": str(stage.get("id") or ""),
            "layer": layer,
            "summary": _truncate(result.get("summary"), 180),
            "source": str(result.get("source") or "model"),
        }
        _set_interrupt_context(
            current_stage=done_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        )
        if _maybe_pause_for_budget(
            args=args,
            run_start_units=run_start_units,
            completed_units=completed_units,
            total_units=total_units,
            current_stage=done_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        ):
            return 0
        try:
            _apply_cooldown(float(args.cooldown_sec or 0.0), reason=f"stage:{stage.get('id') or index}")
        except KeyboardInterrupt:
            return _interrupt_training(
                reason="Interrupted by operator (Ctrl+C)",
                current_stage=done_stage,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            )

        layer_done = index == len(stages) or str((stages[index] if index < len(stages) else {}).get("layer") or "") != layer
        if not layer_done:
            continue
        try:
            brain_map, layer_reports, completed_units = _finalize_layer(
                layer=layer,
                payload=payload,
                args=args,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                started_at=started_at,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                brain_map=brain_map,
                layer_reports=layer_reports,
                layer_stage_results=layer_stage_results,
                layer_stage_defs=layer_stage_defs,
                stage_results=stage_results,
            )
        except TrainingInterrupted as exc:
            return _interrupt_training(
                reason=exc.reason,
                current_stage=exc.current_stage,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            )
        if layer_reports:
            done_layer = {
                "index": completed_units,
                "title": str(layer_reports[-1].get("title") or ""),
                "stage": str(layer_reports[-1].get("stage") or ""),
                "layer": str(layer_reports[-1].get("layer") or ""),
                "summary": _truncate(layer_reports[-1].get("summary"), 180),
                "source": str(layer_reports[-1].get("source") or "model"),
            }
            if _maybe_pause_for_budget(
                args=args,
                run_start_units=run_start_units,
                completed_units=completed_units,
                total_units=total_units,
                current_stage=done_layer,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            ):
                return 0
            _set_interrupt_context(
                current_stage=done_layer,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            )
            try:
                _apply_cooldown(float(args.cooldown_sec or 0.0), reason=f"layer:{layer_reports[-1].get('layer') or layer}")
            except KeyboardInterrupt:
                return _interrupt_training(
                    reason="Interrupted by operator (Ctrl+C)",
                    current_stage=done_layer,
                    agent_id=agent_id,
                    source_path=source_path,
                    model=model,
                    host=host,
                    total_units=total_units,
                    completed_units=completed_units,
                    progress_path=progress_path,
                    checkpoint_path=checkpoint_path,
                    report_path=report_path,
                    stage_results=stage_results,
                    brain_map=brain_map,
                    layer_reports=layer_reports,
                    started_at=started_at,
                )

    try:
        brain_map, layer_reports, completed_units = _finalize_pending_layers(
            layer_names=layer_names,
            layer_stage_defs=layer_stage_defs,
            layer_stage_results=layer_stage_results,
            payload=payload,
            args=args,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            started_at=started_at,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            brain_map=brain_map,
            layer_reports=layer_reports,
            stage_results=stage_results,
        )
    except TrainingInterrupted as exc:
        return _interrupt_training(
            reason=exc.reason,
            current_stage=exc.current_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        )
    if layer_reports:
        final_pending_stage = {
            "index": completed_units,
            "title": str(layer_reports[-1].get("title") or ""),
            "stage": str(layer_reports[-1].get("stage") or ""),
            "layer": str(layer_reports[-1].get("layer") or ""),
            "summary": _truncate(layer_reports[-1].get("summary"), 180),
            "source": str(layer_reports[-1].get("source") or "model"),
        }
        if _maybe_pause_for_budget(
            args=args,
            run_start_units=run_start_units,
            completed_units=completed_units,
            total_units=total_units,
            current_stage=final_pending_stage,
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        ):
            return 0
        try:
            _apply_cooldown(float(args.cooldown_sec or 0.0), reason=f"layer:{layer_reports[-1].get('layer') or 'finalize'}")
        except KeyboardInterrupt:
            return _interrupt_training(
                reason="Interrupted by operator (Ctrl+C)",
                current_stage=final_pending_stage,
                agent_id=agent_id,
                source_path=source_path,
                model=model,
                host=host,
                total_units=total_units,
                completed_units=completed_units,
                progress_path=progress_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                stage_results=stage_results,
                brain_map=brain_map,
                layer_reports=layer_reports,
                started_at=started_at,
            )

    _write_progress(
        progress_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage={
            "index": completed_units + 1,
            "title": "Final Synthesis",
            "stage": "final_synthesis",
            "focus": "Сборка единого brain_profile по layered brain map.",
        },
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status="running",
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage={
            "index": completed_units + 1,
            "title": "Final Synthesis",
            "stage": "final_synthesis",
            "focus": "Сборка единого brain_profile по layered brain map.",
        },
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _set_interrupt_context(
        current_stage={
            "index": completed_units + 1,
            "title": "Final Synthesis",
            "stage": "final_synthesis",
            "focus": "Сборка единого brain_profile по layered brain map.",
        },
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_units=total_units,
        completed_units=completed_units,
        progress_path=progress_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
    )
    _println(_rule())
    _println(_color(f"{_progress(completed_units, total_units)}  Final Synthesis", "33"))
    final_source = "model"
    try:
        final_raw = _generate_json(
            host=host,
            model=model,
            prompt=_final_prompt(payload, stage_results, brain_map, layer_reports),
            timeout=min(
                max(float(args.timeout), 90.0),
                max(0.0, float(args.max_request_timeout or 0.0)) or (75.0 if args.low_resource else max(float(args.timeout), 90.0)),
            ),
            keep_alive=str(args.keep_alive),
            label="final_synthesis",
            retries=1 if args.low_resource else int(args.retries),
            num_predict=max(180, int(FINAL_NUM_PREDICT * (0.65 if args.low_resource else max(0.35, float(args.predict_scale or 1.0))))),
            max_request_timeout=max(0.0, float(args.max_request_timeout or 0.0)) or (75.0 if args.low_resource else 0.0),
        )
        profile = _normalize_profile(final_raw, payload, stage_results, brain_map, layer_reports)
    except KeyboardInterrupt:
        return _interrupt_training(
            reason="Interrupted by operator (Ctrl+C)",
            current_stage={
                "index": completed_units + 1,
                "title": "Final Synthesis",
                "stage": "final_synthesis",
            },
            agent_id=agent_id,
            source_path=source_path,
            model=model,
            host=host,
            total_units=total_units,
            completed_units=completed_units,
            progress_path=progress_path,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            stage_results=stage_results,
            brain_map=brain_map,
            layer_reports=layer_reports,
            started_at=started_at,
        )
    except BrainLabRequestError as exc:
        _println(_color(f"Final fallback: {exc}", "31"))
        profile = _local_profile_from_brain_map(payload, stage_results, brain_map, layer_reports)
        final_source = "fallback"

    profile_summary = summarize_brain_profile(profile)
    elapsed = time.time() - started
    completed_units += 1
    fallback_count = sum(1 for row in stage_results if str(row.get("source") or "model") != "model")
    fallback_count += sum(1 for row in layer_reports if str(row.get("source") or "model") != "model")
    if final_source != "model":
        fallback_count += 1
    report_status = "completed_with_fallback" if fallback_count else "completed"

    report = {
        "trained_at": profile.get("trained_at"),
        "agent_id": agent_id,
        "source_path": str(source_path),
        "model": model,
        "host": host,
        "elapsed_sec": round(elapsed, 2),
        "status": report_status,
        "fallback_count": int(fallback_count),
        "brain_payload_summary": {
            "emotion_baseline": payload.get("emotion_baseline", {}),
            "totals": payload.get("totals", {}),
        },
        "stage_results": stage_results,
        "brain_map": _summarize_brain_map(brain_map),
        "layer_reports": layer_reports,
        "brain_profile": profile,
        "brain_profile_summary": profile_summary,
    }
    _write_json(report_path, report)
    _write_progress(
        progress_path,
        status=report_status,
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage={
            "index": completed_units,
            "title": "Final Synthesis",
            "stage": "final_synthesis",
            "summary": _truncate(((profile.get("identity", {}) or {}).get("summary") or ""), 180),
            "source": final_source,
        },
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )
    _write_checkpoint(
        checkpoint_path,
        status=report_status,
        agent_id=agent_id,
        source_path=source_path,
        model=model,
        host=host,
        total_stages=total_units,
        completed_stages=completed_units,
        current_stage={
            "index": completed_units,
            "title": "Final Synthesis",
            "stage": "final_synthesis",
            "summary": _truncate(((profile.get("identity", {}) or {}).get("summary") or ""), 180),
            "source": final_source,
        },
        stage_results=stage_results,
        brain_map=brain_map,
        layer_reports=layer_reports,
        started_at=started_at,
        final_report=report_path,
    )

    if not args.no_write_back:
        _write_back_profile(source_path, raw_payload, profile, stage_results)

    _print_block(
        "TRAINING COMPLETE",
        [
            f"Agent:   {agent_id}",
            f"Report:  {report_path}",
            f"Saved:   {'source brain updated' if not args.no_write_back else 'report only'}",
            f"Elapsed: {elapsed:0.1f}s",
            f"Status:  {report_status}",
            f"Layers:  {', '.join(layer_names)}",
            f"Fallbacks: {fallback_count}",
            f"Style:   {((profile.get('identity', {}) or {}).get('style') or 'n/a')}",
            f"Summary: {((profile.get('identity', {}) or {}).get('summary') or 'n/a')}",
        ],
        color_code="32",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(_interrupt_from_context("Interrupted by operator (Ctrl+C)"))
