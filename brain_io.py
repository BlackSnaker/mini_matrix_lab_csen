from __future__ import annotations
import os
import json
import time
from typing import Optional, Dict, Any, Tuple, List, Union

from mind_core import ConsciousnessBlock


# =============================================================================
# Глобальные настройки хранения
# =============================================================================

_BRAINS_DIR = "brains"

# v3 (текущий):
#   {
#     "_schema_version": 3,
#     "_lineage": {...},       # агрегированная статистика линии
#     "_runtime": {...},       # фаст-срез текущего тела/страха/энергии и т.д.
#     "_snapshots": [...],     # история последних runtime-срезов
#     "brain": {...},          # полная сериализация сознания (ConsciousnessBlock)
#     "last_saved": "..."
#   }
#
_SCHEMA_VERSION = 3

# Ограничение истории слепков
_SNAPSHOT_LIMIT = 5

# Префикс для print-логов (удобно грепать)
_LOG_PREFIX = "[brain_io]"


# =============================================================================
# Утилиты низкого уровня
# =============================================================================

def _ensure_dir() -> None:
    try:
        os.makedirs(_BRAINS_DIR, exist_ok=True)
    except Exception as e:
        print(f"{_LOG_PREFIX} WARN: can't ensure dir {_BRAINS_DIR}: {e}")


def _sanitize_lineage_id(lineage_id: str) -> str:
    safe_chars = []
    for ch in str(lineage_id):
        if (
            "a" <= ch <= "z"
            or "A" <= ch <= "Z"
            or "0" <= ch <= "9"
            or ch in (".", "_", "-")
        ):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def _brain_path(agent_lineage_id: str) -> str:
    _ensure_dir()
    safe_id = _sanitize_lineage_id(agent_lineage_id)
    return os.path.join(_BRAINS_DIR, f"{safe_id}.json")


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"{_LOG_PREFIX} WARN: can't read {path}: {e}")
        return None


def _write_json_file_atomic(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"{_LOG_PREFIX} ERROR: save failed for {path}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if x != x:
        return default
    if x in (float("inf"), float("-inf")):
        return default
    return x


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return x


def _safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)
    if len(v) > 1024:
        return v[:1024] + "...(+cut)"
    return v


# =============================================================================
# Вытягивание runtime/lineage/snapshot метаданных из ConsciousnessBlock
# =============================================================================

def _extract_runtime_meta(block: ConsciousnessBlock) -> Dict[str, Any]:
    fear_level = _safe_float(getattr(block, "fear_level", 0.0))
    health = _safe_float(getattr(block, "health", 0.0))
    energy = _safe_float(getattr(block, "energy", 0.0))
    hunger = _safe_float(getattr(block, "hunger", 0.0))
    survival_score = _safe_float(getattr(block, "survival_score", 0.0))
    curiosity_charge = _safe_float(getattr(block, "curiosity_charge", 0.0))

    age_ticks = _safe_int(getattr(block, "age_ticks", 0))

    alive_now = bool(getattr(block, "alive", True))
    current_drive = _safe_str(getattr(block, "current_drive", None))
    last_death_reason = _safe_str(getattr(block, "last_death_reason", None))
    last_thought = _safe_str(getattr(block, "last_thought", None))

    ally_anchor = getattr(block, "ally_anchor", None)
    if isinstance(ally_anchor, dict):
        ally_anchor_slim: Dict[str, Any] = {}
        for k, v in ally_anchor.items():
            if isinstance(v, (str, bytes)):
                ally_anchor_slim[k] = _safe_str(v)
            elif isinstance(v, (int, float, bool)):
                ally_anchor_slim[k] = v
            else:
                ally_anchor_slim[k] = f"<{type(v).__name__}>"
        ally_anchor_out: Union[Dict[str, Any], str, None] = ally_anchor_slim
    else:
        ally_anchor_out = _safe_str(ally_anchor)

    trauma_map = getattr(block, "trauma_map", None)
    trauma_spots_known = len(trauma_map) if isinstance(trauma_map, list) else 0

    # Короткий срез по навыкам (для дебага)
    skills_version = _safe_str(getattr(block, "gc_version", None))
    skills_steps = _safe_int(getattr(block, "gc_steps", 0))
    goals = getattr(block, "gc_goal_vocab", None)
    skills_goals = len(goals) if isinstance(goals, dict) else 0

    return {
        "alive_now": alive_now,
        "current_drive": current_drive,
        "fear_level": fear_level,
        "health": health,
        "energy": energy,
        "hunger": hunger,
        "survival_score": survival_score,
        "age_ticks": age_ticks,
        "ally_anchor": ally_anchor_out,
        "last_death_reason": last_death_reason,
        "last_thought": last_thought,
        "curiosity_charge": curiosity_charge,
        "trauma_spots_known": trauma_spots_known,
        "skills_version": skills_version,
        "skills_steps": skills_steps,
        "skills_goals": skills_goals,
    }


def _build_snapshot(block: ConsciousnessBlock) -> Dict[str, Any]:
    rt = _extract_runtime_meta(block)
    return {
        "ts": _now_utc_iso(),
        "alive_now": rt.get("alive_now", True),
        "age_ticks": rt.get("age_ticks", 0),
        "survival_score": rt.get("survival_score", 0.0),
        "fear_level": rt.get("fear_level", 0.0),
        "health": rt.get("health", 0.0),
        "energy": rt.get("energy", 0.0),
        "hunger": rt.get("hunger", 0.0),
        "current_drive": rt.get("current_drive", None),
        "last_death_reason": rt.get("last_death_reason", None),
    }


# =============================================================================
# Чтение / апгрейд форматов v1 → v2 → v3
# =============================================================================

def _unpack_brain_data(raw: Dict[str, Any]) -> Tuple[
    ConsciousnessBlock,
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]]
]:
    if raw is None:
        dummy = ConsciousnessBlock(agent_id="unknown")
        return dummy, {}, {}, []

    if isinstance(raw, dict) and raw.get("_schema_version") == 3:
        brain_dict = raw.get("brain", {}) or {}
        lineage_meta = raw.get("_lineage", {}) or {}
        runtime_meta = raw.get("_runtime", {}) or {}
        snapshots = raw.get("_snapshots", []) or []
        block = ConsciousnessBlock.from_dict(brain_dict)
        ldr = lineage_meta.get("last_death_reason")
        if ldr and not getattr(block, "last_death_reason", None):
            block.last_death_reason = ldr
        return block, lineage_meta, runtime_meta, snapshots

    if isinstance(raw, dict) and "brain" in raw:
        brain_dict = raw.get("brain", {}) or {}
        lineage_meta = raw.get("_lineage", {}) or {}
        block = ConsciousnessBlock.from_dict(brain_dict)
        runtime_meta: Dict[str, Any] = {}
        snapshots: List[Dict[str, Any]] = []
        ldr = lineage_meta.get("last_death_reason")
        if ldr and not getattr(block, "last_death_reason", None):
            block.last_death_reason = ldr
        return block, lineage_meta, runtime_meta, snapshots

    block = ConsciousnessBlock.from_dict(raw)
    return block, {}, {}, []


# =============================================================================
# Агрегация статистики линии (_lineage)
# =============================================================================

def _merge_lineage_stats(
    prev_meta: Dict[str, Any],
    prev_runtime: Dict[str, Any],
    prev_snaps: List[Dict[str, Any]],
    new_block: ConsciousnessBlock,
) -> Dict[str, Any]:
    """
    Обновляем агрегированную статистику линии (_lineage) с учётом свежего мозга.

    Исправления:
      - total_age_ticks накапливается по Δage (age_now - age_prev_saved ≥ 0),
        чтобы не было двойного учёта при частых сохранениях в одном теле.
      - deaths / generation инкрементируются только при переходе Alive→Dead.
    """
    lineage_id = getattr(new_block, "agent_id", prev_meta.get("lineage_id", "unknown"))

    prev_generation = _safe_int(prev_meta.get("generation", 0))
    prev_deaths = _safe_int(prev_meta.get("deaths", 0))
    prev_total_age = _safe_int(prev_meta.get("total_age_ticks", 0))
    prev_best_score = _safe_float(prev_meta.get("best_survival_score", 0.0))
    prev_last_reason = prev_meta.get("last_death_reason")

    age_now = _safe_int(getattr(new_block, "age_ticks", 0))
    survival_now = _safe_float(getattr(new_block, "survival_score", 0.0))
    alive_now = bool(getattr(new_block, "alive", True))
    last_reason_now = getattr(new_block, "last_death_reason", None)

    # Δage считаем относительно последнего сохранённого age_ticks из _runtime
    age_prev_saved = _safe_int(prev_runtime.get("age_ticks", 0))
    delta_age = max(0, age_now - age_prev_saved)
    total_age_ticks = prev_total_age + delta_age
    best_survival_score = max(prev_best_score, survival_now)

    # Был ли агент жив в прошлый раз?
    # Берём из _runtime (при его отсутствии считаем, что жив).
    prev_alive = True
    if isinstance(prev_runtime, dict) and "alive_now" in prev_runtime:
        prev_alive = bool(prev_runtime.get("alive_now", True))
    elif prev_snaps:
        # fallback: смотрим самый свежий снап (он у нас в начале списка)
        prev_alive = bool(prev_snaps[0].get("alive_now", True))

    # Инкремент поколений/смертей только на переходе Alive→Dead
    if prev_alive and not alive_now:
        generation = prev_generation + 1
        deaths = prev_deaths + 1
        last_reason_final = last_reason_now or prev_last_reason
    else:
        generation = prev_generation
        deaths = prev_deaths
        last_reason_final = prev_last_reason

    return {
        "lineage_id": lineage_id,
        "generation": generation,
        "deaths": deaths,
        "total_age_ticks": total_age_ticks,
        "best_survival_score": best_survival_score,
        "last_death_reason": last_reason_final,
    }


def _merge_snapshots(
    old_snapshots: List[Dict[str, Any]],
    block: ConsciousnessBlock,
) -> List[Dict[str, Any]]:
    fresh = _build_snapshot(block)
    new_list = [fresh] + list(old_snapshots or [])
    return new_list[:_SNAPSHOT_LIMIT]


# =============================================================================
# Публичные функции для сервера / тренера
# =============================================================================

def load_brain(agent_lineage_id: str) -> Optional[ConsciousnessBlock]:
    path = _brain_path(agent_lineage_id)
    raw = _read_json_file(path)
    if raw is None:
        return None

    try:
        block, _lineage, _runtime, _snaps = _unpack_brain_data(raw)
        block.agent_id = agent_lineage_id
        return block
    except Exception as e:
        print(f"{_LOG_PREFIX} load_brain error for {agent_lineage_id}: {e}")
        return None


def load_brain_with_meta(agent_lineage_id: str) -> Optional[Dict[str, Any]]:
    path = _brain_path(agent_lineage_id)
    raw = _read_json_file(path)
    if raw is None:
        return None

    try:
        block, lineage_meta, runtime_meta, snapshots = _unpack_brain_data(raw)
        block.agent_id = agent_lineage_id
        return {
            "brain": block,
            "lineage": lineage_meta,
            "runtime": runtime_meta,
            "snapshots": snapshots,
        }
    except Exception as e:
        print(f"{_LOG_PREFIX} load_brain_with_meta error for {agent_lineage_id}: {e}")
        return None


def save_brain(block: ConsciousnessBlock) -> None:
    """
    Сохранить состояние сознания/поведения на диск в формате v3.

    Алгоритм:
      1) читаем старый файл → lineage/runtime/snapshots;
      2) обновляем _lineage (учёт Δage и переход Alive→Dead);
      3) собираем новый _runtime;
      4) добавляем свежий snapshot (с обрезкой истории);
      5) атомарно пишем payload.
    """
    lineage_id = getattr(block, "agent_id", "unknown")
    path = _brain_path(lineage_id)

    prev_raw = _read_json_file(path)
    if prev_raw is None:
        prev_meta: Dict[str, Any] = {}
        prev_runtime: Dict[str, Any] = {}
        prev_snaps: List[Dict[str, Any]] = []
    else:
        try:
            _old_block, prev_meta, prev_runtime, prev_snaps = _unpack_brain_data(prev_raw)
        except Exception as e:
            print(f"{_LOG_PREFIX} WARN: broken previous brain for {lineage_id}: {e}")
            prev_meta = {}
            prev_runtime = {}
            prev_snaps = []

    # 2. lineage-статистика с учётом Δage и перехода Alive→Dead
    new_meta = _merge_lineage_stats(prev_meta, prev_runtime, prev_snaps, block)

    # 3. текущий runtime-срез
    runtime_meta = _extract_runtime_meta(block)

    # 4. история снапшотов
    snapshots = _merge_snapshots(prev_snaps, block)

    # 5. итоговый payload
    try:
        brain_dict = block.to_dict()
    except Exception as e:
        print(f"{_LOG_PREFIX} WARN: block.to_dict() failed for {lineage_id}: {e}")
        brain_dict = {
            "agent_id": getattr(block, "agent_id", None),
            "fallback_dump_error": str(e),
        }

    payload: Dict[str, Any] = {
        "_schema_version": _SCHEMA_VERSION,
        "_lineage": new_meta,
        "_runtime": runtime_meta,
        "_snapshots": snapshots,
        "brain": brain_dict,
        "last_saved": _now_utc_iso(),
    }

    _write_json_file_atomic(path, payload)
