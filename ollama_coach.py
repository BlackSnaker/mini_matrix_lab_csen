from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import os
import re
import urllib.error
import urllib.request

from ollama_brain_profile import summarize_brain_profile
from ollama_behavior_context import build_behavior_context


DEFAULT_OLLAMA_HOST = str(
    os.environ.get("CSEN_OLLAMA_HOST")
    or os.environ.get("OLLAMA_HOST")
    or "http://127.0.0.1:11434"
).rstrip("/")
DEFAULT_MODEL_PRIORITY = (
    "llama3.2:latest",
    "mistral:latest",
    "tinyllama:latest",
)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return fallback
    if math.isnan(out) or math.isinf(out):
        return fallback
    return out


DEFAULT_OLLAMA_TIMEOUT = max(
    10.0,
    _safe_float(os.environ.get("CSEN_OLLAMA_TIMEOUT") or os.environ.get("OLLAMA_TIMEOUT"), 60.0),
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _pct(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value, default)
    if out <= 1.5:
        out *= 100.0
    return _clamp(out, 0.0, 100.0)


def _is_white_room(room: Any) -> bool:
    room_id = str(getattr(room, "room_id", "") or "").casefold()
    object_id = str(getattr(room, "object_id", "") or "").casefold()
    label = str(getattr(room, "label", "") or "")
    return "white_room" in room_id or "white_room" in object_id or "blank_room" in room_id or "Белая_комната" in label


def _normalize_vec(x: float, y: float) -> Tuple[float, float]:
    n = math.hypot(float(x), float(y))
    if n <= 1e-6:
        return (1.0, 0.0)
    return (float(x) / n, float(y) / n)


def _agent_facing_xy(agent: Any) -> Tuple[float, float]:
    vx = _safe_float(getattr(agent, "vx", 0.0), 0.0)
    vy = _safe_float(getattr(agent, "vy", 0.0), 0.0)
    if math.hypot(vx, vy) > 1e-4:
        return _normalize_vec(vx, vy)
    fx = _safe_float(getattr(agent, "_manual_facing_x", 0.0), 0.0)
    fy = _safe_float(getattr(agent, "_manual_facing_y", 0.0), 0.0)
    if math.hypot(fx, fy) > 1e-4:
        return _normalize_vec(fx, fy)
    dx = _safe_float(getattr(agent, "goal_x", 0.0), 0.0) - _safe_float(getattr(agent, "x", 0.0), 0.0)
    dy = _safe_float(getattr(agent, "goal_y", 0.0), 0.0) - _safe_float(getattr(agent, "y", 0.0), 0.0)
    if math.hypot(dx, dy) > 1e-4:
        return _normalize_vec(dx, dy)
    return (1.0, 0.0)


def _estimate_step_unit(agent: Any) -> float:
    move_speed = _safe_float(getattr(agent, "move_speed", 3.0), 3.0)
    return _clamp(move_speed * 0.45, 0.9, 1.8)


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold().replace("ё", "е")).strip()


_STEP_WORDS = {
    "один": 1,
    "одна": 1,
    "одну": 1,
    "раз": 1,
    "два": 2,
    "две": 2,
    "пару": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}

_STOP_COMMAND_PATTERNS = (
    "останови агента",
    "остановись",
    "остановись на месте",
    "остановись здесь",
    "остановись тут",
    "остановись сейчас",
    "остановись немедленно",
    "остановка",
    "стой",
    "замри",
    "не двигайся",
    "не двигаться",
    "останови движение",
    "прекрати движение",
    "пауза",
    "stop",
    "freeze",
    "hold position",
)

_LANDMARK_ALIASES = {
    "morpheus_chair": ("morpheus chair", "left chair", "левое кресло", "левому креслу", "левое кресло морфеуса", "кресло морфеуса", "кресло слева", "левый стул"),
    "neo_chair": ("neo chair", "right chair", "правое кресло", "правому креслу", "правое кресло нео", "кресло нео", "кресло справа", "правый стул"),
    "coffee_table": ("coffee table", "table", "столик", "столику", "кофейный столик", "стол"),
    "floor_center": ("floor center", "center", "центр", "центру", "центр комнаты", "середина комнаты"),
    "green_panel": ("green panel", "panel", "панель", "панели", "зеленая панель", "зелёная панель", "экран"),
    "lamp": ("lamp", "торшер", "торшеру", "лампа", "лампе"),
}

_MOVE_TO_LANDMARK_PATTERNS = (
    "иди к", "подойди к", "перейди к", "двигайся к", "направляйся к", "сходи к",
    "сядь в", "сядь на", "go to", "move to", "walk to", "sit in",
)

_FACE_LANDMARK_PATTERNS = (
    "посмотри на", "смотри на", "повернись к", "развернись к", "взгляни на", "look at", "face",
)

_WAIT_PATTERNS = (
    "подожди", "жди", "постой", "останься на месте", "wait", "hold",
)

_REMEMBER_PATTERNS = (
    "запомни", "запиши", "сохрани в память", "remember", "memorize",
)

_GENERIC_CHAIR_TOKENS = (
    "кресло", "креслу", "кресле", "кресла",
    "стул", "стула", "стулу", "стуле",
    "chair", "seat",
)

_SIT_PATTERNS = (
    "сядь", "садись", "присядь", "сиди", "sit", "take a seat",
)

_HOLD_THERE_PATTERNS = (
    "стой там", "жди там", "останься там", "остановись там", "замри там", "сиди там",
    "stay there", "wait there", "hold there", "sit there",
)

_SLOW_PATTERNS = (
    "замедли", "иди медленнее", "иди медленно", "снизь скорость", "уменьши скорость",
    "тише", "медленнее", "slow down", "walk slower",
)

_FAST_PATTERNS = (
    "ускорься", "ускорь движение", "иди быстрее", "увеличь скорость", "быстрее",
    "speed up", "walk faster",
)

_CALM_PATTERNS = (
    "успокойся", "успокой агента", "не бойся", "перестань бояться", "расслабься",
    "стань спокойнее", "будь спокойнее", "calm down", "relax",
)

_FOCUS_PATTERNS = (
    "сосредоточься", "соберись", "сконцентрируйся", "будь внимательнее",
    "focus", "concentrate",
)

_CURIOUS_PATTERNS = (
    "будь любопытнее", "прояви любопытство", "исследуй смелее", "стань любознательнее",
    "be curious", "explore more",
)

_BRAVE_PATTERNS = (
    "будь смелее", "стань смелее", "будь увереннее", "перестань трусить",
    "be brave", "be confident",
)

_REST_PATTERNS = (
    "отдохни", "усни", "сделай паузу", "передохни", "rest", "take a break",
)

_ENERGIZE_PATTERNS = (
    "взбодрись", "воспрянь", "воодушевись", "стань энергичнее", "набери темп",
    "energize", "wake up", "be more energetic",
)

_FACE_RELATIVE_PATTERNS = {
    "left": ("повернись налево", "развернись налево", "turn left", "look left"),
    "right": ("повернись направо", "развернись направо", "turn right", "look right"),
    "backward": ("повернись назад", "развернись назад", "turn around", "look back"),
    "forward": ("смотри вперед", "смотри вперёд", "повернись вперед", "повернись вперёд", "look forward", "face forward"),
}

_MOVE_DIRECTION_TOKENS = {
    "forward": ("вперед", "вперёд", "прямо"),
    "backward": ("назад",),
    "left": ("влево", "налево"),
    "right": ("вправо", "направо"),
}

_SEQUENCE_SPLIT_RE = re.compile(r"\b(?:и потом|потом|затем|после этого|after that|then)\b")


def _extract_step_count(text: str) -> int:
    raw = _normalize_text(text)
    match = re.search(r"(\d{1,2})\s*(?:шаг|шага|шагов|раз)?", raw)
    if match:
        return int(_clamp(_safe_float(match.group(1), 1.0), 1.0, 12.0))
    for word, count in _STEP_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", raw):
            return int(count)
    if "несколько" in raw:
        return 3
    return 1


def _relative_goal_from_instruction(snapshot: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    text = _normalize_text(snapshot.get("operator_instruction"))
    if not text:
        return None

    agent = snapshot.get("agent", {}) or {}
    room = snapshot.get("room", {}) or {}
    bounds = room.get("bounds", {}) or {}
    pos = agent.get("pos", {}) or {}
    facing = agent.get("facing", {}) or {}
    px = _safe_float(pos.get("x"), 0.0)
    py = _safe_float(pos.get("y"), 0.0)
    fx = _safe_float(facing.get("x"), 1.0)
    fy = _safe_float(facing.get("y"), 0.0)
    fx, fy = _normalize_vec(fx, fy)
    step_unit = _clamp(_safe_float(agent.get("step_unit"), 1.2), 0.6, 2.0)
    steps = _extract_step_count(text)
    distance = max(0.6, float(steps) * step_unit)

    vx = 0.0
    vy = 0.0
    matched = False
    if any(token in text for token in ("вперед", "прямо перед", "прямо")):
        vx += fx
        vy += fy
        matched = True
    if "назад" in text:
        vx -= fx
        vy -= fy
        matched = True
    if any(token in text for token in ("влево", "налево", "слева")):
        vx += -fy
        vy += fx
        matched = True
    if any(token in text for token in ("вправо", "направо", "справа")):
        vx += fy
        vy += -fx
        matched = True
    if not matched:
        return None

    vx, vy = _normalize_vec(vx, vy)
    gx = px + vx * distance
    gy = py + vy * distance
    left = _safe_float(bounds.get("left"), 0.0)
    top = _safe_float(bounds.get("top"), 0.0)
    right = _safe_float(bounds.get("right"), left + 1.0)
    bottom = _safe_float(bounds.get("bottom"), top + 1.0)
    return (
        _clamp(gx, left + 1.2, right - 1.2),
        _clamp(gy, top + 1.2, bottom - 1.2),
    )


def _is_stop_instruction(text: Any) -> bool:
    raw = _normalize_text(text)
    if not raw:
        return False
    for token in _STOP_COMMAND_PATTERNS:
        if token in raw:
            return True
    return False


def _room_landmarks(snapshot: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    room = snapshot.get("room", {}) or {}
    landmarks = room.get("landmarks", {}) or {}
    if not isinstance(landmarks, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in landmarks.items():
        if isinstance(value, dict) and ("x" in value) and ("y" in value):
            out[str(key)] = value
    return out


def _nearest_landmark_key(snapshot: Dict[str, Any], landmark_keys: Iterable[str]) -> Optional[str]:
    landmarks = _room_landmarks(snapshot)
    candidates = [str(key) for key in list(landmark_keys or []) if str(key) in landmarks]
    if not candidates:
        return None
    agent = snapshot.get("agent", {}) or {}
    pos = agent.get("pos", {}) or {}
    px = _safe_float(pos.get("x"), 0.0)
    py = _safe_float(pos.get("y"), 0.0)
    best_key: Optional[str] = None
    best_dist = float("inf")
    for key in candidates:
        target = landmarks.get(key) or {}
        tx = _safe_float(target.get("x"), px)
        ty = _safe_float(target.get("y"), py)
        dist = math.hypot(tx - px, ty - py)
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


def _extract_landmark_key(text: Any, snapshot: Dict[str, Any]) -> Optional[str]:
    raw = _normalize_text(text)
    if not raw:
        return None
    landmarks = _room_landmarks(snapshot)
    if not landmarks:
        return None
    for key in landmarks:
        for alias in _LANDMARK_ALIASES.get(key, ()):
            if _normalize_text(alias) in raw:
                return key
    if any(token in raw for token in _GENERIC_CHAIR_TOKENS):
        return _nearest_landmark_key(snapshot, ("morpheus_chair", "neo_chair"))
    return None


def _extract_wait_ticks(text: Any) -> int:
    raw = _normalize_text(text)
    if not raw:
        return 48
    match = re.search(r"(\d{1,3})\s*(?:тик|тиков|ticks?)", raw)
    if match:
        return int(_clamp(_safe_float(match.group(1), 48.0), 12.0, 240.0))
    match = re.search(r"(\d{1,2})\s*(?:сек|секунд|секунды|seconds?)", raw)
    if match:
        return int(_clamp(_safe_float(match.group(1), 2.0), 1.0, 8.0) * 30.0)
    if "немного" in raw:
        return 36
    return 48


def _extract_remember_text(text: Any) -> Optional[str]:
    raw = str(text or "").strip()
    norm = _normalize_text(raw)
    if not norm:
        return None
    for token in _REMEMBER_PATTERNS:
        norm_token = _normalize_text(token)
        idx = norm.find(norm_token)
        if idx >= 0:
            out = raw[idx + len(token):].strip(" :,-")
            out = out or raw.strip()
            return out[:240]
    return None


def _infer_landmark_action(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = snapshot.get("operator_instruction")
    raw = _normalize_text(text)
    if not raw:
        return None
    landmark = _extract_landmark_key(raw, snapshot)
    if not landmark:
        return None
    if any(token in raw for token in _FACE_LANDMARK_PATTERNS):
        return {"type": "face_landmark", "landmark": landmark}
    if any(token in raw for token in _MOVE_TO_LANDMARK_PATTERNS):
        return {"type": "move_to_landmark", "landmark": landmark}
    return None


def _goal_from_direction(snapshot: Dict[str, Any], direction: str, steps: int) -> Optional[Tuple[float, float]]:
    suffix = {
        "forward": "вперед",
        "backward": "назад",
        "left": "влево",
        "right": "вправо",
    }.get(str(direction or "forward"), "вперед")
    proxy_snapshot = dict(snapshot)
    proxy_snapshot["operator_instruction"] = f"{max(1, int(steps))} шага {suffix}"
    return _relative_goal_from_instruction(proxy_snapshot)


def _infer_speed_action(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw:
        return None
    agent = snapshot.get("agent", {}) or {}
    current = _safe_float(agent.get("move_speed"), 3.0)
    if any(token in raw for token in _SLOW_PATTERNS):
        return {
            "type": "set_move_speed",
            "speed": round(_clamp(current * 0.58, 0.85, 6.0), 3),
            "preset": "slow",
        }
    if any(token in raw for token in _FAST_PATTERNS):
        return {
            "type": "set_move_speed",
            "speed": round(_clamp(current * 1.35, 0.85, 6.5), 3),
            "preset": "fast",
        }
    return None


def _infer_emotion_action(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw:
        return None
    if any(token in raw for token in _CALM_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "calm",
            "fear": 0.06,
            "curiosity": 0.34,
            "drive": "idle",
            "thought": "Я спокоен и собран.",
        }
    if any(token in raw for token in _FOCUS_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "focus",
            "fear": 0.08,
            "curiosity": 0.42,
            "drive": "idle",
            "thought": "Я сосредоточен на задаче.",
        }
    if any(token in raw for token in _CURIOUS_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "curious",
            "fear": 0.08,
            "curiosity": 0.82,
            "drive": "explore",
            "thought": "Мне интересно изучить обстановку.",
        }
    if any(token in raw for token in _BRAVE_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "brave",
            "fear": 0.03,
            "curiosity": 0.58,
            "drive": "explore",
            "thought": "Я действую уверенно.",
        }
    if any(token in raw for token in _REST_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "rest",
            "fear": 0.04,
            "curiosity": 0.22,
            "drive": "rest",
            "energy_delta": 8.0,
            "thought": "Я делаю паузу и восстанавливаюсь.",
        }
    if any(token in raw for token in _ENERGIZE_PATTERNS):
        return {
            "type": "tune_emotion",
            "preset": "energize",
            "fear": 0.08,
            "curiosity": 0.74,
            "drive": "explore",
            "energy_delta": 10.0,
            "thought": "Я чувствую прилив энергии.",
        }
    return None


def _infer_sit_sequence(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw or not any(token in raw for token in _SIT_PATTERNS):
        return None
    landmark = _extract_landmark_key(raw, snapshot)
    if not landmark:
        landmark = _nearest_landmark_key(snapshot, ("morpheus_chair", "neo_chair"))
    if not landmark:
        return None
    return {
        "type": "sequence",
        "steps": [
            {"type": "move_to_landmark", "landmark": landmark},
            {"type": "wait", "ticks": 96},
        ],
    }


def _infer_hold_sequence(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw:
        return None
    hold_idx = -1
    for token in _HOLD_THERE_PATTERNS:
        idx = raw.find(_normalize_text(token))
        if idx >= 0 and (hold_idx < 0 or idx < hold_idx):
            hold_idx = idx
    if hold_idx <= 0:
        return None
    prefix = raw[:hold_idx].strip(" ,.-")
    if not prefix:
        return None
    proxy_snapshot = dict(snapshot)
    proxy_snapshot["operator_instruction"] = prefix
    first = _infer_single_operator_control(proxy_snapshot)
    if not isinstance(first, dict):
        return None
    if str(first.get("type") or "") in {"stop", "wait", "remember_note", "set_move_speed"}:
        return None
    return {
        "type": "sequence",
        "steps": [
            first,
            {"type": "wait", "ticks": _extract_wait_ticks(raw)},
        ],
    }


def _infer_direction_sequence(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw or " и " not in raw:
        return None
    ordered: List[Tuple[int, str]] = []
    for direction, tokens in _MOVE_DIRECTION_TOKENS.items():
        best_idx = -1
        for token in tokens:
            idx = raw.find(_normalize_text(token))
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx = idx
        if best_idx >= 0:
            ordered.append((best_idx, direction))
    ordered.sort(key=lambda row: row[0])
    unique_dirs: List[str] = []
    for _, direction in ordered:
        if direction not in unique_dirs:
            unique_dirs.append(direction)
    if len(unique_dirs) < 2:
        return None
    steps_count = _extract_step_count(raw)
    seq_steps: List[Dict[str, Any]] = []
    for direction in unique_dirs[:3]:
        goal = _goal_from_direction(snapshot, direction, steps_count)
        if goal is None:
            continue
        seq_steps.append({"type": "move", "goal": goal})
    if len(seq_steps) < 2:
        return None
    return {"type": "sequence", "steps": seq_steps}


def _infer_split_sequence(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw or not _SEQUENCE_SPLIT_RE.search(raw):
        return None
    parts = [part.strip(" ,.-") for part in _SEQUENCE_SPLIT_RE.split(raw) if part.strip(" ,.-")]
    if len(parts) < 2:
        return None
    seq_steps: List[Dict[str, Any]] = []
    for part in parts[:4]:
        proxy_snapshot = dict(snapshot)
        proxy_snapshot["operator_instruction"] = part
        ctrl = _infer_single_operator_control(proxy_snapshot)
        if not isinstance(ctrl, dict):
            return None
        seq_steps.append(ctrl)
    return {"type": "sequence", "steps": seq_steps}


def _infer_single_operator_control(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = snapshot.get("operator_instruction")
    emotion_action = _infer_emotion_action(snapshot)
    if emotion_action is not None:
        return emotion_action
    speed_action = _infer_speed_action(snapshot)
    if speed_action is not None:
        return speed_action
    if _is_stop_instruction(text):
        return {"type": "stop"}
    landmark_action = _infer_landmark_action(snapshot)
    if landmark_action is not None:
        return landmark_action
    face_direction = _infer_face_direction_action(snapshot)
    if face_direction is not None:
        return face_direction
    raw = _normalize_text(text)
    if any(token in raw for token in _WAIT_PATTERNS):
        return {"type": "wait", "ticks": _extract_wait_ticks(text)}
    remember_text = _extract_remember_text(text)
    if remember_text:
        return {"type": "remember_note", "text": remember_text}
    goal = _relative_goal_from_instruction(snapshot)
    if goal is not None:
        return {"type": "move", "goal": goal}
    return None


def summarize_control(control: Optional[Dict[str, Any]]) -> str:
    if not isinstance(control, dict):
        return "unknown"
    ctype = str(control.get("type") or control.get("name") or "unknown")
    args = control.get("args") if isinstance(control.get("args"), dict) else {}
    if ctype == "sequence":
        steps = [summarize_control(step) for step in list(control.get("steps", []) or []) if isinstance(step, dict)]
        return " -> ".join(steps) if steps else "sequence"
    if ctype == "move_to_landmark":
        return f"move_to_landmark({str(control.get('landmark') or args.get('landmark') or '?')})"
    if ctype == "face_landmark":
        return f"face_landmark({str(control.get('landmark') or args.get('landmark') or '?')})"
    if ctype == "face_direction":
        return f"face_direction({str(control.get('direction') or args.get('direction') or '?')})"
    if ctype == "wait":
        return f"wait({int(_safe_float(control.get('ticks', args.get('ticks', 48)), 48.0))})"
    if ctype == "remember_note":
        return f"remember_note({str(control.get('text') or args.get('text') or '').strip()[:48]})"
    if ctype == "set_move_speed":
        return f"set_move_speed({round(_safe_float(control.get('speed', args.get('speed')), 0.0), 2)})"
    if ctype == "tune_emotion":
        pieces: List[str] = []
        preset = str(control.get("preset") or args.get("preset") or "").strip()
        if preset:
            pieces.append(preset)
        for key in ("fear", "curiosity", "energy_delta", "drive"):
            value = control.get(key, args.get(key))
            if value is None or value == "":
                continue
            pieces.append(f"{key}={value}")
        return "tune_emotion(" + ", ".join(pieces) + ")"
    if ctype == "move":
        goal = control.get("goal")
        if isinstance(goal, tuple) and len(goal) == 2:
            return f"move({round(float(goal[0]), 2)}, {round(float(goal[1]), 2)})"
        return "move"
    return ctype


def _infer_face_direction_action(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = _normalize_text(snapshot.get("operator_instruction"))
    if not raw:
        return None
    for direction, patterns in _FACE_RELATIVE_PATTERNS.items():
        for token in patterns:
            if _normalize_text(token) in raw:
                return {"type": "face_direction", "direction": direction}
    return None


def _tool_specs(landmarks: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    landmark_keys = sorted(str(k) for k in landmarks.keys())
    return [
        {"name": "stop", "description": "Остановить агента на месте и сбросить текущее движение.", "args": {}},
        {"name": "move_to_landmark", "description": "Отправить агента к ориентиру комнаты.", "args": {"landmark": landmark_keys}},
        {"name": "move_relative", "description": "Сделать несколько шагов относительно текущего взгляда агента.", "args": {"direction": ["forward", "backward", "left", "right"], "steps": "1..12"}},
        {"name": "face_landmark", "description": "Повернуть агента к ориентиру без перемещения.", "args": {"landmark": landmark_keys}},
        {"name": "face_direction", "description": "Повернуть агента относительно текущего взгляда.", "args": {"direction": ["forward", "backward", "left", "right"]}},
        {"name": "set_move_speed", "description": "Изменить скорость ходьбы агента.", "args": {"speed": "0.85..6.5"}},
        {"name": "wait", "description": "Заставить агента стоять на месте некоторое время.", "args": {"ticks": "12..240"}},
        {"name": "remember_note", "description": "Сохранить короткую заметку в память агента.", "args": {"text": "short string"}},
        {"name": "tune_emotion", "description": "Изменить эмоциональный фон, уверенность, любопытство, восстановление или драйв агента.", "args": {"preset": ["calm", "focus", "curious", "brave", "rest", "energize"], "fear": "0..1", "curiosity": "0..1", "energy_delta": "-20..20", "drive": ["idle", "explore", "rest", "heal", "seek_safety"], "thought": "short string"}},
    ]


def _memory_tail_lines(memory_tail: Iterable[Any], *, limit: int = 6) -> List[str]:
    lines: List[str] = []
    for ev in list(memory_tail or [])[-max(1, int(limit)):]:
        if isinstance(ev, dict):
            etype = str(ev.get("type") or ev.get("etype") or "event")
            data = ev.get("data")
            if isinstance(data, dict) and data:
                payload = ", ".join(f"{k}={v}" for k, v in list(data.items())[:3])
                lines.append(f"{etype}: {payload}")
            else:
                lines.append(etype)
            continue
        etype = str(getattr(ev, "etype", getattr(ev, "type", "event")))
        data = getattr(ev, "data", {}) or {}
        if isinstance(data, dict) and data:
            payload = ", ".join(f"{k}={v}" for k, v in list(data.items())[:3])
            lines.append(f"{etype}: {payload}")
        else:
            lines.append(etype)
    return lines


def _belief_lines(brain: Any, *, limit: int = 5) -> List[Dict[str, Any]]:
    beliefs = getattr(brain, "beliefs", []) or []
    out: List[Dict[str, Any]] = []
    for row in list(beliefs)[-max(1, int(limit)):]:
        if isinstance(row, dict):
            out.append({
                "if": str(row.get("if") or row.get("condition") or ""),
                "then": str(row.get("then") or row.get("conclusion") or ""),
                "strength": round(_clamp(_safe_float(row.get("strength", 0.5), 0.5), 0.0, 1.0), 2),
            })
            continue
        out.append({
            "if": str(getattr(row, "condition", "")),
            "then": str(getattr(row, "conclusion", "")),
            "strength": round(_clamp(_safe_float(getattr(row, "strength", 0.5), 0.5), 0.0, 1.0), 2),
        })
    return [row for row in out if row["if"] and row["then"]]


def _post_json(url: str, payload: Dict[str, Any], *, timeout: float) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout))) as resp:
        raw = resp.read().decode("utf-8", "replace")
    return json.loads(raw or "{}")


def _get_json(url: str, *, timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout))) as resp:
        raw = resp.read().decode("utf-8", "replace")
    return json.loads(raw or "{}")


def _extract_json_block(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def discover_local_models(*, host: Optional[str] = None, timeout: float = 1.8) -> List[str]:
    base = str(host or DEFAULT_OLLAMA_HOST).rstrip("/")
    try:
        raw = _get_json(f"{base}/api/tags", timeout=timeout)
    except Exception:
        return []
    models = raw.get("models", [])
    if not isinstance(models, list):
        return []
    out: List[str] = []
    for row in models:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name:
            out.append(name)
    return out


def pick_default_model(*, preferred: Optional[str] = None, host: Optional[str] = None) -> str:
    for candidate in (
        preferred,
        os.environ.get("CSEN_OLLAMA_MODEL"),
        os.environ.get("OLLAMA_MODEL"),
    ):
        if candidate:
            return str(candidate).strip()
    models = discover_local_models(host=host)
    if models:
        lowered = {m.casefold(): m for m in models}
        for wanted in DEFAULT_MODEL_PRIORITY:
            if wanted.casefold() in lowered:
                return lowered[wanted.casefold()]
        return str(models[0])
    return DEFAULT_MODEL_PRIORITY[0]


@dataclass(frozen=True)
class CoachAdvice:
    model: str
    thought: str
    lesson: str
    goal: Optional[Tuple[float, float]]
    belief: Optional[Dict[str, Any]]
    behavior: Dict[str, Optional[float]]
    reward_hint: float
    raw: Dict[str, Any]
    speech: str = ""


def infer_operator_goal(snapshot: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    return _relative_goal_from_instruction(snapshot)


def infer_operator_control(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for resolver in (
        _infer_sit_sequence,
        _infer_hold_sequence,
        _infer_direction_sequence,
        _infer_split_sequence,
        _infer_single_operator_control,
    ):
        control = resolver(snapshot)
        if control is not None:
            return control
    return None


def build_training_snapshot(
    agent: Any,
    world: Any,
    room: Any,
    *,
    active_lesson: Optional[Dict[str, Any]] = None,
    operator_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    bounds = room.bounds_for(world)
    cx, cy = bounds.center
    white_room = _is_white_room(room)
    if white_room:
        landmarks = {
            "floor_center": {"x": round(cx, 2), "y": round(cy, 2)},
            "north_wall": {"x": round(cx, 2), "y": round(bounds.top + 1.6, 2)},
            "south_wall": {"x": round(cx, 2), "y": round(bounds.bottom - 1.6, 2)},
            "west_wall": {"x": round(bounds.left + 1.6, 2), "y": round(cy, 2)},
            "east_wall": {"x": round(bounds.right - 1.6, 2), "y": round(cy, 2)},
        }
    else:
        landmarks = {
            "morpheus_chair": {
                "x": round(bounds.left + bounds.width * 0.28, 2),
                "y": round(bounds.top + bounds.height * 0.62, 2),
            },
            "neo_chair": {
                "x": round(bounds.left + bounds.width * 0.72, 2),
                "y": round(bounds.top + bounds.height * 0.62, 2),
            },
            "coffee_table": {"x": round(cx, 2), "y": round(bounds.top + bounds.height * 0.56, 2)},
            "floor_center": {"x": round(cx, 2), "y": round(cy, 2)},
            "green_panel": {"x": round(cx, 2), "y": round(bounds.top + bounds.height * 0.24, 2)},
            "lamp": {
                "x": round(bounds.left + bounds.width * 0.16, 2),
                "y": round(bounds.top + bounds.height * 0.26, 2),
            },
        }
    tools = _tool_specs(landmarks)

    brain = getattr(agent, "brain", None)
    rules = getattr(brain, "behavior_rules", None)
    room_tick = int(getattr(world, "tick_count", getattr(world, "ticks", getattr(world, "time", 0))) or 0)
    snapshot = {
        "tick": room_tick,
        "room": {
            "theme": "blank_white_room" if white_room else "morpheus_construct",
            "safe": True,
            "bounds": {
                "left": round(bounds.left, 2),
                "top": round(bounds.top, 2),
                "right": round(bounds.right, 2),
                "bottom": round(bounds.bottom, 2),
                "center_x": round(cx, 2),
                "center_y": round(cy, 2),
                "safe_radius": round(bounds.safe_radius, 2),
            },
            "landmarks": landmarks,
        },
        "agent": {
            "id": str(getattr(agent, "id", getattr(agent, "agent_id", "agent"))),
            "name": str(getattr(agent, "name", "Agent")),
            "pos": {
                "x": round(_safe_float(getattr(agent, "x", cx), cx), 2),
                "y": round(_safe_float(getattr(agent, "y", cy), cy), 2),
            },
            "goal": {
                "x": round(_safe_float(getattr(agent, "goal_x", cx), cx), 2),
                "y": round(_safe_float(getattr(agent, "goal_y", cy), cy), 2),
            },
            "facing": {
                "x": round(_agent_facing_xy(agent)[0], 4),
                "y": round(_agent_facing_xy(agent)[1], 4),
            },
            "health": round(_safe_float(getattr(agent, "health", 100.0), 100.0), 2),
            "energy": round(_pct(getattr(agent, "energy", 100.0), 100.0), 2),
            "hunger": round(_pct(getattr(agent, "hunger", 0.0), 0.0), 2),
            "fear": round(_clamp(_safe_float(getattr(agent, "fear", 0.0), 0.0), 0.0, 1.0), 3),
            "age_ticks": int(_safe_float(getattr(agent, "age_ticks", 0), 0.0)),
            "move_speed": round(_safe_float(getattr(agent, "move_speed", 3.0), 3.0), 3),
            "step_unit": round(_estimate_step_unit(agent), 3),
            "tags": [str(t) for t in list(getattr(agent, "tags", []) or [])],
        },
        "brain": {
            "current_drive": str(getattr(brain, "current_drive", "idle")) if brain is not None else "idle",
            "survival_score": round(_clamp(_safe_float(getattr(brain, "survival_score", 1.0), 1.0), 0.0, 1.0), 3) if brain is not None else 1.0,
            "curiosity_charge": round(_clamp(_safe_float(getattr(brain, "curiosity_charge", 0.5), 0.5), 0.0, 1.0), 3) if brain is not None else 0.5,
            "fear_level": round(_clamp(_safe_float(getattr(brain, "fear_level", getattr(agent, "fear", 0.0)), getattr(agent, "fear", 0.0)), 0.0, 1.0), 3) if brain is not None else round(_clamp(_safe_float(getattr(agent, "fear", 0.0), 0.0), 0.0, 1.0), 3),
            "last_thought": str(getattr(brain, "last_thought", "")) if brain is not None else "",
            "behavior_rules": {
                "avoid_hazard_radius": round(_safe_float(getattr(rules, "avoid_hazard_radius", 6.0), 6.0), 3) if rules is not None else 6.0,
                "healing_zone_seek_priority": round(_clamp(_safe_float(getattr(rules, "healing_zone_seek_priority", 0.5), 0.5), 0.0, 1.0), 3) if rules is not None else 0.5,
                "stick_with_ally_if_fear_above": round(_clamp(_safe_float(getattr(rules, "stick_with_ally_if_fear_above", 0.7), 0.7), 0.0, 1.0), 3) if rules is not None else 0.7,
                "exploration_bias": round(_clamp(_safe_float(getattr(rules, "exploration_bias", 0.2), 0.2), 0.0, 1.0), 3) if rules is not None else 0.2,
            },
            "beliefs": _belief_lines(brain),
            "recent_memory": _memory_tail_lines(getattr(brain, "memory_tail", []) if brain is not None else []),
        },
        "tools": tools,
        "active_lesson": dict(active_lesson or {}) if isinstance(active_lesson, dict) else None,
        "operator_instruction": (str(operator_instruction or "").strip() or None),
        "brain_study": summarize_brain_profile(getattr(brain, "ollama_brain_profile", {}) if brain is not None else {}),
    }
    live_dialogue = list(getattr(brain, "ollama_dialogue_tail", []) or []) if brain is not None else []
    live_examples = list(getattr(brain, "ollama_command_examples", []) or []) if brain is not None else []
    snapshot["behavior_context"] = build_behavior_context(snapshot, live_dialogue=live_dialogue, live_examples=live_examples)
    return snapshot


class OllamaCoach:
    def __init__(
        self,
        *,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
        keep_alive: str = "20m",
    ) -> None:
        self.host = str(host or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout = max(2.0, float(timeout))
        self.keep_alive = str(keep_alive or "20m")
        self.model = pick_default_model(preferred=model, host=self.host)

    def describe(self) -> str:
        return f"{self.model} @ {self.host}"

    def refresh_model(self) -> str:
        self.model = pick_default_model(preferred=self.model, host=self.host)
        return self.model

    def request_advice(self, snapshot: Dict[str, Any]) -> CoachAdvice:
        prompt = self._build_prompt(snapshot)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.25,
                "top_p": 0.9,
                "num_predict": 240,
            },
        }
        raw = _post_json(f"{self.host}/api/generate", payload, timeout=self.timeout)
        response_text = str(raw.get("response") or "").strip()
        data = _extract_json_block(response_text)
        if not data:
            raise RuntimeError("Ollama returned an empty or non-JSON lesson")
        bounds = (((snapshot.get("room") or {}).get("bounds")) or {})
        relative_goal = _relative_goal_from_instruction(snapshot)
        action = self._sanitize_action(data.get("action"), snapshot)
        model_goal = self._sanitize_goal(data.get("goal"), bounds)
        action_goal = self._goal_from_action(action, snapshot)
        if relative_goal is not None:
            data["_goal_source"] = "operator_relative"
        if action is not None:
            data["_action"] = action
        speech = str(data.get("speech") or data.get("agent_speech") or data.get("reply") or "").strip()
        return CoachAdvice(
            model=str(raw.get("model") or self.model),
            thought=str(data.get("thought") or "").strip(),
            lesson=str(data.get("lesson") or "Сделай ещё один спокойный шаг и запомни маршрут.").strip(),
            goal=relative_goal if relative_goal is not None else (action_goal if action_goal is not None else model_goal),
            belief=self._sanitize_belief(data.get("belief")),
            behavior=self._sanitize_behavior(data.get("behavior")),
            reward_hint=_clamp(_safe_float(data.get("reward_hint", 0.12), 0.12), -1.0, 1.0),
            raw=data,
            speech=speech,
        )

    def _sanitize_goal(self, goal_raw: Any, bounds: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if not bounds:
            return None
        if not isinstance(goal_raw, dict):
            return None
        if "x" not in goal_raw or "y" not in goal_raw:
            return None
        gx = _safe_float(goal_raw.get("x"), _safe_float(bounds.get("center_x"), 0.0))
        gy = _safe_float(goal_raw.get("y"), _safe_float(bounds.get("center_y"), 0.0))
        left = _safe_float(bounds.get("left"), 0.0)
        top = _safe_float(bounds.get("top"), 0.0)
        right = _safe_float(bounds.get("right"), left + 1.0)
        bottom = _safe_float(bounds.get("bottom"), top + 1.0)
        return (
            _clamp(gx, left + 1.2, right - 1.2),
            _clamp(gy, top + 1.2, bottom - 1.2),
        )

    def _sanitize_belief(self, belief_raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(belief_raw, dict):
            return None
        cond = str(belief_raw.get("if") or belief_raw.get("condition") or "").strip()
        concl = str(belief_raw.get("then") or belief_raw.get("conclusion") or "").strip()
        if not cond or not concl:
            return None
        return {
            "if": cond,
            "then": concl,
            "strength": round(_clamp(_safe_float(belief_raw.get("strength", 0.58), 0.58), 0.0, 1.0), 3),
        }

    def _sanitize_behavior(self, behavior_raw: Any) -> Dict[str, Optional[float]]:
        src = behavior_raw if isinstance(behavior_raw, dict) else {}
        return {
            "avoid_hazard_radius": _clamp(_safe_float(src.get("avoid_hazard_radius", 6.0), 6.0), 2.0, 20.0) if "avoid_hazard_radius" in src else None,
            "healing_zone_seek_priority": _clamp(_safe_float(src.get("healing_zone_seek_priority", 0.5), 0.5), 0.0, 1.0) if "healing_zone_seek_priority" in src else None,
            "stick_with_ally_if_fear_above": _clamp(_safe_float(src.get("stick_with_ally_if_fear_above", 0.7), 0.7), 0.0, 1.0) if "stick_with_ally_if_fear_above" in src else None,
            "exploration_bias": _clamp(_safe_float(src.get("exploration_bias", 0.2), 0.2), 0.0, 1.0) if "exploration_bias" in src else None,
        }

    def _sanitize_action(self, action_raw: Any, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(action_raw, dict):
            return None
        name = _normalize_text(action_raw.get("name"))
        args_raw = action_raw.get("args")
        args = dict(args_raw) if isinstance(args_raw, dict) else {}
        landmarks = _room_landmarks(snapshot)
        if name == "stop":
            return {"name": "stop", "args": {}}
        if name == "move_to_landmark":
            landmark = str(args.get("landmark") or "").strip()
            if landmark in landmarks:
                return {"name": "move_to_landmark", "args": {"landmark": landmark}}
            return None
        if name == "move_relative":
            direction = _normalize_text(args.get("direction"))
            if direction not in {"forward", "backward", "left", "right"}:
                return None
            steps = int(_clamp(_safe_float(args.get("steps", 1), 1.0), 1.0, 12.0))
            return {"name": "move_relative", "args": {"direction": direction, "steps": steps}}
        if name == "face_landmark":
            landmark = str(args.get("landmark") or "").strip()
            if landmark in landmarks:
                return {"name": "face_landmark", "args": {"landmark": landmark}}
            return None
        if name == "face_direction":
            direction = _normalize_text(args.get("direction"))
            if direction not in {"forward", "backward", "left", "right"}:
                return None
            return {"name": "face_direction", "args": {"direction": direction}}
        if name == "set_move_speed":
            speed = round(_clamp(_safe_float(args.get("speed", 3.0), 3.0), 0.85, 6.5), 3)
            return {"name": "set_move_speed", "args": {"speed": speed}}
        if name == "wait":
            ticks = int(_clamp(_safe_float(args.get("ticks", 48), 48.0), 12.0, 240.0))
            return {"name": "wait", "args": {"ticks": ticks}}
        if name == "remember_note":
            text = str(args.get("text") or "").strip()
            if not text:
                return None
            return {"name": "remember_note", "args": {"text": text[:240]}}
        if name in {"tune_emotion", "set_emotion_state"}:
            preset = _normalize_text(args.get("preset"))
            drive = str(args.get("drive") or "").strip()
            thought = str(args.get("thought") or "").strip()
            payload: Dict[str, Any] = {}
            if preset in {"calm", "focus", "curious", "brave", "rest", "energize"}:
                payload["preset"] = preset
            if "fear" in args:
                payload["fear"] = round(_clamp(_safe_float(args.get("fear"), 0.1), 0.0, 1.0), 3)
            if "curiosity" in args:
                payload["curiosity"] = round(_clamp(_safe_float(args.get("curiosity"), 0.5), 0.0, 1.0), 3)
            if "energy_delta" in args:
                payload["energy_delta"] = round(_clamp(_safe_float(args.get("energy_delta"), 0.0), -20.0, 20.0), 2)
            if drive in {"idle", "explore", "rest", "heal", "seek_safety", "stay_with_ally"}:
                payload["drive"] = drive
            if thought:
                payload["thought"] = thought[:140]
            return {"name": "tune_emotion", "args": payload} if payload else None
        return None

    def _goal_from_action(self, action: Optional[Dict[str, Any]], snapshot: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if not isinstance(action, dict):
            return None
        name = str(action.get("name") or "")
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        if name == "move_to_landmark":
            landmark = str(args.get("landmark") or "").strip()
            target = _room_landmarks(snapshot).get(landmark)
            if isinstance(target, dict):
                return (_safe_float(target.get("x"), 0.0), _safe_float(target.get("y"), 0.0))
            return None
        if name == "move_relative":
            direction = str(args.get("direction") or "forward")
            steps = int(_clamp(_safe_float(args.get("steps", 1), 1.0), 1.0, 12.0))
            suffix = {
                "forward": "вперед",
                "backward": "назад",
                "left": "влево",
                "right": "вправо",
            }.get(direction, "вперед")
            proxy_snapshot = dict(snapshot)
            proxy_snapshot["operator_instruction"] = f"{steps} шага {suffix}" if steps != 1 else f"иди {suffix}"
            return _relative_goal_from_instruction(proxy_snapshot)
        return None

    def _build_prompt(self, snapshot: Dict[str, Any]) -> str:
        room = snapshot.get("room", {}) or {}
        agent = snapshot.get("agent", {}) or {}
        brain = snapshot.get("brain", {}) or {}
        tools = list(snapshot.get("tools", []) or [])
        behavior_context = snapshot.get("behavior_context", {}) or {}
        brain_study = snapshot.get("brain_study", {}) or {}
        active_lesson = snapshot.get("active_lesson")
        operator_instruction = str(snapshot.get("operator_instruction") or "").strip()
        room_theme = str(room.get("theme") or "")
        white_room = room_theme == "blank_white_room"
        room_rule = (
            "Комната абсолютно пустая: только белые стены, белый пол и открытое пространство. "
            "Не упоминай кресла, стол, лампу, панели и другой интерьер.\n"
            if white_room
            else "Фразы вида «сядь на стул/кресло» в этой комнате означают подойти к ближайшему креслу и остаться там.\n"
        )
        operator_block = ""
        if operator_instruction:
            operator_block = (
                "Приоритетный запрос оператора. Его нужно выполнить в первую очередь, "
                "если это безопасно и возможно внутри комнаты:\n"
                f"{operator_instruction}\n"
            )
        return (
            "Ты локальный AI-тренер одного лабораторного агента в безопасной комнате Морфеуса.\n"
            "Комната безопасна: здесь нет реальной угрозы, боли, боя, хищников и токсичных зон.\n"
            "Твоя задача: обучать агента спокойному перемещению, концентрации, закреплению полезных правил и осмысленному исследованию комнаты.\n"
            "Пиши мысль и урок только на русском языке, без английских слов и без Markdown.\n"
            "Поле speech — это речь самого агента оператору от первого лица. Делай её осмысленной, связанной с вопросом и состоянием памяти.\n"
            "Если оператор просит шаги вперед, назад, влево или вправо, ориентируйся на agent.facing и agent.step_unit.\n"
            "Используй behavior_context.current_profile как психопрофиль текущего мозга.\n"
            "Если есть brain_study, считай его результатом длительного изучения полного мозга агента и используй как главный долгосрочный профиль характера, реакции на команды и эмоциональные рычаги.\n"
            "Если есть похожие command_examples, интерпретируй похожую команду тем же способом.\n"
            "Если есть style_priors из архивных мозгов, используй их как мягкие поведенческие priors, но не спорь с прямой командой оператора.\n"
            "Фразы вида «подойди ... и стой там» это не stop, а последовательность move_to_landmark -> wait.\n"
            f"{room_rule}"
            "Если оператор просит изменить настроение, уверенность, спокойствие, любопытство, энергию или фокус агента, используй action=tune_emotion.\n"
            "Если в command_examples уже есть успешная похожая команда, повторяй тот же action family и не импровизируй.\n"
            "Если запрос можно выразить как инструментальное действие, заполни поле action и выбери инструмент из tools.\n"
            "Нельзя выводить цель за пределы комнаты.\n"
            "Если у агента уже есть активный урок, не повторяй его буквально, а дай следующий естественный шаг.\n"
            f"{operator_block}"
            "Ответь только JSON-объектом без Markdown и без пояснений.\n"
            "Строгая схема ответа:\n"
            "{\n"
            '  "thought": "короткая мысль агента на русском, до 12 слов",\n'
            '  "speech": "осмысленная фраза агента оператору от первого лица, до 22 слов",\n'
            '  "lesson": "короткая команда тренера на русском, до 18 слов",\n'
            '  "action": {"name": "tool_name", "args": {...}} или null,\n'
            '  "goal": {"x": 0, "y": 0},\n'
            '  "belief": {"if": "условие", "then": "вывод", "strength": 0.6} или null,\n'
            '  "behavior": {\n'
            '    "avoid_hazard_radius": число или null,\n'
            '    "healing_zone_seek_priority": число или null,\n'
            '    "stick_with_ally_if_fear_above": число или null,\n'
            '    "exploration_bias": число или null\n'
            "  },\n"
            '  "reward_hint": число от -1 до 1\n'
            "}\n"
            "Контекст комнаты и агента:\n"
            f"{json.dumps({'room': room, 'agent': agent, 'brain': brain, 'brain_study': brain_study, 'behavior_context': behavior_context, 'tools': tools, 'active_lesson': active_lesson}, ensure_ascii=False, sort_keys=True)}\n"
        )


def apply_advice_to_agent(
    agent: Any,
    world: Any,
    room: Any,
    advice: CoachAdvice,
) -> Dict[str, Any]:
    tick = int(getattr(world, "tick_count", getattr(world, "ticks", getattr(world, "time", 0))) or 0)
    name = str(getattr(agent, "name", getattr(agent, "id", "agent")))
    agent_id = str(getattr(agent, "id", getattr(agent, "agent_id", "agent")))
    applied_goal: Optional[Tuple[float, float]] = None

    if advice.goal is not None:
        gx, gy = advice.goal
        if hasattr(room, "clamp_point_for_agent"):
            gx, gy = room.clamp_point_for_agent(world, agent_id, gx, gy)
        if hasattr(agent, "set_goal"):
            try:
                agent.set_goal(
                    gx,
                    gy,
                    world_size=(float(getattr(world, "width", gx)), float(getattr(world, "height", gy))),
                    reason=f"ollama_lesson:{advice.lesson}",
                    tick=tick,
                )
            except Exception:
                agent.goal_x = float(gx)
                agent.goal_y = float(gy)
        else:
            agent.goal_x = float(gx)
            agent.goal_y = float(gy)
        applied_goal = (float(gx), float(gy))

    belief_payload: Optional[Dict[str, Any]] = None
    if isinstance(advice.belief, dict):
        belief_payload = dict(advice.belief)
    elif advice.belief is not None:
        belief_text = str(advice.belief).strip()
        if belief_text:
            belief_payload = {"if": "ollama_note", "then": belief_text, "strength": 0.55}

    brain = getattr(agent, "brain", None)
    if brain is not None:
        if advice.thought:
            try:
                brain.last_thought = str(advice.thought)
            except Exception:
                pass
        if belief_payload is not None and hasattr(brain, "add_belief"):
            try:
                brain.add_belief(belief_payload)
            except Exception:
                pass
        rules = getattr(brain, "behavior_rules", None)
        if rules is not None:
            for key, value in advice.behavior.items():
                if value is None:
                    continue
                try:
                    setattr(rules, key, float(value))
                except Exception:
                    continue

    payload: Dict[str, Any] = {
        "model": advice.model,
        "lesson": advice.lesson,
        "thought": advice.thought,
        "speech": advice.speech,
        "reward_hint": advice.reward_hint,
    }
    if applied_goal is not None:
        payload["goal"] = (round(applied_goal[0], 2), round(applied_goal[1], 2))
    if belief_payload is not None:
        payload["belief"] = dict(belief_payload)

    remember = getattr(agent, "_remember_event", None)
    if callable(remember):
        try:
            remember(
                "ollama_lesson",
                payload,
                tick=tick,
                actor="ollama",
                pos=applied_goal,
            )
        except Exception:
            pass
    elif brain is not None and hasattr(brain, "add_memory"):
        try:
            brain.add_memory({"type": "ollama_lesson", "tick": tick, "data": payload})
        except Exception:
            pass

    if hasattr(world, "add_chat_line"):
        try:
            goal_text = ""
            if applied_goal is not None:
                goal_text = f" -> ({applied_goal[0]:.1f}, {applied_goal[1]:.1f})"
            spoken_text = str(advice.speech or advice.lesson or "").strip()
            world.add_chat_line(f"[agent:{advice.model}] {name}: {spoken_text}{goal_text}")
        except Exception:
            pass
    if hasattr(world, "add_event"):
        try:
            event = {
                "type": "ollama_lesson",
                "tick": tick,
                "who": name,
                "agent_id": agent_id,
                "model": advice.model,
                "lesson": advice.lesson,
                "speech": advice.speech,
            }
            if applied_goal is not None:
                event["goal"] = {"x": round(applied_goal[0], 2), "y": round(applied_goal[1], 2)}
            world.add_event(event)
        except Exception:
            pass

    return {
        "goal": applied_goal,
        "lesson": advice.lesson,
        "thought": advice.thought,
        "speech": advice.speech,
        "reward_hint": advice.reward_hint,
        "model": advice.model,
    }


def apply_goal_feedback(
    agent: Any,
    world: Any,
    *,
    goal: Optional[Tuple[float, float]],
    lesson: str,
    model: str,
    reward: float,
    success: bool,
) -> None:
    tick = int(getattr(world, "tick_count", getattr(world, "ticks", getattr(world, "time", 0))) or 0)
    brain = getattr(agent, "brain", None)
    state = {
        "tick": tick,
        "pos": (float(getattr(agent, "x", 0.0)), float(getattr(agent, "y", 0.0))),
        "goal": goal,
        "health": float(getattr(agent, "health", 100.0)),
        "energy": _pct(getattr(agent, "energy", 100.0), 100.0),
        "hunger": _pct(getattr(agent, "hunger", 0.0), 0.0),
        "fear": _clamp(_safe_float(getattr(agent, "fear", 0.0), 0.0), 0.0, 1.0),
        "lesson": str(lesson or ""),
        "success": bool(success),
        "ollama_model": str(model or ""),
    }
    if brain is not None and hasattr(brain, "on_step"):
        try:
            brain.on_step(state=state, reward=float(reward))
        except Exception:
            pass

    payload = {
        "model": model,
        "lesson": lesson,
        "reward": round(float(reward), 4),
        "success": bool(success),
    }
    if goal is not None:
        payload["goal"] = (round(float(goal[0]), 2), round(float(goal[1]), 2))
    remember = getattr(agent, "_remember_event", None)
    if callable(remember):
        try:
            remember(
                "ollama_feedback",
                payload,
                tick=tick,
                actor="ollama",
                pos=goal,
            )
        except Exception:
            pass
