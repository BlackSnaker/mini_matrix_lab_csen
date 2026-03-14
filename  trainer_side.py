# trainer_side.py
from __future__ import annotations

from typing import Dict, Any
from brain_io import save_brain


def _extract_taming_meta(brain: Any) -> Dict[str, int]:
    """
    Возвращает агрегаты по событиям приручения/peer-обучения из памяти мозга.
    """
    out = {
        "tame_success_events": 0,
        "tame_progress_events": 0,
        "tame_peer_share_events": 0,
        "seek_partner_events": 0,
        "offspring_born_events": 0,
        "lineage_inherit_events": 0,
    }
    tail = list(getattr(brain, "memory_tail", []) or [])
    for ev in tail:
        etype = ""
        if hasattr(ev, "etype"):
            etype = str(getattr(ev, "etype", ""))
        elif isinstance(ev, dict):
            etype = str(ev.get("etype") or ev.get("type") or "")
        if etype in out:
            out[etype] += 1
    return out


def export_brain(agent_id: str, brain, skills: Dict[str, Any]) -> str:
    """
    Экспортирует мозг и метаданные тренировки.
    Возвращает путь до сохранённого файла, чтобы можно было подхватить hot-swap.
    """
    # Добавляем в экспорт и социальное обучение приручению, чтобы это отражалось
    # в артефактах side-trainer наравне с остальными трекерами.
    skills_out = dict(skills or {})
    skills_out.update(_extract_taming_meta(brain))
    setattr(brain, "trainer_side_meta", {"agent_id": agent_id, "skills": skills_out, "version": "csen-1"})
    save_brain(brain)
    return f"brains/{agent_id}.json"
