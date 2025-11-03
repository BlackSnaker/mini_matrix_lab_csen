# trainer_side.py
from __future__ import annotations

from typing import Dict, Any
from brain_io import save_brain


def export_brain(agent_id: str, brain, skills: Dict[str, Any]) -> str:
    """
    Экспортирует мозг и метаданные тренировки.
    Возвращает путь до сохранённого файла, чтобы можно было подхватить hot-swap.
    """
    meta = {"agent_id": agent_id, "skills": skills, "version": "csen-1"}
    path = f"brains/{agent_id}-latest.npz"
    # save_brain поддерживает meta и опциональный путь
    save_brain(brain, path, meta=meta)
    return path
