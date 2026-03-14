from __future__ import annotations

from typing import Any, Dict

from brain_io import save_brain


def _extract_taming_meta(brain: Any) -> Dict[str, int]:
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
    skills_out = dict(skills or {})
    skills_out.update(_extract_taming_meta(brain))
    setattr(brain, "trainer_side_meta", {"agent_id": agent_id, "skills": skills_out, "version": "csen-1"})
    save_brain(brain)
    return f"brains/{agent_id}.json"
