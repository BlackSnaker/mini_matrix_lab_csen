# server_side.py
from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple

from brain_io import load_brain
from world import push_global_event  # глобальный синк событий из world.py


# Глобальное хранилище активных мозгов
BRAINS: Dict[str, Any] = {}  # agent_id -> brain (объект с методом step(obs, dt))


def attach_brain(agent_id: str, key_or_path: str) -> bool:
    """
    Подключить мозг к агенту.
    key_or_path может быть как agent_id (ключ), так и файловый путь.
    """
    brain = load_brain(key_or_path)
    if brain is None:
        return False
    BRAINS[agent_id] = brain
    # Новый структурированный формат события (поддерживается и legacy-строка)
    push_global_event("brain_updated", agent_id=agent_id)
    return True


# ------------------------ Вспомогалки для генерации наблюдений ------------------------

def _vec_to_goal_norm(agent) -> Tuple[float, float]:
    dx = float(agent.goal_x) - float(agent.x)
    dy = float(agent.goal_y) - float(agent.y)
    L = math.hypot(dx, dy) or 1.0
    return (dx / L, dy / L)


def _nearest_animal_of(world, species_id: str, x: float, y: float, r: float = 15.0) -> Optional[Dict[str, Any]]:
    """
    Найти ближайшее животное указанного вида в радиусе r.
    Ожидается world.animals: dict[uid -> AnimalSim], у AnimalSim: x,y,hp,species.species_id,name,...
    """
    best = None
    best_d = r
    for ani in world.animals.values():
        if not ani.is_alive():
            continue
        sid = getattr(getattr(ani, "species", None), "species_id", None)
        if sid != species_id:
            continue
        d = math.hypot(ani.x - x, ani.y - y)
        if d <= best_d:
            best_d = d
            best = {
                "id": ani.uid,
                "pos": (ani.x, ani.y),
                "hp": getattr(ani, "hp", None),
                "dist": d,
                "tamed_by": getattr(ani, "tamed_by", None),
                "name": getattr(getattr(ani, "species", None), "name", species_id),
            }
    return best


def make_obs(world, agent) -> Dict[str, Any]:
    """
    Адаптер наблюдений под унифицированный action-space.
    Под world/agent ожидаются структуры из текущего server-side мира (см. world.py).
    """
    # В мире agent.health/agent.fear/agent.energy (а не hp)
    obs = {
        "self": {
            "hp": float(agent.health),
            "fear": float(agent.fear),
            "energy": float(agent.energy),
            "hunger": float(agent.hunger),
            "pos": (float(agent.x), float(agent.y)),
            "goal": (float(agent.goal_x), float(agent.goal_y)),
        },
        "vec_to_goal": _vec_to_goal_norm(agent),
        # пример: ищем ближайшего волка
        "nearest_wolf": _nearest_animal_of(world, "wolf", agent.x, agent.y, r=15.0),
        # при необходимости можешь расширять:
        # "nearest_food": _nearest_animal_of(world, "deer", agent.x, agent.y, r=20.0),
        # "danger_cloud_count": len(agent.danger_zones),
    }
    return obs


# ----------------------------- Применение действий мозга -----------------------------

def _apply_action(world, agent, action: Dict[str, Any]) -> None:
    """
    Унифицированный маппер действий в механику мира.
    Если в world нет метода apply_action, применяем безопасные дефолты.
    Поддерживаемые ключи:
      - {"set_goal": (x, y)}            → установить цель агенту (через API мира)
      - {"nudge": (dx, dy)}             → мягкое смещение цели (мини-шаг)
      - {"panic": true}                 → отступить к SAFE_POINT (будет перехвачено логикой агента)
    """
    if hasattr(world, "apply_action"):
        # Если у мира есть собственная реализация — используем её
        world.apply_action(agent, action)
        return

    # Дефолтная, безопасная реализация:
    if not isinstance(action, dict):
        return

    if "set_goal" in action:
        gx, gy = action["set_goal"]
        # используем штатный метод мира (даёт события/чат)
        if hasattr(world, "set_agent_goal"):
            world.set_agent_goal(agent.id, float(gx), float(gy))
        else:
            agent.goal_x = float(gx)
            agent.goal_y = float(gy)

    if "nudge" in action:
        dx, dy = action["nudge"]
        gx = float(agent.goal_x) + float(dx)
        gy = float(agent.goal_y) + float(dy)
        if hasattr(world, "set_agent_goal"):
            world.set_agent_goal(agent.id, gx, gy)
        else:
            agent.goal_x = gx
            agent.goal_y = gy

    if action.get("panic"):
        # Ничего напрямую не делаем: логика паники в агенте сама потянет SAFE_POINT.
        # Можем лишь слегка повысить страх — это учтётся мозгом/поведением.
        agent.fear = min(1.0, float(agent.fear) + 0.1)


# -------------------------------- Основные API-операции ------------------------------

def step_agent(world, agent, dt: float) -> None:
    """
    Один шаг для одного агента: наблюдение → действие → применение.
    """
    brain = BRAINS.get(agent.id)
    if brain is None:
        return
    obs = make_obs(world, agent)
    # У унифицированного интерфейса brains ожидается step(obs, dt) -> action(dict)
    action = brain.step(obs, float(dt))
    _apply_action(world, agent, action)


def hot_swap(agent_id: str, tag: str = "latest") -> bool:
    """
    Горячая замена мозга: подхватываем brains/{agent_id}-{tag}.npz
    """
    return attach_brain(agent_id, f"brains/{agent_id}-{tag}.npz")
