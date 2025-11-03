# bootstrap.py
from __future__ import annotations

import os
import time
import traceback

from world import World, Agent, WorldObject
from server_side import attach_brain as _attach_brain_raw, step_agent

# -----------------------------------------------------------------------------
# Вспомогательная обёртка для совместимости разных сигнатур attach_brain
# -----------------------------------------------------------------------------
def attach_brain_compat(world: World, agent_id: str, path: str) -> bool:
    """
    Пытается вызвать attach_brain в двух вариантах:
      1) attach_brain(agent_id, path)
      2) attach_brain(world, agent_id, path)
    Возвращает True при успехе (или если функция не возвращает ничего, но исключений нет).
    """
    try:
        ok = _attach_brain_raw(agent_id, path)  # вариант без world
        ok = True if ok is None else bool(ok)
        world.push_global_event("brain_updated", agent_id=agent_id)
        world.add_chat_line(f"[brain] Загружен мозг для {agent_id} из '{path}'")
        return ok
    except TypeError:
        # Попробуем сигнатуру с world
        try:
            ok = _attach_brain_raw(world, agent_id, path)  # вариант с world
            ok = True if ok is None else bool(ok)
            world.push_global_event("brain_updated", agent_id=agent_id)
            world.add_chat_line(f"[brain] Загружен мозг для {agent_id} из '{path}' (через world)")
            return ok
        except Exception as e:  # noqa: F841
            traceback.print_exc()
            world.add_chat_line(f"[brain:ERR] Не удалось загрузить мозг для {agent_id}: {e}")
            return False
    except Exception as e:  # noqa: F841
        traceback.print_exc()
        world.add_chat_line(f"[brain:ERR] Не удалось загрузить мозг для {agent_id}: {e}")
        return False


def main():
    # -------------------------------------------------------------------------
    # Создаём мир
    # -------------------------------------------------------------------------
    world = World(200.0, 200.0)

    # Агент(ы)
    world.add_agent(Agent("alice", "Алиса", 10, 10, 60, 60, persona="caring/support"))
    world.add_agent(Agent("bob",   "Боб",   30, 20, 70, 10, persona="protective"))

    # Safe/опасные зоны
    world.add_object(WorldObject("camp", "Лагерь", "safe",   100, 100, 12, comfort_level=0.9))
    world.add_object(WorldObject("fire", "Костёр", "hazard",  80, 100,  6, danger_level=0.6))

    # -------------------------------------------------------------------------
    # Загружаем «мозги» (горячее подключение)
    # -------------------------------------------------------------------------
    attach_brain_compat(world, "alice", "brains/alice-latest.npz")
    attach_brain_compat(world, "bob",   "brains/bob-latest.npz")

    # -------------------------------------------------------------------------
    # Главный цикл симуляции
    # -------------------------------------------------------------------------
    DT = 0.1
    SLEEP = 0.016  # ~60 FPS для HUD/рендера
    max_ticks_env = os.getenv("MAX_TICKS")
    MAX_TICKS = int(max_ticks_env) if (max_ticks_env and max_ticks_env.isdigit()) else None

    try:
        while True:
            # 1) шаг «мозга» → экшены (могут скорректировать цель/ускорения/паник-уход)
            for a in list(world.agents.values()):
                try:
                    step_agent(world, a, DT)
                except Exception as e:  # мозг не ответил/не загружен — не падаем
                    # Лог в чат и продолжаем: агент живёт на своих встроенных правилах (world.tick)
                    world.add_chat_line(f"[brain:WARN] {a.name}: step_agent пропущен ({e})")

            # 2) шаг мира (животные/агенты/смерти/чаты/ивенты/соц.обмен)
            world.tick()

            # 3) Экспорт синхро-пакета для 3D/HUD
            payload = world.export_for_engine3d()
            # TODO: отправь payload в свой транспорт (WebSocket/IPC/файл и т.п.)
            # send_payload(payload)

            # (необязательно) Небольшой троттлинг, чтобы не жечь 100% CPU
            time.sleep(SLEEP)

            # (необязательно) Ограничитель по тикам для тестов/CI
            if MAX_TICKS is not None and world.tick_count >= MAX_TICKS:
                break

    except KeyboardInterrupt:
        world.add_chat_line("[sys] Остановлено пользователем (Ctrl+C).")

if __name__ == "__main__":
    main()
