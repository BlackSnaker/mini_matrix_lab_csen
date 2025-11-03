# memory.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable, Set
from dataclasses import dataclass
from collections import deque

XY = Tuple[float, float]

# Сколько символов показывать в публичном UI внутри data
PUBLIC_STR_MAX = 200


@dataclass
class MemoryEvent:
    """
    Одна запись памяти агента.

    Поля:
      type   : строка категории ("damage", "pain", "heal", "animal_attack",
                                  "repel_animal", "tame_progress", "tame_success",
                                  "escape_from_hazard", "fear_spike", "ally_panic",
                                  "msg_in", "msg_out", "move", "external_command",
                                  "new_goal", "death", "saw_safe", "enter_safe_zone", ...)
      data   : произвольный словарь деталей (hp, текст, id источника, координаты и т.п.)
      tick   : игровой тик, на котором это случилось (может быть None)
      private: если True — событие не уходит наружу в публичный GUI
      level  : важность для UI: "info" | "warning" | "critical"
      actor  : внешний источник ("ally:a2", "animal:wolf_3", "hazard_7", "self", ...)
      pos    : (x, y) где это произошло (для карт травмы/убежища)
    """

    type: str
    data: Dict[str, Any]
    tick: Optional[int] = None
    private: bool = False
    level: str = "info"
    actor: Optional[str] = None
    pos: Optional[XY] = None

    # Совместимость с кодом, который обращается к .etype
    @property
    def etype(self) -> str:
        return self.type

    @staticmethod
    def from_any(obj: Any) -> "MemoryEvent":
        """
        Унифицированный конструктор: принимает уже готовый MemoryEvent
        или словарь формата {'type'/ 'etype', 'data', 'tick', 'private', 'level', 'actor', 'pos'}.
        Нужен для безопасной конвертации хвостов памяти разного типа.
        """
        if isinstance(obj, MemoryEvent):
            return obj
        if isinstance(obj, dict):
            t = obj.get("type") or obj.get("etype") or "event"
            d = obj.get("data") or {}
            return MemoryEvent(
                type=str(t),
                data=dict(d),
                tick=obj.get("tick"),
                private=bool(obj.get("private", False)),
                level=str(obj.get("level", "info")),
                actor=obj.get("actor"),
                pos=tuple(obj.get("pos")) if isinstance(obj.get("pos"), (list, tuple)) and len(obj.get("pos")) == 2 else None,
            )
        # последний шанс — привести к строке
        return MemoryEvent(type=str(obj), data={})

    def to_plain(self) -> Dict[str, Any]:
        """
        Плоское представление события для сериализации в файл (без обрезки строк).
        Совместимо с кодом вида: {"tick": ev.tick, "etype": ev.etype, "data": ev.data}
        """
        return {
            "tick": self.tick,
            "etype": self.etype,
            "data": dict(self.data),
            "private": self.private,
            "level": self.level,
            "actor": self.actor,
            "pos": self.pos,
        }

    def public_view(self) -> Dict[str, Any]:
        """
        Безопасный вид для GUI.
        - приватные эвенты отфильтруются на уровне dump_public_view()
        - длинные строки режем
        - прокидываем level, actor, pos
        """
        safe_data: Dict[str, Any] = {}
        for k, v in self.data.items():
            if isinstance(v, str) and len(v) > PUBLIC_STR_MAX:
                safe_data[k] = v[:PUBLIC_STR_MAX] + "...(+cut)"
            else:
                safe_data[k] = v
        return {
            "type": self.type,
            "tick": self.tick,
            "level": self.level,
            "actor": self.actor,
            "pos": self.pos,
            "data": safe_data,
        }


class AgentMemory:
    """
    Краткосрочная память агента (бортовой регистратор).

    Задачи:
      - хранить фиксированный хвост событий;
      - быстрая выдача безопасного "публичного хвоста" для UI;
      - сводка recent-сигналов для мозга (боль/лечение/паника/команды/карты);
      - быстрый доступ к последнему событию типа.
    """

    def __init__(self, max_events: int = 200):
        self.max_events = max_events
        self._events: deque[MemoryEvent] = deque(maxlen=max_events)

    # ------------------------------------------------------------------
    # важность событий
    # ------------------------------------------------------------------

    _CRITICAL_TYPES: Set[str] = {
        "death", "damage", "pain",
        "animal_attack",              # укус/атака зверя
    }
    _WARNING_TYPES: Set[str] = {
        "escape", "escape_from_hazard",
        "fear_spike", "ally_panic",
        "msg_in", "enter_safe_zone",
        "repel", "repel_animal",      # отпугивание угрозы
    }
    _INFO_TYPES_EXTRA: Set[str] = {
        "heal", "saw_safe", "rest_at_safe", "sleep", "ate_food",
        "tame_try", "tame_progress", "tame_success",  # приручение
        "pet_defend",                                  # питомец защищал
        "seeking_shelter",
        "external_command", "new_goal",
        "move", "msg_out",
    }

    def _auto_level_for_type(self, event_type: str) -> str:
        if event_type in self._CRITICAL_TYPES:
            return "critical"
        if event_type in self._WARNING_TYPES:
            return "warning"
        return "info"

    # ------------------------------------------------------------------
    # запись события (+автодополнение actor/pos)
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_actor(data: Dict[str, Any]) -> Optional[str]:
        """
        Если actor не передан явно — пытаемся угадать по типичным ключам.
        """
        for key in ("actor", "source", "from", "by", "attacker"):
            val = data.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    @staticmethod
    def _guess_pos(data: Dict[str, Any]) -> Optional[XY]:
        """
        Если pos не передан явно — пытаемся угадать по ключам 'pos' или 'at'.
        Должна быть пара чисел (x, y).
        """
        for key in ("pos", "at", "where"):
            val = data.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                try:
                    x = float(val[0])
                    y = float(val[1])
                    return (x, y)
                except Exception:
                    continue
        return None

    def remember(
        self,
        event_type: str,
        data: Dict[str, Any],
        *,
        tick: Optional[int] = None,
        private: bool = False,
        level: Optional[str] = None,
        actor: Optional[str] = None,
        pos: Optional[XY] = None,
    ) -> None:
        """
        Добавить событие в память.

        Можно передавать actor/pos как именованные аргументы.
        Если они не указаны — попытаемся извлечь из data.
        """
        if level is None:
            level = self._auto_level_for_type(event_type)

        if actor is None:
            actor = self._guess_actor(data)

        if pos is None:
            pos = self._guess_pos(data)

        ev = MemoryEvent(
            type=event_type,
            data=data,
            tick=tick,
            private=private,
            level=level,
            actor=actor,
            pos=pos,
        )
        self._events.append(ev)

    # Быстрая массовая загрузка (например, из сохранённого хвоста мозга)
    def extend_from_any(self, items: Iterable[Any]) -> None:
        """
        Принимает последовательность MemoryEvent или dict — и добавляет их в хвост.
        """
        for it in items:
            self._events.append(MemoryEvent.from_any(it))

    # ------------------------------------------------------------------
    # прямой доступ
    # ------------------------------------------------------------------

    def get_recent(self, n: int) -> List[MemoryEvent]:
        if n <= 0:
            return []
        return list(self._events)[-n:]

    def get_last_event_of_type(
        self,
        event_type: str,
        *,
        include_private: bool = False,
    ) -> Optional[MemoryEvent]:
        for ev in reversed(self._events):
            if ev.type != event_type:
                continue
            if not include_private and ev.private:
                continue
            return ev
        return None

    # ------------------------------------------------------------------
    # вспомогательные: что считать "недавно"
    # ------------------------------------------------------------------

    def _compute_cutoff_tick(self, max_ticks_ago: int) -> Optional[int]:
        latest_tick_seen: Optional[int] = None
        for ev in reversed(self._events):
            if ev.tick is not None:
                latest_tick_seen = ev.tick
                break
        if latest_tick_seen is None:
            # нет абсолютного времени — считаем всё недавним
            return None
        return latest_tick_seen - max_ticks_ago

    # ------------------------------------------------------------------
    # сводка для мозга / принятия решений
    # ------------------------------------------------------------------

    def summarize_recent(
        self,
        max_ticks_ago: int = 50,
    ) -> Dict[str, Any]:
        """
        Короткая выжимка "что было недавно".
        """
        cutoff_tick = self._compute_cutoff_tick(max_ticks_ago)

        def _recent(ev: MemoryEvent) -> bool:
            return cutoff_tick is None or ev.tick is None or ev.tick >= cutoff_tick

        took_damage = False
        healed = False
        visited_safe = False
        saw_panic = False
        repelled_threat = False

        last_command_target: Optional[Any] = None
        taming_progress: Optional[float] = None
        tamed_success = False

        hazard_spots: List[XY] = []
        safe_spots: List[XY] = []
        seen_hazard_xy: Set[XY] = set()
        seen_safe_xy: Set[XY] = set()
        recent_attackers: List[str] = []
        seen_attackers: Set[str] = set()

        for ev in reversed(self._events):
            if not _recent(ev):
                break

            et = ev.type

            # боль/урон (включая атаки зверей)
            if et in ("damage", "pain", "animal_attack"):
                took_damage = True

                # атакующий как "кто обидел"
                attacker = ev.actor or str(ev.data.get("by") or ev.data.get("source") or "")
                if attacker and attacker not in seen_attackers:
                    seen_attackers.add(attacker)
                    recent_attackers.append(attacker)

                # где было больно
                pos = ev.pos or self._guess_pos(ev.data)
                if pos and pos not in seen_hazard_xy:
                    seen_hazard_xy.add(pos)
                    hazard_spots.append(pos)

            # паника / побег
            if et in ("fear_spike", "ally_panic", "escape_from_hazard"):
                saw_panic = True
                if et == "escape_from_hazard":
                    pos = ev.pos or self._guess_pos(ev.data)
                    if pos and pos not in seen_hazard_xy:
                        seen_hazard_xy.add(pos)
                        hazard_spots.append(pos)

            # отпугивание
            if et in ("repel", "repel_animal"):
                repelled_threat = True

            # лечение / безопасные места / еда / сон
            if et in ("heal", "rest_at_safe", "enter_safe_zone", "saw_safe", "ate_food", "sleep"):
                healed = healed or (et == "heal")
                visited_safe = True
                pos = ev.pos or self._guess_pos(ev.data)
                if pos and pos not in seen_safe_xy:
                    seen_safe_xy.add(pos)
                    safe_spots.append(pos)

            # команды/цели
            if et in ("external_command", "new_goal"):
                goal = (ev.data.get("new_goal")
                        or ev.data.get("goal")
                        or ev.data.get("target"))
                if goal is not None and last_command_target is None:
                    last_command_target = goal

            # приручение
            if et in ("tame_try", "tame_progress"):
                prog = ev.data.get("progress")
                try:
                    if prog is not None:
                        taming_progress = float(prog)
                except Exception:
                    pass
            elif et == "tame_success":
                tamed_success = True
                taming_progress = 1.0

        return {
            "took_damage": took_damage,
            "healed": healed,
            "visited_safe": visited_safe,
            "saw_panic": saw_panic,
            "repelled_threat": repelled_threat,
            "had_external_command": last_command_target,
            "taming_progress": taming_progress,
            "tamed_success": tamed_success,
            "recent_attackers": recent_attackers,
            "hazard_spots": hazard_spots,
            "safe_spots": safe_spots,
        }

    # ------------------------------------------------------------------
    # публичный хвост для GUI (с шумоподавлением)
    # ------------------------------------------------------------------

    def dump_public_view(
        self,
        tail: int = 5,
        *,
        since_tick: Optional[int] = None,
        types: Optional[Iterable[str]] = None,
        exclude_types: Optional[Iterable[str]] = None,
        collapse_noise: bool = True,
        move_stride: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Публичный хвост событий для UI.
        Возвращает последние `tail` НЕприватных событий (от старых к новым),
        в безопасном виде (режет длинные строки).

        Доп. параметры:
          - since_tick: показывать только события не старше этого тика
          - types: включить только эти типы
          - exclude_types: исключить перечисленные типы
          - collapse_noise: если True — не показывать слишком частые 'move'
                            (не чаще 1 раз в `move_stride` тиков)
        """
        allow_types: Optional[Set[str]] = set(types) if types else None
        deny_types: Set[str] = set(exclude_types) if exclude_types else set()

        public_events: List[Dict[str, Any]] = []

        last_move_tick_included: Optional[int] = None  # для шумоподавления

        # идём от свежего к старому, собираем, затем разворачиваем
        for ev in reversed(self._events):
            if ev.private:
                continue
            if since_tick is not None and ev.tick is not None and ev.tick < since_tick:
                continue
            if allow_types is not None and ev.type not in allow_types:
                continue
            if ev.type in deny_types:
                continue

            # шумоподавление движения
            if collapse_noise and ev.type == "move":
                if last_move_tick_included is None:
                    last_move_tick_included = ev.tick if ev.tick is not None else None
                else:
                    # Мы идём от новых к старым → ev.tick уменьшается.
                    if ev.tick is not None and last_move_tick_included is not None:
                        if (last_move_tick_included - ev.tick) < move_stride:
                            continue
                    # если тика нет — считаем событие шумом и пропускаем
                    else:
                        continue

            public_events.append(ev.public_view())
            if len(public_events) >= tail:
                break

        public_events.reverse()
        return public_events
