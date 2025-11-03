from __future__ import annotations
from typing import Optional, Dict, Any, Literal, List, Tuple
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ТИПЫ, КОТОРЫЕ ХОДЯТ ЧЕРЕЗ СЕТЬ
# =============================================================================


class Vec2(BaseModel):
    """
    Простой 2D-вектор (координаты в мире).
    Используем для pos / vel / goal.
    """
    x: float = Field(..., description="Координата X")
    y: float = Field(..., description="Координата Y")

    @field_validator("x", "y")
    @classmethod
    def finite(cls, v: float) -> float:
        # защита от NaN/inf
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("coordinate must be a finite number")
        return float(v)


class MemoryEventPublic(BaseModel):
    """
    Безопасная запись краткосрочной памяти агента, пригодная для UI.

    Это примерно MemoryEvent.public_view() с сервера.

    Поля:
      - type: что произошло ("damage","heal","ally_panic","move","msg_out"...)
      - tick: тик симуляции, когда это случилось
      - level: "info" | "warning" | "critical"
               (UI красит этим цветом)
      - actor: кто был источником ("self","ally:a2","hazard_3"...)
      - pos:   [x,y] где это произошло (если есть)
      - data:  произвольные детали (обрезанные до разумной длины на сервере)
    """
    type: str = Field(..., description="Код события памяти ('damage', 'heal', ...)")
    tick: Optional[int] = Field(
        None,
        description="Игровой тик события",
    )
    level: str = Field(
        "info",
        description="UI-важность: info / warning / critical",
    )
    actor: Optional[str] = Field(
        None,
        description="Источник события ('ally:a2', 'hazard_7', 'self', ...)",
    )
    pos: Optional[Tuple[float, float]] = Field(
        None,
        description="Координаты (x,y), где это случилось. None если неизвестно.",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Безопасное описание события (урон, текст реплики и т.д.)",
    )


class AgentMindBlockPublic(BaseModel):
    """
    Публичный срез 'сознания' агента.
    Это то, что мы показываем во вкладке Mind в viewer.

    Поля:
      - survival_score: числовая оценка 'насколько я думаю, что выживу'
      - current_drive:  текущий драйв ('seek_food','hide','explore'...)
      - behavior_rules: правила поведения (словарь простых if/then)
      - beliefs:        убеждения / выводы ('если больно там → туда не ходить')
      - memory_tail:    последние события, как их видит сознание
                        (тот же формат, что MemoryEventPublic)
    """
    survival_score: Optional[float] = Field(
        None,
        description="Оценка шансов на выживание по мнению сознания агента",
    )
    current_drive: Optional[str] = Field(
        None,
        description="Доминирующий мотив поведения агента сейчас",
    )
    behavior_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Текущие активные правила поведения (key -> value)",
    )
    beliefs: List[Any] = Field(
        default_factory=list,
        description="Список убеждений/выводов агента",
    )
    memory_tail: List[MemoryEventPublic] = Field(
        default_factory=list,
        description="Последние события (по мнению сознания)",
    )


# =============================================================================
# КЛИЕНТ → СЕРВЕР
# =============================================================================


class GoalUpdate(BaseModel):
    """
    Новая целевая точка для агента.
    Пример: {"x": 10.0, "y": 90.0}
    """
    x: float = Field(..., description="Желаемая цель X в координатах мира")
    y: float = Field(..., description="Желаемая цель Y в координатах мира")

    @field_validator("x", "y")
    @classmethod
    def finite(cls, v: float) -> float:
        # защита от NaN/inf
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("coordinate must be a finite number")
        return float(v)


class ClientMessage(BaseModel):
    """
    Сообщение, которое фронтенд (GUI / человек-наблюдатель) шлёт
    серверу по WebSocket.

    type:
      - "subscribe": клиент хочет подписаться на обновления мира
      - "set_goal":  сменить цель конкретного агента
      - "ping":      keepalive ("я жив")

    Примеры:
      {"type":"subscribe"}

      {"type":"set_goal","agent_id":"a1","goal":{"x":10,"y":90}}

      {"type":"ping"}
    """

    type: Literal["subscribe", "set_goal", "ping"] = Field(
        ...,
        description="Тип команды от клиента",
    )

    agent_id: Optional[str] = Field(
        None,
        description=(
            "ID агента, к которому относится команда (например, 'a1'). "
            "Требуется для type='set_goal'."
        ),
    )

    goal: Optional[GoalUpdate] = Field(
        None,
        description=(
            "Новые координаты цели агента. "
            "Требуется для type='set_goal'."
        ),
    )

    # На будущее (корреляция запрос/ответ, телеметрия и т.п.)
    # client_ts: Optional[int] = Field(
    #     None,
    #     description="Клиентский timestamp, можно использовать для отладки лагов"
    # )


# =============================================================================
# СЕРВЕР → КЛИЕНТ (общие полезные нагрузики)
# =============================================================================


class AckPayload(BaseModel):
    """
    Ответ-успех на команду клиента.
    """
    message: str = Field(
        "...",
        description="Человеко-читаемое подтверждение",
    )
    agent_id: Optional[str] = Field(
        None,
        description="Если подтверждали команду над агентом — его ID",
    )


class ErrorPayload(BaseModel):
    """
    Ответ-ошибка на команду клиента.
    """
    error: str = Field(
        ...,
        description="Текст ошибки, например 'agent_not_found_or_bad_goal'",
    )


class AgentPublicState(BaseModel):
    """
    Публичный срез одного агента из снапшота мира.
    Это примерно то, что world.Agent.serialize_public_state() должен возвращать.

    Поля:
      - id / name
      - pos / vel / goal          : координаты/скорость/цель (Vec2)
      - fear                      : 0..1 (мы защитно клампим)
      - health / energy / hunger  : 0..100 (%)
      - alive / cause_of_death    : жив ли агент и почему умер (если умер)
      - age_ticks                 : сколько тиков прожил
      - danger_zones_count        : сколько "опасных мест" он помнит
      - hazards_known             : сколько конкретных угроз он знает
      - memory_tail               : последние MemoryEventPublic для GUI
      - mind                      : AgentMindBlockPublic (сознание), опционально
    """
    id: str = Field(..., description="Уникальный ID агента, например 'a1'")
    name: str = Field(..., description="Имя агента, человеко-читаемо")

    pos: Vec2 = Field(..., description="Текущая позиция агента в мире")
    vel: Vec2 = Field(..., description="Текущая скорость агента")
    goal: Vec2 = Field(..., description="Текущая целевая точка агента")

    fear: float = Field(
        0.0,
        description="Уровень страха [0..1]",
    )
    health: float = Field(
        100.0,
        description="Очки здоровья [% 0..100]",
    )
    energy: Optional[float] = Field(
        None,
        description="Энергия/выносливость [% 0..100] (может отсутствовать)",
    )
    hunger: Optional[float] = Field(
        None,
        description="Голод [% 0..100], где больше = хуже (может отсутствовать)",
    )

    alive: Optional[bool] = Field(
        None,
        description="Жив ли агент сейчас. Если None — клиент может вывести health>0?",
    )
    cause_of_death: Optional[str] = Field(
        None,
        description="Причина смерти (если умер)",
    )

    age_ticks: int = Field(
        0,
        description="Сколько тиков агент прожил",
    )

    danger_zones_count: int = Field(
        0,
        description="Сколько опасных зон он считает опасными",
    )
    hazards_known: int = Field(
        0,
        description="Сколько конкретных угроз он знает (ядовитые лужи, хищники и т.п.)",
    )

    memory_tail: List[MemoryEventPublic] = Field(
        default_factory=list,
        description="Публичный хвост краткосрочной памяти агента для UI",
    )

    mind: Optional[AgentMindBlockPublic] = Field(
        None,
        description="Срез сознания агента (мотивация, убеждения)",
    )

    # --- валидация чисел -----------------------------------------------------

    @field_validator("fear")
    @classmethod
    def clamp_fear(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        # страх не должен выходить за границы
        return max(0.0, min(1.0, float(v)))

    @field_validator("health", "energy", "hunger")
    @classmethod
    def clamp_pct(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return 0.0
        return max(0.0, min(100.0, float(v)))


class WorldObjectPublic(BaseModel):
    """
    Публичный срез объекта окружения.

    Обычно это всякие POI:
      - костёр / лагерь
      - мастерская
      - токсичная зона
      - склад еды
      - руины (опасно)
    """
    id: str = Field(..., description="ID объекта/POI")
    name: str = Field(..., description="Название ('Campfire','Токсичная лужа',...)")
    kind: str = Field(..., description="safe / neutral / hazard / ...")
    pos: Vec2 = Field(..., description="Центр POI")
    radius: float = Field(..., description="Радиус влияния/зоны")

    @field_validator("radius")
    @classmethod
    def clamp_radius(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("radius must be finite")
        return max(0.0, float(v))


class WorldInfo(BaseModel):
    """
    Габариты мира.
    """
    width: float = Field(..., description="Ширина мира")
    height: float = Field(..., description="Высота мира")

    @field_validator("width", "height")
    @classmethod
    def positive(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("size must be finite")
        return max(1.0, float(v))


class WorldStatePayload(BaseModel):
    """
    Полный снапшот симуляции, который сервер рассылает клиентам.
    Это то, что вернёт /state и что идёт по WebSocket в type="world_state".

    Поля:
      - tick: текущий тик
      - agents: список публичных состояний всех агентов
      - world: размеры мира
      - objects: POI / опасности / убежища
      - chat: глобальный чат (крики боли, предупреждения и т.д.)
      - events: важные события мира (смерти, приказы оператора, и т.п.)
                Это сырые словари вида {"type":"death","tick":123,...}
                Клиент сам отрисует human-friendly строки.
    """
    tick: int = Field(
        ...,
        description="Текущий тик симуляции",
    )

    agents: List[AgentPublicState] = Field(
        default_factory=list,
        description="Публичное состояние всех агентов",
    )

    world: WorldInfo = Field(
        ...,
        description="Габариты мира",
    )

    objects: List[WorldObjectPublic] = Field(
        default_factory=list,
        description="Известные объекты окружения (опасные зоны, убежища и т.д.)",
    )

    chat: List[str] = Field(
        default_factory=list,
        description="Последние реплики глобального чата мира",
    )

    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Журнал ключевых событий мира: смерти, внешние приказы и т.д. "
            "Каждый элемент — словарь с полями вроде "
            "{type:'death',tick:123,name:'Nova',reason:'acid'...}"
        ),
    )


# =============================================================================
# ОБОЛОЧКА WEBSOCKET-СООБЩЕНИЙ ОТ СЕРВЕРА
# =============================================================================


class ServerMessage(BaseModel):
    """
    Сообщение, которое сервер шлёт клиенту по WebSocket.

    type:
      - "ack"         : подтверждение, что команда клиента принята
      - "error"       : ошибка обработки клиентской команды
      - "world_state" : полное состояние мира на данный тик
      - "pong"        : ответ на ping клиента

    data:
      - для "ack"         → AckPayload
      - для "error"       → ErrorPayload
      - для "world_state" → WorldStatePayload
      - для "pong"        → {"alive": true} (минимальный keepalive)

    tick:
      - дублируем текущий тик для удобства клиента
        (чтобы UI мог просто посмотреть msg.tick, не лезя в data)
    """

    type: Literal["ack", "error", "world_state", "pong"] = Field(
        ...,
        description="Тип сообщения от сервера",
    )

    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Полезная нагрузка: снапшот мира / ack / ошибка / pong / ...",
    )

    tick: Optional[int] = Field(
        None,
        description=(
            "Текущий тик симуляции на момент отправки. "
            "Дублируется здесь для удобства,"
            "чтобы не всегда лезть внутрь data."
        ),
    )

    # -------------------------------------------------------------------------
    # Конструкторы-помощники (quality of life для сервера)
    # -------------------------------------------------------------------------

    @classmethod
    def ack(cls, msg: str, agent_id: Optional[str] = None) -> "ServerMessage":
        """
        Собрать ack-ответ.
        Пример:
            return ServerMessage.ack("set_goal ok", agent_id="a1")
        """
        payload = AckPayload(message=msg, agent_id=agent_id)
        return cls(type="ack", data=payload.model_dump())

    @classmethod
    def err(cls, err_text: str) -> "ServerMessage":
        """
        Собрать error-ответ.
        Пример:
            return ServerMessage.err("agent_not_found_or_bad_goal")
        """
        payload = ErrorPayload(error=err_text)
        return cls(type="error", data=payload.model_dump())

    @classmethod
    def world_state(cls, snapshot: Dict[str, Any]) -> "ServerMessage":
        """
        Превратить словарь снапшота мира (из world.snapshot())
        в типизированный ServerMessage.

        Обычно world.snapshot() уже возвращает структуру,
        совместимую с WorldStatePayload. Тут мы валидируем её
        и нормализуем числа (через pydantic).
        """
        payload = WorldStatePayload.model_validate(snapshot)
        return cls(
            type="world_state",
            data=payload.model_dump(),
            tick=payload.tick,
        )

    @classmethod
    def pong(cls, tick: Optional[int] = None) -> "ServerMessage":
        """
        Ответ на ping-клиента.
        Минимум данных, но клиенту уже достаточно понять, что
        соединение живо.
        """
        return cls(
            type="pong",
            data={"alive": True},
            tick=tick,
        )
