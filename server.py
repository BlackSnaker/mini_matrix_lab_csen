# server.py
import asyncio
import random
import math
from typing import Dict, List, Set, Optional, Tuple, Iterable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from pydantic import BaseModel, Field
import numpy as np  # для упаковки реплеев из трейнера

import config
from world import World, Agent, WorldObject
from schema import ClientMessage, ServerMessage
from village_map import attach_village
from mind_core import Transition  # реплеи навыков из трейнера (rehearsal → brain)

# =============================================================================
# 0. МИР + ГЛОБАЛЬНАЯ БЛОКИРОВКА
# =============================================================================
#
# Важно:
# - FastAPI крутится в одном event loop, но у нас есть:
#     * simulation_loop(), который мутирует мир world.tick()
#     * REST / WS ручки, которые читают snapshot(), set_agent_goal(), и т.д.
#
#   Если дать им лезть в world одновременно, можно словить гонки
#   (частичный снапшот посреди тика и т.п.).
#
# - Поэтому вводим world_lock = asyncio.Lock()
#   и ВСЕ обращения к world.{tick(), snapshot(), set_agent_goal(), ...}
#   оборачиваем в `async with world_lock: ...`.
#
world = World(width=config.WORLD_WIDTH, height=config.WORLD_HEIGHT)
world_lock = asyncio.Lock()


def _iter_agents() -> List[Agent]:
    """
    Унифицированный итератор агентов:
    поддерживает world.agents как list[Agent] или dict[str, Agent].
    """
    agents_attr = getattr(world, "agents", [])
    if isinstance(agents_attr, dict):
        return list(agents_attr.values())
    return list(agents_attr)


def _find_agent(agent_id: str) -> Optional[Agent]:
    """
    Утилита поиска агента по id (без знания внутренностей World).
    Корректно работает и при world.agents: list или dict.
    """
    # Быстрый путь: если dict
    agents_attr = getattr(world, "agents", None)
    if isinstance(agents_attr, dict):
        a = agents_attr.get(agent_id)
        if a:
            return a
    # Универсальный обход
    for a in _iter_agents():
        if getattr(a, "agent_id", None) == agent_id or getattr(a, "id", None) == agent_id:
            return a
    return None


# =============================================================================
# 1. ПРОЦЕДУРНО СТРОИМ ДЕРЕВНЮ / ОКРУЖЕНИЕ
# =============================================================================

def _get_xy_from_registry(
    reg: Dict[str, dict],
    poi_id: str,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Достаём координаты центра POI по его id.
    Если такого POI нет, возвращаем fallback.
    """
    rec = reg.get(poi_id)
    if not rec:
        return fallback
    area = rec.get("area", {})
    return (
        float(area.get("x", fallback[0])),
        float(area.get("y", fallback[1])),
    )


# Базовый центр мира — пригодится как запасной вариант.
world_center = (world.width * 0.5, world.height * 0.5)

# Строим окружение и реестр активностей
activity_registry = attach_village(world)

# Координаты "домов" и "общей точки сбора".
echo_spawn_xy = _get_xy_from_registry(activity_registry, "house_echo", world_center)
nova_spawn_xy = _get_xy_from_registry(activity_registry, "house_nova", world_center)
camp_goal_xy = _get_xy_from_registry(activity_registry, "campfire", world_center)


# -----------------------------------------------------------------------------
# 2. ДОПОЛНИТЕЛЬНЫЕ АКТИВНОСТИ/POI, ЧТОБЫ МИР БЫЛ БОГАЧЕ
# -----------------------------------------------------------------------------
def _rand_xy_in_world(margin: float = 10.0) -> Tuple[float, float]:
    """Случайные координаты внутри мира с небольшим отступом от краёв."""
    x = random.uniform(margin, max(margin, world.width - margin))
    y = random.uniform(margin, max(margin, world.height - margin))
    return (x, y)


def _register_extra_activity_spots(world: World, registry: Dict[str, dict]):
    """
    Создаём дополнительные точки интереса:
      - food_cache: запас еды
      - workshop: импровизированная мастерская/укрытие
      - watchtower: вышка наблюдения (чуть успокаивает, можно вздремнуть)
      - wild_forest: тихое место с ягодами
      - old_ruins: полусломанные руины с лутом (слегка опасно, не лечит)
    """

    def _add_poi(
        poi_id: str,
        name: str,
        kind: str,
        x: float,
        y: float,
        radius: float,
        danger_level: float,
        comfort_level: float,
        activity_tags: List[str],
        resource_tag: Optional[str] = None,
        resource_abundance: float = 0.0,
    ):
        """
        Создаёт WorldObject + вносит его в world и в реестр активностей.
        """
        obj = WorldObject(
            obj_id=poi_id,
            name=name,
            kind=kind,
            x=x,
            y=y,
            radius=radius,
            danger_level=danger_level,
            comfort_level=comfort_level,
            resource_tag=resource_tag,
            resource_abundance=resource_abundance,
        )
        world.add_object(obj)

        registry[poi_id] = {
            "name": name,
            "activity_tags": list(activity_tags),
            "comfort_level": comfort_level,
            "danger_level": danger_level,
            "area": {
                "x": x,
                "y": y,
                "radius": radius,
            },
        }

    # 1) Склад провизии / Food Cache.
    fx, fy = _rand_xy_in_world()
    _add_poi(
        poi_id="food_cache",
        name="Склад провизии",
        kind="safe",
        x=fx,
        y=fy,
        radius=random.uniform(4.0, 6.0),
        danger_level=0.0,
        comfort_level=0.6,  # еда + чуть лечит/успокаивает
        activity_tags=["eat", "restock_food", "rest"],
        resource_tag="food",
        resource_abundance=1.0,
    )

    # 2) Мастерская / Workshop.
    wx, wy = _rand_xy_in_world()
    _add_poi(
        poi_id="workshop",
        name="Мастерская",
        kind="safe",
        x=wx,
        y=wy,
        radius=random.uniform(4.0, 6.0),
        danger_level=0.0,
        comfort_level=0.7,  # чинит/лечит лучше
        activity_tags=["repair_self", "rest", "sleep", "calm"],
        resource_tag="scrap",
        resource_abundance=0.5,
    )

    # 3) Наблюдательная вышка / Watchtower.
    tx, ty = _rand_xy_in_world()
    _add_poi(
        poi_id="watchtower",
        name="Старая вышка",
        kind="safe",
        x=tx,
        y=ty,
        radius=random.uniform(3.0, 5.0),
        danger_level=0.0,
        comfort_level=0.4,  # слегка безопасно, можно передохнуть
        activity_tags=["rest", "sleep", "calm"],
        resource_tag=None,
        resource_abundance=0.0,
    )

    # 4) Дикий лес с ягодами / Wild Forest.
    fx2, fy2 = _rand_xy_in_world()
    _add_poi(
        poi_id="wild_forest",
        name="Ягодная роща",
        kind="safe",
        x=fx2,
        y=fy2,
        radius=random.uniform(5.0, 8.0),
        danger_level=0.0,
        comfort_level=0.4,  # природа лечит нервы
        activity_tags=["eat", "restock_food", "rest", "calm"],
        resource_tag="food",
        resource_abundance=0.6,
    )

    # 5) Старые руины / Old Ruins — слегка опасно, но можно полутать.
    rx, ry = _rand_xy_in_world()
    ruins_radius = random.uniform(6.0, 8.0)
    obj_ruins = WorldObject(
        obj_id="old_ruins",
        name="Старые руины",
        kind="neutral",
        x=rx,
        y=ry,
        radius=ruins_radius,
        danger_level=0.4,      # чуть больно
        comfort_level=0.0,     # но не лечит
        resource_tag="scrap",
        resource_abundance=0.8,
    )
    world.add_object(obj_ruins)

    registry["old_ruins"] = {
        "name": "Старые руины",
        "activity_tags": ["scavenge", "scrap", "explore"],
        "comfort_level": 0.0,
        "danger_level": 0.4,
        "area": {
            "x": rx,
            "y": ry,
            "radius": ruins_radius,
        },
    }


# накидываем дополнительные POI
_register_extra_activity_spots(world, activity_registry)

# после расширения activity_registry обновляем world.activities,
# чтобы агенты могли пользоваться этими зонами в pick_new_goal()
world.set_activity_registry(activity_registry)


# -----------------------------------------------------------------------------
# 3. SAFE_POINT (лагерь/костёр для паники и критических состояний)
# -----------------------------------------------------------------------------
try:
    cx, cy = camp_goal_xy
    cx = max(0.0, min(world.width, float(cx)))
    cy = max(0.0, min(world.height, float(cy)))
    config.SAFE_POINT = (cx, cy)
except Exception:
    # fallback — центр мира
    config.SAFE_POINT = world_center


# -----------------------------------------------------------------------------
# 4. СПАВН АГЕНТОВ
# -----------------------------------------------------------------------------
def _spawn_agent(
    agent_id: str,
    name: str,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    persona: str,
) -> Agent:
    """
    Удобный конструктор агента.
    Агент:
      - стартует в своей зоне (дом / укрытие / лагерь)
      - получает первую цель (например, площадь у костра)
      - грузит прошлый мозг через load_brain(agent_id) внутри Agent.__init__
        (если файл brains/<agent_id>.json уже существует).
    """
    sx, sy = start_xy
    gx, gy = goal_xy
    return Agent(
        agent_id=agent_id,
        name=name,
        x=sx,
        y=sy,
        goal_x=gx,
        goal_y=gy,
        persona=persona,
    )


def _jittered_near(base_xy: Tuple[float, float], spread: float = 2.5) -> Tuple[float, float]:
    """
    Чуть-чуть раскинуть спавн вокруг точки (например костра),
    чтобы агенты не стояли строго в одном пикселе.
    """
    bx, by = base_xy
    jx = bx + random.uniform(-spread, spread)
    jy = by + random.uniform(-spread, spread)
    jx = max(0.0, min(world.width, jx))
    jy = max(0.0, min(world.height, jy))
    return (jx, jy)


# Echo спавнится у "house_echo", Nova у "house_nova".
# Обеим ставим goal = "campfire" (социальный хаб).
echo = _spawn_agent(
    agent_id="a1",
    name="Echo",
    start_xy=echo_spawn_xy,
    goal_xy=camp_goal_xy,
    persona=(
        "Ты Echo. Осторожный, тревожный выживальщик. "
        "Тебе страшно умереть, ты переживаешь за Nova. "
        "Ты стараешься держать всех в безопасности и предупреждать об угрозах."
    ),
)
world.add_agent(echo)

nova = _spawn_agent(
    agent_id="a2",
    name="Nova",
    start_xy=nova_spawn_xy,
    goal_xy=camp_goal_xy,
    persona=(
        "Ты Nova. Ты смелая исследовательница. "
        "Ты любишь разведку, но не хочешь, чтобы Echo пострадал. "
        "Ты звучишь уверенно и тёпло, даже когда больно."
    ),
)
world.add_agent(nova)

# Третий выживший (A0) — спавним у костра с джиттером.
a0_spawn_xy = _jittered_near(camp_goal_xy, spread=2.5)
a0 = _spawn_agent(
    agent_id="agent_0",
    name="A0",
    start_xy=a0_spawn_xy,
    goal_xy=camp_goal_xy,
    persona="caring/supportive",
)
world.add_agent(a0)


# =============================================================================
# 5. ConnectionManager — централизуем работу с WebSocket-клиентами
# =============================================================================
class ConnectionManager:
    """
    Управляет активными WebSocket-подписчиками (наблюдателями).
    Делает:
      - connect() / disconnect()
      - safe_send_text(): отправка одному, с автокиком при ошибке
      - broadcast_text(): рассылка снапшота всем живым
    """

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def safe_send_text(self, ws: WebSocket, text: str):
        """Безопасно отправить текст этому конкретному клиенту."""
        try:
            await ws.send_text(text)
        except Exception:
            self.disconnect(ws)

    async def broadcast_text(self, text: str):
        """Отправить текст всем подписчикам."""
        dead: List[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# =============================================================================
# 5.1 SSE (Server-Sent Events) — для терминалов (curl -N)
# =============================================================================
class SSEManager:
    """
    Лёгкий менеджер клиентов SSE: каждому держим asyncio.Queue[str].
    broadcast(msg) кладёт сообщение во все очереди; переполненные — сдвигаем.
    """
    def __init__(self):
        self.clients: Set[asyncio.Queue[str]] = set()

    def register(self) -> asyncio.Queue:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
        self.clients.add(q)
        return q

    def unregister(self, q: asyncio.Queue):
        self.clients.discard(q)

    async def broadcast(self, text: str):
        dead: List[asyncio.Queue] = []
        for q in list(self.clients):
            try:
                if q.full():
                    _ = q.get_nowait()
                await q.put(text)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unregister(q)


sse = SSEManager()


# =============================================================================
# 5.2 Универсальная рассылка произвольных событий (WS + SSE)
# =============================================================================
async def push_global_event(kind: str, payload: Dict):
    """
    Отправить событие всем подписчикам.
    Параллельно — положить его в лог мира, если он есть.
    """
    try:
        async with world_lock:
            if hasattr(world, "push_event") and callable(getattr(world, "push_event")):
                world.push_event(kind=kind, payload=payload)
            elif hasattr(world, "event_log"):
                ev = {"tick": getattr(world, "tick_count", 0), "kind": kind, "data": payload}
                world.event_log.append(ev)
                lim = getattr(world, "MAX_EVENT_LOG", 200)
                if isinstance(lim, int) and lim > 0:
                    world.event_log[:] = world.event_log[-lim:]
    except Exception:
        pass

    msg = ServerMessage(type="event", data={"kind": kind, "payload": payload}).model_dump_json()
    if manager.active:
        await manager.broadcast_text(msg)
    if sse.clients:
        await sse.broadcast(msg)


# =============================================================================
# 6. FastAPI app
# =============================================================================
app = FastAPI(
    title="Mini-Matrix World Server",
    description=(
        "Симуляция мира с агентами, питомцами/хищниками, болью/лечением, "
        "страхом и социальной передачей знаний. Мир живёт в процедурно "
        "сгенерированной деревне + загружает обученные мозги."
    ),
    version="1.0.0",
)


# =============================================================================
# 7. Аутентификация токеном (опционально)
# =============================================================================
API_TOKEN = getattr(config, "API_TOKEN", "")  # пустая строка = токен отключён


async def _require_token(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Если API_TOKEN задан, требуем заголовок x-api-key со значением токена.
    Бросаем 401 при несовпадении/отсутствии.
    """
    if API_TOKEN and x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


# =============================================================================
# 8. УТИЛИТЫ ДЛЯ ДОСТУПА К МИРУ И ОТВЕТОВ КЛИЕНТУ
# =============================================================================
def _clamp_to_world(x: float, y: float) -> Tuple[float, float]:
    """Ограничить координаты приказа в пределах карты."""
    return (
        max(0.0, min(world.width, x)),
        max(0.0, min(world.height, y)),
    )


# ------------------------ NEW: Combat overlay injection -----------------------
def _inject_combat_into_snapshot(snap: Dict, overlay: Dict[str, Dict]) -> Dict:
    """
    Вплетаем combat-поля в список агентов снапшота (in-place).
    overlay: {agent_id: {"state":..., "enemy_id":..., "skill":..., "just_hit":...}}
    """
    try:
        for a in snap.get("agents", []):
            aid = a.get("id") or a.get("agent_id")
            if not aid:
                continue
            cmb = overlay.get(aid)
            if cmb:
                a["combat"] = {
                    "state": str(cmb.get("state", "idle")),
                    "enemy_id": cmb.get("enemy_id"),
                    "just_hit": bool(cmb.get("just_hit", False)),
                    "skill": float(cmb.get("skill", 0.0)),
                }
    except Exception:
        pass
    return snap


async def _safe_snapshot_json() -> str:
    """Снять снапшот мира под world_lock и упаковать в ServerMessage."""
    async with world_lock:
        snap = world.snapshot()
        # Инъекция боевого контекста из CombatSystem:
        snap = _inject_combat_into_snapshot(snap, combat.overlay)
    msg = ServerMessage(
        type="world_state",
        data=snap,
    )
    return msg.model_dump_json()


# =============================================================================
# 9. СХЕМЫ ДЛЯ НОВЫХ ЭНДПОИНТОВ (skills/events)
# =============================================================================
class EventIn(BaseModel):
    kind: str
    payload: Dict = Field(default_factory=dict)


class TransitionIn(BaseModel):
    obs: List[float]
    action: Tuple[float, float]
    reward: float
    next_obs: List[float]
    done: bool = False
    goal_tag: Optional[str] = None
    goal_id: Optional[int] = None


class RehearsalIn(BaseModel):
    transitions: List[TransitionIn] = Field(default_factory=list)


class OnlineToggleIn(BaseModel):
    enable: Optional[bool] = None
    steps_per_tick: Optional[int] = None


# =============================================================================
# 10. REST ЭНДПОИНТЫ
# =============================================================================
@app.get("/health")
async def health():
    """Жив ли сервер и сколько тиков прошло."""
    async with world_lock:
        tick_now = world.tick_count
    return JSONResponse({"status": "ok", "tick": tick_now})


@app.get("/state")
async def get_state():
    """
    Полный снапшот мира (агенты, животные, объекты, события и т.д.).
    """
    async with world_lock:
        snap = world.snapshot()
        snap = _inject_combat_into_snapshot(snap, combat.overlay)
    return JSONResponse(snap)


@app.get("/stream")
async def stream():
    """
    SSE-стрим для терминала:
      curl -N http://<ip>:8000/stream
    Можно пропускать через jq: ... | jq --unbuffered 'fromjson? // .'
    """
    q = sse.register()

    async def gen():
        # мгновенно отдать актуальный снапшот при подключении
        first = await _safe_snapshot_json()
        yield f"data: {first}\n\n"
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            sse.unregister(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/agent/{agent_id}/goal")
async def set_goal_rest(
    agent_id: str,
    x: float,
    y: float,
    _token_ok: None = Depends(_require_token),  # проверка токена (если включён)
):
    """
    Удобная ручка для задания цели агенту из терминала (curl).
      POST /agent/a1/goal?x=12.3&y=45.6
    Если API_TOKEN задан — добавь заголовок: -H 'x-api-key: <token>'
    """
    gx, gy = _clamp_to_world(x, y)
    async with world_lock:
        changed = world.set_agent_goal(agent_id, gx, gy)
    return JSONResponse({"ok": bool(changed), "agent_id": agent_id, "goal": {"x": gx, "y": gy}})


# ---- Новые: события/навыки ---------------------------------------------------
@app.post("/event")
async def post_event(evt: EventIn, _token_ok: None = Depends(_require_token)):
    """
    Бросить произвольное глобальное событие в лог/WS/SSE.
    """
    await push_global_event(evt.kind, evt.payload)
    return JSONResponse({"ok": True})


@app.post("/agent/{agent_id}/skills/rehearsal")
async def upload_rehearsal(agent_id: str, payload: RehearsalIn, _token_ok: None = Depends(_require_token)):
    """
    Подать мозгу агента пакет реплеев из трейнера.
    Каждый Transition маппится на goal_id (по tag или напрямую).
    """
    async with world_lock:
        agent = _find_agent(agent_id)
        if not agent or not hasattr(agent, "brain"):
            return JSONResponse({"ok": False, "error": "agent_not_found_or_no_brain"}, status_code=404)
        brain = agent.brain
        to_add: List[Transition] = []
        for t in payload.transitions:
            # маппинг цели: tag → id через словарь мозга (расширяем словарь при необходимости)
            if t.goal_id is not None:
                gid = int(t.goal_id)
            else:
                tag = t.goal_tag or "explore"
                _ = brain._gc_goal_onehot(tag)  # создаёт при необходимости
                gid = int(brain.gc_goal_vocab.get(tag, 0))
            tr = Transition(
                obs=np.array(t.obs, dtype=np.float32),
                goal_id=gid,
                action=np.array(t.action, dtype=np.float32),
                reward=float(t.reward),
                next_obs=np.array(t.next_obs, dtype=np.float32),
                done=bool(t.done),
            )
            to_add.append(tr)
        brain.policy_preload_rehearsal(to_add)
        return JSONResponse({"ok": True, "added": len(to_add), "goal_vocab": brain.gc_goal_vocab})


@app.get("/agent/{agent_id}/skills")
async def get_agent_skills(agent_id: str):
    """
    Публичное состояние навыков агента (для UI/дебага).
    """
    async with world_lock:
        agent = _find_agent(agent_id)
        if not agent or not hasattr(agent, "brain"):
            return JSONResponse({"ok": False, "error": "agent_not_found_or_no_brain"}, status_code=404)
        mind_pub = agent.brain.export_public_state_for_ui()
        return JSONResponse({"ok": True, "skills": mind_pub.get("skills")})


@app.post("/skills/online")
async def toggle_online_learning(cfg: OnlineToggleIn, _token_ok: None = Depends(_require_token)):
    """
    Включить/выключить онлайн-дообучение и настроить шаги на тик.
    """
    if cfg.enable is not None:
        setattr(config, "LIVE_SKILL_LEARNING", bool(cfg.enable))
    if cfg.steps_per_tick is not None:
        steps = max(0, int(cfg.steps_per_tick))
        setattr(config, "LIVE_SKILL_LEARNING_STEPS", steps)
    return JSONResponse({
        "ok": True,
        "enable": bool(getattr(config, "LIVE_SKILL_LEARNING", True)),
        "steps_per_tick": int(getattr(config, "LIVE_SKILL_LEARNING_STEPS", 1)),
    })


# =============================================================================
# NEW (11): Простейшая система боя и обучения отбиваться
# =============================================================================
class CombatSystem:
    """
    Лёгкая ε-greedy боевая надстройка поверх мира:
      - обнаруживает угрозы (агрессивные животные в радиусе)
      - выбирает действие из {attack, block, evade, run}
      - ставит цели агенту в world.set_agent_goal(...)
      - оценивает награду, учится (Q-learning) и накапливает skill
      - инъецирует в снапшот public-поле "combat"
    Никаких изменений классов World/Agent не требуется.
    """

    def __init__(self):
        # параметры
        self.enabled = True
        self.threat_radius = getattr(config, "COMBAT_THREAT_RADIUS", 6.0)
        self.epsilon = getattr(config, "COMBAT_EPSILON", 0.20)
        self.alpha = getattr(config, "COMBAT_ALPHA", 0.35)
        self.gamma = getattr(config, "COMBAT_GAMMA", 0.90)
        self.idle_cooldown = 0.6  # сек, не дёргать каждую итерацию без нужды

        self.overlay: Dict[str, Dict] = {}     # публичный контекст на отправку
        self._state: Dict[str, Dict] = {}      # приватное состояние/память агента
        self._pending_events: List[Tuple[str, Dict]] = []

    # ----------------- утилиты -----------------
    @staticmethod
    def _xy(obj: Dict) -> Tuple[float, float]:
        """Безопасно вытащить (x,y) из записи снапшота."""
        p = obj.get("pos") or {}
        return float(p.get("x", 0.0)), float(p.get("y", p.get("z", 0.0)))

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def _is_aggressive(an: Dict) -> bool:
        return str(an.get("temperament", "")).lower() == "aggressive"

    def _state_key(self, fear: float, dist: float) -> str:
        db = int(min(9, dist // 1.5))
        fb = int(min(5, fear * 5.0))
        return f"d{db}|f{fb}"

    def _choose_action(self, aid: str, s_key: str) -> str:
        q_for_s = self._state.setdefault(aid, {}).setdefault("q", {}).setdefault(s_key, {})
        actions = ["attack", "block", "evade", "run"]
        # ε-greedy
        if random.random() < self.epsilon or not q_for_s:
            return random.choice(actions)
        # argmax
        best_a = max(actions, key=lambda a: q_for_s.get(a, 0.0))
        return best_a

    def _update_q(self, aid: str, s: str, a: str, r: float, s2: str):
        q_all = self._state.setdefault(aid, {}).setdefault("q", {})
        q_s = q_all.setdefault(s, {})
        old = q_s.get(a, 0.0)
        next_best = 0.0
        q_s2 = q_all.get(s2, {})
        if q_s2:
            next_best = max(q_s2.values())
        new = old + self.alpha * (r + self.gamma * next_best - old)
        q_s[a] = new

    def _boost_skill(self, aid: str, r: float):
        st = self._state.setdefault(aid, {})
        skill = float(st.get("skill", 0.0))
        # медленно растим при положительной награде, медленно падаем при отрицательной
        skill = max(0.0, min(1.0, skill + 0.02 * (1.0 if r > 0 else -1.0)))
        st["skill"] = skill
        return skill

    def _remember(self, aid: str, hp: float, fear: float, dist: float):
        st = self._state.setdefault(aid, {})
        st["prev_hp"] = hp
        st["prev_dist"] = dist

    def _previous(self, aid: str) -> Tuple[float, float]:
        st = self._state.setdefault(aid, {})
        return float(st.get("prev_hp", 100.0)), float(st.get("prev_dist", 999.0))

    def _cooldown_ok(self, aid: str, now_tick: int, tick_hz: float) -> bool:
        st = self._state.setdefault(aid, {})
        last_tick = int(st.get("last_tick", -999999))
        need = int(self.idle_cooldown * tick_hz)
        if now_tick - last_tick >= max(1, need):
            st["last_tick"] = now_tick
            return True
        return False

    # ----------------- действия → цели -----------------
    def _apply_action_goal(self, action: str, aid: str, ax: float, ay: float,
                           ex: float, ey: float, safe_point: Tuple[float, float]):
        """
        Ставит цель агенту в зависимости от выбранного действия.
        Выполняется под world_lock в simulation_loop (через world.set_agent_goal).
        """
        # вектор на врага и от врага
        dx, dy = ex - ax, ey - ay
        d = math.hypot(dx, dy) + 1e-6
        ux, uy = dx / d, dy / d
        if action == "attack":
            # идём прямо к противнику
            world.set_agent_goal(aid, ex, ey)
        elif action == "block":
            # остаёмся примерно на месте, маленький сдвиг назад
            world.set_agent_goal(aid, ax - ux * 0.7, ay - uy * 0.7)
        elif action == "evade":
            # смещение перпендикулярно (уклонение)
            px, py = -uy, ux
            world.set_agent_goal(aid, ax + px * 2.0, ay + py * 2.0)
        elif action == "run":
            # бежим к безопасной точке или просто от противника
            sx, sy = safe_point
            # если далеко — просто отскочим на 4 метра
            tgt_x = ax - ux * 4.0
            tgt_y = ay - uy * 4.0
            # чуть направляем в сторону лагеря
            tgt_x = (tgt_x * 0.6) + (sx * 0.4)
            tgt_y = (tgt_y * 0.6) + (sy * 0.4)
            world.set_agent_goal(aid, tgt_x, tgt_y)

    # ----------------- основной апдейт -----------------
    def apply(self, world_obj: World, snap: Dict, dt: float, tick_hz: float):
        """
        Выполняется под world_lock в simulation_loop.
        Обновляет политику и при необходимости меняет цели агентов в мире.
        """
        if not self.enabled:
            self.overlay.clear()
            return

        agents = snap.get("agents", [])
        animals = [an for an in snap.get("animals", []) if self._is_aggressive(an)]

        overlay_now: Dict[str, Dict] = {}

        # быстрый словарь агрессивных животных
        aggr_map = {}
        for an in animals:
            an_id = an.get("id") or an.get("animal_id")
            if not an_id:
                continue
            aggr_map[an_id] = an

        safe_point = getattr(config, "SAFE_POINT", (world_obj.width * 0.5, world_obj.height * 0.5))
        now_tick = getattr(world_obj, "tick_count", 0)

        for a in agents:
            aid = a.get("id") or a.get("agent_id")
            if not aid:
                continue
            if not a.get("alive", True):
                overlay_now[aid] = {"state": "idle", "enemy_id": None, "skill": float(self._state.get(aid, {}).get("skill", 0.0)), "just_hit": False}
                continue

            ax, ay = self._xy(a)
            hp = float(a.get("health", a.get("hp", 100.0)))
            fear = float(a.get("fear", 0.0))

            # найдём ближайшего агрессивного зверя
            nearest_id, nearest_dist, ex, ey = None, 1e9, 0.0, 0.0
            for an in animals:
                an_id = an.get("id") or an.get("animal_id")
                if not an_id:
                    continue
                xz, yz = self._xy(an)
                d = self._dist(ax, ay, xz, yz)
                if d < nearest_dist:
                    nearest_dist, nearest_id, ex, ey = d, an_id, xz, yz

            if nearest_id is None or nearest_dist > self.threat_radius:
                # нет боя
                overlay_now[aid] = {"state": "idle", "enemy_id": None, "skill": float(self._state.get(aid, {}).get("skill", 0.0)), "just_hit": False}
                # сбросим прошлую дистанцию, чтобы в следующий раз корректно оценить
                self._remember(aid, hp, fear, 999.0)
                continue

            # есть угроза — формируем состояние, выбираем действие
            s_key = self._state_key(fear, nearest_dist)
            action = self._choose_action(aid, s_key)

            # чтобы не спамить world.set_agent_goal слишком часто — лёгкий кулдаун
            if self._cooldown_ok(aid, now_tick, tick_hz):
                try:
                    self._apply_action_goal(action, aid, ax, ay, ex, ey, safe_point)
                except Exception:
                    pass

            # оценим награду: штраф за потерю HP, небольшой бонус за увеличение дистанции
            prev_hp, prev_dist = self._previous(aid)
            delta_hp = hp - prev_hp                 # отрицательно если нас побили
            delta_dist = nearest_dist - prev_dist   # положительно если стали дальше
            reward = (0.10 * delta_dist) + (0.0 if delta_hp >= 0 else (delta_hp * 0.6))

            # формируем следующее состояние
            s2_key = self._state_key(fear, nearest_dist)

            # Q-learning
            self._update_q(aid, s_key, action, reward, s2_key)
            skill = self._boost_skill(aid, reward)

            just_hit = (delta_hp < -0.1)

            # события для VFX/лога (вынесем за лок дальше)
            if just_hit:
                self._pending_events.append((
                    "combat_hit",
                    {"agent_id": aid, "enemy_id": nearest_id, "hp_loss": abs(delta_hp)}
                ))
            # не спамим одинаковыми "action" каждую итерацию
            last_action = self._state.setdefault(aid, {}).get("last_action")
            if last_action != action:
                self._pending_events.append((
                    "combat_state",
                    {"agent_id": aid, "state": action, "enemy_id": nearest_id}
                ))
                self._state[aid]["last_action"] = action

            # обновим память
            self._remember(aid, hp, fear, nearest_dist)

            # положим в публичное оверлей-поле
            overlay_now[aid] = {
                "state": action,
                "enemy_id": nearest_id,
                "skill": float(skill),
                "just_hit": bool(just_hit),
            }

        # сохранить слой для инъекции
        self.overlay = overlay_now

    def pop_events(self) -> List[Tuple[str, Dict]]:
        evs = self._pending_events[:]
        self._pending_events.clear()
        return evs


# глобальный экземпляр
combat = CombatSystem()


# =============================================================================
# 12. WEBSOCKET API
# =============================================================================
async def handle_subscribe(ws: WebSocket):
    """
    Клиент отправил type="subscribe".
    1) Отправляем ack (type="ack").
    2) Мгновенно шлём текущее состояние (type="world_state").
    """
    ack = ServerMessage(
        type="ack",
        data={"subscribed": True},
    ).model_dump_json()
    await manager.safe_send_text(ws, ack)

    snap_msg = await _safe_snapshot_json()
    await manager.safe_send_text(ws, snap_msg)


async def handle_ping(ws: WebSocket):
    """Клиент прислал ping → отвечаем pong (keepalive)."""
    pong = ServerMessage(
        type="pong",
        data={},
    ).model_dump_json()
    await manager.safe_send_text(ws, pong)


async def handle_set_goal(ws: WebSocket, msg: ClientMessage):
    """
    Клиент (observer в UI) указывает цель агенту (ПКМ на карте).
    Server-authoritative: clamp координат, world.set_agent_goal(...), ack/error.
    """
    changed = False

    if msg.agent_id and msg.goal:
        try:
            gx_raw = float(msg.goal.get("x"))
            gy_raw = float(msg.goal.get("y"))
            gx, gy = _clamp_to_world(gx_raw, gy_raw)
        except Exception:
            gx, gy = None, None

        if gx is not None and gy is not None:
            async with world_lock:
                changed = world.set_agent_goal(msg.agent_id, gx, gy)

    if changed:
        resp = ServerMessage(
            type="ack",
            data={"set_goal": "ok", "agent_id": msg.agent_id},
        )
    else:
        resp = ServerMessage(
            type="error",
            data={"error": "agent_not_found_or_bad_goal"},
        )

    await manager.safe_send_text(ws, resp.model_dump_json())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """
    Главный WebSocket-эндпоинт.
    ВХОД:
      - "subscribe"
      - "ping"
      - "set_goal": {agent_id, goal:{x,y}}
    ВЫХОД:
      - "ack" / "error" / "pong" / "world_state"
    """
    await manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = ClientMessage.model_validate_json(raw)
            except Exception as e:
                err = ServerMessage(
                    type="error",
                    data={"error": f"bad_request {e}"},
                ).model_dump_json()
                await manager.safe_send_text(ws, err)
                continue

            if msg.type == "subscribe":
                await handle_subscribe(ws)
            elif msg.type == "ping":
                await handle_ping(ws)
            elif msg.type == "set_goal":
                await handle_set_goal(ws, msg)
            else:
                err = ServerMessage(
                    type="error",
                    data={"error": f"unknown_type {msg.type}"},
                ).model_dump_json()
                await manager.safe_send_text(ws, err)

    except WebSocketDisconnect:
        manager.disconnect(ws)


# =============================================================================
# 13. ЦИКЛ СИМУЛЯЦИИ
# =============================================================================
async def simulation_loop():
    """
    Главный цикл симуляции:
      - world.tick() двигает агентов и животных,
        обновляет страх/голод/цели, фиксит укусы/смерти
        и заполняет world.event_log / chat_log.
      - формируем snapshot
      - шлём snapshot всем подписчикам (WS и SSE)
      - при включённом LIVE_SKILL_LEARNING выполняем мягкое дообучение
      - NEW: применяем CombatSystem и вплетаем контекст «боёвки»
    """
    tick_hz = getattr(config, "TICK_RATE_HZ", 5.0)
    if tick_hz <= 0:
        tick_hz = 5.0
    tick_delay = 1.0 / tick_hz

    while True:
        try:
            # 1) стикать мир и сразу снять снапшот под локом
            async with world_lock:
                world.tick()

                # Мягкое онлайн-дообучение навыков (если включено)
                if bool(getattr(config, "LIVE_SKILL_LEARNING", True)):
                    steps = int(getattr(config, "LIVE_SKILL_LEARNING_STEPS", 1))
                    # Итерируемся по агентам независимо от структуры world.agents
                    for ag in _iter_agents():
                        br = getattr(ag, "brain", None)
                        if br and hasattr(br, "policy_learn_online"):
                            try:
                                br.policy_learn_online(steps=steps)
                            except Exception:
                                # не роняем мир из-за одного агента
                                pass

                snap = world.snapshot()

                # NEW: применяем CombatSystem прямо под локом (он ставит цели)
                combat.apply(world, snap, dt=tick_delay, tick_hz=tick_hz)

            # 2) инъекция боевого контекста уже за пределами лока
            snap = _inject_combat_into_snapshot(snap, combat.overlay)

            # 3) подготовить json-пакет
            msg_out = ServerMessage(
                type="world_state",
                data=snap,
            ).model_dump_json()

            # 4) широковещательно отдать подписчикам WebSocket
            if manager.active:
                await manager.broadcast_text(msg_out)

            # 5) и подписчикам SSE (curl -N)
            if sse.clients:
                await sse.broadcast(msg_out)

            # 6) NEW: рассылаем накопленные combat-события
            pending = combat.pop_events()
            for kind, payload in pending:
                await push_global_event(kind, payload)

        except Exception as loop_err:
            # Не даём игре упасть из-за одной ошибки, но логируем.
            print(f"[simulation_loop] error: {loop_err}")

        # пауза до следующего тика
        await asyncio.sleep(tick_delay)


@app.on_event("startup")
async def on_startup():
    """При старте сервера — запускаем симуляционный цикл."""
    asyncio.create_task(simulation_loop())


# =============================================================================
# 14. Локальный запуск через uvicorn (python server.py)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
