# village_map.py
"""
Процедурный генератор «деревни» и окружения для мира.

Новое:
- Никаких жёстких координат: всё масштабируется под world_w/world_h.
- Расширенный набор POI (дом Echo, дом Nova, костровая площадь, мастерская,
  склад еды, смотровая вышка, колодец, святилище, охотничья будка,
  ягодное поле, тёмный лес, старые руины, токсичная лужа, логово зверя).
- Каждый POI → WorldObject (+ danger/comfort/resource поля).
- activity_registry дополняется «effects» (что даёт посещение точки: hp/energy/fear/hunger).
- Процедурные пути: waypoints от базы к периферийным POI + кольцевая тропа.
  Строится nav-граф (nodes/edges). Пишем в world.set_nav_graph(...) если есть,
  иначе кладём под ключ '_nav_graph' в world.activities.
- seed для детерминизма.

Агенты могут использовать world.activities для выбора целей
(по activity_tags, resource_tag, effects), а nav-граф — для простого роутинга.

Ожидаемые API:
- class World: width, height; add_object(WorldObject); set_activity_registry(dict)
  (опционально) set_nav_graph(dict) или (опционально) set_meta(key, val)
- class WorldObject(obj_id, name, kind, x, y, radius,
                    danger_level=..., comfort_level=...,
                    resource_tag=None, resource_abundance=0.0)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import random
import math

from world import World, WorldObject


# =============================================================================
# Внутренние хелперы генерации координат/геометрии
# =============================================================================

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _jitter(base_x: float, base_y: float, spread: float) -> Tuple[float, float]:
    """Чуть шумим координату (для «живости» раскладки)."""
    return (
        base_x + random.uniform(-spread, spread),
        base_y + random.uniform(-spread, spread),
    )


def _ring_point(
    cx: float,
    cy: float,
    world_w: float,
    world_h: float,
    min_frac: float,
    max_frac: float,
    margin: float,
) -> Tuple[float, float]:
    """
    Случайная точка в кольце вокруг центра.
    min_frac/max_frac — доли от min(world_w, world_h).
    """
    base_scale = min(world_w, world_h)
    r = random.uniform(base_scale * min_frac, base_scale * max_frac)
    ang = random.uniform(0.0, 2.0 * math.pi)
    x = cx + math.cos(ang) * r
    y = cy + math.sin(ang) * r
    x = _clamp(x, margin, world_w - margin)
    y = _clamp(y, margin, world_h - margin)
    return x, y


def _line_waypoints(
    ax: float, ay: float, bx: float, by: float,
    step: float,
) -> List[Tuple[float, float]]:
    """
    Дискретизация отрезка A→B узлами через каждые ~step.
    Не включает стартовую точку (чтобы не дублировать POI у базы).
    """
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return []
    n = max(1, int(dist // step))
    res: List[Tuple[float, float]] = []
    for i in range(1, n + 1):
        t = i / (n + 1) if n > 1 else i / (n + 1)
        res.append((ax + dx * t, ay + dy * t))
    return res


def _ring_waypoints(
    cx: float, cy: float, radius: float, count: int
) -> List[Tuple[float, float]]:
    """Кольцо точек вокруг (cx,cy)."""
    out: List[Tuple[float, float]] = []
    for i in range(count):
        a = (2.0 * math.pi) * (i / count)
        out.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
    return out


# =============================================================================
# POI модель
# =============================================================================

@dataclass
class VillagePOI:
    """
    Точка интереса (Point Of Interest).

    id, name, semantic_role        — идентичность/смысл
    activity_tags                  — что можно делать (rest, sleep, heal, eat, craft, loot, gather_wood,...)
    x, y, radius                   — геометрия зоны
    comfort_level, danger_level    — ощущения/риск для мозга агента (0..1)
    kind_for_client                — визуальный тип для клиента: "safe" / "hazard" / "neutral"
    resource_tag, resource_abundance — ресурс и его «наполненность» (0..1)

    Дополнительно: effects — предполагаемые эффекты посещения точки
      (hp_delta, energy_delta, fear_delta, hunger_delta) — могут быть флотами или 0.
    """
    id: str
    name: str
    semantic_role: str
    activity_tags: List[str]

    x: float
    y: float
    radius: float

    comfort_level: float
    danger_level: float
    kind_for_client: str  # "safe", "hazard", "neutral"

    resource_tag: Optional[str] = None
    resource_abundance: float = 0.0

    # новые подсказки для планировщика
    effects: Dict[str, float] = None  # {'hp_delta':..., 'energy_delta':..., 'fear_delta':..., 'hunger_delta':...}

    def to_world_object(self) -> WorldObject:
        return WorldObject(
            obj_id=self.id,
            name=self.name,
            kind=self.kind_for_client,
            x=self.x,
            y=self.y,
            radius=self.radius,
            danger_level=self.danger_level,
            comfort_level=self.comfort_level,
            resource_tag=self.resource_tag,
            resource_abundance=self.resource_abundance,
        )

    def to_activity_record(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "semantic_role": self.semantic_role,
            "activity_tags": list(self.activity_tags),
            "area": {"x": self.x, "y": self.y, "radius": self.radius},
            "danger_level": self.danger_level,
            "comfort_level": self.comfort_level,
            "resource_tag": self.resource_tag,
            "resource_abundance": self.resource_abundance,
            "effects": dict(self.effects or {}),
        }


# =============================================================================
# Процедурная сборка списка POI
# =============================================================================

def _build_poi_list(world_w: float, world_h: float) -> List[VillagePOI]:
    """
    Генерирует «базу» (кластер около центра) и «периферию» (ресурсы/опасности) по кольцу.
    """

    poi: List[VillagePOI] = []

    # масштабы
    min_dim = min(world_w, world_h)
    base_spread = min_dim * 0.02      # шум построек
    base_scale = min_dim * 0.10       # «разнесение» построек базы
    margin = 5.0
    cx = world_w * 0.5
    cy = world_h * 0.5

    def cluster(dx: float, dy: float, spread_mul: float = 1.0) -> Tuple[float, float]:
        bx = cx + dx * base_scale
        by = cy + dy * base_scale
        return _jitter(bx, by, base_spread * spread_mul)

    # радиусы зон (относительно размера мира)
    r_house     = min_dim * 0.05
    r_camp      = min_dim * 0.06
    r_workshop  = min_dim * 0.04
    r_storage   = min_dim * 0.04
    r_tower     = min_dim * 0.03
    r_well      = min_dim * 0.035
    r_shrine    = min_dim * 0.035
    r_hut       = min_dim * 0.04

    r_field     = min_dim * 0.06
    r_forest    = min_dim * 0.10
    r_ruins     = min_dim * 0.07
    r_toxic     = min_dim * 0.05
    r_den       = min_dim * 0.045

    # -----------------------------------------------------------------
    # БАЗА
    # -----------------------------------------------------------------

    # Дом Echo — лечение/сон/успокоение
    hx, hy = cluster(-1.0, +0.5)
    poi.append(VillagePOI(
        id="house_echo", name="Дом Echo", semantic_role="home",
        activity_tags=["rest", "sleep", "heal", "calm", "social"],
        x=hx, y=hy, radius=r_house,
        comfort_level=0.85, danger_level=0.0, kind_for_client="safe",
        effects={"hp_delta": +0.15, "energy_delta": +0.10, "fear_delta": -0.20, "hunger_delta": 0.0},
    ))

    # Дом Nova
    nx, ny = cluster(+1.0, +0.5)
    poi.append(VillagePOI(
        id="house_nova", name="Дом Nova", semantic_role="home",
        activity_tags=["rest", "sleep", "heal", "calm", "social"],
        x=nx, y=ny, radius=r_house,
        comfort_level=0.85, danger_level=0.0, kind_for_client="safe",
        effects={"hp_delta": +0.12, "energy_delta": +0.12, "fear_delta": -0.18, "hunger_delta": 0.0},
    ))

    # Костровая площадь — социальный хаб
    fx, fy = cluster(0.0, 0.0)
    poi.append(VillagePOI(
        id="campfire", name="Площадь у костра", semantic_role="campfire",
        activity_tags=["rest", "calm", "social", "share_info"],
        x=fx, y=fy, radius=r_camp,
        comfort_level=0.6, danger_level=0.0, kind_for_client="safe",
        effects={"hp_delta": +0.05, "energy_delta": +0.06, "fear_delta": -0.25, "hunger_delta": 0.0},
    ))

    # Мастерская — крафт/ремонт
    wx, wy = cluster(0.0, -1.0)
    poi.append(VillagePOI(
        id="workshop", name="Мастерская", semantic_role="workshop",
        activity_tags=["craft", "repair_self", "repair_tools"],
        x=wx, y=wy, radius=r_workshop,
        comfort_level=0.2, danger_level=0.0, kind_for_client="safe",
        resource_tag="scrap", resource_abundance=random.uniform(0.4, 0.8),
        effects={"hp_delta": +0.02, "energy_delta": -0.03, "fear_delta": -0.05, "hunger_delta": 0.0},
    ))

    # Склад еды
    sx, sy = cluster(0.0, +1.0)
    poi.append(VillagePOI(
        id="food_storage", name="Склад еды", semantic_role="food_storage",
        activity_tags=["eat", "restock_food", "share_food"],
        x=sx, y=sy, radius=r_storage,
        comfort_level=0.4, danger_level=0.0, kind_for_client="safe",
        resource_tag="food", resource_abundance=random.uniform(0.6, 1.0),
        effects={"hp_delta": +0.03, "energy_delta": +0.08, "fear_delta": -0.02, "hunger_delta": -0.35},
    ))

    # Смотровая вышка
    tx, ty = cluster(+2.0, -0.5, spread_mul=1.5)
    poi.append(VillagePOI(
        id="watch_tower", name="Смотровая вышка", semantic_role="scout_tower",
        activity_tags=["scout", "warn", "share_info"],
        x=tx, y=ty, radius=r_tower,
        comfort_level=0.1, danger_level=0.0, kind_for_client="safe",
        effects={"hp_delta": 0.0, "energy_delta": -0.04, "fear_delta": -0.06, "hunger_delta": 0.0},
    ))

    # Колодец — вода/энергия, небольшой комфорт
    wx2, wy2 = cluster(-0.6, -1.4)
    poi.append(VillagePOI(
        id="water_well", name="Колодец", semantic_role="water",
        activity_tags=["drink", "refill_water", "rest"],
        x=wx2, y=wy2, radius=r_well,
        comfort_level=0.35, danger_level=0.0, kind_for_client="safe",
        resource_tag="water", resource_abundance=random.uniform(0.7, 1.0),
        effects={"hp_delta": +0.02, "energy_delta": +0.10, "fear_delta": -0.03, "hunger_delta": 0.0},
    ))

    # Святилище — сильное снижение страха, чуть лечит
    shx, shy = cluster(+1.2, -1.6)
    poi.append(VillagePOI(
        id="shrine", name="Святилище", semantic_role="shrine_calm",
        activity_tags=["pray", "calm", "rest"],
        x=shx, y=shy, radius=r_shrine,
        comfort_level=0.55, danger_level=0.0, kind_for_client="safe",
        effects={"hp_delta": +0.04, "energy_delta": +0.02, "fear_delta": -0.35, "hunger_delta": 0.0},
    ))

    # Охотничья будка — рядом с базой, про «инструменты/еду»
    hhx, hhy = cluster(-1.8, -0.2)
    poi.append(VillagePOI(
        id="hunter_hut", name="Охотничья будка", semantic_role="hunter_hut",
        activity_tags=["craft", "prepare_food", "scout"],
        x=hhx, y=hhy, radius=r_hut,
        comfort_level=0.25, danger_level=0.05, kind_for_client="neutral",
        resource_tag="tools", resource_abundance=random.uniform(0.3, 0.7),
        effects={"hp_delta": +0.01, "energy_delta": -0.02, "fear_delta": -0.04, "hunger_delta": -0.05},
    ))

    # -----------------------------------------------------------------
    # ПЕРИФЕРИЯ
    # -----------------------------------------------------------------

    # Поле ягод
    bfx, bfy = _ring_point(cx, cy, world_w, world_h, min_frac=0.25, max_frac=0.35, margin=margin)
    poi.append(VillagePOI(
        id="berry_field", name="Поле ягод", semantic_role="gather_food_zone",
        activity_tags=["gather_food", "forage", "eat"],
        x=bfx, y=bfy, radius=r_field,
        comfort_level=0.10, danger_level=0.15, kind_for_client="neutral",
        resource_tag="food", resource_abundance=random.uniform(0.5, 0.9),
        effects={"hp_delta": +0.02, "energy_delta": +0.03, "fear_delta": -0.02, "hunger_delta": -0.25},
    ))

    # Тёмный лес
    dfx, dfy = _ring_point(cx, cy, world_w, world_h, min_frac=0.30, max_frac=0.45, margin=margin)
    poi.append(VillagePOI(
        id="dark_forest", name="Тёмный лес", semantic_role="wood_zone",
        activity_tags=["gather_wood", "forage", "scout"],
        x=dfx, y=dfy, radius=r_forest,
        comfort_level=0.0, danger_level=0.40, kind_for_client="hazard",
        resource_tag="wood", resource_abundance=random.uniform(0.5, 1.0),
        effects={"hp_delta": -0.03, "energy_delta": -0.03, "fear_delta": +0.05, "hunger_delta": 0.0},
    ))

    # Старые руины
    rux, ruy = _ring_point(cx, cy, world_w, world_h, min_frac=0.40, max_frac=0.50, margin=margin)
    poi.append(VillagePOI(
        id="old_ruins", name="Старые руины", semantic_role="ruins_loot",
        activity_tags=["loot", "scout", "share_info"],
        x=rux, y=ruy, radius=r_ruins,
        comfort_level=0.0, danger_level=1.0, kind_for_client="hazard",
        resource_tag="loot", resource_abundance=random.uniform(0.4, 0.8),
        effects={"hp_delta": -0.10, "energy_delta": -0.10, "fear_delta": +0.20, "hunger_delta": 0.0},
    ))

    # Токсичная лужа
    tox_x, tox_y = _ring_point(cx, cy, world_w, world_h, min_frac=0.35, max_frac=0.55, margin=margin)
    poi.append(VillagePOI(
        id="toxic_pool", name="Токсичная лужа", semantic_role="toxic_zone",
        activity_tags=["death_zone", "avoid", "warn_others"],
        x=tox_x, y=tox_y, radius=r_toxic,
        comfort_level=0.0, danger_level=0.9, kind_for_client="hazard",
        effects={"hp_delta": -0.20, "energy_delta": -0.05, "fear_delta": +0.25, "hunger_delta": 0.0},
    ))

    # Логово зверя — источник опасности рядом с ресурсами
    den_x, den_y = _ring_point(cx, cy, world_w, world_h, min_frac=0.28, max_frac=0.42, margin=margin)
    poi.append(VillagePOI(
        id="beast_den", name="Логово зверя", semantic_role="beast_den",
        activity_tags=["avoid", "warn_others", "hunt"],
        x=den_x, y=den_y, radius=r_den,
        comfort_level=0.0, danger_level=0.75, kind_for_client="hazard",
        effects={"hp_delta": -0.12, "energy_delta": -0.08, "fear_delta": +0.18, "hunger_delta": 0.0},
    ))

    return poi


# =============================================================================
# Навигационный граф (узлы/рёбра) и пути
# =============================================================================

def _build_nav_graph(
    poi_list: List[VillagePOI],
    world_w: float,
    world_h: float,
) -> Dict[str, Any]:
    """
    Строим простой граф:
      - узлы: все POI (type='poi') + waypoints вдоль путей (type='wp')
      - рёбра: между последовательными точками на каждом пути + POI ↔ ближайший wp
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Tuple[str, str]] = []

    # удобный индекс по id → POI
    by_id = {p.id: p for p in poi_list}
    if "campfire" not in by_id:
        # без базы пути не строим
        return {"nodes": nodes, "edges": edges}

    camp = by_id["campfire"]
    min_dim = min(world_w, world_h)
    step = max(min_dim * 0.05, 3.0)  # шаг дискретизации дороги
    wp_radius = max(min_dim * 0.01, 1.2)

    # 1) Добавляем узлы-POI
    for p in poi_list:
        nodes[p.id] = {"id": p.id, "x": p.x, "y": p.y, "type": "poi"}

    # 2) Пути от костра к удалённым зонам
    targets = [
        "berry_field", "dark_forest", "old_ruins", "toxic_pool", "beast_den",
        "water_well", "workshop", "food_storage", "house_echo", "house_nova", "shrine", "hunter_hut",
    ]
    wp_counter = 0

    def add_path(a_id: str, b_id: str):
        nonlocal wp_counter
        if a_id not in by_id or b_id not in by_id:
            return
        ax, ay = by_id[a_id].x, by_id[a_id].y
        bx, by = by_id[b_id].x, by_id[b_id].y
        points = _line_waypoints(ax, ay, bx, by, step=step)
        prev = a_id
        for (x, y) in points:
            wp_id = f"wp_{wp_counter}"
            wp_counter += 1
            nodes[wp_id] = {"id": wp_id, "x": x, "y": y, "type": "wp", "r": wp_radius}
            edges.append((prev, wp_id))
            prev = wp_id
        edges.append((prev, b_id))

    for t in targets:
        add_path("campfire", t)

    # 3) Кольцевая тропа вокруг базы (для патрулей/прогулок)
    ring_r = step * 2.5
    ring_pts = _ring_waypoints(camp.x, camp.y, ring_r, count=8)
    ring_ids: List[str] = []
    for (x, y) in ring_pts:
        wp_id = f"wp_{wp_counter}"
        wp_counter += 1
        nodes[wp_id] = {"id": wp_id, "x": x, "y": y, "type": "wp", "r": wp_radius}
        ring_ids.append(wp_id)
    for i in range(len(ring_ids)):
        a = ring_ids[i]
        b = ring_ids[(i + 1) % len(ring_ids)]
        edges.append((a, b))
    # связать кольцо с костром
    if ring_ids:
        edges.append((ring_ids[0], "campfire"))
        edges.append(("campfire", ring_ids[len(ring_ids) // 2]))

    return {"nodes": nodes, "edges": edges}


# =============================================================================
# Публичная точка входа
# =============================================================================

def attach_village(world: World, *, seed: Optional[int] = None, with_paths: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Создаёт окружение для данного мира.

    1) Генерируем список POI (база + периферия).
    2) Каждый POI → WorldObject и добавляем через world.add_object(...).
    3) Собираем activity_registry:
       - activity_tags, semantic_role
       - comfort_level/danger_level
       - resource_tag/resource_abundance
       - effects (hp/energy/fear/hunger дельты)
       - area (x,y,radius)
       world.set_activity_registry(registry)
    4) (опц.) Строим навигационный граф (POI + waypoints):
       - если есть world.set_nav_graph → положим туда,
         иначе registry['_nav_graph'] = {...}

    Возвращает сам registry.
    """
    if seed is not None:
        random.seed(seed)

    poi_list = _build_poi_list(world.width, world.height)

    # 1. Физические зоны в мир
    for poi in poi_list:
        world.add_object(poi.to_world_object())

    # 2. Реестр активностей
    registry: Dict[str, Dict[str, Any]] = {}
    for poi in poi_list:
        registry[poi.id] = poi.to_activity_record()

    # 3. Навигация
    if with_paths:
        nav_graph = _build_nav_graph(poi_list, world.width, world.height)
        # предпочтительно — отдельным полем мира
        if hasattr(world, "set_nav_graph"):
            try:
                world.set_nav_graph(nav_graph)
            except Exception:
                # fallback — сохранить в activities под спецключом
                registry["_nav_graph"] = nav_graph
        else:
            # другой возможный API
            if hasattr(world, "set_meta"):
                try:
                    world.set_meta("nav_graph", nav_graph)
                except Exception:
                    registry["_nav_graph"] = nav_graph
            else:
                registry["_nav_graph"] = nav_graph

    # 4. Регистрируем активности в мире
    world.set_activity_registry(registry)

    return registry
