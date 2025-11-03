# procgen.py
#
# Генерация окружения мира:
#   - безопасные лагеря разного качества
#   - опасные зоны
#   - ресурсные точки
#   - activity_registry для агентов (heal/rest/eat/calm/...),
#     которое world.pick_new_goal() уже понимает.
#
# Идея:
#   - ближе к центру: "средненькие" убежища, мало еды
#   - дальше: "жирные" убежища (больше comfort_level, еда),
#     но рядом сильные hazard-ы
#
# Это создаёт естественный стимул уходить в экспедиции:
#   чтобы хорошо восстановиться или поесть,
#   надо оторваться от стартовой зоны и рискнуть.

from __future__ import annotations
from typing import Dict, Any, List, Tuple
import random
import math

from world import World, WorldObject

def _rand_circle_around(cx: float, cy: float, min_r: float, max_r: float) -> Tuple[float, float]:
    """
    Возвращает (x,y), равномерно в кольце радиусов [min_r, max_r] вокруг (cx,cy).
    """
    ang = random.uniform(0.0, 2.0 * math.pi)
    r = random.uniform(min_r, max_r)
    return (cx + math.cos(ang)*r, cy + math.sin(ang)*r)

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def procedural_populate_world(
    world: World,
    seed: int,
    *,
    num_safe_clusters: int = 3,
    num_hazard_clusters: int = 4,
    num_resource_spots: int = 4,
) -> Dict[str, Dict[str, Any]]:
    """
    Наполняет world.objects и возвращает activity_registry.
    Возвращаем то, что потом можно скормить world.set_activity_registry().

    Алгоритм:
    - Берём центр мира как "базовый лагерь".
    - Делаем несколько safe-зон ближе и дальше от центра, с разным comfort_level.
    - Делаем hazard-зоны вокруг вкусных safe-зон (risk / reward).
    - Кидаем ресурсные точки.
    - Для каждой safe-зоны регистрируем activity_tags (heal/rest/eat/calm),
      чтобы агент мог целиться туда как в "медпункт" / "еду" / "отдых".
    """

    random.seed(seed)

    w = float(world.width)
    h = float(world.height)
    cx = w * 0.5
    cy = h * 0.5

    activity_registry: Dict[str, Dict[str, Any]] = {}

    objects_local: List[WorldObject] = []

    # -----------------------------------------------
    # 1. Безопасные кластеры (убежища / костры / склад еды)
    #    Ближе к центру -> comfort_level пониже
    #    Дальше -> comfort_level повыше (то есть выгоднее туда сходить)
    # -----------------------------------------------

    for i in range(num_safe_clusters):
        # радиус от центра: внутренние кэмпы и дальние базы
        # кластер i=0 может быть около центра, i=последний - далеко
        base_min_r = (i / max(1, num_safe_clusters-1)) * (min(w, h) * 0.25)
        base_max_r = base_min_r + (min(w, h) * 0.15)
        sx, sy = _rand_circle_around(cx, cy, base_min_r, base_max_r)

        # clamp координаты в пределах мира
        sx = _clamp(sx, 5.0, w - 5.0)
        sy = _clamp(sy, 5.0, h - 5.0)

        radius = random.uniform(4.0, 9.0)

        # комфорт тем выше, чем дальше этот лагерь от центра:
        # (примерно = "экспедиционная база с норм лечилкой/едой")
        dist_from_center = math.hypot(sx - cx, sy - cy)
        dist_norm = dist_from_center / (0.5 * math.hypot(w, h))
        comfort = _clamp(0.3 + dist_norm * 0.7, 0.3, 1.0)

        # наличие еды зависит от comfort тоже
        has_food = comfort > 0.5

        obj_id = f"safe_{i}"
        name = random.choice([
            "Лагерь",
            "Укрытие",
            "Стоянка",
            "Склад провианта",
            "Полевой медпункт",
        ]) + f"_{i}"

        safe_obj = WorldObject(
            obj_id=obj_id,
            name=name,
            kind="safe",
            x=sx,
            y=sy,
            radius=radius,
            danger_level=0.0,
            comfort_level=comfort,
            resource_tag="food" if has_food else None,
            resource_abundance=(0.6 + 0.4*random.random()) if has_food else 0.0,
        )
        objects_local.append(safe_obj)

        # activity_tags подбираем по сути "чем это место полезно"
        tags = ["rest", "calm", "heal"]
        if has_food:
            tags.append("eat")

        activity_registry[obj_id] = {
            "name": name,
            "activity_tags": tags,
            "comfort_level": comfort,
            "danger_level": 0.0,
            "area": {
                "x": sx,
                "y": sy,
                "radius": radius,
            },
        }

    # -----------------------------------------------
    # 2. Опасные кластеры (ядовитые пятна, огонь, аномалии)
    #    Делаем их:
    #      - частью "естественного шума"
    #      - и также прижимаем некоторые прям к вкусным safe-зонам
    #        => "ресурсы есть, но страшно"
    # -----------------------------------------------

    for i in range(num_hazard_clusters):
        # половина hazard'ов будет привязана к какому-то safe-лагерю, чтобы создать риск-награда
        if objects_local and random.random() < 0.5:
            anchor = random.choice([o for o in objects_local if o.kind == "safe"])
            # hazard ближе к лагерю, но не прямо сверху
            hx, hy = _rand_circle_around(anchor.x, anchor.y,
                                         min_r=anchor.radius + 2.0,
                                         max_r=anchor.radius + 12.0)
        else:
            # свободно в мире, чуть рандомно
            hx = random.uniform(5.0, w - 5.0)
            hy = random.uniform(5.0, h - 5.0)

        hazard_radius = random.uniform(3.0, 7.0)
        hazard_power = random.uniform(0.4, 1.0)  # насколько больно

        hname = random.choice([
            "огонь",
            "ядовитая лужа",
            "радиационное пятно",
            "электрическая арка",
            "газовая утечка",
        ]) + f"_{i}"

        haz_obj = WorldObject(
            obj_id=f"hazard_{i}",
            name=hname,
            kind="hazard",
            x=hx,
            y=hy,
            radius=hazard_radius,
            danger_level=hazard_power,
            comfort_level=0.0,
        )
        objects_local.append(haz_obj)

    # -----------------------------------------------
    # 3. Ресурсные точки (neutral)
    #    Это такие "сундуки лута". Они не лечат, но могут использоваться
    #    в будущем под крафт/ремонт. Пока они агенту не критичны,
    #    но мы уже можем включить их в activities с тегом "restock_food"
    #    или чем-то похожим, чтобы pick_new_goal() мог их рассматривать как еду.
    # -----------------------------------------------

    for i in range(num_resource_spots):
        rx = random.uniform(5.0, w - 5.0)
        ry = random.uniform(5.0, h - 5.0)
        rr = random.uniform(2.0, 4.0)

        rtype = random.choice([
            ("food_cache",  "Тайник с едой",        "eat"),
            ("med_crate",   "Медицинский ящик",     "heal"),
            ("scrap_heap",  "Куча металлолома",     None),
            ("supply_drop", "Сброшенный контейнер", "eat"),
        ])

        tag_internal, human_name, maybe_tag = rtype

        abundance = 0.3 + 0.7 * random.random()  # насколько этот тайник ещё "богат"

        res_obj = WorldObject(
            obj_id=f"res_{i}",
            name=f"{human_name}_{i}",
            kind="neutral",
            x=rx,
            y=ry,
            radius=rr,
            danger_level=0.0,
            comfort_level=(0.2 if maybe_tag == "heal" else 0.1 if maybe_tag == "eat" else 0.0),
            resource_tag=("food" if maybe_tag == "eat" else "meds" if maybe_tag == "heal" else "scrap"),
            resource_abundance=abundance,
        )
        objects_local.append(res_obj)

        # если это еда/медпомощь — даём его в activity_registry,
        # чтобы агенты тоже его могли рассматривать как цель.
        if maybe_tag in ("eat", "heal"):
            activity_registry[res_obj.obj_id] = {
                "name": res_obj.name,
                "activity_tags": [maybe_tag, "rest", "calm"],
                "comfort_level": res_obj.comfort_level,
                "danger_level": res_obj.danger_level,
                "area": {
                    "x": rx,
                    "y": ry,
                    "radius": rr,
                },
            }

    # -----------------------------------------------
    # Применяем в мир
    # -----------------------------------------------

    for obj in objects_local:
        world.add_object(obj)

    return activity_registry
