# env_lowpoly.py
"""
Генератор лоуполи-деревни для движка MiniMatrixEngine.

Фичи:
- Автопозиционирование объектов относительно размеров мира (world_w/world_h)
- Кластер домов + костёр
- Дорожки (roads) от деревни к озеру и роще
- Озеро овальной формы, «опасная» зона у воды
- Роща с анти-слипанием (простая репульсия), случайные камни/бревна
- «Безопасная» зона у костра
- Сид random для детерминизма (опционально)

Типы kind, которые отдаём в движок:
  "house", "fire", "lake", "tree", "rock", "log", "road",
  "zone_safe", "zone_hazard"
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math
import random

from engine3d import Vec3, StaticMeshInstance

TAU = 2.0 * math.pi


# ---------------------------- helpers ---------------------------------

def _yaw_towards(ax: float, az: float, bx: float, bz: float) -> float:
    """yaw (радианы) от A к B в плоскости XZ"""
    return math.atan2((bz - az), (bx - ax))

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _in_bounds(x: float, z: float, w: float, h: float, margin: float = 1.0) -> bool:
    return (margin <= x <= w - margin) and (margin <= z <= h - margin)

def _mk(kind: str, x: float, z: float, yaw: float = 0.0,
        sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> StaticMeshInstance:
    return StaticMeshInstance(kind=kind, pos=Vec3(x, 0.0, z), yaw=yaw, scale=Vec3(sx, sy, sz))

def _mk_zone(kind: str, x: float, z: float, rx: float, rz: float) -> StaticMeshInstance:
    """Эллипс-зона: масштабом кодируем радиусы по X/Z, Y не важен"""
    return StaticMeshInstance(kind=kind, pos=Vec3(x, 0.0, z), yaw=0.0, scale=Vec3(rx, 1.0, rz))

def _scatter_non_overlapping(
    count: int,
    center: Tuple[float, float],
    radius: float,
    min_dist: float,
    world_w: float,
    world_h: float,
    max_attempts_per_point: int = 20,
) -> List[Tuple[float, float]]:
    """
    Простое «Пуассон-подобное» разбрасывание точек в круге:
    стараемся держать минимум расстояния между точками.
    """
    cx, cz = center
    pts: List[Tuple[float, float]] = []
    for _ in range(count):
        placed = False
        for _try in range(max_attempts_per_point):
            ang = random.random() * TAU
            r = radius * (0.15 + 0.85 * random.random())
            x = cx + r * math.cos(ang)
            z = cz + r * math.sin(ang)

            if not _in_bounds(x, z, world_w, world_h, margin=0.5):
                continue

            ok = True
            for (px, pz) in pts:
                if (px - x) * (px - x) + (pz - z) * (pz - z) < (min_dist * min_dist):
                    ok = False
                    break
            if ok:
                pts.append((x, z))
                placed = True
                break

        if not placed and pts:
            # fallback: возьмём ближайшую уже существующую + небольшой шум
            bx, bz = random.choice(pts)
            pts.append((bx + random.uniform(-min_dist, min_dist)*0.4,
                        bz + random.uniform(-min_dist, min_dist)*0.4))
    return pts


# -------------------------- themed builders ----------------------------

def _cluster_houses(cx: float, cz: float) -> List[StaticMeshInstance]:
    meshes: List[StaticMeshInstance] = []
    # лёгкая «органика» в расстановке, но стабильно вокруг центра
    layout = [
        (cx - 3.0 + random.uniform(-0.6, 0.6), cz - 2.0 + random.uniform(-0.6, 0.6), math.radians(10.0 + random.uniform(-8, 8))),
        (cx + 4.0 + random.uniform(-0.6, 0.6), cz + 1.5 + random.uniform(-0.6, 0.6), math.radians(-20.0 + random.uniform(-8, 8))),
        (cx - 2.0 + random.uniform(-0.6, 0.6), cz + 4.5 + random.uniform(-0.6, 0.6), math.radians(45.0 + random.uniform(-8, 8))),
    ]
    for (hx, hz, yaw) in layout:
        meshes.append(
            StaticMeshInstance(
                kind="house",
                pos=Vec3(hx, 0.0, hz),
                yaw=yaw,
                scale=Vec3(2.5, 2.0, 2.5),
            )
        )
    # костёр в центре деревни
    meshes.append(_mk("fire", cx + 1.2, cz + 0.6, yaw=0.0, sx=1.0, sy=1.0, sz=1.0))
    return meshes

def _campfire_decor(cx: float, cz: float) -> List[StaticMeshInstance]:
    """Камни кольцом и одно бревно рядом с костром."""
    meshes: List[StaticMeshInstance] = []
    ring_r = 1.4
    for i in range(6):
        ang = i * TAU / 6.0 + random.uniform(-0.08, 0.08)
        rx = cx + 1.2 + ring_r * math.cos(ang)
        rz = cz + 0.6 + ring_r * math.sin(ang)
        meshes.append(_mk("rock", rx, rz, yaw=random.random() * TAU, sx=0.5, sy=0.5, sz=0.5))
    # бревно-лавочка
    lx = cx - 0.4
    lz = cz + 2.2
    meshes.append(_mk("log", lx, lz, yaw=math.radians(15), sx=1.8, sy=0.5, sz=0.6))
    return meshes

def _lake_oval(x: float, z: float, rx: float, rz: float) -> StaticMeshInstance:
    """Овальный водоём: rx/rz — полуоси по X/Z. Чуть опускаем, чтобы читался берег."""
    inst = StaticMeshInstance(
        kind="lake",
        pos=Vec3(x, -0.02, z),   # слегка ниже пола
        yaw=0.0,
        scale=Vec3(rx, 1.0, rz),
    )
    return inst

def _tree_grove(center: Tuple[float, float],
                radius: float,
                count: int,
                world_w: float,
                world_h: float) -> List[StaticMeshInstance]:
    """
    Роща с простым анти-слипанием. Высоты слегка варьируются.
    """
    meshes: List[StaticMeshInstance] = []
    pts = _scatter_non_overlapping(
        count=count,
        center=center,
        radius=radius,
        min_dist=1.2,
        world_w=world_w,
        world_h=world_h,
    )
    for (tx, tz) in pts:
        h_scale = random.uniform(1.35, 2.1)
        meshes.append(
            StaticMeshInstance(
                kind="tree",
                pos=Vec3(tx, 0.0, tz),
                yaw=random.random() * TAU,
                scale=Vec3(0.8, h_scale, 0.8),
            )
        )
    return meshes

def _rocks_and_logs_area(center: Tuple[float, float],
                         radius: float,
                         rocks: int,
                         logs: int,
                         world_w: float,
                         world_h: float) -> List[StaticMeshInstance]:
    meshes: List[StaticMeshInstance] = []
    rxz = _scatter_non_overlapping(rocks, center, radius, 1.0, world_w, world_h)
    for (x, z) in rxz:
        s = random.uniform(0.5, 0.9)
        meshes.append(_mk("rock", x, z, yaw=random.random() * TAU, sx=s, sy=s, sz=s))
    lxz = _scatter_non_overlapping(logs, center, radius, 1.5, world_w, world_h)
    for (x, z) in lxz:
        meshes.append(_mk("log", x, z, yaw=random.random() * TAU, sx=1.6, sy=0.5, sz=0.5))
    return meshes

def _road_strip(ax: float, az: float, bx: float, bz: float, width: float) -> StaticMeshInstance:
    """
    Прямой «пласт» дороги от A к B:
    - центр — середина сегмента
    - yaw — вдоль направления
    - scale.x — половина длины (визуал зависит от движка)
    - scale.z — половина ширины
    """
    mx = 0.5 * (ax + bx)
    mz = 0.5 * (az + bz)
    length = max(0.1, math.hypot(bx - ax, bz - az))
    yaw = _yaw_towards(ax, az, bx, bz)
    # Примем convention: scale.x ~ половина длины; scale.z ~ половина ширины
    return _mk("road", mx, mz, yaw=yaw, sx=length * 0.5, sy=0.2, sz=width * 0.5)


# ----------------------------- public API --------------------------------

def build_lowpoly_village(
    world_w: float = 100.0,
    world_h: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> List[StaticMeshInstance]:
    """
    Сборка набора статических мешей под размеры мира.

    Параметры:
      world_w, world_h — размеры мира
      seed — фиксированный сид для воспроизводимости (по умолчанию None)

    Возвращает:
      List[StaticMeshInstance]
    """
    if seed is not None:
        random.seed(seed)

    meshes: List[StaticMeshInstance] = []

    # --- опорные точки сцены (с отступами от краёв мира)
    margin = 8.0
    vw = _clamp(world_w, 30.0, 10_000.0)
    vh = _clamp(world_h, 30.0, 10_000.0)

    # Центр деревни — в левой нижней трети мира
    village_cx = _clamp(vw * 0.22, margin, vw - margin)
    village_cz = _clamp(vh * 0.25, margin, vh - margin)

    # Озеро — правая половина, ближе к низу
    lake_cx = _clamp(vw * 0.62, margin + 6.0, vw - margin - 6.0)
    lake_cz = _clamp(vh * 0.30, margin + 4.0, vh - margin - 4.0)
    lake_rx = min(12.0, vw * 0.10)  # полуось X
    lake_rz = min(8.0, vh * 0.08)   # полуось Z

    # Роща — правый верхний квадрант
    grove_cx = _clamp(vw * 0.75, margin, vw - margin)
    grove_cz = _clamp(vh * 0.76, margin, vh - margin)
    grove_radius = min(10.0, 0.09 * min(vw, vh))

    # Небольшая группа в центре
    mid_cx = vw * 0.48 + random.uniform(-2.0, 2.0)
    mid_cz = vh * 0.52 + random.uniform(-2.0, 2.0)

    # --- деревня
    meshes += _cluster_houses(village_cx, village_cz)
    meshes += _campfire_decor(village_cx, village_cz)

    # безопасная зона у костра (приятный радиус)
    meshes.append(_mk_zone("zone_safe", village_cx + 1.2, village_cz + 0.6, rx=4.0, rz=4.0))

    # --- озеро и «опасная» зона поблизости
    meshes.append(_lake_oval(lake_cx, lake_cz, rx=lake_rx, rz=lake_rz))
    # hazard шире зеркала воды
    meshes.append(_mk_zone("zone_hazard", lake_cx, lake_cz, rx=lake_rx * 1.25, rz=lake_rz * 1.25))

    # --- рощи/деревья
    meshes += _tree_grove((grove_cx, grove_cz), grove_radius, count=14, world_w=vw, world_h=vh)
    meshes += _tree_grove((mid_cx, mid_cz), radius=4.0, count=4, world_w=vw, world_h=vh)

    # немного окружения: камни/брёвна возле рощи
    meshes += _rocks_and_logs_area((grove_cx - 3.0, grove_cz - 2.0), radius=5.0,
                                   rocks=6, logs=2, world_w=vw, world_h=vh)

    # --- дорожки: от деревни к озеру и к роще
    road_w = 2.2
    meshes.append(_road_strip(village_cx, village_cz, lake_cx, lake_cz, width=road_w))
    meshes.append(_road_strip(village_cx, village_cz, grove_cx, grove_cz, width=road_w * 0.9))

    # короткая тропинка от костра к «лавке-бревну»
    meshes.append(_road_strip(village_cx + 1.2, village_cz + 0.6,
                              village_cx - 0.4, village_cz + 2.2, width=1.2))

    return meshes
