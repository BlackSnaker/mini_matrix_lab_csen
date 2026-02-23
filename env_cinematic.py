"""
env_cinematic.py
Более насыщенное окружение для Combined App:
- крупнее и читаемее планировка;
- заметные ориентиры (башни/святилища/колодцы);
- сеть дорог и фонари;
- лесные пояса и декоративные детали.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import math
import random

from engine3d import Vec3, StaticMeshInstance
from env_lowpoly import build_lowpoly_village


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _mk(
    kind: str,
    x: float,
    z: float,
    *,
    yaw: float = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
    sz: float = 1.0,
    y: float = 0.0,
) -> StaticMeshInstance:
    return StaticMeshInstance(kind=kind, pos=Vec3(x, y, z), yaw=yaw, scale=Vec3(sx, sy, sz))


def _road(ax: float, az: float, bx: float, bz: float, width: float) -> StaticMeshInstance:
    mx = 0.5 * (ax + bx)
    mz = 0.5 * (az + bz)
    length = max(0.1, math.hypot(bx - ax, bz - az))
    yaw = math.atan2((bz - az), (bx - ax))
    # В рендере дороги ожидают scale.x как "полу-длина", scale.z как "полу-ширина".
    return _mk("road", mx, mz, yaw=yaw, sx=length * 0.5, sy=0.2, sz=width * 0.5)


def _scatter_circle(
    rnd: random.Random,
    center: Tuple[float, float],
    radius: float,
    count: int,
    world_w: float,
    world_h: float,
    margin: float,
) -> List[Tuple[float, float]]:
    cx, cz = center
    out: List[Tuple[float, float]] = []
    for _ in range(max(1, count)):
        ang = rnd.random() * math.tau
        rr = radius * (0.2 + 0.8 * rnd.random())
        x = _clamp(cx + math.cos(ang) * rr, margin, world_w - margin)
        z = _clamp(cz + math.sin(ang) * rr, margin, world_h - margin)
        out.append((x, z))
    return out


def build_cinematic_environment(
    world_w: float = 100.0,
    world_h: float = 100.0,
    *,
    seed: Optional[int] = None,
) -> List[StaticMeshInstance]:
    """
    Возвращает более детализированный набор статики для 3D-сцены.
    """
    rnd = random.Random(seed if seed is not None else 2026)
    w = _clamp(float(world_w), 40.0, 10000.0)
    h = _clamp(float(world_h), 40.0, 10000.0)

    meshes: List[StaticMeshInstance] = []
    # Базовый "каркас" деревни оставляем, но дальше сильно наращиваем детали.
    meshes.extend(build_lowpoly_village(world_w=w, world_h=h, seed=seed))

    cx, cz = w * 0.5, h * 0.5

    # --- Центральный район с ориентиром и сервисными точками
    plaza_r = max(10.0, min(w, h) * 0.10)
    homes = 8
    for i in range(homes):
        a = (math.tau * i / homes) + rnd.uniform(-0.16, 0.16)
        x = cx + math.cos(a) * plaza_r
        z = cz + math.sin(a) * plaza_r
        yaw = a + math.pi * 0.5 + rnd.uniform(-0.25, 0.25)
        hs = rnd.uniform(2.4, 3.2)
        meshes.append(_mk("house", x, z, yaw=yaw, sx=hs, sy=rnd.uniform(2.0, 2.5), sz=hs * rnd.uniform(0.9, 1.1)))

    # Ориентиры в центре — чтобы карта читалась "с первого взгляда".
    meshes.append(_mk("tower", cx + plaza_r * 0.1, cz - plaza_r * 0.2, sx=2.2, sy=5.0, sz=2.2))
    meshes.append(_mk("shrine", cx - plaza_r * 0.7, cz + plaza_r * 0.35, sx=2.0, sy=1.8, sz=2.0))
    meshes.append(_mk("well", cx + plaza_r * 0.75, cz + plaza_r * 0.45, sx=1.7, sy=1.3, sz=1.7))

    # --- Основные дороги (крест + диагонали)
    road_w_main = max(2.2, min(w, h) * 0.018)
    road_w_sub = road_w_main * 0.75
    meshes.append(_road(6.0, cz, w - 6.0, cz, road_w_main))
    meshes.append(_road(cx, 6.0, cx, h - 6.0, road_w_main))
    meshes.append(_road(8.0, 8.0, w - 8.0, h - 8.0, road_w_sub))
    meshes.append(_road(w - 8.0, 8.0, 8.0, h - 8.0, road_w_sub))

    # Короткие локальные дорожки от центра к домам.
    for i in range(homes):
        a = math.tau * i / homes
        hx = cx + math.cos(a) * plaza_r
        hz = cz + math.sin(a) * plaza_r
        meshes.append(_road(cx, cz, hx, hz, road_w_sub * 0.7))

    # --- Фонари вдоль главных осей
    lantern_step = max(9.0, min(w, h) * 0.08)
    t = 10.0
    while t <= w - 10.0:
        meshes.append(_mk("lantern", t, cz + road_w_main * 0.9, sx=0.55, sy=2.0, sz=0.55))
        meshes.append(_mk("lantern", t, cz - road_w_main * 0.9, sx=0.55, sy=2.0, sz=0.55))
        t += lantern_step
    t = 10.0
    while t <= h - 10.0:
        meshes.append(_mk("lantern", cx + road_w_main * 0.9, t, sx=0.55, sy=2.0, sz=0.55))
        meshes.append(_mk("lantern", cx - road_w_main * 0.9, t, sx=0.55, sy=2.0, sz=0.55))
        t += lantern_step

    # --- Вторичный водоём в другом квартале (для читаемой географии)
    lake2_x = _clamp(w * 0.18, 7.0, w - 7.0)
    lake2_z = _clamp(h * 0.72, 7.0, h - 7.0)
    meshes.append(_mk("lake", lake2_x, lake2_z, sx=max(6.0, w * 0.06), sy=1.0, sz=max(4.0, h * 0.045), y=-0.02))
    meshes.append(_mk("road", (lake2_x + cx) * 0.5, (lake2_z + cz) * 0.5,
                      yaw=math.atan2(cz - lake2_z, cx - lake2_x),
                      sx=math.hypot(cx - lake2_x, cz - lake2_z) * 0.25,
                      sy=0.2, sz=road_w_sub * 0.5))

    # --- Пояс деревьев по периметру
    belt_n = max(38, int((w + h) * 0.38))
    for i in range(belt_n):
        side = i % 4
        if side == 0:      # низ
            x = rnd.uniform(2.5, w - 2.5); z = rnd.uniform(1.8, 5.5)
        elif side == 1:    # верх
            x = rnd.uniform(2.5, w - 2.5); z = rnd.uniform(h - 5.5, h - 1.8)
        elif side == 2:    # лево
            x = rnd.uniform(1.8, 5.5); z = rnd.uniform(2.5, h - 2.5)
        else:              # право
            x = rnd.uniform(w - 5.5, w - 1.8); z = rnd.uniform(2.5, h - 2.5)
        meshes.append(_mk("tree", x, z, yaw=rnd.uniform(0.0, math.tau), sx=rnd.uniform(0.8, 1.25), sy=rnd.uniform(1.35, 2.3), sz=rnd.uniform(0.8, 1.25)))

    # --- Локальные рощи
    grove_centers = [
        (w * 0.22, h * 0.24),
        (w * 0.77, h * 0.24),
        (w * 0.75, h * 0.80),
    ]
    for gcx, gcz in grove_centers:
        pts = _scatter_circle(rnd, (gcx, gcz), radius=min(w, h) * 0.09, count=18, world_w=w, world_h=h, margin=2.2)
        for tx, tz in pts:
            meshes.append(_mk("tree", tx, tz, yaw=rnd.uniform(0.0, math.tau), sx=rnd.uniform(0.85, 1.3), sy=rnd.uniform(1.3, 2.4), sz=rnd.uniform(0.85, 1.3)))

    # --- Камни и брёвна для рельефа
    deco_pts = _scatter_circle(rnd, (w * 0.55, h * 0.35), radius=min(w, h) * 0.28, count=24, world_w=w, world_h=h, margin=2.0)
    for i, (dx, dz) in enumerate(deco_pts):
        if i % 3 == 0:
            meshes.append(_mk("log", dx, dz, yaw=rnd.uniform(0.0, math.tau), sx=rnd.uniform(1.2, 2.0), sy=rnd.uniform(0.35, 0.6), sz=rnd.uniform(0.45, 0.7)))
        else:
            rs = rnd.uniform(0.5, 1.0)
            meshes.append(_mk("rock", dx, dz, yaw=rnd.uniform(0.0, math.tau), sx=rs, sy=rs * rnd.uniform(0.8, 1.3), sz=rs * rnd.uniform(0.8, 1.25)))

    # --- Лёгкие стены-периметр (визуально очерчивают карту)
    wall_h = 0.9
    wall_t = 0.6
    meshes.append(_mk("wall", w * 0.5, 1.1, sx=w * 0.5 - 1.2, sy=wall_h, sz=wall_t, yaw=0.0))
    meshes.append(_mk("wall", w * 0.5, h - 1.1, sx=w * 0.5 - 1.2, sy=wall_h, sz=wall_t, yaw=0.0))
    meshes.append(_mk("wall", 1.1, h * 0.5, sx=h * 0.5 - 1.2, sy=wall_h, sz=wall_t, yaw=math.pi * 0.5))
    meshes.append(_mk("wall", w - 1.1, h * 0.5, sx=h * 0.5 - 1.2, sy=wall_h, sz=wall_t, yaw=math.pi * 0.5))

    return meshes

