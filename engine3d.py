# engine3d.py
# Мини-движок визуализации для нашей "мини-матрицы" c HUD, dead-reckoning и VFX.
# (расширенная версия: пульс выбора, FOV-конус, линии целей, числа урона, LOD HUD)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import math
import random

# OpenGL immediate mode
from OpenGL.GL import (
    glBegin, glEnd,
    glVertex3f, glColor3f, glColor4f,
    glLineWidth,
    glEnable, glBlendFunc,
    glClearColor, glClear,
    glMatrixMode, glLoadIdentity, glViewport,
    glDepthFunc,
    glGetDoublev, glGetIntegerv, glRasterPos3f,
    GL_MODELVIEW, GL_PROJECTION,
    GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX, GL_VIEWPORT,
    GL_DEPTH_TEST, GL_BLEND,
    GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_LINES, GL_TRIANGLE_FAN, GL_QUADS,
    GL_LEQUAL,
)
from OpenGL.GLU import gluPerspective, gluLookAt, gluProject

# Пытаемся подключить GLUT для текстовых меток HUD. Если нет — рисуем без текста.
try:
    from OpenGL.GLUT import (
        glutInit,
        glutBitmapCharacter,
        GLUT_BITMAP_HELVETICA_12,
        GLUT_BITMAP_HELVETICA_18,
    )
    _HAS_GLUT = True
except Exception:
    _HAS_GLUT = False


# ---------------------------------------------------------------------
# Константы поведения клиентского движка
# ---------------------------------------------------------------------

SMOOTH_LERP_SPEED = 10.0        # скорость сглаживания поз к целям
PERSONAL_SPACE_RADIUS = 0.6     # радиус "капсулы" агента для разведения
ANIMAL_SPACE_RADIUS = 0.4       # личный радиус зверя
SEPARATION_PUSH = 0.5           # сила разведения при пересечении капсул

# Dead-reckoning
DEAD_RECKONING = True
DR_MAX_PREDICT_SEC = 0.35       # ограничиваем прогноз, чтобы не уезжать далеко

# HUD
HUD_SHOW_TEXT = True            # будет отключено автоматически, если нет GLUT
HUD_HP_BAR_W = 1.2
HUD_HP_BAR_H = 0.08
HUD_FEAR_BAR_W = 1.2
HUD_FEAR_BAR_H = 0.06

# VFX
VFX_RING_TTL = 0.8
VFX_RING_R0 = 0.2
VFX_RING_R1 = 1.4

# Новые визуальные флаги/параметры
SHOW_THREAT_RINGS   = True      # пульсирующие кольца-угрозы у агрессивных зверей
SHOW_TARGET_LINES   = True      # линии от атакующего к цели
SHOW_FOV_CONES      = True      # конус обзора у выбранного агента
SHOW_DAMAGE_NUMBERS = True      # всплывающие числа урона/лечения
SELECTED_PULSE      = True      # пульсирующее кольцо у выбранных сущностей
MAX_HUD_DISTANCE    = 80.0      # LOD: дальше этого HUD не рисуем (кроме выбранного)
FOV_DEG             = 80.0      # базовая ширина FOV-конуса
FOV_RANGE           = 7.0       # базовая дальность FOV-конуса


# ---------------------------------------------------------------------
# Вспомогательная математика
# ---------------------------------------------------------------------

@dataclass
class Vec3:
    x: float
    y: float
    z: float

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        L = self.length()
        if L < 1e-8:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / L, self.y / L, self.z / L)

    def lerp(self, other: "Vec3", alpha: float) -> "Vec3":
        return Vec3(
            self.x + (other.x - self.x) * alpha,
            self.y + (other.y - self.y) * alpha,
            self.z + (other.z - self.z) * alpha,
        )


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def angle_lerp(a: float, b: float, t: float) -> float:
    """
    Плавный поворот по кругу [-pi,pi]: интерполируем a → b с учётом wrap-around.
    """
    diff = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + diff * t


def _xy_from_any(v: Any) -> Tuple[float, float]:
    """
    Универсальный парсер координат:
    - {"x": 12, "y": 30}
    - [12, 30]
    - (12, 30)
    - None → (0,0)
    """
    if isinstance(v, dict):
        return float(v.get("x", 0.0)), float(v.get("y", 0.0))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    return 0.0, 0.0


# ---------------------------------------------------------------------
# Сущности сцены
# ---------------------------------------------------------------------

@dataclass
class Transform:
    """
    Поза, в которой мы реально рисуем модель.
    pos    — сглаженная позиция.
    yaw    — текущий поворот корпуса вокруг оси Y (радианы).
    """
    pos: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    yaw: float = 0.0


@dataclass
class RigidBody:
    """
    Кинематические данные (с точки зрения визуалки):
      - vel        : скорость в плоскости XZ (серверная)
      - radius     : радиус "коллайдера" (для локального разведения)
    """
    vel: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    radius: float = PERSONAL_SPACE_RADIUS


@dataclass
class NetState:
    """Сетевое состояние для dead-reckoning."""
    server_pos: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    server_vel: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    since_snap: float = 0.0  # сек с момента последнего снапшота


@dataclass
class AgentAnimState:
    """
    Локальная анимация агента:
    - walk_phase   : фаза шага
    - fear, health : эмоции и состояние тела → наклон/сутулость
    - alive        : жив ли
    - prev_fear    : для детекции всплесков (VFX)
    - prev_hp      : для всплывающих чисел урона/хила
    """
    walk_phase: float = 0.0
    fear: float = 0.0
    health: float = 100.0
    alive: bool = True
    prev_fear: float = 0.0
    prev_hp: float = 100.0


@dataclass
class AgentBrainLike:
    """
    "Куда смотрю / куда иду".
    Обычно — нормализованная скорость. Если стоим, то держим прежнее направление.
    """
    desired_dir: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))


@dataclass
class AgentEntity:
    """
    Визуально-физическая проекция серверного агента.
    """
    agent_id: str
    name: str

    transform: Transform
    target_pos: Vec3

    body: RigidBody
    brain: AgentBrainLike
    anim: AgentAnimState

    goal: Vec3
    public_state: Dict[str, Any] = field(default_factory=dict)

    net: NetState = field(default_factory=NetState)

    selected: bool = False


# --- ЖИВОТНЫЕ --------------------------------------------------------

@dataclass
class AnimalAnimState:
    """
    Анимация зверя / его состояние.
    - walk_phase    : фаза шага/покачивания
    - health, alive : жив ли
    - temperament   : "aggressive" / "tameable" / "neutral"
    - tamed         : приручён ли (есть хозяин)
    - owner_id      : id хозяина
    - last_action   : текст последнего действия (для HUD)
    - prev_hp       : для всплывающих чисел урона/хила
    """
    walk_phase: float = 0.0
    health: float = 100.0
    alive: bool = True
    temperament: str = "neutral"
    tamed: bool = False
    owner_id: Optional[str] = None
    last_action: str = ""
    prev_hp: float = 100.0


@dataclass
class AnimalEntity:
    """
    Визуально-физическая проекция зверя.
    """
    animal_id: str
    name: str
    species: Optional[str]

    transform: Transform
    target_pos: Vec3

    body: RigidBody
    brain: AgentBrainLike
    anim: AnimalAnimState

    public_state: Dict[str, Any] = field(default_factory=dict)

    net: NetState = field(default_factory=lambda: NetState(server_pos=Vec3(0,0,0), server_vel=Vec3(0,0,0)))

    selected: bool = False
    last_action_prev: str = ""


@dataclass
class ZoneObject:
    """
    Статические зоны из сервера (safe/hazard/neutral),
    рисуем в виде цветных дисков.
    """
    obj_id: str
    name: str
    kind: str
    x: float
    z: float
    radius: float


@dataclass
class WorldStatic:
    """
    Данные окружения мира, известные клиенту.
    """
    width: float
    height: float
    zones: List[ZoneObject] = field(default_factory=list)


@dataclass
class StaticMeshInstance:
    """
    Лоуполи окружение (дом, дерево, костёр, озеро и т.д.).
    """
    kind: str
    pos: Vec3
    yaw: float
    scale: Vec3


# ---------------------------------------------------------------------
# HUD / VFX типы
# ---------------------------------------------------------------------

@dataclass
class VFXRing:
    x: float
    z: float
    y: float
    r0: float
    r1: float
    ttl: float = VFX_RING_TTL
    age: float = 0.0
    color: Tuple[float, float, float, float] = (1.0, 1.0, 0.2, 0.9)

    def alive(self) -> bool:
        return self.age < self.ttl

    def radius(self) -> float:
        t = clamp(self.age / max(self.ttl, 1e-6), 0.0, 1.0)
        return self.r0 + (self.r1 - self.r0) * t

    def alpha(self) -> float:
        t = clamp(self.age / max(self.ttl, 1e-6), 0.0, 1.0)
        return (1.0 - t) * self.color[3]


@dataclass
class DamageNumber:
    """
    Всплывающая надпись урона/лечения.
    """
    x: float
    z: float
    y0: float
    value: float
    color: Tuple[float, float, float, float] = (1.0, 0.2, 0.2, 1.0)  # красный по умолчанию
    ttl: float = 1.1
    age: float = 0.0

    def alive(self) -> bool:
        return self.age < self.ttl

    def y(self) -> float:
        # лёгкий подъём
        t = clamp(self.age / max(self.ttl, 1e-6), 0.0, 1.0)
        return self.y0 + 0.8 * t

    def alpha(self) -> float:
        t = clamp(self.age / max(self.ttl, 1e-6), 0.0, 1.0)
        return (1.0 - t)


# ---------------------------------------------------------------------
# Рендер-помощники (иммедиат режим)
# ---------------------------------------------------------------------

def _fake_lighting_color(base: Tuple[float, float, float], normal_y: float) -> Tuple[float, float, float]:
    r, g, b = base
    light = clamp(0.3 + normal_y * 0.7, 0.2, 1.0)
    return (r * light, g * light, b * light)


def _draw_floor_grid(world_w: float, world_h: float):
    base_col = (0.08, 0.08, 0.10)
    lit_col = _fake_lighting_color(base_col, normal_y=1.0)

    glColor3f(*lit_col)
    glBegin(GL_QUADS)
    glVertex3f(0.0,      0.0,      0.0)
    glVertex3f(world_w,  0.0,      0.0)
    glVertex3f(world_w,  0.0,      world_h)
    glVertex3f(0.0,      0.0,      world_h)
    glEnd()

    glColor3f(0.2, 0.2, 0.3)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    step = 10.0
    x = 0.0
    while x <= world_w + 0.001:
        glVertex3f(x, 0.001, 0.0)
        glVertex3f(x, 0.001, world_h)
        x += step
    z = 0.0
    while z <= world_h + 0.001:
        glVertex3f(0.0,      0.001, z)
        glVertex3f(world_w,  0.001, z)
        z += step
    glEnd()


def _draw_disc_zone(x: float, z: float, radius: float,
                    kind: str,
                    y: float = 0.02):
    if kind == "hazard":
        col = (1.0, 0.2, 0.2, 0.4)
    elif kind == "safe":
        col = (0.2, 1.0, 1.0, 0.35)
    else:
        col = (0.6, 0.6, 0.8, 0.18)

    glColor4f(*col)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(x, y, z)
    steps = 48
    for i in range(steps + 1):
        ang = (2.0 * math.pi) * (i / steps)
        vx = x + math.cos(ang) * radius
        vz = z + math.sin(ang) * radius
        glVertex3f(vx, y, vz)
    glEnd()


def _draw_ring(x: float, z: float, radius: float, y: float,
               rgb: Tuple[float, float, float],
               width: float = 2.0,
               steps: int = 64,
               alpha: float = 1.0):
    glColor4f(rgb[0], rgb[1], rgb[2], alpha)
    glLineWidth(width)
    glBegin(GL_LINES)
    prev = None
    for i in range(steps + 1):
        ang = (2.0 * math.pi) * (i / steps)
        vx = x + math.cos(ang) * radius
        vz = z + math.sin(ang) * radius
        if prev:
            glVertex3f(prev[0], y, prev[1])
            glVertex3f(vx,      y, vz)
        prev = (vx, vz)
    glEnd()


def _draw_oriented_box(
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    yaw_rad: float,
    base_color: Tuple[float, float, float],
    pitch_forward: float = 0.0,
):
    """
    Очень примитивный прямоугольный "меш" из 6 граней.
    """
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    # лёгкий наклон корпуса вперёд/вниз (pitch_forward)
    lean_z = pitch_forward * 0.2

    verts = []
    for sx in (-hx, hx):
        for sz in (-hz, hz):
            for sy in (-hy, hy):
                sz2 = sz + lean_z * (sy / hy if hy > 1e-6 else 1.0)
                wx = cx + sx * cos_y - sz2 * sin_y
                wz = cz + sx * sin_y + sz2 * cos_y
                wy = cy + sy
                verts.append((wx, wy, wz))

    def V(i: int):
        return verts[i]

    faces = [
        (0, 4, 5, 1),  # -z
        (2, 3, 7, 6),  # +z
        (0, 1, 3, 2),  # -x
        (4, 6, 7, 5),  # +x
        (0, 2, 6, 4),  # -y
        (1, 5, 7, 3),  # +y
    ]

    glBegin(GL_QUADS)
    for f in faces:
        p0 = V(f[0]); p1 = V(f[1]); p2 = V(f[2])
        ux = p1[0] - p0[0]; uy = p1[1] - p0[1]; uz = p1[2] - p0[2]
        vx = p2[0] - p0[0]; vy = p2[1] - p0[1]; vz = p2[2] - p0[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        nlen = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        ny /= nlen

        lit = _fake_lighting_color(base_color, normal_y=ny)
        glColor3f(*lit)

        for idx in f:
            glVertex3f(*V(idx))
    glEnd()


def _draw_head_disc(
    cx: float,
    cy: float,
    cz: float,
    yaw_rad: float,
    radius: float,
    forward_lean: float,
):
    """
    "Голова" человека — просто белый диск над корпусом.
    """
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    head_cx = cx + math.sin(forward_lean) * cos_y * 0.2
    head_cz = cz + math.sin(forward_lean) * sin_y * 0.2
    head_cy = cy

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(head_cx, head_cy, head_cz)

    steps = 20
    for i in range(steps + 1):
        ang = (2.0 * math.pi) * (i / steps)
        vx = head_cx + math.cos(ang) * radius
        vz = head_cz + math.sin(ang) * radius
        glVertex3f(vx, head_cy, vz)
    glEnd()


def _color_from_state(fear: float, alive: bool) -> Tuple[float, float, float]:
    """
    Цвет туловища человека:
      много страха → краснеет,
      мёртвый → серый.
    """
    if not alive:
        return (0.3, 0.3, 0.35)
    r = min(1.0, 0.2 + fear * 0.8)
    g = max(0.0, 0.9 - fear * 0.9)
    b = 0.3
    return (r, g, b)


def _animal_body_color(temperament: str, tamed: bool, alive: bool) -> Tuple[float, float, float]:
    """
    Цвет тела зверя:
      питомец → сине-голубой,
      агрессивный → красный,
      дружелюбный дикий → зелёный,
      мёртвый → серый.
    """
    if not alive:
        return (0.3, 0.3, 0.3)
    if tamed:
        return (0.3, 0.5, 1.0)
    if temperament == "aggressive":
        return (1.0, 0.25, 0.25)
    return (0.2, 1.0, 0.3)


def _animal_ring_color(temperament: str, tamed: bool, alive: bool) -> Tuple[float, float, float]:
    """
    Цвет кольца под зверем:
      питомец → голубой,
      агрессивный → красный,
      дружелюбный → зелёный,
      труп → серый.
    """
    if not alive:
        return (0.4, 0.4, 0.4)
    if tamed:
        return (0.2, 0.8, 1.0)
    if temperament == "aggressive":
        return (1.0, 0.2, 0.2)
    return (0.2, 1.0, 0.3)


def _draw_pet_marker(px: float, py: float, pz: float):
    """
    Маленький "бейдж питомца" над зверем.
    Просто ромбик из треугольного фана.
    """
    glColor3f(0.2, 0.8, 1.0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex3f(px, py + 1.00, pz)
    glVertex3f(px + 0.10, py + 0.80, pz)
    glVertex3f(px,        py + 0.60, pz)
    glVertex3f(px - 0.10, py + 0.80, pz)
    glVertex3f(px,        py + 1.00, pz)
    glEnd()


# --- лоуполи окружение helpers --------------------------------------

def _draw_pyramid_roof(
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_rad: float,
    base_color: Tuple[float,float,float],
    top_color: Optional[Tuple[float,float,float]] = None,
):
    """
    Крыша домика / хвоя дерева / пламя — 4 треугольника-пирамидки.
    """
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)

    base_pts_local = [
        (-sx, 0.0, -sz),
        ( sx, 0.0, -sz),
        ( sx, 0.0,  sz),
        (-sx, 0.0,  sz),
    ]
    apex_local = (0.0, sy, 0.0)

    def to_world(px, py, pz):
        wx = cx + px * cos_y - pz * sin_y
        wz = cz + px * sin_y + pz * cos_y
        wy = cy + py
        return (wx, wy, wz)

    base_w = [to_world(*p) for p in base_pts_local]
    apex_w = to_world(*apex_local)

    glBegin(GL_TRIANGLE_FAN)
    for (a, b, c) in [
        (base_w[0], base_w[1], apex_w),
        (base_w[1], base_w[2], apex_w),
        (base_w[2], base_w[3], apex_w),
        (base_w[3], base_w[0], apex_w),
    ]:
        ux = b[0] - a[0]; uy = b[1] - a[1]; uz = b[2] - a[2]
        vx = c[0] - a[0]; vy = c[1] - a[1]; vz = c[2] - a[2]
        nx = uy*vz - uz*vy
        ny = uz*vx - ux*vz
        nz = ux*vy - uy*vx
        nlen = math.sqrt(nx*nx + ny*ny + nz*nz) + 1e-9
        ny /= nlen

        face_col = base_color
        if top_color:
            mix_t = clamp((ny + 1.0) * 0.5, 0.0, 1.0)
            face_col = (
                base_color[0]*(1-mix_t) + top_color[0]*mix_t,
                base_color[1]*(1-mix_t) + top_color[1]*mix_t,
                base_color[2]*(1-mix_t) + top_color[2]*mix_t,
            )

        lit = _fake_lighting_color(face_col, normal_y=ny)
        glColor3f(*lit)

        glVertex3f(*a)
        glVertex3f(*b)
        glVertex3f(*c)
    glEnd()


def _draw_house(inst: StaticMeshInstance):
    """
    Маленький домик (коробка + пирамидальная крыша).
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    sx = inst.scale.x * 0.5
    sy = inst.scale.y * 0.5
    sz = inst.scale.z * 0.5

    wall_color = (0.5, 0.45, 0.4)
    roof_color = (0.4, 0.1, 0.1)
    roof_high  = (0.8, 0.2, 0.2)

    _draw_oriented_box(
        cx=x, cy=y + sy, cz=z,
        hx=sx, hy=sy, hz=sz,
        yaw_rad=yaw,
        base_color=wall_color,
        pitch_forward=0.0,
    )

    _draw_pyramid_roof(
        cx=x,
        cy=y + sy*2.0,
        cz=z,
        sx=sx*1.05,
        sy=sy*0.8,
        sz=sz*1.05,
        yaw_rad=yaw,
        base_color=roof_color,
        top_color=roof_high,
    )


def _draw_tree(inst: StaticMeshInstance):
    """
    Дерево: коричневый ствол (box) + зелёная шапка (pyramid).
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw

    trunk_h = 0.6 * inst.scale.y
    crown_h = 0.9 * inst.scale.y
    radius  = 0.2 * inst.scale.x

    trunk_color = (0.35, 0.22, 0.15)
    leaf_color  = (0.1, 0.45, 0.1)
    leaf_high   = (0.3, 0.8, 0.3)

    _draw_oriented_box(
        cx=x,
        cy=y + trunk_h * 0.5,
        cz=z,
        hx=radius * 0.5,
        hy=trunk_h * 0.5,
        hz=radius * 0.5,
        yaw_rad=yaw,
        base_color=trunk_color,
        pitch_forward=0.0,
    )

    _draw_pyramid_roof(
        cx=x,
        cy=y + trunk_h,
        cz=z,
        sx=radius,
        sy=crown_h,
        sz=radius,
        yaw_rad=yaw,
        base_color=leaf_color,
        top_color=leaf_high,
    )


def _draw_lake(inst: StaticMeshInstance):
    """
    Плоский "озёрный" прямоугольник.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    sx = inst.scale.x
    sz = inst.scale.z

    glColor4f(0.1, 0.4, 0.8, 0.6)
    glBegin(GL_QUADS)
    glVertex3f(x - sx, y, z - sz)
    glVertex3f(x + sx, y, z - sz)
    glVertex3f(x + sx, y, z + sz)
    glVertex3f(x - sx, y, z + sz)
    glEnd()

    glColor3f(0.2, 0.5, 0.9)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(x - sx, y+0.01, z - sz); glVertex3f(x + sx, y+0.01, z - sz)
    glVertex3f(x + sx, y+0.01, z - sz); glVertex3f(x + sx, y+0.01, z + sz)
    glVertex3f(x + sx, y+0.01, z + sz); glVertex3f(x - sx, y+0.01, z + sz)
    glVertex3f(x - sx, y+0.01, z + sz); glVertex3f(x - sx, y+0.01, z - sz)
    glEnd()


def _draw_fire(inst: StaticMeshInstance, global_time: float):
    """
    Костёр: серое кольцо камней + мерцающее "пламя" (pyramid).
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw

    flicker = 0.8 + 0.2 * math.sin(global_time * 7.0 + x * 3.0 + z * 5.0)

    stone_col = (0.3, 0.3, 0.33)
    _draw_oriented_box(
        cx=x,
        cy=y + 0.15,
        cz=z,
        hx=0.6 * inst.scale.x,
        hy=0.15,
        hz=0.6 * inst.scale.z,
        yaw_rad=yaw,
        base_color=stone_col,
        pitch_forward=0.0,
    )

    flame_col = (1.0, 0.5, 0.05)
    flame_hot = (1.0, 0.8, 0.3)

    _draw_pyramid_roof(
        cx=x,
        cy=y + 0.3,
        cz=z,
        sx=0.4 * flicker,
        sy=1.0 * flicker,
        sz=0.4 * flicker,
        yaw_rad=yaw,
        base_color=flame_col,
        top_color=flame_hot,
    )


def _draw_static_mesh(inst: StaticMeshInstance, global_time: float):
    if inst.kind == "house":
        _draw_house(inst)
    elif inst.kind == "tree":
        _draw_tree(inst)
    elif inst.kind == "lake":
        _draw_lake(inst)
    elif inst.kind == "fire":
        _draw_fire(inst, global_time)


# --- отрисовка агента ------------------------------------------------

def draw_agent_humanoid(agent: AgentEntity, t: float):
    yaw = agent.transform.yaw
    fear = agent.anim.fear
    hp = agent.anim.health
    alive = agent.anim.alive
    phase = agent.anim.walk_phase

    # сутулость / "я ранен"
    crouch = 0.0
    if hp < 50.0:
        crouch += 0.2
    if fear > 0.7:
        crouch += 0.1

    # наклон корпуса вперёд/вниз (паника или безжизненность)
    fwd_lean = 0.0
    if fear > 0.6:
        fwd_lean += 0.2
    if hp < 30.0:
        fwd_lean += 0.3
    if not alive:
        fwd_lean = 0.8
        crouch = 0.4

    base_y_offset = 0.0
    if not alive:
        base_y_offset = -0.1

    body_col = _color_from_state(fear, alive)

    leg_amp = 0.30
    arm_amp = 0.30
    leg_sw = math.sin(phase) * leg_amp
    arm_sw = math.sin(phase + math.pi) * arm_amp

    px = agent.transform.pos.x
    pz = agent.transform.pos.z

    # ТОРСО
    torso_mid_y  = 1.2 - crouch + base_y_offset
    torso_half_w = 0.45
    torso_half_d = 0.25
    torso_half_h = 0.45
    _draw_oriented_box(
        cx=px,
        cy=torso_mid_y,
        cz=pz,
        hx=torso_half_w,
        hy=torso_half_h,
        hz=torso_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=fwd_lean,
    )

    # ГОЛОВА
    head_y = torso_mid_y + torso_half_h + 0.35 - crouch
    _draw_head_disc(
        cx=px,
        cy=head_y,
        cz=pz,
        yaw_rad=yaw,
        radius=0.35,
        forward_lean=fwd_lean,
    )

    # НОГИ
    hip_y_mid   = 0.6 - crouch + base_y_offset
    hip_half_h  = 0.35
    hip_half_w  = 0.22
    hip_half_d  = 0.22

    shin_y_mid  = 0.2 - crouch + base_y_offset
    shin_half_h = 0.2
    shin_half_w = 0.18
    shin_half_d = 0.18

    leg_gap_x   = 0.25

    # левая
    _draw_oriented_box(
        cx=px - leg_gap_x,
        cy=hip_y_mid,
        cz=pz + leg_sw,
        hx=hip_half_w,
        hy=hip_half_h,
        hz=hip_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=px - leg_gap_x,
        cy=shin_y_mid,
        cz=pz + leg_sw,
        hx=shin_half_w,
        hy=shin_half_h,
        hz=shin_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    # правая
    _draw_oriented_box(
        cx=px + leg_gap_x,
        cy=hip_y_mid,
        cz=pz - leg_sw,
        hx=hip_half_w,
        hy=hip_half_h,
        hz=hip_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=px + leg_gap_x,
        cy=shin_y_mid,
        cz=pz - leg_sw,
        hx=shin_half_w,
        hy=shin_half_h,
        hz=shin_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    # РУКИ
    shoulder_y_mid = torso_mid_y + 0.15 - crouch
    arm_half_h     = 0.25
    arm_half_w     = 0.15
    arm_half_d     = 0.15
    arm_offset_x   = torso_half_w + arm_half_w + 0.05

    # левая
    _draw_oriented_box(
        cx=px - arm_offset_x,
        cy=shoulder_y_mid,
        cz=pz + arm_sw,
        hx=arm_half_w,
        hy=arm_half_h,
        hz=arm_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )
    # правая
    _draw_oriented_box(
        cx=px + arm_offset_x,
        cy=shoulder_y_mid,
        cz=pz - arm_sw,
        hx=arm_half_w,
        hy=arm_half_h,
        hz=arm_half_d,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    # КОЛЬЦО ВЫБОРА (с пульсом)
    if agent.selected:
        base_r = 1.15
        if SELECTED_PULSE:
            pulse = (math.sin(t * 4.0) * 0.5 + 0.5)  # 0..1
            r = base_r + pulse * 0.25
            a = 0.55 + 0.35 * pulse
        else:
            r = base_r
            a = 0.9
        _draw_ring(px, pz, radius=r, y=0.05, rgb=(0.2, 0.8, 1.0), width=2.0, alpha=a)


def draw_agent_direction_arrow(agent: AgentEntity):
    """
    Маленькая стрелка, показывающая вектор движения агента.
    Жёлтая линия над головой.
    """
    vx = agent.body.vel.x
    vz = agent.body.vel.z
    speed = math.hypot(vx, vz)
    if speed < 0.01:
        return
    px = agent.transform.pos.x
    pz = agent.transform.pos.z
    nx = px + (vx / speed) * 1.2
    nz = pz + (vz / speed) * 1.2

    glColor3f(1.0, 1.0, 0.2)
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glVertex3f(px, 2.2, pz)
    glVertex3f(nx, 2.2, nz)
    glEnd()


# --- отрисовка зверя -------------------------------------------------

def draw_animal_quadruped(an: AnimalEntity, t: float):
    """
    Примитивная "четвероногая зверюшка":
    - прямоугольный торс
    - маленькая голова
    - кольцо состояния
    - маркер питомца (если приручён)
    """
    yaw = an.transform.yaw
    hp = an.anim.health
    alive = an.anim.alive
    temperament = an.anim.temperament
    tamed = an.anim.tamed

    phase = an.anim.walk_phase
    bob = math.sin(phase) * 0.03

    px = an.transform.pos.x
    pz = an.transform.pos.z

    body_col = _animal_body_color(temperament, tamed, alive)
    ring_col = _animal_ring_color(temperament, tamed, alive)

    # кольцо состояния под зверем
    _draw_ring(px, pz, radius=0.6, y=0.03, rgb=ring_col, width=1.5)

    # пульсирующее "кольцо угрозы" для агрессивных
    if SHOW_THREAT_RINGS and temperament == "aggressive" and alive:
        pulse = (math.sin(t * 3.2) * 0.5 + 0.5)
        _draw_ring(px, pz, radius=0.85 + pulse * 0.25, y=0.031, rgb=(1.0, 0.25, 0.25), width=2.0, alpha=0.45 + 0.35 * pulse)

    # тело
    body_mid_y  = 0.5 + bob
    body_hx = 0.35
    body_hy = 0.20
    body_hz = 0.60
    _draw_oriented_box(
        cx=px,
        cy=body_mid_y,
        cz=pz,
        hx=body_hx,
        hy=body_hy,
        hz=body_hz,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    # голова (маленький куб спереди)
    fx = math.cos(yaw)
    fz = math.sin(yaw)
    head_cx = px + fx * (body_hz + 0.25)
    head_cz = pz + fz * (body_hz + 0.25)
    head_mid_y = body_mid_y + 0.05

    _draw_oriented_box(
        cx=head_cx,
        cy=head_mid_y,
        cz=head_cz,
        hx=0.18,
        hy=0.15,
        hz=0.18,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    # маркер питомца
    if tamed and alive:
        _draw_pet_marker(px, body_mid_y + 0.4, pz)

    # пульс выбора
    if an.selected and SELECTED_PULSE:
        pulse = (math.sin(t * 4.0) * 0.5 + 0.5)
        _draw_ring(px, pz, radius=1.0 + pulse * 0.2, y=0.05, rgb=(0.2, 0.8, 1.0), width=2.0, alpha=0.5 + 0.35 * pulse)


def draw_animal_direction_arrow(an: AnimalEntity):
    vx = an.body.vel.x
    vz = an.body.vel.z
    speed = math.hypot(vx, vz)
    if speed < 0.01:
        return
    px = an.transform.pos.x
    pz = an.transform.pos.z
    nx = px + (vx / speed) * 1.0
    nz = pz + (vz / speed) * 1.0

    glColor3f(0.85, 0.85, 1.0)
    glLineWidth(1.7)
    glBegin(GL_LINES)
    glVertex3f(px, 1.5, pz)
    glVertex3f(nx, 1.5, nz)
    glEnd()


# ---------------------------------------------------------------------
# MiniMatrixEngine
# ---------------------------------------------------------------------

class MiniMatrixEngine:
    """
    Главный класс движка.
    Хранит локальные визуальные сущности и рисует сцену.
    """

    def __init__(self):
        self.world = WorldStatic(width=100.0, height=100.0, zones=[])
        self.agents: Dict[str, AgentEntity] = {}
        self.animals: Dict[str, AnimalEntity] = {}

        self.static_meshes: List[StaticMeshInstance] = []

        # события мира с сервера (смерти, укусы, приручения...)
        self.global_events: List[str] = []
        self._seen_events: set[str] = set()

        # активные VFX-ринги и damage numbers
        self.vfx: List[VFXRing] = []
        self.numbers: List[DamageNumber] = []

        self._time_accum: float = 0.0

        # камера (для HUD LOD)
        self._cam_pos: Tuple[float, float, float] = (0.0, 20.0, -20.0)
        self._cam_look: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        # HUD / текст
        self._hud_text_enabled = HUD_SHOW_TEXT and _HAS_GLUT
        if _HAS_GLUT:
            try:
                glutInit()
            except Exception:
                self._hud_text_enabled = False

    # -----------------------------------------------------------------
    # ОКРУЖЕНИЕ
    # -----------------------------------------------------------------

    def load_static_environment(self, meshes: List[StaticMeshInstance]):
        """
        Подгружаем заранее расставленные меши окружения (лес, дома, костёр...).
        """
        self.static_meshes = meshes[:]

    # -----------------------------------------------------------------
    # СИНХРОНИЗАЦИЯ С СОСТОЯНИЕМ СЕРВЕРА
    # -----------------------------------------------------------------

    def sync_from_world(self, snapshot: Dict[str, Any]):
        """
        Принимаем снапшот мира от симуляции (сервера) и обновляем локальные
        визуальные сущности (агенты, звери, зоны и т.д.).
        """

        # --- глобальные события
        new_events = list(snapshot.get("global_events", []))
        self.global_events = new_events[-100:]
        # Спавним VFX для новых событий
        for ev in new_events:
            if ev not in self._seen_events:
                self._spawn_vfx_from_event(ev)
        self._seen_events = set(self.global_events)

        # --- мир и зоны
        w = snapshot.get("world", {})
        self.world.width = float(w.get("width", self.world.width))
        self.world.height = float(w.get("height", self.world.height))

        self.world.zones = []
        for obj in snapshot.get("objects", []):
            try:
                ox, oz = _xy_from_any(obj.get("pos", {"x": 0.0, "y": 0.0}))
                self.world.zones.append(
                    ZoneObject(
                        obj_id=obj["id"],
                        name=obj.get("name", obj["id"]),
                        kind=obj.get("kind", "neutral"),
                        x=float(ox),
                        z=float(oz),
                        radius=float(obj.get("radius", 1.0)),
                    )
                )
            except Exception:
                continue

        # --- агенты
        live_agent_ids = set()
        for a in snapshot.get("agents", []):
            try:
                aid = a["id"]
            except Exception:
                continue
            live_agent_ids.add(aid)

            # позиция / скорость / цель
            pos_x, pos_z = _xy_from_any(a.get("pos", {"x": 0.0, "y": 0.0}))
            vel_x, vel_z = _xy_from_any(a.get("vel", {"x": 0.0, "y": 0.0}))
            goal_x, goal_z = _xy_from_any(
                a.get("goal", a.get("pos", {"x": pos_x, "y": pos_z}))
            )

            fear = float(a.get("fear", 0.0))
            hp = float(a.get("health", a.get("hp", 100.0)))
            alive = bool(a.get("alive", hp > 0.0))

            age_ticks = float(a.get("age_ticks", a.get("age", 0.0)))

            # направление взгляда по скорости
            speed = math.hypot(vel_x, vel_z)
            if speed > 1e-6:
                desired_dir = Vec3(vel_x / speed, 0.0, vel_z / speed)
            else:
                if aid in self.agents:
                    desired_dir = self.agents[aid].brain.desired_dir
                else:
                    ang = random.random() * 2.0 * math.pi
                    desired_dir = Vec3(math.cos(ang), 0.0, math.sin(ang))

            if aid not in self.agents:
                start_pos = Vec3(pos_x, 0.0, pos_z)
                self.agents[aid] = AgentEntity(
                    agent_id=aid,
                    name=a.get("name", aid),

                    transform=Transform(pos=start_pos.copy(), yaw=0.0),
                    target_pos=start_pos.copy(),

                    body=RigidBody(
                        vel=Vec3(vel_x, 0.0, vel_z),
                        radius=PERSONAL_SPACE_RADIUS,
                    ),
                    brain=AgentBrainLike(desired_dir=desired_dir),
                    anim=AgentAnimState(
                        walk_phase=age_ticks * 0.3,
                        fear=fear,
                        health=hp,
                        alive=alive,
                        prev_fear=fear,
                        prev_hp=hp,
                    ),
                    goal=Vec3(goal_x, 0.0, goal_z),
                    public_state=dict(a),
                    net=NetState(server_pos=start_pos.copy(), server_vel=Vec3(vel_x,0.0,vel_z), since_snap=0.0),
                    selected=False,
                )
            else:
                ent = self.agents[aid]

                # сетевое состояние для DR
                ent.net.server_pos.x = pos_x
                ent.net.server_pos.y = 0.0
                ent.net.server_pos.z = pos_z
                ent.net.server_vel.x = vel_x
                ent.net.server_vel.y = 0.0
                ent.net.server_vel.z = vel_z
                ent.net.since_snap = 0.0  # сбрасываем таймер снапшота

                # целевая позиция тоже поддерживаем (для совместимости)
                ent.target_pos.x = pos_x
                ent.target_pos.y = 0.0
                ent.target_pos.z = pos_z

                # скорость
                ent.body.vel.x = vel_x
                ent.body.vel.y = 0.0
                ent.body.vel.z = vel_z

                # цель
                ent.goal.x = goal_x
                ent.goal.y = 0.0
                ent.goal.z = goal_z

                # направление взгляда
                ent.brain.desired_dir = desired_dir

                # состояние/анимация
                ent.anim.prev_fear = ent.anim.fear
                ent.anim.fear = fear

                # числа урона/хила
                if SHOW_DAMAGE_NUMBERS and self._hud_text_enabled:
                    if abs(hp - ent.anim.prev_hp) >= 0.5:
                        delta = hp - ent.anim.prev_hp
                        col = (0.2, 1.0, 0.3, 1.0) if delta > 0 else (1.0, 0.25, 0.25, 1.0)
                        self._spawn_damage_number(ent.transform.pos.x, ent.transform.pos.z, y=2.7, value=delta, color=col)
                ent.anim.prev_hp = hp

                ent.anim.health = hp
                ent.anim.alive = alive

                ent.public_state = dict(a)

                # VFX: всплеск страха
                if ent.anim.prev_fear < 0.6 and fear >= 0.6:
                    self._spawn_ring(ent.transform.pos.x, ent.transform.pos.z, y=0.04, color=(1.0,0.4,0.2,0.8))

        # удалить агентов, которых больше нет
        for old_id in list(self.agents.keys()):
            if old_id not in live_agent_ids:
                del self.agents[old_id]

        # --- звери
        live_animal_ids = set()
        for adata in snapshot.get("animals", []):
            zid = adata.get("id") or adata.get("animal_id")
            if zid is None:
                continue
            live_animal_ids.add(zid)

            pos_x, pos_z = _xy_from_any(adata.get("pos", {"x": 0.0, "y": 0.0}))
            vel_x, vel_z = _xy_from_any(adata.get("vel", {"x": 0.0, "y": 0.0}))

            hp_an = float(adata.get("health", adata.get("hp", 100.0)))
            alive_an = bool(adata.get("is_alive", hp_an > 0.0))

            age_ticks = float(adata.get("age_ticks", adata.get("age", 0.0)))

            temperament = str(adata.get("temperament", "neutral"))
            owner_id = adata.get("owner_id", None)
            tamed_flag = bool(adata.get("tamed", (owner_id is not None)))

            last_action = str(adata.get("last_action", ""))

            # направление взгляда
            spd = math.hypot(vel_x, vel_z)
            if spd > 1e-6:
                desired_dir = Vec3(vel_x / spd, 0.0, vel_z / spd)
            else:
                if zid in self.animals:
                    desired_dir = self.animals[zid].brain.desired_dir
                else:
                    ang2 = random.random() * 2.0 * math.pi
                    desired_dir = Vec3(math.cos(ang2), 0.0, math.sin(ang2))

            if zid not in self.animals:
                start_pos = Vec3(pos_x, 0.0, pos_z)
                self.animals[zid] = AnimalEntity(
                    animal_id=zid,
                    name=adata.get("name", zid),
                    species=adata.get("species", None),

                    transform=Transform(pos=start_pos.copy(), yaw=0.0),
                    target_pos=start_pos.copy(),

                    body=RigidBody(
                        vel=Vec3(vel_x, 0.0, vel_z),
                        radius=ANIMAL_SPACE_RADIUS,
                    ),
                    brain=AgentBrainLike(desired_dir=desired_dir),
                    anim=AnimalAnimState(
                        walk_phase=age_ticks * 0.3,
                        health=hp_an,
                        alive=alive_an,
                        temperament=temperament,
                        tamed=tamed_flag,
                        owner_id=owner_id,
                        last_action=last_action,
                        prev_hp=hp_an,
                    ),
                    public_state=dict(adata),
                    net=NetState(server_pos=start_pos.copy(), server_vel=Vec3(vel_x,0.0,vel_z), since_snap=0.0),
                    selected=False,
                    last_action_prev=last_action,
                )
            else:
                ent_an = self.animals[zid]

                # сетевое состояние для DR
                ent_an.net.server_pos.x = pos_x
                ent_an.net.server_pos.y = 0.0
                ent_an.net.server_pos.z = pos_z
                ent_an.net.server_vel.x = vel_x
                ent_an.net.server_vel.y = 0.0
                ent_an.net.server_vel.z = vel_z
                ent_an.net.since_snap = 0.0

                # целевая позиция (совместимость)
                ent_an.target_pos.x = pos_x
                ent_an.target_pos.y = 0.0
                ent_an.target_pos.z = pos_z

                ent_an.body.vel.x = vel_x
                ent_an.body.vel.y = 0.0
                ent_an.body.vel.z = vel_z

                ent_an.brain.desired_dir = desired_dir

                # числа урона/хила
                if SHOW_DAMAGE_NUMBERS and self._hud_text_enabled:
                    if abs(hp_an - ent_an.anim.prev_hp) >= 0.5:
                        delta = hp_an - ent_an.anim.prev_hp
                        col = (0.2, 1.0, 0.3, 1.0) if delta > 0 else (1.0, 0.25, 0.25, 1.0)
                        self._spawn_damage_number(ent_an.transform.pos.x, ent_an.transform.pos.z, y=1.9, value=delta, color=col)
                ent_an.anim.prev_hp = hp_an

                ent_an.anim.health = hp_an
                ent_an.anim.alive = alive_an
                ent_an.anim.temperament = temperament
                ent_an.anim.tamed = tamed_flag
                ent_an.anim.owner_id = owner_id
                ent_an.anim.last_action = last_action

                # VFX: смена действия
                if last_action and last_action != ent_an.last_action_prev:
                    self._spawn_ring(ent_an.transform.pos.x, ent_an.transform.pos.z, y=0.03, color=(0.8,1.0,0.3,0.85))
                ent_an.last_action_prev = last_action

                ent_an.public_state = dict(adata)

        # удалить зверей, которых больше нет
        for old_zid in list(self.animals.keys()):
            if old_zid not in live_animal_ids:
                del self.animals[old_zid]

    # -----------------------------------------------------------------
    # ЛОКАЛЬНЫЕ СИСТЕМЫ ОБНОВЛЕНИЯ
    # -----------------------------------------------------------------

    def _smooth_positions_towards_targets(self, dt: float):
        """
        Плавно тянем визуальную позицию к предсказанной/серверной target_pos.
        Сначала считаем predicted server_pos + server_vel * since_snap (dead-reckoning),
        затем визуально интерполируем к нему.
        """
        if dt <= 0.0:
            return
        alpha = clamp(dt * SMOOTH_LERP_SPEED, 0.0, 1.0)

        # Обновляем таймеры снапшотов
        for ent in self.agents.values():
            if DEAD_RECKONING:
                ent.net.since_snap = clamp(ent.net.since_snap + dt, 0.0, 10.0)
                tpr = min(ent.net.since_snap, DR_MAX_PREDICT_SEC)
                predicted = Vec3(
                    ent.net.server_pos.x + ent.net.server_vel.x * tpr,
                    0.0,
                    ent.net.server_pos.z + ent.net.server_vel.z * tpr,
                )
                ent.target_pos = predicted
            # Сглаживание к цели
            ent.transform.pos = ent.transform.pos.lerp(ent.target_pos, alpha)
            ent.transform.pos.y = 0.0
            ent.transform.pos.x = clamp(ent.transform.pos.x, 0.0, self.world.width)
            ent.transform.pos.z = clamp(ent.transform.pos.z, 0.0, self.world.height)

        for ent_an in self.animals.values():
            if DEAD_RECKONING:
                ent_an.net.since_snap = clamp(ent_an.net.since_snap + dt, 0.0, 10.0)
                tpr = min(ent_an.net.since_snap, DR_MAX_PREDICT_SEC)
                predicted = Vec3(
                    ent_an.net.server_pos.x + ent_an.net.server_vel.x * tpr,
                    0.0,
                    ent_an.net.server_pos.z + ent_an.net.server_vel.z * tpr,
                )
                ent_an.target_pos = predicted
            ent_an.transform.pos = ent_an.transform.pos.lerp(ent_an.target_pos, alpha)
            ent_an.transform.pos.y = 0.0
            ent_an.transform.pos.x = clamp(ent_an.transform.pos.x, 0.0, self.world.width)
            ent_an.transform.pos.z = clamp(ent_an.transform.pos.z, 0.0, self.world.height)

    def _apply_social_avoidance_agents(self):
        """
        Локальное "разведение" агентов, чтобы в кадре не стояли в одной точке.
        Это не влияет на сервер — чисто визуальный щиток.
        """
        ids = list(self.agents.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.agents[ids[i]]
                b = self.agents[ids[j]]
                if not a.anim.alive or not b.anim.alive:
                    continue
                dx = a.transform.pos.x - b.transform.pos.x
                dz = a.transform.pos.z - b.transform.pos.z
                dist2 = dx * dx + dz * dz
                if dist2 < 1e-9:
                    continue
                min_dist = a.body.radius + b.body.radius
                if dist2 < (min_dist * min_dist):
                    dist = math.sqrt(dist2)
                    if dist > 1e-6:
                        push = (min_dist - dist) * 0.5 * SEPARATION_PUSH
                        nx = dx / dist
                        nz = dz / dist
                        a.transform.pos.x += nx * push
                        a.transform.pos.z += nz * push
                        b.transform.pos.x -= nx * push
                        b.transform.pos.z -= nz * push

                        a.transform.pos.x = clamp(a.transform.pos.x, 0.0, self.world.width)
                        a.transform.pos.z = clamp(a.transform.pos.z, 0.0, self.world.height)
                        b.transform.pos.x = clamp(b.transform.pos.x, 0.0, self.world.width)
                        b.transform.pos.z = clamp(b.transform.pos.z, 0.0, self.world.height)

    def _apply_social_avoidance_animals(self):
        """
        То же самое для зверей.
        """
        ids = list(self.animals.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self.animals[ids[i]]
                b = self.animals[ids[j]]
                if not a.anim.alive or not b.anim.alive:
                    continue
                dx = a.transform.pos.x - b.transform.pos.x
                dz = a.transform.pos.z - b.transform.pos.z
                dist2 = dx * dx + dz * dz
                if dist2 < 1e-9:
                    continue
                min_dist = a.body.radius + b.body.radius
                if dist2 < (min_dist * min_dist):
                    dist = math.sqrt(dist2)
                    if dist > 1e-6:
                        push = (min_dist - dist) * 0.5 * SEPARATION_PUSH
                        nx = dx / dist
                        nz = dz / dist
                        a.transform.pos.x += nx * push
                        a.transform.pos.z += nz * push
                        b.transform.pos.x -= nx * push
                        b.transform.pos.z -= nz * push

                        a.transform.pos.x = clamp(a.transform.pos.x, 0.0, self.world.width)
                        a.transform.pos.z = clamp(a.transform.pos.z, 0.0, self.world.height)
                        b.transform.pos.x = clamp(b.transform.pos.x, 0.0, self.world.width)
                        b.transform.pos.z = clamp(b.transform.pos.z, 0.0, self.world.height)

    def _apply_social_avoidance_cross(self):
        """
        Разведение "агент ↔ зверь".
        Нужно, чтобы волк визуально не залезал в туловище человека.
        """
        a_ids = list(self.agents.keys())
        z_ids = list(self.animals.keys())

        for aid in a_ids:
            for zid in z_ids:
                ag = self.agents[aid]
                an = self.animals[zid]

                # если кто-то мёртв, можно не толкать
                if not ag.anim.alive or not an.anim.alive:
                    continue

                dx = ag.transform.pos.x - an.transform.pos.x
                dz = ag.transform.pos.z - an.transform.pos.z
                dist2 = dx * dx + dz * dz
                if dist2 < 1e-9:
                    continue

                min_dist = ag.body.radius + an.body.radius
                if dist2 < (min_dist * min_dist):
                    dist = math.sqrt(dist2)
                    if dist > 1e-6:
                        push = (min_dist - dist) * 0.5 * SEPARATION_PUSH
                        nx = dx / dist
                        nz = dz / dist

                        # раздвигаем оба тела поровну
                        ag.transform.pos.x += nx * push
                        ag.transform.pos.z += nz * push
                        an.transform.pos.x -= nx * push
                        an.transform.pos.z -= nz * push

                        ag.transform.pos.x = clamp(ag.transform.pos.x, 0.0, self.world.width)
                        ag.transform.pos.z = clamp(ag.transform.pos.z, 0.0, self.world.height)
                        an.transform.pos.x = clamp(an.transform.pos.x, 0.0, self.world.width)
                        an.transform.pos.z = clamp(an.transform.pos.z, 0.0, self.world.height)

    def _orient_and_animate_agents(self, dt: float):
        """
        Плавно крутим корпус агента в сторону движения.
        Обновляем фазу шага по скорости.
        """
        if dt <= 0.0:
            return
        for ent in self.agents.values():
            dir_vec = ent.brain.desired_dir
            target_yaw = math.atan2(dir_vec.z, dir_vec.x)
            ent.transform.yaw = angle_lerp(ent.transform.yaw, target_yaw, t=min(1.0, dt * 8.0))

            speed_flat = math.hypot(ent.body.vel.x, ent.body.vel.z)
            ent.anim.walk_phase += speed_flat * dt * 0.15 * (2.0 * math.pi)
            if ent.anim.walk_phase > 2.0 * math.pi:
                ent.anim.walk_phase -= 2.0 * math.pi

    def _orient_and_animate_animals(self, dt: float):
        """
        То же самое для зверей: поворот корпуса и лёгкое покачивание.
        """
        if dt <= 0.0:
            return
        for ent in self.animals.values():
            dir_vec = ent.brain.desired_dir
            target_yaw = math.atan2(dir_vec.z, dir_vec.x)
            ent.transform.yaw = angle_lerp(ent.transform.yaw, target_yaw, t=min(1.0, dt * 8.0))

            speed_flat = math.hypot(ent.body.vel.x, ent.body.vel.z)
            ent.anim.walk_phase += speed_flat * dt * 0.22 * (2.0 * math.pi)
            if ent.anim.walk_phase > 2.0 * math.pi:
                ent.anim.walk_phase -= 2.0 * math.pi

    def _update_vfx(self, dt: float):
        if dt <= 0.0:
            return
        alive_list = []
        for ring in self.vfx:
            ring.age += dt
            if ring.alive():
                alive_list.append(ring)
        self.vfx = alive_list

        # damage numbers
        nums_alive = []
        for dn in self.numbers:
            dn.age += dt
            if dn.alive():
                nums_alive.append(dn)
        self.numbers = nums_alive

    def _spawn_ring(self, x: float, z: float, y: float = 0.03,
                    r0: float = VFX_RING_R0, r1: float = VFX_RING_R1,
                    color: Tuple[float, float, float, float] = (1.0, 1.0, 0.2, 0.9),
                    ttl: float = VFX_RING_TTL):
        self.vfx.append(VFXRing(x=x, z=z, y=y, r0=r0, r1=r1, ttl=ttl, color=color))

    def _spawn_damage_number(self, x: float, z: float, y: float, value: float,
                             color: Tuple[float, float, float, float]):
        if not self._hud_text_enabled:
            return
        self.numbers.append(DamageNumber(x=x, z=z, y0=y, value=value, color=color))

    def _spawn_vfx_from_event(self, ev: str):
        """
        Пытаемся сопоставить id сущностей в тексте события и показать кольцо на них.
        Если не нашли — рисуем в центре мира.
        """
        placed = False
        tokens = [t.strip(",.:;()") for t in ev.split()]
        # ищем точные id
        for t in tokens:
            if t in self.agents:
                p = self.agents[t].transform.pos
                self._spawn_ring(p.x, p.z, y=0.04, color=(1.0, 0.8, 0.2, 0.9))
                placed = True
            if t in self.animals:
                p = self.animals[t].transform.pos
                self._spawn_ring(p.x, p.z, y=0.04, color=(0.8, 1.0, 0.6, 0.9))
                placed = True
        if not placed:
            # кольцо в центре, чтоб хотя бы визуально отметить событие
            self._spawn_ring(self.world.width * 0.5, self.world.height * 0.5, y=0.02, color=(1.0, 1.0, 1.0, 0.6))

    # -----------------------------------------------------------------
    # HUD: текст и бары
    # -----------------------------------------------------------------

    def _draw_text3d(self, s: str, x: float, y: float, z: float, big: bool = False, alpha: float = 1.0,
                     rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        if not self._hud_text_enabled or not s:
            return
        try:
            glColor4f(rgb[0], rgb[1], rgb[2], alpha)
            glRasterPos3f(x, y, z)
            font = GLUT_BITMAP_HELVETICA_18 if big else GLUT_BITMAP_HELVETICA_12
            for ch in s:
                glutBitmapCharacter(font, ord(ch))
        except Exception:
            # если драйвер не позволяет — тихо пропускаем
            pass

    def _within_hud_lod(self, x: float, z: float, force: bool = False) -> bool:
        if force:
            return True
        if MAX_HUD_DISTANCE <= 0:
            return True
        # расстояние по XZ до камеры
        cx, cy, cz = self._cam_pos
        dx = x - cx
        dz = z - cz
        return (dx * dx + dz * dz) <= (MAX_HUD_DISTANCE * MAX_HUD_DISTANCE)

    def _draw_agent_hud(self, agent: AgentEntity):
        px, pz = agent.transform.pos.x, agent.transform.pos.z

        if not self._within_hud_lod(px, pz, force=agent.selected):
            return

        base_y = 2.35

        hp = clamp(agent.anim.health / 100.0, 0.0, 1.0)
        fear = clamp(agent.anim.fear, 0.0, 1.0)

        # HP bar (фон)
        w = HUD_HP_BAR_W; h = HUD_HP_BAR_H
        glColor4f(0.1, 0.1, 0.1, 0.75)
        glBegin(GL_QUADS)
        glVertex3f(px - w/2, base_y, pz)
        glVertex3f(px + w/2, base_y, pz)
        glVertex3f(px + w/2, base_y + h, pz)
        glVertex3f(px - w/2, base_y + h, pz)
        glEnd()
        # HP fill (зелёный→красный)
        glColor4f(1.0 - 0.6*hp, 0.2 + 0.8*hp, 0.2, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(px - w/2, base_y, pz)
        glVertex3f(px - w/2 + w*hp, base_y, pz)
        glVertex3f(px - w/2 + w*hp, base_y + h, pz)
        glVertex3f(px - w/2, base_y + h, pz)
        glEnd()

        # FEAR bar ниже
        base_y2 = base_y - 0.12
        w2 = HUD_FEAR_BAR_W; h2 = HUD_FEAR_BAR_H
        glColor4f(0.1, 0.1, 0.1, 0.65)
        glBegin(GL_QUADS)
        glVertex3f(px - w2/2, base_y2, pz)
        glVertex3f(px + w2/2, base_y2, pz)
        glVertex3f(px + w2/2, base_y2 + h2, pz)
        glVertex3f(px - w2/2, base_y2 + h2, pz)
        glEnd()

        glColor4f(0.9, 0.25, 0.25, 0.85)
        glBegin(GL_QUADS)
        glVertex3f(px - w2/2, base_y2, pz)
        glVertex3f(px - w2/2 + w2*fear, base_y2, pz)
        glVertex3f(px - w2/2 + w2*fear, base_y2 + h2, pz)
        glVertex3f(px - w2/2, base_y2 + h2, pz)
        glEnd()

        # Имя над HP
        if self._hud_text_enabled:
            self._draw_text3d(agent.name, px - w/2, base_y + h + 0.06, pz, big=False)

    def _draw_animal_hud(self, an: AnimalEntity):
        px, pz = an.transform.pos.x, an.transform.pos.z

        if not self._within_hud_lod(px, pz, force=an.selected):
            return

        base_y = 1.6

        hp = clamp(an.anim.health / 100.0, 0.0, 1.0)
        w = 0.9; h = 0.07

        # HP bar
        glColor4f(0.1, 0.1, 0.1, 0.65)
        glBegin(GL_QUADS)
        glVertex3f(px - w/2, base_y, pz)
        glVertex3f(px + w/2, base_y, pz)
        glVertex3f(px + w/2, base_y + h, pz)
        glVertex3f(px - w/2, base_y + h, pz)
        glEnd()

        glColor4f(0.2 + 0.8*hp, 0.7*hp, 0.2, 0.85)
        glBegin(GL_QUADS)
        glVertex3f(px - w/2, base_y, pz)
        glVertex3f(px - w/2 + w*hp, base_y, pz)
        glVertex3f(px - w/2 + w*hp, base_y + h, pz)
        glVertex3f(px - w/2, base_y + h, pz)
        glEnd()

        if self._hud_text_enabled:
            name_line = an.species + " • " + an.name if an.species else an.name
            self._draw_text3d(name_line, px - w/2, base_y + h + 0.05, pz, big=False)
            if an.anim.last_action:
                self._draw_text3d(an.anim.last_action, px - w/2, base_y - 0.15, pz, big=False)

    # -----------------------------------------------------------------
    # ОБНОВЛЕНИЕ/РЕНДЕР
    # -----------------------------------------------------------------

    def update(self, dt: float):
        """
        Кадровое обновление клиента.
        Порядок:
          1) увеличиваем внутренний таймер (для анимации огня и т.д.)
          2) плавно тянем позы к серверным координатам (dead-reckoning + lerp)
          3) разводим сущности визуально (антислипание)
          4) плавно поворачиваем модели и обновляем walk_phase
          5) обновляем VFX
        """
        if dt <= 0.0:
            return

        self._time_accum += dt

        self._smooth_positions_towards_targets(dt)
        self._apply_social_avoidance_agents()
        self._apply_social_avoidance_animals()
        self._apply_social_avoidance_cross()
        self._orient_and_animate_agents(dt)
        self._orient_and_animate_animals(dt)
        self._update_vfx(dt)

    def setup_viewport_and_camera(
        self,
        w: int,
        h: int,
        cam_pos: Tuple[float, float, float],
        cam_look: Tuple[float, float, float],
        fov_deg: float = 45.0,
    ):
        """
        Настройка камеры и матриц проекции/вида под текущий размер окна.
        """
        if h <= 0:
            h = 1
        aspect = w / float(h)

        # сохраняем камеру для LOD HUD / отрисовки вспомогательных гизмов
        self._cam_pos = cam_pos
        self._cam_look = cam_look

        glViewport(0, 0, int(w), int(h))

        glClearColor(0.03, 0.03, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov_deg, aspect, 0.1, 2000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            cam_pos[0],  cam_pos[1],  cam_pos[2],
            cam_look[0], cam_look[1], cam_look[2],
            0.0,         1.0,         0.0
        )

    def _draw_vfx(self):
        for ring in self.vfx:
            a = ring.alpha()
            r = ring.radius()
            col = (ring.color[0], ring.color[1], ring.color[2])
            _draw_ring(ring.x, ring.z, radius=r, y=ring.y, rgb=col, width=2.0, alpha=a)

    def _draw_damage_numbers(self):
        if not (self._hud_text_enabled and SHOW_DAMAGE_NUMBERS):
            return
        for dn in self.numbers:
            alpha = clamp(dn.alpha(), 0.0, 1.0)
            val = dn.value
            sign = "+" if val > 0 else ""
            text = f"{sign}{val:.0f}"
            self._draw_text3d(text, dn.x, dn.y(), dn.z, big=True, alpha=alpha, rgb=(dn.color[0], dn.color[1], dn.color[2]))

    def _draw_agent_fov(self, agent: AgentEntity):
        if not (SHOW_FOV_CONES and agent.selected and agent.anim.alive):
            return
        px, pz = agent.transform.pos.x, agent.transform.pos.z
        yaw = agent.transform.yaw
        fov_deg = float(agent.public_state.get("fov_deg", FOV_DEG))
        fov_range = float(agent.public_state.get("fov_range", FOV_RANGE))
        half = math.radians(fov_deg * 0.5)

        steps = 16
        glColor4f(0.2, 0.8, 1.0, 0.12)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(px, 0.02, pz)
        for i in range(steps + 1):
            a = yaw - half + (2 * half) * (i / steps)
            vx = px + math.cos(a) * fov_range
            vz = pz + math.sin(a) * fov_range
            glVertex3f(vx, 0.02, vz)
        glEnd()

        glColor4f(0.2, 0.8, 1.0, 0.35)
        glLineWidth(1.5)
        _draw_ring(px, pz, radius=fov_range, y=0.021, rgb=(0.2, 0.8, 1.0), width=1.2, alpha=0.25)

    def _draw_target_lines(self):
        if not SHOW_TARGET_LINES:
            return

        def _pos_of(entity_id: str) -> Optional[Tuple[float, float]]:
            if entity_id in self.agents:
                p = self.agents[entity_id].transform.pos
                return p.x, p.z
            if entity_id in self.animals:
                p = self.animals[entity_id].transform.pos
                return p.x, p.z
            return None

        glLineWidth(1.5)
        for an in self.animals.values():
            # подхватить разные ключи цели
            tgt = an.public_state.get("target_id") \
                  or an.public_state.get("attack_target_id") \
                  or an.public_state.get("attack_target") \
                  or an.public_state.get("target")
            if isinstance(tgt, str):
                tp = _pos_of(tgt)
                if tp is None:
                    continue
                sx, sz = an.transform.pos.x, an.transform.pos.z
                tx, tz = tp
                # цвет: красный для агрессивных, голубой — если приручён идёт к хозяину
                if an.anim.tamed:
                    col = (0.2, 0.8, 1.0, 0.65)
                else:
                    col = (1.0, 0.25, 0.25, 0.75)
                glColor4f(*col)
                glBegin(GL_LINES)
                glVertex3f(sx, 1.0, sz)
                glVertex3f(tx, 1.0, tz)
                glEnd()

        # для агентов — если есть явная "target_id" в public_state
        for ag in self.agents.values():
            tgt = ag.public_state.get("target_id") or ag.public_state.get("target")
            if isinstance(tgt, str):
                tp = _pos_of(tgt)
                if tp is None:
                    continue
                sx, sz = ag.transform.pos.x, ag.transform.pos.z
                tx, tz = tp
                glColor4f(1.0, 1.0, 0.2, 0.75)
                glBegin(GL_LINES)
                glVertex3f(sx, 1.8, sz)
                glVertex3f(tx, 1.8, tz)
                glEnd()

    def render_opengl(self):
        """
        Рисуем мир:
          - пол + сетка
          - статика окружения (лес, дома, костёр, озеро)
          - зоны (safe / hazard) цветными дисками
          - цели агентов жёлтым кольцом
          - вспомогательные гизмы (FOV, линии-цели)
          - сами агенты + HUD
          - звери + HUD
          - VFX (вспышки) + числа урона/хила
        """
        # пол/сетка
        _draw_floor_grid(self.world.width, self.world.height)

        # окружение
        for inst in self.static_meshes:
            _draw_static_mesh(inst, self._time_accum)

        # зоны с сервера
        for zone in self.world.zones:
            _draw_disc_zone(zone.x, zone.z, zone.radius, zone.kind, y=0.02)

        # цели агентов (кольца)
        for agent in self.agents.values():
            _draw_ring(
                agent.goal.x,
                agent.goal.z,
                radius=0.8,
                y=0.03,
                rgb=(1.0, 1.0, 0.2),
                width=1.0,
            )

        # вспомогательные гизмы
        self._draw_target_lines()
        for agent in self.agents.values():
            self._draw_agent_fov(agent)

        # агенты
        for agent in self.agents.values():
            draw_agent_humanoid(agent, self._time_accum)
            draw_agent_direction_arrow(agent)
            self._draw_agent_hud(agent)

        # звери
        for animal in self.animals.values():
            draw_animal_quadruped(animal, self._time_accum)
            draw_animal_direction_arrow(animal)
            self._draw_animal_hud(animal)

        # эффекты
        self._draw_vfx()
        self._draw_damage_numbers()
