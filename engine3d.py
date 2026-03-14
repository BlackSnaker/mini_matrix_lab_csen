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
MAX_ENTITY_RENDER_DISTANCE = 120.0  # дальше этого сущности вообще не рисуем
MAX_AGENT_DETAIL_DISTANCE = 78.0    # после этого агенты переходят в простой LOD
MAX_ANIMAL_DETAIL_DISTANCE = 72.0   # после этого звери переходят в простой LOD
MAX_STATIC_DISTANCE = 135.0         # дальность статических мешей
MAX_ZONE_DISTANCE = 110.0           # дальность цветных зон
MAX_GOAL_RING_DISTANCE = 82.0       # дальность колец целей
MAX_DIRECTION_ARROW_DISTANCE = 52.0 # дальность стрелок направления
MAX_TARGET_LINE_DISTANCE = 88.0     # дальность линий целей
MAX_VFX_DISTANCE = 95.0             # дальность VFX и damage numbers
SOCIAL_AVOIDANCE_CELL = 2.25        # размер ячейки spatial hash для разведения
FOV_DEG             = 80.0      # базовая ширина FOV-конуса
FOV_RANGE           = 7.0       # базовая дальность FOV-конуса

# Sun / daylight
SUN_ENABLED = True
SUN_CYCLE_SEC = 180.0           # полный цикл "день" (сек) — без ночи, только дневной дрейф
SUN_MIN_ELEV_DEG = 28.0         # минимальная высота солнца над горизонтом
SUN_MAX_ELEV_DEG = 76.0         # максимальная высота солнца над горизонтом
SUN_BASE_HEIGHT = 26.0          # базовая высота солнца над картой

# Текущее состояние солнца (обновляется движком каждый кадр)
_SUN_DIR = (0.50, 0.84, 0.18)   # нормализованный вектор света
_SUN_COLOR = (1.0, 0.96, 0.88)  # тёплый дневной оттенок
_SUN_AMBIENT = 0.36
_SUN_DIFFUSE = 0.76
_SKY_COLOR = (0.53, 0.73, 0.96)


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

def _fake_lighting_color(
    base: Tuple[float, float, float],
    normal_x: float = 0.0,
    normal_y: float = 1.0,
    normal_z: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Упрощённый "свет солнца":
    - ambient + diffuse по dot(normal, sun_dir)
    - лёгкий up-bias, чтобы верхние грани всегда читались.
    """
    nx, ny, nz = float(normal_x), float(normal_y), float(normal_z)
    nlen = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    nx /= nlen
    ny /= nlen
    nz /= nlen

    sx, sy, sz = _SUN_DIR
    dot = max(0.0, nx * sx + ny * sy + nz * sz)

    light = _SUN_AMBIENT + _SUN_DIFFUSE * dot + 0.10 * max(0.0, ny)
    light = clamp(light, 0.14, 1.35)

    warm_r, warm_g, warm_b = _SUN_COLOR
    r = clamp(base[0] * light * warm_r, 0.0, 1.0)
    g = clamp(base[1] * light * warm_g, 0.0, 1.0)
    b = clamp(base[2] * light * warm_b, 0.0, 1.0)
    return (r, g, b)


def _draw_floor_grid(
    world_w: float,
    world_h: float,
    cam_pos: Optional[Tuple[float, float, float]] = None,
):
    """
    Слой земли + сетка.
    Сделано чуть "богаче", чтобы карта лучше читалась на больших масштабах.
    """
    ground_boost = clamp(_SUN_AMBIENT + 0.40 * _SUN_DIFFUSE, 0.25, 1.10)
    detail_pressure = 0.0
    if cam_pos is not None:
        cx, cy, cz = cam_pos
        center_x = world_w * 0.5
        center_z = world_h * 0.5
        detail_pressure = max(abs(cy) * 1.25, math.hypot(center_x - cx, center_z - cz))

    # Базовая подложка.
    c0 = _fake_lighting_color((0.12 * ground_boost, 0.17 * ground_boost, 0.14 * ground_boost), 0.0, 1.0, 0.0)
    glColor3f(*c0)
    glBegin(GL_QUADS)
    glVertex3f(0.0,      0.0,      0.0)
    glVertex3f(world_w,  0.0,      0.0)
    glVertex3f(world_w,  0.0,      world_h)
    glVertex3f(0.0,      0.0,      world_h)
    glEnd()

    # Продольные "полосы" цвета по Z, чтобы плоскость не была плоской по тону.
    if detail_pressure >= 140.0:
        bands = 8
    elif detail_pressure >= 80.0:
        bands = 12
    else:
        bands = 18
    for i in range(bands):
        t0 = i / float(bands)
        t1 = (i + 1) / float(bands)
        z0 = world_h * t0
        z1 = world_h * t1
        c = 0.05 + 0.03 * math.sin((t0 + t1) * math.pi * 2.0)
        cc = _fake_lighting_color(
            (0.10 + c * 0.09, 0.15 + c * 0.14, 0.11 + c * 0.07),
            0.0, 1.0, 0.0
        )
        glColor3f(*cc)
        glBegin(GL_QUADS)
        glVertex3f(0.0,      0.0002, z0)
        glVertex3f(world_w,  0.0002, z0)
        glVertex3f(world_w,  0.0002, z1)
        glVertex3f(0.0,      0.0002, z1)
        glEnd()

    _draw_ground_patches(world_w, world_h, detail_pressure)

    # Минорная сетка.
    base_step = max(5.0, min(world_w, world_h) / 22.0)
    if detail_pressure >= 140.0:
        minor_step = base_step * 4.0
    elif detail_pressure >= 80.0:
        minor_step = base_step * 2.0
    else:
        minor_step = base_step
    draw_minor = detail_pressure < 170.0
    if draw_minor:
        grid_minor = _fake_lighting_color((0.23, 0.25, 0.24), 0.0, 1.0, 0.0)
        glColor3f(*grid_minor)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        x = 0.0
        while x <= world_w + 0.001:
            glVertex3f(x, 0.001, 0.0)
            glVertex3f(x, 0.001, world_h)
            x += minor_step
        z = 0.0
        while z <= world_h + 0.001:
            glVertex3f(0.0,      0.001, z)
            glVertex3f(world_w,  0.001, z)
            z += minor_step
        glEnd()

    # Мажорная сетка (толще и светлее) каждые 4 шага.
    major_step = base_step * (8.0 if detail_pressure >= 140.0 else 4.0)
    grid_major = _fake_lighting_color((0.32, 0.35, 0.33), 0.0, 1.0, 0.0)
    glColor3f(*grid_major)
    glLineWidth(1.4)
    glBegin(GL_LINES)
    x = 0.0
    while x <= world_w + 0.001:
        glVertex3f(x, 0.0015, 0.0)
        glVertex3f(x, 0.0015, world_h)
        x += major_step
    z = 0.0
    while z <= world_h + 0.001:
        glVertex3f(0.0,      0.0015, z)
        glVertex3f(world_w,  0.0015, z)
        z += major_step
    glEnd()

    # Контур мира.
    border_col = _fake_lighting_color((0.75, 0.79, 0.82), 0.0, 1.0, 0.0)
    glColor3f(*border_col)
    glLineWidth(2.4)
    glBegin(GL_LINES)
    glVertex3f(0.0,      0.003, 0.0);      glVertex3f(world_w,  0.003, 0.0)
    glVertex3f(world_w,  0.003, 0.0);      glVertex3f(world_w,  0.003, world_h)
    glVertex3f(world_w,  0.003, world_h);  glVertex3f(0.0,      0.003, world_h)
    glVertex3f(0.0,      0.003, world_h);  glVertex3f(0.0,      0.003, 0.0)
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
        nx /= nlen
        ny /= nlen
        nz /= nlen

        lit = _fake_lighting_color(base_color, normal_x=nx, normal_y=ny, normal_z=nz)
        glColor3f(*lit)

        for idx in f:
            glVertex3f(*V(idx))
    glEnd()


def _mix_rgb(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    tt = clamp(t, 0.0, 1.0)
    return (
        a[0] * (1.0 - tt) + b[0] * tt,
        a[1] * (1.0 - tt) + b[1] * tt,
        a[2] * (1.0 - tt) + b[2] * tt,
    )


def _hash01(x: float, y: float, z: float = 0.0) -> float:
    v = math.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453123
    return v - math.floor(v)


def _vary_rgb(
    base: Tuple[float, float, float],
    seed_a: float,
    seed_b: float = 0.0,
    amount: float = 0.08,
) -> Tuple[float, float, float]:
    r = clamp(base[0] + (_hash01(seed_a, seed_b, 1.0) - 0.5) * 2.0 * amount, 0.0, 1.0)
    g = clamp(base[1] + (_hash01(seed_a, seed_b, 2.0) - 0.5) * 2.0 * amount, 0.0, 1.0)
    b = clamp(base[2] + (_hash01(seed_a, seed_b, 3.0) - 0.5) * 2.0 * amount, 0.0, 1.0)
    return (r, g, b)


def _yaw_world_offset(cx: float, cz: float, yaw_rad: float, lx: float, lz: float) -> Tuple[float, float]:
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    return cx + lx * cos_y - lz * sin_y, cz + lx * sin_y + lz * cos_y


def _draw_ground_patches(world_w: float, world_h: float, detail_pressure: float):
    patch_step = max(10.0, min(world_w, world_h) / (8.0 if detail_pressure < 100.0 else 6.0))
    cols = max(3, int(world_w / patch_step) + 1)
    rows = max(3, int(world_h / patch_step) + 1)
    steps = 14 if detail_pressure < 110.0 else 10
    base_col = (0.14, 0.19, 0.12)

    for ix in range(cols):
        for iz in range(rows):
            n = _hash01(ix * 0.73, iz * 0.91, patch_step)
            cx = (ix + 0.5) * patch_step + (n - 0.5) * patch_step * 0.38
            cz = (iz + 0.5) * patch_step + (_hash01(ix * 1.19, iz * 0.67, patch_step) - 0.5) * patch_step * 0.34
            rx = patch_step * (0.22 + 0.12 * n)
            rz = patch_step * (0.18 + 0.14 * _hash01(ix * 0.49, iz * 1.31, 8.0))
            col = _fake_lighting_color(_vary_rgb(base_col, cx * 0.11, cz * 0.13, amount=0.05), 0.0, 1.0, 0.0)
            alpha = 0.06 + 0.08 * n

            glColor4f(col[0], col[1], col[2], alpha)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(cx, 0.0008, cz)
            for step in range(steps + 1):
                ang = (2.0 * math.pi) * (step / steps)
                wobble = 0.88 + 0.22 * _hash01(ix * 3.1 + step, iz * 2.7, ang)
                vx = cx + math.cos(ang) * rx * wobble
                vz = cz + math.sin(ang) * rz * wobble
                glVertex3f(vx, 0.0008, vz)
            glEnd()


def _draw_shingled_roof(
    cx: float,
    cy: float,
    cz: float,
    sx: float,
    sy: float,
    sz: float,
    yaw_rad: float,
    base_color: Tuple[float, float, float],
    top_color: Tuple[float, float, float],
    tiers: int = 5,
):
    tiers = max(3, int(tiers))
    slab_h = max(0.02, sy * 0.045)
    for i in range(tiers):
        t = i / float(max(1, tiers - 1))
        tier_col = _vary_rgb(_mix_rgb(base_color, top_color, 0.15 + t * 0.45), cx * 0.2 + i, cz * 0.2, 0.04)
        _draw_oriented_box(
            cx=cx,
            cy=cy + slab_h * (0.5 + i * 0.95),
            cz=cz,
            hx=sx * (1.08 - 0.10 * t),
            hy=slab_h,
            hz=sz * (1.08 - 0.10 * t),
            yaw_rad=yaw_rad,
            base_color=tier_col,
            pitch_forward=0.0,
        )

    _draw_pyramid_roof(
        cx=cx,
        cy=cy + slab_h * (tiers * 0.90),
        cz=cz,
        sx=sx,
        sy=sy,
        sz=sz,
        yaw_rad=yaw_rad,
        base_color=base_color,
        top_color=top_color,
    )


def _draw_stone_courses(
    cx: float,
    cy: float,
    cz: float,
    hx: float,
    hy: float,
    hz: float,
    yaw_rad: float,
    base_color: Tuple[float, float, float],
    courses: int = 4,
):
    courses = max(2, int(courses))
    course_full_h = (hy * 2.0) / courses
    bottom_y = cy - hy
    for i in range(courses):
        center_y = bottom_y + course_full_h * (i + 0.5)
        inset = 0.96 - 0.04 * (i % 2)
        stone_col = _vary_rgb(base_color, cx * 0.33 + i * 1.7, cz * 0.41, 0.05)
        _draw_oriented_box(
            cx=cx,
            cy=center_y,
            cz=cz,
            hx=hx * inset,
            hy=course_full_h * 0.45,
            hz=hz * (0.96 - 0.03 * ((i + 1) % 2)),
            yaw_rad=yaw_rad,
            base_color=stone_col,
            pitch_forward=0.0,
        )


def _draw_water_surface(
    x: float,
    y: float,
    z: float,
    sx: float,
    sz: float,
    time_seed: float,
):
    layers = [
        ((0.10, 0.28, 0.50), 0.72, 0.00),
        ((0.16, 0.42, 0.68), 0.48, 0.08),
        ((0.34, 0.63, 0.86), 0.24, 0.15),
    ]
    for rgb, alpha, inset in layers:
        glColor4f(rgb[0], rgb[1], rgb[2], alpha)
        glBegin(GL_QUADS)
        glVertex3f(x - sx + inset, y, z - sz + inset)
        glVertex3f(x + sx - inset, y, z - sz + inset)
        glVertex3f(x + sx - inset, y, z + sz - inset)
        glVertex3f(x - sx + inset, y, z + sz - inset)
        glEnd()

    glColor4f(0.80, 0.93, 1.00, 0.26)
    glLineWidth(1.1)
    for row in range(5):
        zz = z - sz * 0.72 + (row / 4.0) * sz * 1.44
        glBegin(GL_LINES)
        segments = 12
        prev = None
        for seg in range(segments + 1):
            t = seg / float(segments)
            xx = x - sx * 0.80 + t * sx * 1.60
            wave = math.sin(time_seed * 2.7 + row * 0.9 + t * math.pi * 3.2) * sz * 0.035
            pt = (xx, y + 0.004, zz + wave)
            if prev is not None:
                glVertex3f(*prev)
                glVertex3f(*pt)
            prev = pt
        glEnd()


def _draw_sphere_lowpoly(
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    base_color: Tuple[float, float, float],
    *,
    lat_steps: int = 7,
    lon_steps: int = 10,
):
    """
    Низкополигональная сфера (голова/суставы), но визуально мягче плоского диска.
    """
    lat_steps = max(3, int(lat_steps))
    lon_steps = max(6, int(lon_steps))
    for i in range(lat_steps):
        v0 = i / float(lat_steps)
        v1 = (i + 1) / float(lat_steps)
        lat0 = -0.5 * math.pi + v0 * math.pi
        lat1 = -0.5 * math.pi + v1 * math.pi
        y0 = math.sin(lat0); r0 = math.cos(lat0)
        y1 = math.sin(lat1); r1 = math.cos(lat1)

        for j in range(lon_steps):
            u0 = j / float(lon_steps)
            u1 = (j + 1) / float(lon_steps)
            lon0 = u0 * 2.0 * math.pi
            lon1 = u1 * 2.0 * math.pi

            p00 = (cx + radius * r0 * math.cos(lon0), cy + radius * y0, cz + radius * r0 * math.sin(lon0))
            p10 = (cx + radius * r0 * math.cos(lon1), cy + radius * y0, cz + radius * r0 * math.sin(lon1))
            p11 = (cx + radius * r1 * math.cos(lon1), cy + radius * y1, cz + radius * r1 * math.sin(lon1))
            p01 = (cx + radius * r1 * math.cos(lon0), cy + radius * y1, cz + radius * r1 * math.sin(lon0))

            mx = 0.25 * (p00[0] + p10[0] + p11[0] + p01[0]) - cx
            my = 0.25 * (p00[1] + p10[1] + p11[1] + p01[1]) - cy
            mz = 0.25 * (p00[2] + p10[2] + p11[2] + p01[2]) - cz
            lit = _fake_lighting_color(base_color, normal_x=mx, normal_y=my, normal_z=mz)
            glColor3f(*lit)
            glBegin(GL_QUADS)
            glVertex3f(*p00)
            glVertex3f(*p10)
            glVertex3f(*p11)
            glVertex3f(*p01)
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
    Совместимый API: рисуем не диск, а упрощённую сферу головы.
    """
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    head_cx = cx + math.sin(forward_lean) * cos_y * 0.20
    head_cz = cz + math.sin(forward_lean) * sin_y * 0.20
    _draw_sphere_lowpoly(
        head_cx,
        cy,
        head_cz,
        radius=max(0.06, radius),
        base_color=(0.92, 0.87, 0.82),
        lat_steps=7,
        lon_steps=10,
    )


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
        nx /= nlen
        ny /= nlen
        nz /= nlen

        face_col = base_color
        if top_color:
            mix_t = clamp((ny + 1.0) * 0.5, 0.0, 1.0)
            face_col = (
                base_color[0]*(1-mix_t) + top_color[0]*mix_t,
                base_color[1]*(1-mix_t) + top_color[1]*mix_t,
                base_color[2]*(1-mix_t) + top_color[2]*mix_t,
            )

        lit = _fake_lighting_color(face_col, normal_x=nx, normal_y=ny, normal_z=nz)
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

    stone_base_h = max(0.12, sy * 0.24)
    wall_h = max(0.25, sy - stone_base_h)
    wall_color = (0.66, 0.59, 0.50)
    beam_color = (0.39, 0.25, 0.16)
    roof_color = (0.36, 0.15, 0.11)
    roof_high = (0.63, 0.27, 0.18)
    window_color = (0.30, 0.42, 0.52)

    _draw_stone_courses(
        cx=x,
        cy=y + stone_base_h,
        cz=z,
        hx=sx * 1.02,
        hy=stone_base_h,
        hz=sz * 1.02,
        yaw_rad=yaw,
        base_color=(0.49, 0.49, 0.51),
        courses=2,
    )

    _draw_oriented_box(
        cx=x,
        cy=y + stone_base_h * 2.0 + wall_h,
        cz=z,
        hx=sx,
        hy=wall_h,
        hz=sz,
        yaw_rad=yaw,
        base_color=wall_color,
        pitch_forward=0.0,
    )

    for lx in (-sx * 0.84, sx * 0.84):
        wx, wz = _yaw_world_offset(x, z, yaw, lx, 0.0)
        _draw_oriented_box(
            cx=wx,
            cy=y + stone_base_h * 2.0 + wall_h,
            cz=wz,
            hx=max(0.05, sx * 0.10),
            hy=wall_h,
            hz=max(0.05, sz * 0.10),
            yaw_rad=yaw,
            base_color=beam_color,
            pitch_forward=0.0,
        )
    for lz in (-sz * 0.84, sz * 0.84):
        wx, wz = _yaw_world_offset(x, z, yaw, 0.0, lz)
        _draw_oriented_box(
            cx=wx,
            cy=y + stone_base_h * 2.0 + wall_h,
            cz=wz,
            hx=max(0.05, sx * 0.92),
            hy=max(0.04, wall_h * 0.09),
            hz=max(0.05, sz * 0.10),
            yaw_rad=yaw,
            base_color=beam_color,
            pitch_forward=0.0,
        )

    door_x, door_z = _yaw_world_offset(x, z, yaw, 0.0, sz * 0.92)
    _draw_oriented_box(
        cx=door_x,
        cy=y + stone_base_h * 2.0 + wall_h * 0.48,
        cz=door_z,
        hx=max(0.12, sx * 0.18),
        hy=max(0.22, wall_h * 0.48),
        hz=max(0.03, sz * 0.05),
        yaw_rad=yaw,
        base_color=(0.32, 0.21, 0.14),
        pitch_forward=0.0,
    )
    for lx in (-sx * 0.42, sx * 0.42):
        wx, wz = _yaw_world_offset(x, z, yaw, lx, sz * 0.94)
        _draw_oriented_box(
            cx=wx,
            cy=y + stone_base_h * 2.0 + wall_h * 0.76,
            cz=wz,
            hx=max(0.11, sx * 0.14),
            hy=max(0.10, wall_h * 0.16),
            hz=max(0.03, sz * 0.04),
            yaw_rad=yaw,
            base_color=window_color,
            pitch_forward=0.0,
        )

    _draw_shingled_roof(
        cx=x,
        cy=y + sy*2.0,
        cz=z,
        sx=sx*1.05,
        sy=sy*0.8,
        sz=sz*1.05,
        yaw_rad=yaw,
        base_color=roof_color,
        top_color=roof_high,
        tiers=5,
    )


def _draw_tree(inst: StaticMeshInstance):
    """
    Дерево: коричневый ствол (box) + зелёная шапка (pyramid).
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw

    trunk_h = 0.64 * inst.scale.y
    crown_h = 0.95 * inst.scale.y
    radius  = 0.22 * inst.scale.x

    trunk_color = (0.34, 0.23, 0.15)
    bark_dark = (0.22, 0.14, 0.10)
    leaf_color  = (0.12, 0.35, 0.12)
    leaf_high   = (0.36, 0.62, 0.22)

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
    _draw_oriented_box(
        cx=x,
        cy=y + trunk_h * 0.52,
        cz=z,
        hx=radius * 0.18,
        hy=trunk_h * 0.48,
        hz=radius * 0.72,
        yaw_rad=yaw + 0.15,
        base_color=bark_dark,
        pitch_forward=0.0,
    )

    _draw_pyramid_roof(
        cx=x,
        cy=y + trunk_h * 0.82,
        cz=z,
        sx=radius * 1.28,
        sy=crown_h * 0.66,
        sz=radius * 1.24,
        yaw_rad=yaw,
        base_color=leaf_color,
        top_color=leaf_high,
    )
    _draw_pyramid_roof(
        cx=x,
        cy=y + trunk_h + crown_h * 0.18,
        cz=z,
        sx=radius * 0.92,
        sy=crown_h * 0.70,
        sz=radius * 0.88,
        yaw_rad=yaw + 0.23,
        base_color=_vary_rgb(leaf_color, x, z, 0.04),
        top_color=leaf_high,
    )


def _draw_lake(inst: StaticMeshInstance):
    """
    Плоский "озёрный" прямоугольник.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    sx = inst.scale.x
    sz = inst.scale.z

    glColor4f(0.20, 0.24, 0.19, 0.24)
    glBegin(GL_QUADS)
    glVertex3f(x - sx * 1.05, y - 0.002, z - sz * 1.05)
    glVertex3f(x + sx * 1.05, y - 0.002, z - sz * 1.05)
    glVertex3f(x + sx * 1.05, y - 0.002, z + sz * 1.05)
    glVertex3f(x - sx * 1.05, y - 0.002, z + sz * 1.05)
    glEnd()

    _draw_water_surface(x, y, z, sx, sz, time_seed=x * 0.17 + z * 0.23)

    glColor3f(0.34, 0.62, 0.86)
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


def _draw_road(inst: StaticMeshInstance):
    """
    Плоская дорога-полоса с боковой кромкой.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    half_len = max(0.4, abs(inst.scale.x))
    half_w = max(0.25, abs(inst.scale.z))

    _draw_oriented_box(
        cx=x, cy=y + 0.02, cz=z,
        hx=half_len, hy=0.02, hz=half_w,
        yaw_rad=yaw,
        base_color=(0.30, 0.25, 0.20),
        pitch_forward=0.0,
    )

    _draw_oriented_box(
        cx=x, cy=y + 0.024, cz=z,
        hx=half_len * 0.96, hy=0.006, hz=half_w * 0.78,
        yaw_rad=yaw,
        base_color=(0.22, 0.18, 0.14),
        pitch_forward=0.0,
    )
    for lx in (-half_len * 0.28, half_len * 0.28):
        wx, wz = _yaw_world_offset(x, z, yaw, lx, 0.0)
        _draw_oriented_box(
            cx=wx,
            cy=y + 0.026,
            cz=wz,
            hx=half_len * 0.16,
            hy=0.005,
            hz=half_w * 0.64,
            yaw_rad=yaw,
            base_color=(0.36, 0.31, 0.25),
            pitch_forward=0.0,
        )
    # Светлая центральная полоса для читаемости.
    _draw_oriented_box(
        cx=x, cy=y + 0.028, cz=z,
        hx=half_len * 0.98, hy=0.005, hz=half_w * 0.14,
        yaw_rad=yaw,
        base_color=(0.72, 0.68, 0.56),
        pitch_forward=0.0,
    )


def _draw_rock(inst: StaticMeshInstance):
    """
    Камень: несколько наложенных "глыб".
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    hx = max(0.2, abs(inst.scale.x) * 0.45)
    hy = max(0.2, abs(inst.scale.y) * 0.40)
    hz = max(0.2, abs(inst.scale.z) * 0.45)

    _draw_oriented_box(
        cx=x,
        cy=y + hy * 0.5,
        cz=z,
        hx=hx,
        hy=hy,
        hz=hz,
        yaw_rad=yaw,
        base_color=(0.42, 0.44, 0.47),
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=x + 0.10 * hx,
        cy=y + hy * 1.1,
        cz=z - 0.08 * hz,
        hx=hx * 0.62,
        hy=hy * 0.58,
        hz=hz * 0.62,
        yaw_rad=yaw + 0.7,
        base_color=(0.50, 0.52, 0.56),
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=x - 0.12 * hx,
        cy=y + hy * 0.92,
        cz=z + 0.08 * hz,
        hx=hx * 0.42,
        hy=hy * 0.14,
        hz=hz * 0.34,
        yaw_rad=yaw + 0.25,
        base_color=(0.26, 0.34, 0.19),
        pitch_forward=0.0,
    )


def _draw_log(inst: StaticMeshInstance):
    """
    Бревно.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    hx = max(0.4, abs(inst.scale.x) * 0.5)
    hy = max(0.12, abs(inst.scale.y) * 0.45)
    hz = max(0.16, abs(inst.scale.z) * 0.38)

    _draw_oriented_box(
        cx=x,
        cy=y + hy + 0.01,
        cz=z,
        hx=hx,
        hy=hy,
        hz=hz,
        yaw_rad=yaw,
        base_color=(0.42, 0.26, 0.15),
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=x,
        cy=y + hy + 0.04,
        cz=z,
        hx=hx * 0.96,
        hy=hy * 0.18,
        hz=hz * 0.62,
        yaw_rad=yaw,
        base_color=(0.56, 0.37, 0.21),
        pitch_forward=0.0,
    )


def _draw_tower(inst: StaticMeshInstance):
    """
    Смотровая башня.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    base_hx = max(0.6, abs(inst.scale.x) * 0.38)
    base_hz = max(0.6, abs(inst.scale.z) * 0.38)
    tower_h = max(2.5, abs(inst.scale.y))

    post_col = (0.45, 0.35, 0.24)
    deck_col = (0.30, 0.24, 0.18)
    for lx in (-base_hx * 0.78, base_hx * 0.78):
        for lz in (-base_hz * 0.78, base_hz * 0.78):
            wx, wz = _yaw_world_offset(x, z, yaw, lx, lz)
            _draw_oriented_box(
                cx=wx, cy=y + tower_h * 0.45, cz=wz,
                hx=max(0.08, base_hx * 0.12), hy=tower_h * 0.45, hz=max(0.08, base_hz * 0.12),
                yaw_rad=yaw,
                base_color=post_col,
                pitch_forward=0.0,
            )
    _draw_oriented_box(
        cx=x, cy=y + tower_h * 0.90, cz=z,
        hx=base_hx * 1.15, hy=tower_h * 0.08, hz=base_hz * 1.15,
        yaw_rad=yaw,
        base_color=deck_col,
        pitch_forward=0.0,
    )
    _draw_shingled_roof(
        cx=x, cy=y + tower_h * 0.98, cz=z,
        sx=base_hx * 1.25, sy=tower_h * 0.30, sz=base_hz * 1.25,
        yaw_rad=yaw,
        base_color=(0.28, 0.14, 0.10),
        top_color=(0.56, 0.24, 0.17),
        tiers=4,
    )


def _draw_well(inst: StaticMeshInstance):
    """
    Колодец: каменное основание + вода + крыша.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    base_r = max(0.45, abs(inst.scale.x) * 0.42)
    h = max(0.8, abs(inst.scale.y))

    _draw_stone_courses(
        cx=x, cy=y + h * 0.28, cz=z,
        hx=base_r, hy=h * 0.28, hz=base_r,
        yaw_rad=yaw,
        base_color=(0.55, 0.56, 0.60),
        courses=4,
    )
    _draw_water_surface(
        x=x,
        y=y + h * 0.40,
        z=z,
        sx=base_r * 0.76,
        sz=base_r * 0.76,
        time_seed=x * 0.11 + z * 0.09,
    )
    left_x, left_z = _yaw_world_offset(x, z, yaw, -base_r * 0.62, 0.0)
    right_x, right_z = _yaw_world_offset(x, z, yaw, base_r * 0.62, 0.0)
    _draw_oriented_box(
        cx=left_x, cy=y + h * 0.80, cz=left_z,
        hx=0.08, hy=h * 0.35, hz=0.08,
        yaw_rad=yaw,
        base_color=(0.42, 0.30, 0.20),
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=right_x, cy=y + h * 0.80, cz=right_z,
        hx=0.08, hy=h * 0.35, hz=0.08,
        yaw_rad=yaw,
        base_color=(0.42, 0.30, 0.20),
        pitch_forward=0.0,
    )
    _draw_shingled_roof(
        cx=x, cy=y + h * 1.28, cz=z,
        sx=base_r * 1.25, sy=h * 0.42, sz=base_r * 1.25,
        yaw_rad=yaw,
        base_color=(0.33, 0.16, 0.11),
        top_color=(0.62, 0.27, 0.17),
        tiers=4,
    )


def _draw_shrine(inst: StaticMeshInstance, global_time: float):
    """
    Святилище: постамент + крыша + мягкое свечение.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    sx = max(0.8, abs(inst.scale.x) * 0.52)
    sy = max(0.8, abs(inst.scale.y))
    sz = max(0.8, abs(inst.scale.z) * 0.52)

    _draw_stone_courses(
        cx=x, cy=y + sy * 0.22, cz=z,
        hx=sx, hy=sy * 0.22, hz=sz,
        yaw_rad=yaw,
        base_color=(0.56, 0.55, 0.52),
        courses=3,
    )
    _draw_oriented_box(
        cx=x, cy=y + sy * 0.55, cz=z,
        hx=sx * 0.52, hy=sy * 0.25, hz=sz * 0.52,
        yaw_rad=yaw,
        base_color=(0.78, 0.76, 0.72),
        pitch_forward=0.0,
    )
    _draw_shingled_roof(
        cx=x, cy=y + sy * 0.82, cz=z,
        sx=sx * 1.08, sy=sy * 0.52, sz=sz * 1.08,
        yaw_rad=yaw,
        base_color=(0.22, 0.26, 0.35),
        top_color=(0.36, 0.58, 0.92),
        tiers=4,
    )
    pulse = 0.5 + 0.5 * math.sin(global_time * 2.5 + x * 0.5 + z * 0.5)
    _draw_ring(x, z, radius=max(sx, sz) * (0.9 + pulse * 0.35), y=y + 0.03,
               rgb=(0.35, 0.75, 1.0), width=1.6, alpha=0.26 + pulse * 0.24)


def _draw_lantern(inst: StaticMeshInstance, global_time: float):
    """
    Фонарь: стойка + лампа + световое кольцо.
    """
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    pole_h = max(1.2, abs(inst.scale.y))
    pole_r = max(0.05, abs(inst.scale.x) * 0.12)

    _draw_oriented_box(
        cx=x, cy=y + pole_h * 0.45, cz=z,
        hx=pole_r, hy=pole_h * 0.45, hz=pole_r,
        yaw_rad=yaw,
        base_color=(0.30, 0.22, 0.15),
        pitch_forward=0.0,
    )

    lamp_y = y + pole_h * 0.92
    _draw_oriented_box(
        cx=x, cy=lamp_y, cz=z,
        hx=pole_r * 2.2, hy=pole_r * 2.2, hz=pole_r * 2.2,
        yaw_rad=yaw,
        base_color=(0.92, 0.78, 0.46),
        pitch_forward=0.0,
    )
    glow = 0.5 + 0.5 * math.sin(global_time * 5.5 + x + z)
    _draw_ring(x, z, radius=0.75 + glow * 0.30, y=y + 0.035,
               rgb=(1.0, 0.85, 0.55), width=1.2, alpha=0.14 + 0.18 * glow)


def _draw_wall(inst: StaticMeshInstance):
    x, y, z = inst.pos.x, inst.pos.y, inst.pos.z
    yaw = inst.yaw
    hx = max(0.6, abs(inst.scale.x))
    hy = max(0.3, abs(inst.scale.y) * 0.45)
    hz = max(0.16, abs(inst.scale.z) * 0.35)
    _draw_stone_courses(
        cx=x,
        cy=y + hy,
        cz=z,
        hx=hx,
        hy=hy,
        hz=hz,
        yaw_rad=yaw,
        base_color=(0.46, 0.47, 0.50),
        courses=4,
    )
    for lx in (-hx * 0.66, 0.0, hx * 0.66):
        wx, wz = _yaw_world_offset(x, z, yaw, lx, 0.0)
        _draw_oriented_box(
            cx=wx,
            cy=y + hy * 2.02,
            cz=wz,
            hx=max(0.10, hx * 0.14),
            hy=max(0.07, hy * 0.20),
            hz=hz * 0.90,
            yaw_rad=yaw,
            base_color=(0.56, 0.57, 0.60),
            pitch_forward=0.0,
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
    elif inst.kind == "road":
        _draw_road(inst)
    elif inst.kind == "rock":
        _draw_rock(inst)
    elif inst.kind == "log":
        _draw_log(inst)
    elif inst.kind == "tower":
        _draw_tower(inst)
    elif inst.kind == "well":
        _draw_well(inst)
    elif inst.kind == "shrine":
        _draw_shrine(inst, global_time)
    elif inst.kind == "lantern":
        _draw_lantern(inst, global_time)
    elif inst.kind == "wall":
        _draw_wall(inst)
    elif inst.kind == "zone_safe":
        rr = 0.5 * (abs(inst.scale.x) + abs(inst.scale.z))
        _draw_disc_zone(inst.pos.x, inst.pos.z, radius=max(0.5, rr), kind="safe", y=0.021)
    elif inst.kind == "zone_hazard":
        rr = 0.5 * (abs(inst.scale.x) + abs(inst.scale.z))
        _draw_disc_zone(inst.pos.x, inst.pos.z, radius=max(0.5, rr), kind="hazard", y=0.021)


# --- отрисовка агента ------------------------------------------------

def draw_agent_humanoid(agent: AgentEntity, t: float):
    yaw = float(agent.transform.yaw)
    fear = clamp(float(agent.anim.fear), 0.0, 1.0)
    hp = clamp(float(agent.anim.health), 0.0, 100.0)
    alive = bool(agent.anim.alive)
    phase = float(agent.anim.walk_phase)

    px = float(agent.transform.pos.x)
    pz = float(agent.transform.pos.z)

    hp_ratio = hp / 100.0
    life_t = 1.0 if alive else 0.0

    # Чем хуже состояние, тем более согнутая и «тяжёлая» поза.
    crouch = (1.0 - hp_ratio) * 0.20 + clamp((fear - 0.55) * 0.22, 0.0, 0.12)
    fwd_lean = clamp((fear - 0.45) * 0.50, 0.0, 0.30) + clamp((0.65 - hp_ratio) * 0.28, 0.0, 0.20)
    if not alive:
        crouch = 0.40
        fwd_lean = 0.72

    base_y_offset = -0.10 if not alive else 0.0
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    def _world_pos(lx: float, ly: float, lz: float) -> Tuple[float, float, float]:
        wx = px + lx * cos_y - lz * sin_y
        wz = pz + lx * sin_y + lz * cos_y
        wy = base_y_offset + ly
        return wx, wy, wz

    def _part_box(
        lx: float, ly: float, lz: float,
        hx: float, hy: float, hz: float,
        color: Tuple[float, float, float],
        pitch: float = 0.0,
    ):
        wx, wy, wz = _world_pos(lx, ly, lz)
        _draw_oriented_box(
            cx=wx, cy=wy, cz=wz,
            hx=hx, hy=hy, hz=hz,
            yaw_rad=yaw,
            base_color=color,
            pitch_forward=pitch,
        )

    def _joint(
        lx: float, ly: float, lz: float,
        r: float,
        color: Tuple[float, float, float],
        lat: int = 5,
        lon: int = 8,
    ):
        wx, wy, wz = _world_pos(lx, ly, lz)
        _draw_sphere_lowpoly(wx, wy, wz, radius=max(0.03, r), base_color=color, lat_steps=lat, lon_steps=lon)

    body_core = _color_from_state(fear, alive)
    cloth_main = _mix_rgb(body_core, (0.10, 0.18, 0.36), 0.45)
    cloth_dark = _mix_rgb(cloth_main, (0.08, 0.09, 0.12), 0.55)
    skin = _mix_rgb((0.96, 0.85, 0.74), (0.78, 0.63, 0.54), fear * 0.55)
    if not alive:
        skin = (0.60, 0.61, 0.64)

    # -------------------------------------------------------------
    # Туловище: таз + грудь + шея + голова (округлая)
    # -------------------------------------------------------------
    pelvis_y = 0.90 - crouch
    chest_y = 1.46 - crouch * 0.70
    neck_y = 1.98 - crouch * 0.55
    head_y = 2.27 - crouch * 0.50

    _part_box(0.0, pelvis_y, 0.01, 0.30, 0.20, 0.20, cloth_dark, pitch=fwd_lean * 0.35)
    _part_box(0.0, chest_y, 0.0, 0.34, 0.36, 0.21, cloth_main, pitch=fwd_lean)
    _part_box(0.0, neck_y, 0.0, 0.09, 0.10, 0.09, skin, pitch=fwd_lean * 0.5)

    # Плечевые шарниры добавляют "человечность" силуэта.
    _joint(-0.36, chest_y + 0.18, 0.0, 0.09, cloth_main, lat=5, lon=8)
    _joint(+0.36, chest_y + 0.18, 0.0, 0.09, cloth_main, lat=5, lon=8)

    head_shift = math.sin(fwd_lean) * 0.16
    head_wx, head_wy, head_wz = _world_pos(head_shift, head_y, 0.03)
    _draw_sphere_lowpoly(
        head_wx, head_wy, head_wz,
        radius=0.24,
        base_color=skin,
        lat_steps=8,
        lon_steps=12,
    )
    # Волосы/шапка (верхняя полусфера потемнее).
    _draw_sphere_lowpoly(
        head_wx, head_wy + 0.07, head_wz - 0.01,
        radius=0.16,
        base_color=(0.18, 0.16, 0.14) if alive else (0.42, 0.42, 0.45),
        lat_steps=6,
        lon_steps=10,
    )

    # -------------------------------------------------------------
    # Анимация шага: руки и ноги
    # -------------------------------------------------------------
    walk_amp = (0.25 + 0.12 * life_t) * (0.40 + 0.60 * hp_ratio)
    leg_swing = math.sin(phase) * walk_amp
    arm_swing = math.sin(phase + math.pi) * (walk_amp * 0.95)
    knee_lift = abs(math.sin(phase)) * 0.07
    elbow_fold = abs(math.sin(phase + math.pi * 0.5)) * 0.06

    # Ноги
    hip_x = 0.19
    thigh_hy = 0.24
    shin_hy = 0.22
    boot_hy = 0.07

    left_thigh_z = leg_swing * 0.55
    right_thigh_z = -leg_swing * 0.55

    _part_box(-hip_x, 0.62 - crouch, left_thigh_z, 0.12, thigh_hy, 0.12, cloth_dark, pitch=+leg_swing * 1.25)
    _part_box(+hip_x, 0.62 - crouch, right_thigh_z, 0.12, thigh_hy, 0.12, cloth_dark, pitch=-leg_swing * 1.25)

    _part_box(-hip_x, 0.23 - crouch + knee_lift, left_thigh_z * 1.18, 0.10, shin_hy, 0.10, cloth_main, pitch=-leg_swing * 0.60)
    _part_box(+hip_x, 0.23 - crouch + (0.07 - knee_lift), right_thigh_z * 1.18, 0.10, shin_hy, 0.10, cloth_main, pitch=+leg_swing * 0.60)

    _joint(-hip_x, 0.37 - crouch + knee_lift * 0.5, left_thigh_z * 0.9, 0.06, cloth_dark, lat=4, lon=7)
    _joint(+hip_x, 0.37 - crouch + (0.035 - knee_lift * 0.5), right_thigh_z * 0.9, 0.06, cloth_dark, lat=4, lon=7)

    _part_box(-hip_x, 0.05 - crouch, left_thigh_z * 1.30, 0.13, boot_hy, 0.20, (0.12, 0.12, 0.14), pitch=0.0)
    _part_box(+hip_x, 0.05 - crouch, right_thigh_z * 1.30, 0.13, boot_hy, 0.20, (0.12, 0.12, 0.14), pitch=0.0)

    # Руки
    shoulder_y = chest_y + 0.08
    upper_arm_h = 0.21
    fore_arm_h = 0.19
    arm_x = 0.46

    _part_box(-arm_x, shoulder_y, arm_swing * 0.45, 0.09, upper_arm_h, 0.10, cloth_dark, pitch=-arm_swing * 1.15)
    _part_box(+arm_x, shoulder_y, -arm_swing * 0.45, 0.09, upper_arm_h, 0.10, cloth_dark, pitch=+arm_swing * 1.15)

    _part_box(-arm_x, shoulder_y - 0.29, arm_swing * 0.88 - elbow_fold, 0.08, fore_arm_h, 0.09, cloth_main, pitch=-arm_swing * 0.70)
    _part_box(+arm_x, shoulder_y - 0.29, -arm_swing * 0.88 + elbow_fold, 0.08, fore_arm_h, 0.09, cloth_main, pitch=+arm_swing * 0.70)

    _joint(-arm_x, shoulder_y - 0.52, arm_swing * 1.02 - elbow_fold, 0.06, skin, lat=4, lon=7)
    _joint(+arm_x, shoulder_y - 0.52, -arm_swing * 1.02 + elbow_fold, 0.06, skin, lat=4, lon=7)

    # Лёгкая "динамика дыхания" для живого агента.
    if alive:
        breath = 0.01 * math.sin(t * 2.4 + phase * 0.5)
        _part_box(0.0, chest_y + breath, -0.06, 0.20, 0.03, 0.05, _mix_rgb(cloth_main, (0.85, 0.85, 0.88), 0.25), pitch=fwd_lean * 0.6)

    # КОЛЬЦО ВЫБОРА (с пульсом)
    if agent.selected:
        base_r = 1.10
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


def draw_agent_impostor(agent: AgentEntity, t: float):
    """
    Дальний LOD для агента: дешёвый силуэт вместо полной гуманоидной сборки.
    """
    yaw = float(agent.transform.yaw)
    fear = clamp(float(agent.anim.fear), 0.0, 1.0)
    alive = bool(agent.anim.alive)
    px = float(agent.transform.pos.x)
    pz = float(agent.transform.pos.z)

    body_core = _color_from_state(fear, alive)
    cloth = _mix_rgb(body_core, (0.10, 0.15, 0.23), 0.42)
    skin = _mix_rgb((0.95, 0.85, 0.74), (0.78, 0.63, 0.54), fear * 0.45)
    if not alive:
        skin = (0.60, 0.61, 0.64)
    lean = clamp((fear - 0.45) * 0.22, 0.0, 0.10)

    _draw_oriented_box(
        cx=px,
        cy=0.95 if alive else 0.72,
        cz=pz,
        hx=0.24,
        hy=0.48 if alive else 0.28,
        hz=0.18,
        yaw_rad=yaw,
        base_color=cloth,
        pitch_forward=lean,
    )
    _draw_sphere_lowpoly(
        px + math.cos(yaw) * 0.05,
        1.62 if alive else 0.96,
        pz + math.sin(yaw) * 0.05,
        radius=0.16,
        base_color=skin,
        lat_steps=5,
        lon_steps=8,
    )

    if agent.selected:
        pulse = (math.sin(t * 4.0) * 0.5 + 0.5) if SELECTED_PULSE else 0.0
        _draw_ring(
            px,
            pz,
            radius=1.0 + pulse * 0.18,
            y=0.05,
            rgb=(0.2, 0.8, 1.0),
            width=1.8,
            alpha=0.55 + 0.25 * pulse,
        )


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


def draw_animal_impostor(an: AnimalEntity, t: float):
    """
    Дальний LOD для зверя: дешёвое тело и голова без тяжёлых колец/эффектов.
    """
    yaw = float(an.transform.yaw)
    alive = bool(an.anim.alive)
    px = float(an.transform.pos.x)
    pz = float(an.transform.pos.z)

    body_col = _animal_body_color(an.anim.temperament, an.anim.tamed, alive)
    _draw_oriented_box(
        cx=px,
        cy=0.48,
        cz=pz,
        hx=0.28,
        hy=0.16,
        hz=0.48,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )
    _draw_oriented_box(
        cx=px + math.cos(yaw) * 0.46,
        cy=0.56,
        cz=pz + math.sin(yaw) * 0.46,
        hx=0.12,
        hy=0.10,
        hz=0.12,
        yaw_rad=yaw,
        base_color=body_col,
        pitch_forward=0.0,
    )

    if an.selected:
        pulse = (math.sin(t * 4.0) * 0.5 + 0.5) if SELECTED_PULSE else 0.0
        _draw_ring(
            px,
            pz,
            radius=0.9 + pulse * 0.16,
            y=0.05,
            rgb=(0.2, 0.8, 1.0),
            width=1.8,
            alpha=0.50 + 0.22 * pulse,
        )


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
        # Стартовая инициализация дневного света для первого кадра.
        self._update_sun_state(dt=0.0)

        # камера (для HUD LOD)
        self._cam_pos: Tuple[float, float, float] = (0.0, 20.0, -20.0)
        self._cam_look: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.hidden_agent_ids: set[str] = set()

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

    def _dist2_xz(self, x: float, z: float) -> float:
        cx, _cy, cz = self._cam_pos
        dx = x - cx
        dz = z - cz
        return dx * dx + dz * dz

    def _within_render_lod(self, x: float, z: float, max_distance: float, force: bool = False) -> bool:
        if force or max_distance <= 0.0:
            return True
        return self._dist2_xz(x, z) <= (max_distance * max_distance)

    def _static_mesh_radius(self, inst: StaticMeshInstance) -> float:
        scale_r = max(abs(inst.scale.x), abs(inst.scale.z), abs(inst.scale.y))
        kind_scale = {
            "tree": 1.2,
            "house": 1.8,
            "road": 2.6,
            "wall": 2.8,
            "lake": 2.4,
            "tower": 2.0,
            "shrine": 1.4,
        }.get(inst.kind, 1.0)
        return max(1.0, scale_r * kind_scale)

    def _collect_render_lists(self):
        visible_static: List[StaticMeshInstance] = []
        for inst in self.static_meshes:
            lod_distance = MAX_STATIC_DISTANCE + self._static_mesh_radius(inst)
            if self._within_render_lod(inst.pos.x, inst.pos.z, lod_distance):
                visible_static.append(inst)

        visible_zones: List[ZoneObject] = []
        for zone in self.world.zones:
            if self._within_render_lod(zone.x, zone.z, MAX_ZONE_DISTANCE + zone.radius):
                visible_zones.append(zone)

        visible_agents: List[AgentEntity] = []
        detailed_agent_ids: set[str] = set()
        for agent in self.agents.values():
            if agent.agent_id in self.hidden_agent_ids:
                continue
            force = agent.selected
            px = agent.transform.pos.x
            pz = agent.transform.pos.z
            if not self._within_render_lod(px, pz, MAX_ENTITY_RENDER_DISTANCE, force=force):
                continue
            visible_agents.append(agent)
            if self._within_render_lod(px, pz, MAX_AGENT_DETAIL_DISTANCE, force=force):
                detailed_agent_ids.add(agent.agent_id)

        visible_animals: List[AnimalEntity] = []
        detailed_animal_ids: set[str] = set()
        for animal in self.animals.values():
            force = animal.selected
            px = animal.transform.pos.x
            pz = animal.transform.pos.z
            if not self._within_render_lod(px, pz, MAX_ENTITY_RENDER_DISTANCE, force=force):
                continue
            visible_animals.append(animal)
            if self._within_render_lod(px, pz, MAX_ANIMAL_DETAIL_DISTANCE, force=force):
                detailed_animal_ids.add(animal.animal_id)

        return visible_static, visible_zones, visible_agents, detailed_agent_ids, visible_animals, detailed_animal_ids

    def _cell_key(self, x: float, z: float, cell_size: float) -> Tuple[int, int]:
        return int(math.floor(x / cell_size)), int(math.floor(z / cell_size))

    def _iter_spatial_pairs(self, entities: List[Any], cell_size: float):
        buckets: Dict[Tuple[int, int], List[Any]] = {}
        for ent in entities:
            if not ent.anim.alive:
                continue
            key = self._cell_key(ent.transform.pos.x, ent.transform.pos.z, cell_size)
            buckets.setdefault(key, []).append(ent)

        neighbor_offsets = ((1, 0), (0, 1), (1, 1), (-1, 1))
        for key, bucket in buckets.items():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    yield bucket[i], bucket[j]
            for dx, dz in neighbor_offsets:
                other_bucket = buckets.get((key[0] + dx, key[1] + dz))
                if not other_bucket:
                    continue
                for a in bucket:
                    for b in other_bucket:
                        yield a, b

    def _iter_cross_spatial_pairs(
        self,
        agents: List[AgentEntity],
        animals: List[AnimalEntity],
        cell_size: float,
    ):
        animal_buckets: Dict[Tuple[int, int], List[AnimalEntity]] = {}
        for an in animals:
            if not an.anim.alive:
                continue
            key = self._cell_key(an.transform.pos.x, an.transform.pos.z, cell_size)
            animal_buckets.setdefault(key, []).append(an)

        for ag in agents:
            if not ag.anim.alive:
                continue
            base_key = self._cell_key(ag.transform.pos.x, ag.transform.pos.z, cell_size)
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for an in animal_buckets.get((base_key[0] + dx, base_key[1] + dz), []):
                        yield ag, an

    def _apply_pair_separation(self, a: Any, b: Any):
        if not a.anim.alive or not b.anim.alive:
            return
        dx = a.transform.pos.x - b.transform.pos.x
        dz = a.transform.pos.z - b.transform.pos.z
        dist2 = dx * dx + dz * dz
        if dist2 < 1e-9:
            return
        min_dist = a.body.radius + b.body.radius
        if dist2 >= (min_dist * min_dist):
            return

        dist = math.sqrt(dist2)
        if dist <= 1e-6:
            return

        push = (min_dist - dist) * 0.5 * SEPARATION_PUSH
        nx = dx / dist
        nz = dz / dist
        a.transform.pos.x = clamp(a.transform.pos.x + nx * push, 0.0, self.world.width)
        a.transform.pos.z = clamp(a.transform.pos.z + nz * push, 0.0, self.world.height)
        b.transform.pos.x = clamp(b.transform.pos.x - nx * push, 0.0, self.world.width)
        b.transform.pos.z = clamp(b.transform.pos.z - nz * push, 0.0, self.world.height)

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
            facing_x, facing_z = _xy_from_any(a.get("facing", {"x": 0.0, "y": 0.0}))
            facing_len = math.hypot(facing_x, facing_z)
            if speed > 1e-6:
                desired_dir = Vec3(vel_x / speed, 0.0, vel_z / speed)
            elif facing_len > 1e-6:
                desired_dir = Vec3(facing_x / facing_len, 0.0, facing_z / facing_len)
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
        entities = list(self.agents.values())
        for a, b in self._iter_spatial_pairs(entities, SOCIAL_AVOIDANCE_CELL):
            self._apply_pair_separation(a, b)

    def _apply_social_avoidance_animals(self):
        """
        То же самое для зверей.
        """
        entities = list(self.animals.values())
        for a, b in self._iter_spatial_pairs(entities, SOCIAL_AVOIDANCE_CELL):
            self._apply_pair_separation(a, b)

    def _apply_social_avoidance_cross(self):
        """
        Разведение "агент ↔ зверь".
        Нужно, чтобы волк визуально не залезал в туловище человека.
        """
        agents = list(self.agents.values())
        animals = list(self.animals.values())
        for ag, an in self._iter_cross_spatial_pairs(agents, animals, SOCIAL_AVOIDANCE_CELL):
            self._apply_pair_separation(ag, an)

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

    def _update_sun_state(self, dt: float):
        """
        Обновляем дневной цикл:
        - направление солнца
        - оттенок света
        - интенсивность ambient/diffuse
        - цвет неба.
        """
        if not SUN_ENABLED:
            return

        # Параметры читаются рендер-хелперами напрямую.
        global _SUN_DIR, _SUN_COLOR, _SUN_AMBIENT, _SUN_DIFFUSE, _SKY_COLOR

        cycle_sec = max(30.0, float(SUN_CYCLE_SEC))
        day_t = (self._time_accum / cycle_sec) % 1.0
        az = day_t * (2.0 * math.pi)

        # Высота солнца в дневном диапазоне (без ночи).
        elev_span = SUN_MAX_ELEV_DEG - SUN_MIN_ELEV_DEG
        elev_deg = SUN_MIN_ELEV_DEG + elev_span * (0.5 + 0.5 * math.sin(az))
        elev = math.radians(elev_deg)

        sy = max(0.18, math.sin(elev))
        flat = max(0.01, math.cos(elev))
        sx = math.cos(az) * flat
        sz = math.sin(az) * flat
        nlen = math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-9
        _SUN_DIR = (sx / nlen, sy / nlen, sz / nlen)

        # Ниже солнце -> теплее оттенок.
        warm_t = clamp(1.0 - sy, 0.0, 1.0)
        _SUN_COLOR = (
            1.0,
            clamp(0.98 - 0.12 * warm_t, 0.0, 1.0),
            clamp(0.94 - 0.28 * warm_t, 0.0, 1.0),
        )

        _SUN_AMBIENT = clamp(0.28 + 0.16 * sy, 0.20, 0.52)
        _SUN_DIFFUSE = clamp(0.58 + 0.34 * sy, 0.45, 0.96)

        # Небо: больше синего в верхней фазе солнца и мягкая теплота в нижней.
        sky_mix = clamp((sy - 0.18) / 0.82, 0.0, 1.0)
        _SKY_COLOR = _mix_rgb((0.98, 0.76, 0.58), (0.50, 0.72, 0.97), sky_mix)

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
        if agent.agent_id in self.hidden_agent_ids:
            return
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
          2) обновляем дневной цикл солнца
          3) плавно тянем позы к серверным координатам (dead-reckoning + lerp)
          4) разводим сущности визуально (антислипание)
          5) плавно поворачиваем модели и обновляем walk_phase
          6) обновляем VFX
        """
        if dt <= 0.0:
            return

        self._time_accum += dt
        self._update_sun_state(dt)

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

        glClearColor(_SKY_COLOR[0], _SKY_COLOR[1], _SKY_COLOR[2], 1.0)
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
            if not self._within_render_lod(ring.x, ring.z, MAX_VFX_DISTANCE):
                continue
            a = ring.alpha()
            r = ring.radius()
            col = (ring.color[0], ring.color[1], ring.color[2])
            _draw_ring(ring.x, ring.z, radius=r, y=ring.y, rgb=col, width=2.0, alpha=a)

    def _draw_damage_numbers(self):
        if not (self._hud_text_enabled and SHOW_DAMAGE_NUMBERS):
            return
        for dn in self.numbers:
            if not self._within_render_lod(dn.x, dn.z, MAX_VFX_DISTANCE):
                continue
            alpha = clamp(dn.alpha(), 0.0, 1.0)
            val = dn.value
            sign = "+" if val > 0 else ""
            text = f"{sign}{val:.0f}"
            self._draw_text3d(text, dn.x, dn.y(), dn.z, big=True, alpha=alpha, rgb=(dn.color[0], dn.color[1], dn.color[2]))

    def _draw_sun(self):
        """
        Рисуем солнце в небе (сфера + мягкая корона + короткие лучи).
        """
        if not SUN_ENABLED:
            return

        world_w = max(1.0, float(self.world.width))
        world_h = max(1.0, float(self.world.height))
        center_x = world_w * 0.5
        center_z = world_h * 0.5

        dome_r = max(world_w, world_h) * 0.92 + 40.0
        sx, sy, sz = _SUN_DIR
        sun_x = center_x + sx * dome_r
        sun_y = SUN_BASE_HEIGHT + sy * (dome_r * 0.75)
        sun_z = center_z + sz * dome_r

        sun_radius = max(2.8, max(world_w, world_h) * 0.040)
        sun_core = _mix_rgb(_SUN_COLOR, (1.0, 0.98, 0.86), 0.45)
        _draw_sphere_lowpoly(
            sun_x, sun_y, sun_z,
            radius=sun_radius,
            base_color=sun_core,
            lat_steps=8,
            lon_steps=12,
        )

        # Полупрозрачный ореол.
        halo_r = sun_radius * 2.6
        glColor4f(_SUN_COLOR[0], _SUN_COLOR[1], _SUN_COLOR[2], 0.16)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(sun_x, sun_y, sun_z)
        steps = 36
        for i in range(steps + 1):
            a = (2.0 * math.pi) * (i / steps)
            glVertex3f(
                sun_x + math.cos(a) * halo_r,
                sun_y,
                sun_z + math.sin(a) * halo_r,
            )
        glEnd()

        # Короткие лучи для читаемого силуэта.
        glColor4f(_SUN_COLOR[0], _SUN_COLOR[1], _SUN_COLOR[2], 0.34)
        glLineWidth(1.4)
        glBegin(GL_LINES)
        rays = 12
        for i in range(rays):
            a = (2.0 * math.pi) * (i / rays)
            in_r = sun_radius * 1.35
            out_r = sun_radius * 2.15
            glVertex3f(sun_x + math.cos(a) * in_r, sun_y, sun_z + math.sin(a) * in_r)
            glVertex3f(sun_x + math.cos(a) * out_r, sun_y, sun_z + math.sin(a) * out_r)
        glEnd()

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

    def _draw_target_lines(
        self,
        visible_agents: Optional[List[AgentEntity]] = None,
        visible_animals: Optional[List[AnimalEntity]] = None,
    ):
        if not SHOW_TARGET_LINES:
            return

        animals = visible_animals if visible_animals is not None else list(self.animals.values())
        agents = visible_agents if visible_agents is not None else list(self.agents.values())

        def _pos_of(entity_id: str) -> Optional[Tuple[float, float]]:
            if entity_id in self.agents:
                p = self.agents[entity_id].transform.pos
                return p.x, p.z
            if entity_id in self.animals:
                p = self.animals[entity_id].transform.pos
                return p.x, p.z
            return None

        def _line_is_relevant(sx: float, sz: float, tx: float, tz: float, force: bool = False) -> bool:
            if self._within_render_lod(sx, sz, MAX_TARGET_LINE_DISTANCE, force=force):
                return True
            if self._within_render_lod(tx, tz, MAX_TARGET_LINE_DISTANCE, force=force):
                return True
            return self._within_render_lod((sx + tx) * 0.5, (sz + tz) * 0.5, MAX_TARGET_LINE_DISTANCE, force=force)

        glLineWidth(1.5)
        for an in animals:
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
                if not _line_is_relevant(sx, sz, tx, tz, force=an.selected):
                    continue
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
        for ag in agents:
            if ag.agent_id in self.hidden_agent_ids:
                continue
            tgt = ag.public_state.get("target_id") or ag.public_state.get("target")
            if isinstance(tgt, str):
                tp = _pos_of(tgt)
                if tp is None:
                    continue
                sx, sz = ag.transform.pos.x, ag.transform.pos.z
                tx, tz = tp
                if not _line_is_relevant(sx, sz, tx, tz, force=ag.selected):
                    continue
                glColor4f(1.0, 1.0, 0.2, 0.75)
                glBegin(GL_LINES)
                glVertex3f(sx, 1.8, sz)
                glVertex3f(tx, 1.8, tz)
                glEnd()

    def render_opengl(self):
        """
        Рисуем мир:
          - солнце (ядро + ореол + лучи)
          - пол + сетка
          - статика окружения (лес, дома, костёр, озеро)
          - зоны (safe / hazard) цветными дисками
          - цели агентов жёлтым кольцом
          - вспомогательные гизмы (FOV, линии-цели)
          - сами агенты + HUD
          - звери + HUD
          - VFX (вспышки) + числа урона/хила
        """
        self._draw_sun()

        # пол/сетка
        _draw_floor_grid(self.world.width, self.world.height, cam_pos=self._cam_pos)

        (
            visible_static,
            visible_zones,
            visible_agents,
            detailed_agent_ids,
            visible_animals,
            detailed_animal_ids,
        ) = self._collect_render_lists()

        # окружение
        for inst in visible_static:
            _draw_static_mesh(inst, self._time_accum)

        # зоны с сервера
        for zone in visible_zones:
            _draw_disc_zone(zone.x, zone.z, zone.radius, zone.kind, y=0.02)

        # цели агентов (кольца)
        for agent in visible_agents:
            if not self._within_render_lod(agent.goal.x, agent.goal.z, MAX_GOAL_RING_DISTANCE, force=agent.selected):
                continue
            _draw_ring(
                agent.goal.x,
                agent.goal.z,
                radius=0.8,
                y=0.03,
                rgb=(1.0, 1.0, 0.2),
                width=1.0,
            )

        # вспомогательные гизмы
        self._draw_target_lines(visible_agents=visible_agents, visible_animals=visible_animals)
        for agent in visible_agents:
            self._draw_agent_fov(agent)

        # агенты
        for agent in visible_agents:
            if agent.agent_id in detailed_agent_ids:
                draw_agent_humanoid(agent, self._time_accum)
            else:
                draw_agent_impostor(agent, self._time_accum)
            if agent.agent_id in detailed_agent_ids and self._within_render_lod(
                agent.transform.pos.x,
                agent.transform.pos.z,
                MAX_DIRECTION_ARROW_DISTANCE,
                force=agent.selected,
            ):
                draw_agent_direction_arrow(agent)
            self._draw_agent_hud(agent)

        # звери
        for animal in visible_animals:
            if animal.animal_id in detailed_animal_ids:
                draw_animal_quadruped(animal, self._time_accum)
            else:
                draw_animal_impostor(animal, self._time_accum)
            if animal.animal_id in detailed_animal_ids and self._within_render_lod(
                animal.transform.pos.x,
                animal.transform.pos.z,
                MAX_DIRECTION_ARROW_DISTANCE,
                force=animal.selected,
            ):
                draw_animal_direction_arrow(animal)
            self._draw_animal_hud(animal)

        # эффекты
        self._draw_vfx()
        self._draw_damage_numbers()
