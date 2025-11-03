# animals.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import math
import random
import config

ANIMAL_BODY_RADIUS = 0.8
ANIMAL_MAX_SPEED   = 0.9  # базовая скорость (животные чуть медленнее агента по умолчанию)

def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

@dataclass
class AnimalSpecies:
    species_id: str            # "wolf", "fox", "deer", ...
    name: str                  # "Волк", "Лис", "Олень"
    base_hp: float
    aggressive: bool           # True = хищник (кусает сам), False = не нападает первым
    tamable: bool              # можно ли приручить
    tame_difficulty: float     # 0..1 (ниже = легче подружить)
    bite_damage: float         # урон за один укус в упор
    fear_radius: float         # радиус психологического давления/страха на людей
    follow_distance: float     # дистанция сопровождения хозяина, если приручён
    aggro_radius: float = 10.0 # радиус, на котором агрессивный зверь замечает цель и выбирает жертву

class Animal:
    def __init__(self, uid: str, species: AnimalSpecies, x: float, y: float):
        self.uid = uid
        self.species = species

        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

        self.hp = species.base_hp
        self.age_ticks = 0

        # если зверь приручён → id агента-хозяина
        self.tamed_by: Optional[str] = None

        # если зверь агрессивный → кого именно мы сейчас преследуем и кусаем
        # сюда кладём agent_id
        self.aggression_target: Optional[str] = None

        # для отладки/клиента (HUD, лог действий)
        self.last_action: str = "idle"

    # -------------------------------------------------
    # базовые проверки / вспомогалки
    # -------------------------------------------------

    def is_alive(self) -> bool:
        return self.hp > 0.0

    def distance_to(self, ox: float, oy: float) -> float:
        return math.hypot(ox - self.x, oy - self.y)

    # -------------------------------------------------
    # приручение
    # -------------------------------------------------

    def try_tame(self, agent_id: str, social_bonus: float) -> bool:
        """
        Попытка приручения (агент его кормит / успокаивает).
        social_bonus ~ доверие, спокойствие сцены и т.д. (0..1).
        Успех → зверь закрепляется за agent_id как питомец.
        """
        if not self.species.tamable:
            return False

        # если уже приручён этим же агентом — всё ок
        if self.tamed_by is not None:
            return self.tamed_by == agent_id

        # шанс = social_bonus - сложность
        # если social_bonus 0.8 и tame_difficulty 0.4 → шанс 0.4 (40%)
        chance = max(0.0, (social_bonus - self.species.tame_difficulty))

        if random.random() < chance:
            self.tamed_by = agent_id
            self.aggression_target = None  # приручённый больше не агрится
            self.last_action = f"tamed_by:{agent_id}"
            return True

        return False

    # -------------------------------------------------
    # поиск цели и страховая аура у хищников
    # -------------------------------------------------

    def _pick_aggression_target(self, world):
        """
        Если зверь агрессивный и ещё не приручён:
        выбираем ближайшего живого агента в радиусе агро.
        """
        if not self.species.aggressive:
            return
        if self.tamed_by is not None:
            return  # питомец не охотится на людей
        best_id = None
        best_d2 = None

        # предполагаем, что у мира есть world.agents: Dict[str, Agent]
        for aid, ag in world.agents.items():
            if not ag.is_alive():
                continue
            # не охотимся на собственного хозяина (на будущее)
            if aid == self.tamed_by:
                continue
            d2 = (ag.x - self.x) ** 2 + (ag.y - self.y) ** 2
            if d2 <= (self.species.aggro_radius ** 2):
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_id = aid

        if best_id is not None:
            self.aggression_target = best_id
            self.last_action = f"aggro->{best_id}"

    def _emit_fear_aura(self, world):
        """
        Агрессивный зверь пугает людей рядом просто фактом присутствия.
        Это поднимает страх, даже если он ещё не укусил.
        """
        if not self.species.aggressive:
            return
        if not self.is_alive():
            return

        r2 = self.species.fear_radius ** 2
        for aid, ag in world.agents.items():
            if not ag.is_alive():
                continue
            # питомец не должен пугать своего хозяина после приручения
            if aid == self.tamed_by:
                continue

            dx = ag.x - self.x
            dy = ag.y - self.y
            d2 = dx * dx + dy * dy
            if d2 <= r2:
                # лёгкая тревога
                ag.fear = _clamp(ag.fear + 0.02, 0.0, 1.0)

    # -------------------------------------------------
    # патруль / блуждание
    # -------------------------------------------------

    def _wander(self):
        """
        Свободное блуждание (животное не нападает и не следует за хозяином).
        """
        ang = random.random() * 6.28318  # 2*pi
        self.vx += math.cos(ang) * 0.1
        self.vy += math.sin(ang) * 0.1

        speed = math.hypot(self.vx, self.vy)
        if speed > ANIMAL_MAX_SPEED:
            k = ANIMAL_MAX_SPEED / (speed + 1e-6)
            self.vx *= k
            self.vy *= k

        step = 0.5
        self.x += self.vx * step
        self.y += self.vy * step

        self.last_action = "wander"

    # -------------------------------------------------
    # сопровождение хозяина и защита
    # -------------------------------------------------

    def _follow_owner(self, world):
        """
        Если зверь приручён → держимся рядом с хозяином.
        Если кто-то нападает на хозяина, в будущем тут можно агриться в ответ.
        """
        owner = world.get_agent_by_id(self.tamed_by)
        if (owner is None) or (not owner.is_alive()):
            # хозяин умер → зверь снова дикий
            self.tamed_by = None
            self.last_action = "lost_owner"
            return

        dist_to_owner = self.distance_to(owner.x, owner.y)

        # держим дистанцию
        desired = self.species.follow_distance
        if dist_to_owner > desired:
            dx = owner.x - self.x
            dy = owner.y - self.y
            L = math.hypot(dx, dy) + 1e-6
            self.vx = (dx / L) * ANIMAL_MAX_SPEED
            self.vy = (dy / L) * ANIMAL_MAX_SPEED
            self.x += self.vx
            self.y += self.vy
            self.last_action = f"follow {self.tamed_by}"
        else:
            # стоим рядом
            self.vx *= 0.5
            self.vy *= 0.5
            self.last_action = "guard_idle"

        # хук защиты хозяина:
        # здесь можно просканировать ближайших агрессивных животных,
        # и если они кусают хозяина — мы тоже начинаем их кусать.
        # пока просто оставляем точку расширения.

    # -------------------------------------------------
    # охота / укус
    # -------------------------------------------------

    def _chase_and_bite(self, world):
        """
        Если есть self.aggression_target → догоняем и кусаем агента.
        """
        if self.aggression_target is None:
            return

        target = world.get_agent_by_id(self.aggression_target)
        if target is None or (not target.is_alive()):
            self.aggression_target = None
            self.last_action = "lost_target"
            return

        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.hypot(dx, dy)

        # двигаемся к цели
        if dist > 0:
            self.vx = dx / dist * ANIMAL_MAX_SPEED
            self.vy = dy / dist * ANIMAL_MAX_SPEED

        self.x += self.vx
        self.y += self.vy

        self.last_action = f"chase {self.aggression_target}"

        # укус, если вплотную
        if dist < (ANIMAL_BODY_RADIUS + 1.2):
            dmg = self.species.bite_damage
            before_hp = target.health

            # нанести урон
            target.health = max(0.0, target.health - dmg)

            # поднять страх агенту
            target.fear = min(1.0, target.fear + 0.15)

            # логируем атаку через мир (для HUD/журнала событий)
            world._agent_log_attack_from_animal(
                attacker=self,
                victim=target,
                damage=dmg,
                health_before=before_hp,
            )

            # фиксируем действие для отладки
            self.last_action = f"bite {self.aggression_target} (-{dmg}hp)"

    # -------------------------------------------------
    # главный тик животного
    # -------------------------------------------------

    def tick(self, world):
        """
        Один тик симуляции для этого животного.
        world должен предоставлять:
          - world.agents: Dict[str, Agent]
          - world.get_agent_by_id(agent_id) -> Agent | None
          - world._agent_log_attack_from_animal(...)
          - world.width, world.height (границы карты)
        """
        if not self.is_alive():
            # труп не двигается
            self.last_action = "dead"
            return

        self.age_ticks += 1

        # 1. питомец → следуем за хозяином, не кусаем людей
        if self.tamed_by is not None:
            self._follow_owner(world)

        # 2. дикий агрессивный → ищем цель и атакуем
        elif self.species.aggressive:
            # если нет цели — выбрать новую в радиусе агро
            if self.aggression_target is None:
                self._pick_aggression_target(world)

            if self.aggression_target is not None:
                self._chase_and_bite(world)
            else:
                self._wander()

            # даже если не укусил — он создаёт страховую ауру
            self._emit_fear_aura(world)

        # 3. неагрессивный / просто дружелюбный
        else:
            self._wander()

        # 4. удерживаем животное в пределах мира
        # (предполагаем прямоугольник [0..width] x [0..height])
        self.x = max(0.0, min(self.x, world.width))
        self.y = max(0.0, min(self.y, world.height))

    # -------------------------------------------------
    # экспорт состояния зверя для клиента (отрисовка)
    # -------------------------------------------------

    def build_public_state(self) -> Dict[str, Any]:
        """
        Собираем инфу, которую можно отправить по WebSocket клиенту,
        чтобы там отрисовать зверя и подсветку:
        - temperament: "aggressive", "tameable", "neutral"
        - owner_id, чтобы рисовать маркер "питомец"
        - last_action для HUD дебага
        """
        if self.species.aggressive and self.tamed_by is None:
            temperament = "aggressive"
        elif self.species.tamable:
            temperament = "tameable"
        else:
            temperament = "neutral"

        return {
            "id": self.uid,
            "species": self.species.species_id,
            "name": self.species.name,
            "temperament": temperament,
            "pos": [self.x, self.y],
            "hp": self.hp,
            "owner_id": self.tamed_by,
            "is_alive": self.is_alive(),
            "last_action": self.last_action,
        }
