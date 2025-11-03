# combat_system.py
# Простая обучаемая боевая система для агентов и зверей в локальном мире тренера.
# Подключение: см. раздел "ИНТЕГРАЦИЯ" ниже.

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math
import random


# === УТИЛИТЫ ===

def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


# === МОДЕЛЬ ВЕСОВ (очень легковесное «обучение») ===
# Храним веса выбора действий в brain.behavior_rules["combat_weights"],
# обновляем по принципу REINFORCE с простым базисом.

class _Weights:
    ACTIONS = ("attack", "dodge", "kite", "flee")

    @staticmethod
    def get(agent) -> Dict[str, float]:
        brain = getattr(agent, "brain", None)
        rules = getattr(brain, "behavior_rules", None)
        if rules is None:
            return {a: 0.0 for a in _Weights.ACTIONS}
        cw = getattr(rules, "combat_weights", None)
        if isinstance(cw, dict):
            # copy to avoid side‑effects
            return {a: float(cw.get(a, 0.0)) for a in _Weights.ACTIONS}
        return {a: 0.0 for a in _Weights.ACTIONS}

    @staticmethod
    def set(agent, weights: Dict[str, float]) -> None:
        brain = getattr(agent, "brain", None)
        if brain is None:
            return
        rules = getattr(brain, "behavior_rules", None)
        if rules is None:
            # поведение может быть простым объектом без словаря — создадим карман
            class _BR:  # tiny container
                pass
            rules = _BR()
            setattr(brain, "behavior_rules", rules)
        if not hasattr(rules, "combat_weights") or not isinstance(getattr(rules, "combat_weights"), dict):
            setattr(rules, "combat_weights", {})
        d = getattr(rules, "combat_weights")
        for a in _Weights.ACTIONS:
            d[a] = float(weights.get(a, 0.0))

    @staticmethod
    def update(agent, action: str, reward: float, lr: float = 0.1, decay: float = 0.0005) -> None:
        w = _Weights.get(agent)
        base = sum(w.values()) / (len(w) or 1)
        w[action] = w.get(action, 0.0) + lr * (reward - base)
        # лёгкая регуляризация, чтобы веса не улетали
        for k in list(w.keys()):
            w[k] *= (1.0 - decay)
            w[k] = _clamp(w[k], -5.0, 5.0)
        _Weights.set(agent, w)


# === БОЕВАЯ СИСТЕМА ===

class CombatSystem:
    """Самодостаточный модуль боя. Не привязан к конкретным классам мира:
    работает через getattr/setattr и мягко обходит отсутствующие поля.
    """

    def __init__(self, world: Any) -> None:
        self.world = world
        # настройка по умолчанию
        self.detect_radius = 12.0
        self.kite_distance = 8.0
        self.attack_range = 1.6
        self.agent_base_damage = 8.0
        self.animal_base_damage = 6.0
        self.agent_speed = 4.0
        self.animal_speed = 3.2
        self.cooldown_sec = 0.8
        self.random = random.Random(42)

        # гарантируем карманы
        if not hasattr(self.world, "animals") or getattr(self.world, "animals") is None:
            self.world.animals = []
        if not hasattr(self.world, "chat") or getattr(self.world, "chat") is None:
            self.world.chat = []

    # --- публичные утилиты ---

    def spawn_wave(self, species: str = "wolf", n: int = 3, around: Optional[Tuple[float, float]] = None, r: float = 6.0) -> None:
        W = float(getattr(self.world, "width", 100.0))
        H = float(getattr(self.world, "height", 100.0))
        cx, cy = around if around else (W * 0.5, H * 0.5)
        arr = self._animals_list()
        for i in range(int(n)):
            ang = self.random.random() * math.tau
            rad = r * (0.5 + self.random.random())
            x = _clamp(cx + math.cos(ang) * rad, 0.0, W)
            y = _clamp(cy + math.sin(ang) * rad, 0.0, H)
            beast = self._make_animal(species=species, x=x, y=y)
            arr.append(beast)
        self._chat(f"[spawn] {n} {species}(s) около ({cx:.1f},{cy:.1f})")

    # --- основной шаг симуляции боя ---

    def step(self, dt: float = 0.1) -> None:
        if not self.world:
            return
        agents = self._agents_list()
        animals = self._animals_list()
        if not agents:
            return

        # 1) базовые атрибуты/кулдауны
        for a in agents:
            self._ensure_agent(a)
            cd = float(getattr(a, "combat_cd", 0.0))
            cd = max(0.0, cd - dt)
            setattr(a, "combat_cd", cd)

        for b in animals:
            self._ensure_animal(b)
            cd = float(getattr(b, "atk_cd", 0.0))
            cd = max(0.0, cd - dt)
            setattr(b, "atk_cd", cd)

        # 2) поведение зверей: агрессивные преследуют ближайшего живого агента
        for b in list(animals):
            if not bool(getattr(b, "aggressive", False)):
                continue
            if not self._is_alive_animal(b):
                continue
            target = self._nearest_living_agent(b, agents)
            if target is None:
                continue
            self._chase(b, target, dt)

        # 3) решение агента и действия
        for ag in list(agents):
            if not self._is_alive_agent(ag):
                continue
            threats = self._nearby_animals(ag, animals, radius=self.detect_radius)
            if not threats:
                # плавное восстановление энергии и немного страха
                self._recover_out_of_combat(ag, dt)
                continue

            action = self._decide_action(ag, threats)
            if action == "attack":
                self._act_attack(ag, threats, dt)
            elif action == "dodge":
                self._act_dodge(ag, threats, dt)
            elif action == "kite":
                self._act_kite(ag, threats, dt)
            elif action == "flee":
                self._act_flee(ag, threats, dt)

        # 4) атаки зверей по близким агентам
        for b in list(animals):
            if not self._is_alive_animal(b):
                continue
            if float(getattr(b, "atk_cd", 0.0)) > 0.0:
                continue
            tgt = self._nearest_living_agent(b, agents, max_r=self.attack_range + 0.2)
            if tgt is None:
                continue
            self._deal_damage_to_agent(b, tgt, self._animal_damage(b))
            setattr(b, "atk_cd", self.cooldown_sec)

        # 5) зачистка трупов зверей (мягко: оставляем пару тиков для визуала)
        for b in list(animals):
            if self._is_alive_animal(b):
                continue
            # шанс «разложиться»
            if self.random.random() < 0.05:
                try:
                    animals.remove(b)
                except Exception:
                    pass

    # --- частные методы ---

    def _agents_list(self) -> List[Any]:
        A = getattr(self.world, "agents", None)
        if isinstance(A, dict):
            return list(A.values())
        if isinstance(A, list):
            return A
        return []

    def _animals_list(self) -> List[Any]:
        Z = getattr(self.world, "animals", None)
        if isinstance(Z, dict):
            # переключим на list (проще управлять жизненным циклом)
            lst = list(Z.values())
            try:
                self.world.animals = lst
            except Exception:
                pass
            return lst
        if isinstance(Z, list):
            return Z
        # создадим список
        self.world.animals = []
        return self.world.animals

    def _make_animal(self, species: str, x: float, y: float) -> Any:
        class _Beast:  # минимальный контейнер совместимый со снапшотом
            pass
        b = _Beast()
        setattr(b, "uid", f"{species}_{int(random.random()*1e9):09d}")
        setattr(b, "species", type("_Spec", (), {"species_id": species, "aggressive": True, "tamable": False, "base_hp": 50.0})())
        setattr(b, "x", float(x))
        setattr(b, "y", float(y))
        setattr(b, "hp", 50.0)
        setattr(b, "atk_cd", 0.0)
        setattr(b, "speed", self.animal_speed)
        setattr(b, "alive", True)
        return b

    def _ensure_agent(self, a: Any) -> None:
        if not hasattr(a, "health"): setattr(a, "health", 100.0)
        if not hasattr(a, "energy"): setattr(a, "energy", 100.0)
        if not hasattr(a, "fear"): setattr(a, "fear", 0.0)
        if not hasattr(a, "x"): setattr(a, "x", float(getattr(a, "pos_x", 0.0)))
        if not hasattr(a, "y"): setattr(a, "y", float(getattr(a, "pos_y", 0.0)))
        if not hasattr(a, "vx"): setattr(a, "vx", 0.0)
        if not hasattr(a, "vy"): setattr(a, "vy", 0.0)
        if not hasattr(a, "combat_cd"): setattr(a, "combat_cd", 0.0)
        if not hasattr(a, "combat_skill"): setattr(a, "combat_skill", 0.0)  # влияет на урон и уклонение
        if not hasattr(a, "alive"): setattr(a, "alive", True)

    def _ensure_animal(self, b: Any) -> None:
        if not hasattr(b, "hp"): setattr(b, "hp", 50.0)
        if not hasattr(b, "x"): setattr(b, "x", 0.0)
        if not hasattr(b, "y"): setattr(b, "y", 0.0)
        if not hasattr(b, "speed"): setattr(b, "speed", self.animal_speed)
        if not hasattr(b, "atk_cd"): setattr(b, "atk_cd", 0.0)
        if not hasattr(b, "species"):
            setattr(b, "species", type("_Spec", (), {"species_id": "beast", "aggressive": True, "tamable": False, "base_hp": 50.0})())
        if not hasattr(b, "alive"): setattr(b, "alive", True)

    def _is_alive_agent(self, a: Any) -> bool:
        alive = getattr(a, "alive", True)
        if callable(alive):
            try:
                return bool(alive())
            except Exception:
                pass
        return bool(alive) and float(getattr(a, "health", 0.0)) > 0.0

    def _is_alive_animal(self, b: Any) -> bool:
        alive = getattr(b, "alive", True)
        if callable(alive):
            try:
                return bool(alive())
            except Exception:
                pass
        return bool(alive) and float(getattr(b, "hp", 0.0)) > 0.0

    def _nearest_living_agent(self, beast: Any, agents: List[Any], max_r: Optional[float] = None) -> Optional[Any]:
        bx, by = float(getattr(beast, "x", 0.0)), float(getattr(beast, "y", 0.0))
        best, best_d2 = None, float("inf")
        for a in agents:
            if not self._is_alive_agent(a):
                continue
            d2 = _dist2(bx, by, float(getattr(a, "x", 0.0)), float(getattr(a, "y", 0.0)))
            if d2 < best_d2:
                best_d2 = d2; best = a
        if best is None:
            return None
        if max_r is not None and best_d2 > max_r * max_r:
            return None
        return best

    def _chase(self, beast: Any, target: Any, dt: float) -> None:
        bx, by = float(getattr(beast, "x", 0.0)), float(getattr(beast, "y", 0.0))
        tx, ty = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
        ang = math.atan2(ty - by, tx - bx)
        spd = float(getattr(beast, "speed", self.animal_speed))
        nx = bx + math.cos(ang) * spd * dt
        ny = by + math.sin(ang) * spd * dt
        self._move_entity(beast, nx, ny)

    def _move_entity(self, ent: Any, nx: float, ny: float) -> None:
        W = float(getattr(self.world, "width", 100.0))
        H = float(getattr(self.world, "height", 100.0))
        setattr(ent, "x", _clamp(nx, 0.0, W))
        setattr(ent, "y", _clamp(ny, 0.0, H))

    def _nearby_animals(self, ag: Any, animals: List[Any], radius: float) -> List[Any]:
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        r2 = radius * radius
        res = []
        for b in animals:
            if not self._is_alive_animal(b):
                continue
            if _dist2(ax, ay, float(getattr(b, "x", 0.0)), float(getattr(b, "y", 0.0))) <= r2:
                res.append(b)
        return res

    def _decide_action(self, ag: Any, threats: List[Any]) -> str:
        # оценка угрозы
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        threat_score = 0.0
        nearest_d = float("inf")
        for b in threats:
            bx, by = float(getattr(b, "x", 0.0)), float(getattr(b, "y", 0.0))
            d = math.sqrt(max(1e-6, _dist2(ax, ay, bx, by)))
            nearest_d = min(nearest_d, d)
            hp = float(getattr(b, "hp", 50.0))
            threat_score += (hp / d)

        # базовые правила из мозга
        brain = getattr(ag, "brain", None)
        rules = getattr(brain, "behavior_rules", None)
        fight_thr = getattr(rules, "fight_threshold", 1.5) if rules else 1.5
        kite_dist = getattr(rules, "kite_distance", self.kite_distance) if rules else self.kite_distance

        # веса действий
        w = _Weights.get(ag)

        # эвристики + веса -> выбор
        scores = {
            "attack": (1.0 if nearest_d <= self.attack_range + 0.2 else 0.0) + w["attack"] - 0.2 * (threat_score > fight_thr),
            "dodge": 0.5 + w["dodge"] + 0.2 * (nearest_d <= self.attack_range + 0.5),
            "kite": 0.6 + w["kite"] + 0.3 * (nearest_d < kite_dist),
            "flee": -0.2 + w["flee"] + 0.4 * (threat_score > fight_thr * 1.2),
        }
        # небольшая стохастика
        for k in scores:
            scores[k] += (self.random.random() - 0.5) * 0.05
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return best

    def _recover_out_of_combat(self, ag: Any, dt: float) -> None:
        e = float(getattr(ag, "energy", 100.0))
        h = float(getattr(ag, "health", 100.0))
        f = float(getattr(ag, "fear", 0.0))
        setattr(ag, "energy", _clamp(e + 6.0 * dt, 0.0, 100.0))
        setattr(ag, "fear", _clamp(f - 2.0 * dt, 0.0, 100.0))
        # лёгкая регенерация, если не голоден и не боится
        if getattr(ag, "hunger", 0.0) < 50.0 and f < 10.0:
            setattr(ag, "health", _clamp(h + 1.0 * dt, 0.0, 100.0))

    # --- действия агента ---

    def _act_attack(self, ag: Any, threats: List[Any], dt: float) -> None:
        if float(getattr(ag, "combat_cd", 0.0)) > 0.0:
            return
        # атакуем ближайшего
        target = self._nearest(threats, ag)
        if target is None:
            return
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        bx, by = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
        d = math.sqrt(max(1e-6, _dist2(ax, ay, bx, by)))
        if d > self.attack_range + 0.2:
            # подойти
            self._step_towards(ag, bx, by, dt, speed=self.agent_speed)
            return
        # удар
        dmg = self._agent_damage(ag)
        self._deal_damage_to_animal(ag, target, dmg)
        setattr(ag, "combat_cd", self.cooldown_sec)
        # вознаграждение за успешный удар
        _Weights.update(ag, "attack", reward=+2.0)

    def _act_dodge(self, ag: Any, threats: List[Any], dt: float) -> None:
        # рывок перпендикулярно направлению на ближайшего
        target = self._nearest(threats, ag)
        if target is None:
            return
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        bx, by = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
        ang = math.atan2(ay - by, ax - bx) + math.pi * 0.5
        dash = self.agent_speed * 1.6
        nx = ax + math.cos(ang) * dash * dt
        ny = ay + math.sin(ang) * dash * dt
        self._move_entity(ag, nx, ny)
        # частичная награда за уклонение
        _Weights.update(ag, "dodge", reward=+0.3)

    def _act_kite(self, ag: Any, threats: List[Any], dt: float) -> None:
        # отходим на комфортную дистанцию, по пути бьём если в радиусе
        target = self._nearest(threats, ag)
        if target is None:
            return
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        bx, by = float(getattr(target, "x", 0.0)), float(getattr(target, "y", 0.0))
        d = math.sqrt(max(1e-6, _dist2(ax, ay, bx, by)))
        if d < self.kite_distance:
            # отходим
            ang = math.atan2(ay - by, ax - bx)
            self._step_dir(ag, ang, dt, speed=self.agent_speed * 1.1)
        if d <= self.attack_range + 0.2 and float(getattr(ag, "combat_cd", 0.0)) <= 0.0:
            dmg = self._agent_damage(ag)
            self._deal_damage_to_animal(ag, target, dmg)
            setattr(ag, "combat_cd", self.cooldown_sec)
            _Weights.update(ag, "kite", reward=+1.0)

    def _act_flee(self, ag: Any, threats: List[Any], dt: float) -> None:
        # максимальный рывок от центра масс угроз
        if not threats:
            return
        cx = sum(float(getattr(b, "x", 0.0)) for b in threats) / len(threats)
        cy = sum(float(getattr(b, "y", 0.0)) for b in threats) / len(threats)
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        ang = math.atan2(ay - cy, ax - cx)
        self._step_dir(ag, ang, dt, speed=self.agent_speed * 1.8)
        _Weights.update(ag, "flee", reward=+0.1)

    # --- вспомогательные вещи для действий ---

    def _nearest(self, beasts: List[Any], ag: Any) -> Optional[Any]:
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        best, best_d2 = None, float("inf")
        for b in beasts:
            d2 = _dist2(ax, ay, float(getattr(b, "x", 0.0)), float(getattr(b, "y", 0.0)))
            if d2 < best_d2:
                best_d2 = d2; best = b
        return best

    def _step_towards(self, ag: Any, tx: float, ty: float, dt: float, speed: float) -> None:
        ax, ay = float(getattr(ag, "x", 0.0)), float(getattr(ag, "y", 0.0))
        ang = math.atan2(ty - ay, tx - ax)
        self._step_dir(ag, ang, dt, speed)

    def _step_dir(self, ag: Any, ang: float, dt: float, speed: float) -> None:
        nx = float(getattr(ag, "x", 0.0)) + math.cos(ang) * speed * dt
        ny = float(getattr(ag, "y", 0.0)) + math.sin(ang) * speed * dt
        self._move_entity(ag, nx, ny)

    # --- урон/смерть/награда ---

    def _agent_damage(self, ag: Any) -> float:
        skill = float(getattr(ag, "combat_skill", 0.0))
        return self.agent_base_damage * (1.0 + 0.12 * _clamp(skill, 0.0, 5.0))

    def _animal_damage(self, b: Any) -> float:
        return float(getattr(b, "damage", self.animal_base_damage))

    def _deal_damage_to_animal(self, ag: Any, b: Any, dmg: float) -> None:
        hp = float(getattr(b, "hp", 0.0)) - max(0.0, dmg)
        setattr(b, "hp", hp)
        if hp <= 0.0 and self._is_alive_animal(b):
            setattr(b, "alive", False)
            self._chat(f"{getattr(ag, 'name', getattr(ag, 'id', 'agent'))} убил {self._animal_name(b)}")
            # сильная награда + рост навыка
            _Weights.update(ag, "attack", reward=+6.0)
            _Weights.update(ag, "kite", reward=+3.0)
            setattr(ag, "combat_skill", float(getattr(ag, "combat_skill", 0.0)) + 0.25)
            # мозгу — общий survival_score
            brain = getattr(ag, "brain", None)
            if brain is not None:
                ss = getattr(brain, "survival_score", 0.0)
                setattr(brain, "survival_score", ss + 2.0)

    def _deal_damage_to_agent(self, b: Any, ag: Any, dmg: float) -> None:
        # простой шанс уклонения от навыка
        skill = float(getattr(ag, "combat_skill", 0.0))
        evade_chance = _clamp(0.05 + 0.03 * skill, 0.05, 0.45)
        if random.random() < evade_chance:
            # маленькая награда dodge за удачное уклонение
            _Weights.update(ag, "dodge", reward=+0.8)
            return
        hp = float(getattr(ag, "health", 0.0)) - max(0.0, dmg)
        setattr(ag, "health", hp)
        setattr(ag, "fear", _clamp(float(getattr(ag, "fear", 0.0)) + dmg * 0.6, 0.0, 100.0))
        _Weights.update(ag, "attack", reward=-0.3)
        _Weights.update(ag, "flee", reward=+0.2)
        if hp <= 0.0 and self._is_alive_agent(ag):
            setattr(ag, "alive", False)
            cod = f"killed_by_{self._animal_name(b)}"
            setattr(ag, "cause_of_death", cod)
            self._chat(f"{getattr(ag, 'name', getattr(ag, 'id', 'agent'))} погиб от {self._animal_name(b)}")
            # сильное наказание атаке, поощрение бегства на будущее
            _Weights.update(ag, "attack", reward=-5.0)
            _Weights.update(ag, "flee", reward=+3.0)
            brain = getattr(ag, "brain", None)
            if brain is not None:
                ss = getattr(brain, "survival_score", 0.0)
                setattr(brain, "survival_score", ss - 5.0)

    def _animal_name(self, b: Any) -> str:
        sp = getattr(b, "species", None)
        if sp is None:
            return "beast"
        return getattr(sp, "species_id", getattr(sp, "name", "beast"))

    def _chat(self, line: str) -> None:
        try:
            self.world.chat.append(str(line))
            # ограничим хвост
            if len(self.world.chat) > 500:
                del self.world.chat[:-500]
        except Exception:
            pass


# === ИНТЕГРАЦИЯ В combined_app.py ===
# 1) Добавь импорт:
#     from combat_system import CombatSystem
# 2) После создания self.trainer в CombinedMainWindow.__init__:
#     self.combat = CombatSystem(self.trainer.world)
#     self._combat_timer = QtCore.QTimer(self)
#     self._combat_timer.setInterval(50)  # 20 Гц
#     self._combat_timer.timeout.connect(self._on_combat_tick)
#     self._combat_timer.start()
# 3) Добавь метод в CombinedMainWindow:
#     def _on_combat_tick(self):
#         if getattr(self, 'combat', None) and getattr(self.trainer, 'world', None):
#             self.combat.step(0.05)
#             # пушим снапшот для 3D
#             self.bridge._push_snapshot()
# 4) (Опционально) хоткей на спаун волков рядом с выделенным агентом:
#     act = QtGui.QAction("Spawn wolves", self); act.setShortcut("Ctrl+W")
#     act.triggered.connect(self._spawn_wolves)
#     self.addAction(act)
#     def _spawn_wolves(self):
#         sel = self.shared.get_selected_agent_id()
#         w = self.trainer.world
#         if not w: return
#         cx = getattr(w.get_agent_by_id(sel), 'x', getattr(w, 'width', 50.0)*0.5) if sel and hasattr(w, 'get_agent_by_id') else getattr(w, 'width', 50.0)*0.5
#         cy = getattr(w.get_agent_by_id(sel), 'y', getattr(w, 'height', 50.0)*0.5) if sel and hasattr(w, 'get_agent_by_id') else getattr(w, 'height', 50.0)*0.5
#         self.combat.spawn_wave('wolf', n=3, around=(cx, cy))
#         self.bridge._push_snapshot()
