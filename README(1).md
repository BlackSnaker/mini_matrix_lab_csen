# Mini‑Matrix Lab

> 3D‑мир + Mind Trainer (эволюция/метрики/мозг агентов) с интерактивным просмотром в OpenGL и управлением через PySide6.


## Ключевые возможности


- Единое окно: **3D‑сцена** (QOpenGLWidget) + **панель эволюции/метрик** + **инспектор мозга агента**.
- Синхронный выбор агента между 3D и инспектором.
- **ПКМ по земле** — постановка цели `goal` выбранному агенту.
- **Боевой модуль**: спавн волн (горячая клавиша `Ctrl+W`), шаг симуляции боя с частотой ~20 Гц.
- Низкополигональная сцена окружения (village) и мост Trainer → Engine через снапшоты.
- «Стеклянный» UI со стилизацией, подсказками и статус‑панелью.


## Быстрый старт


**Зависимости (минимум):**
- Python 3.11+
- PySide6
- PyOpenGL
- numpy
- (опционально) moderngl / glfw / pyrr — если используются в ваших модулях рендера

**Установка окружения (пример с uv):**
```bash
uv venv && source .venv/bin/activate
uv pip install PySide6 PyOpenGL numpy
# при необходимости
uv pip install moderngl glfw pyrr
```

**Запуск:**
```bash
python combined_app.py
```
Если у вас другой entry‑point, запустите соответствующий файл.


## Структура проекта

```text
mini_matrixV012_src/
└── mini_matrix
    ├── brains
    │   ├── a1.json
    │   ├── a2.json
    │   └── agent_0.json
    ├── legacy
    ├── trained_brains
    │   ├── a1.mind.json
    │   ├── a2.mind.json
    │   └── agent_0.mind.json
    ├──  trainer_side.py
    ├── agent.py
    ├── animals.py
    ├── bootstrap.py
    ├── brain_io.py
    ├── combat_system.py
    ├── combined_app.py
    ├── config.py
    ├── engine3d.py
    ├── env_lowpoly.py
    ├── gui_client.py
    ├── memory.py
    ├── mind_core.py
    ├── mind_trainer.py
    ├── mind_trainer_gui.py
    ├── procgen.py
    ├── requirements.txt
    ├── schema.py
    ├── server.py
    ├── server_side.py
    ├── structure.lua
    ├── viewer_3d.py
    ├── village_map.py
    └── world.py
```


## Архитектура


Проект состоит из 3 основных слоёв:

1. **Mind Trainer** — логика локального мира/эволюции, хранит `world`, испускает сигнал `world_changed`.
2. **Мост (Trainer → Engine)** — формирует *снапшот* (словарь) из состояния `world` и вызывает `engine.sync_from_world(snapshot)`.
3. **3D‑движок + Виджет** — `MiniMatrixEngine` + `World3DView` (QOpenGLWidget) отрисовывают сцену и агентов.

Панели GUI:
- `TrainerStatsWidget` — метрики/эволюция.
- `AgentBrainWidget` — инспектор текущего агента (beliefs, memory, rules).

Боевой модуль:
- `CombatSystem` — управляет сущностями боя в `world`, спавнит волны, обновляет их состояние на каждом тике.


## Горячие клавиши и управление


- **ЛКМ** — выбрать агента (пикинг по XZ на плоскости `y=0`).
- **ПКМ по земле** — задать `goal (x,z)` выбранному агенту.
- **Колёсико мыши** — зум камеры.
- **Зажатая ПКМ + движение** — орбита камеры.
- **Зажатая СКМ + движение** — панорамирование.
- **Tab** — выбрать следующего агента.
- **F** — сфокусироваться на выбранном агенте.
- **R** — сброс камеры в дефолт.
- **Ctrl+W** — спавн волков около выбранного агента.


## Снапшот мира (Trainer → Engine)


Мост вызывает `_build_engine_snapshot(world, tick=...)`, результат — словарь вида:
```json
{
  "tick": 123,
  "world": {"width": 100.0, "height": 100.0},
  "agents": [
    {
      "id": "a1",
      "name": "Echo",
      "pos": {"x": 12.3, "y": 45.6},
      "goal": {"x": 20.0, "y": 40.0},
      "vel": {"x": 0.1, "y": -0.2},
      "fear": 0.05,
      "health": 98.0,
      "energy": 80.0,
      "hunger": 0.0,
      "age_ticks": 450,
      "alive": true,
      "cause_of_death": null,
      "mind": {
        "current_drive": "explore",
        "survival_score": 0.72,
        "behavior_rules": { "...": "..." },
        "beliefs": [{"if": "...", "then": "...", "strength": 0.8}],
        "memory_tail": ["...", "..."]
      }
    }
  ],
  "objects": [
    { "id": "o1", "name": "Fire", "kind": "hazard", "pos": {"x": 30, "y": 50}, "radius": 3.0,
      "danger_level": 0.9, "comfort_level": 0.0 }
  ],
  "animals": [
    { "id": "w1", "species": "wolf", "pos": {"x": 55, "y": 52}, "hp": 40.0,
      "aggressive": true, "tamable": false, "tamed_by": null }
  ],
  "chat": ["[system] ..."],
  "events": []
}
```
**Важно:** если вы меняете формат, адаптируйте конвертер `_build_engine_snapshot(...)` и метод `MiniMatrixEngine.sync_from_world(...)`.


## Ключевые модули

| Модуль | Назначение |
|---|---|
| `combined_app.py` | Точка входа с окном из трёх колонок (Stats | 3D | Brain), «стеклянным» UI и хоткеями. |
| `engine3d.py` | 3D‑движок и сущности рендера. Ожидает снапшоты мира и умеет `render_opengl()`. |
| `env_lowpoly.py` | Генерация низкополигонального окружения (посёлок/виллэдж). |
| `combat_system.py` | Примитивная боевая система: волны, шаг, коллизии/урон (в зависимости от реализации). |
| `mind_trainer_gui.py` | Мир тренера, метрики, инспектор мозга, сигнал `world_changed`. |
| `mind_core.py` | Психика/поведение агентов (параметры, эмоции, память), если присутствует. |
| `world.py` | Базовые структуры мира, агенты/объекты/животные (если присутствует). |

## Как расширять


### Добавить новый вид животного
1. Определите **species** с параметрами (`base_hp`, `aggressive`, `tamable`, и т. п.).
2. Добавьте инстансы в `world.animals` и обновляйте их в игровом цикле.
3. Отразите их в снапшоте (`animals`), чтобы они визуализировались в 3D.

### Добавить правило поведения / эмоцию
1. Расширьте `mind_core` (beliefs/behavior_rules/current_drive).
2. Прокиньте новые поля в `_brain_to_dict(...)` и далее в снапшот.
3. В `AgentBrainWidget` отобразите/отредактируйте новые параметры.

### Добавить объект мира
1. Создайте объект с `id/name/kind/pos/radius/...` и добавьте в `world.objects`.
2. Обновите логику взаимодействий (опасность/комфорт).
3. Проверьте визуализацию в `engine3d` (иконки, меши, цвета).

### Добавить кнопку/панель в UI
1. Оберните новый виджет в `make_card(title, widget)` для единого стиля.
2. Добавьте карточку в `QSplitter` и настройте `setStretchFactor`.


## Точки интеграции и сигналы


- `MindTrainerInteractive.world_changed: Signal()` — вызывается при изменениях мира; мост собирает снапшот.
- `World3DView.requestSetGoal(agent_id, x, z)` — эмитится при ПКМ по земле, перехватывается окном и транслируется в `world`.
- `MiniMatrixEngine.sync_from_world(snapshot)` — синхронизация состояния рендера с логикой мира.


## Устранение неполадок


- **AttributeError: QPalette has no attribute 'Window'** (Qt6): используйте `QtGui.QPalette.ColorRole.Window` (уже исправлено).
- **AA_UseHighDpiPixmaps is deprecated**: в Qt6 можно не устанавливать, Qt сам поддерживает HiDPI.
- **Черный экран/ничего не рисуется**: проверьте, что `render_opengl()` вызывается и матрицы камеры корректны.
- **Проблемы с OpenGL в Linux**: установите системные драйверы и `PyOpenGL`. Попробуйте запуск в X11 (не Wayland) при старых драйверах.
- **Нет модуля moderngl/glfw**: если вы задействовали их в рендере, установите пакеты или уберите зависимости из кода.


## Требования

```text
PyOpenGL
PySide6
numpy
```


## Лицензия
