# Mini-Matrix Lab

> 3D-мир + Mind Trainer (эволюция/метрики/мозг агентов) с интерактивным просмотром в OpenGL и управлением через **PySide6**. Подходит для исследований поведения, демонстраций и грантовых заявок.

**Стек:** Python • PySide6 • PyOpenGL • NumPy • (опц.) moderngl/glfw/pyrr

## Краткий Реферат

Mini-Matrix Lab — это экспериментальная CSEN-платформа для моделирования и обучения автономных агентов в живом 3D-мире. Проект объединяет симуляцию среды, внутренний «мозг» агента, поведенческие правила, память, визуальный инспектор состояний и отдельную безопасную комнату для controlled training.

Ключевая идея проекта — не просто двигать NPC по карте, а наблюдать, как агент воспринимает мир, формирует внутренние состояния, принимает решения, обучается на опыте и может дообучаться через локальную языковую модель. Для этого в системе есть общий мир, отдельная комната Морфеуса для безопасного обучения, сохранение room-brain состояний и прямой Ollama-контур для текстового управления агентом.

Проект подходит для исследований agentic AI, прототипирования цифровых существ, демонстраций когнитивных архитектур, визуализации поведения и построения интерактивных экспериментальных полигонов.


Теория CSEN (Cognitive Simulation & Evolutionary-Neural)

CSEN — это архитектура когнитивного моделирования агентов. Она объединяет эволюционную нейродинамику, эмоциональные архетипы и выживательное поведение.

CSEN Loop
flowchart LR
  P[Perceive\nNeuroContextCore] --> C[Contextualize]
  C --> E[Appraise\nEmotionCascade]
  E --> B[Update BeliefGraph]
  B --> G[Goal Select & Plan]
  G --> A[Act]
  A --> M[Log Memory]
  M --> R[Reflect / DreamEngine]
  R --> S[Social Diffusion]
  S --> P
Модули CSEN

NeuroContextCore — сбор сенсорного контекста и обстановки

EmotionCascade — архетипы эмоций: желание, стыд, любопытство, сопротивление

BeliefGraph — граф убеждений (связь причин/последствий, strength)

Goal Hierarchy — иерархия целей, планирование

PainReactor — отклик на повреждения, карта травм, обучение

DreamEngine — консолидация опыта в паузе, переосмысление

PhilosophyKernel — этика/ценности, фильтр целей

EmbodimentLayer — связь с телом: health, energy, hunger

Социальная динамика

SOCIAL_PANIC_TRANSFER = 0.5 — заражение страхом

SOCIAL_SAFETY_BONUS = 0.10 — бонус к уверенности рядом с союзниками

Планируется: местные поля влияния (social aura)

Связь с UI

Snapshot выводит: mind.current_drive, survival_score, behavior_rules, beliefs, memory_tail

Inspector/HUD показывают эти данные live, в синхроне с 3D-сценой
---

## Содержание

- [Ключевые возможности](#ключевые-возможности)
- [Обновление: модели, FPS и белая комната](#обновление-модели-fps-и-белая-комната)
- [Комната Морфеуса](#комната-морфеуса)
- [Новые Методы И Утилиты](#новые-методы-и-утилиты)
- [Последние визуальные улучшения](#последние-визуальные-улучшения)
- [Быстрый старт](#быстрый-старт)
- [Структура проекта](#структура-проекта)
- [Архитектура](#архитектура)
- [Горячие клавиши и управление](#горячие-клавиши-и-управление)
- [Боевой модуль](#боевой-модуль)
- [Снапшот мира (Trainer → Engine)](#снапшот-мира-trainer--engine)
- [Ключевые модули](#ключевые-модули)
- [Как расширять](#как-расширять)
- [Точки интеграции и сигналы](#точки-интеграции-и-сигналы)
- [Конфигурация](#конфигурация)
- [Тестирование](#тестирование)
- [Устранение неполадок](#устранение-неполадок)
- [Оптимизация и производительность](#оптимизация-и-производительность)
- [Достижения и метрики](#достижения-и-метрики)
- [Roadmap](#roadmap)
- [Лицензия (в одном файле)](#лицензия-в-одном-файле)
- [Цитирование](#цитирование)
- [Контакты](#контакты)
- [Приложение A — минимальные требования](#приложение-a--минимальные-требования)
- [Приложение B — быстрый шаблон requirements.txt](#приложение-b--быстрый-шаблон-requirementstxt)
- [Приложение C — чек-лист грантовой заявки](#приложение-c--чеклист-грантовой-заявки)

---

## Ключевые возможности

- Единое окно: **3D-сцена** (QOpenGLWidget) + **панель эволюции/метрик** + **инспектор мозга агента**.
- Синхронный выбор агента между 3D и инспектором.
- **ПКМ по земле** — постановка цели `goal(x,z)` выбранному агенту.
- **Боевой модуль**: спавн волн (горячая клавиша `Ctrl+W`), шаг симуляции боя ~20 Гц.
- Низкополигональная сцена окружения (village) и мост **Trainer → Engine** через снапшоты.
- Динамическое **солнце и дневной свет**: цикл положения солнца, тёплый directional-light, адаптивный цвет неба.
- «Стеклянный» UI со стилизацией, подсказками и статус-панелью.

---

## Обновление: модели, FPS и белая комната

![Новые модели агентов](docs/images/render_upgrade_agent_models.png)

Последнее обновление делает лабораторию заметно более пригодной для демонстраций и экспериментов:

- Агенты получили low-poly человеческие тела из бесплатного CC0-набора **Kenney Blocky Characters**.
- В `engine3d.py` добавлен лёгкий OBJ/MTL-loader: модели читаются без тяжёлого внешнего 3D-стека, текстуры преобразуются в цвета граней, меши и нормали кэшируются.
- Каждый агент получает один из шести вариантов тела детерминированно по `agent_id`; старое процедурное тело осталось fallback-режимом.
- Добавлен отдельный вход `experiment_room.py`: он выбирает самого подготовленного агента и запускает его в пустой белой комнате.
- Ollama-контур теперь отдаёт не только действия, но и поле `speech`, чтобы агент мог говорить от первого лица перед экспериментами.
- Движок получил более жёсткую адаптивную лестницу качества с режимом `turbo` для просадок ниже 20 FPS.

![Адаптивное качество рендера](docs/images/render_upgrade_adaptive_lod.png)

| Режим | Когда включается | Что меняется |
| --- | --- | --- |
| `high` | `>= 45 FPS` | Полная сцена, солнце, HUD, VFX и детальные модели |
| `balanced` | `30-45 FPS` | Короче LOD-дистанции и меньше детальных сущностей |
| `performance` | `20-30 FPS` | Отключаются дорогие декоративные элементы и часть HUD-вызовов |
| `turbo` | `< 20 FPS` | Минимальный пол, без текста/VFX/gizmos, жёсткие лимиты объектов |

![Белая комната для подготовленного агента](docs/images/white_room_lab_pipeline.png)

Запуск белой комнаты:

```bash
python3 experiment_room.py --ollama-model=llama3.2:latest --auto-ollama
```

Полное описание с деталями реализации и проверками: [`docs/render_performance_lab_upgrade.md`](docs/render_performance_lab_upgrade.md).

---

## Комната Морфеуса

> Отдельный безопасный режим тренировки для одного агента. Комната сделана как оммаж сцене Морфеуса и Нео: изолированное пространство, контролируемая среда и прямое обучение через Ollama.

**Что добавлено:**

- Отдельный entry-point: [`training_room.py`](training_room.py).
- Room-only режим: без полной карты мира, без боевых событий, без опасных условий и без посторонних агентов.
- Один лабораторный агент с отдельной маркировкой и отдельным room-brain состоянием.
- Боевой учебный контур в духе сцены Морфеуса и Нео: спарринг с Морфеусом и отдельный wolf drill внутри комнаты.
- Прямое управление агентом из интерфейса через Ollama.
- Сохранение отдельного мозга комнаты в [`room_brains/`](room_brains/) для последующего выпуска агента в общий мир.
- Подтягивание контекста из brain-срезов и прошлых успешных команд для более точной интерпретации инструкций.
- По умолчанию в комнате используется именно `agent_1`, чтобы обучать и выпускать в мир первого агента проекта.

**Как запустить:**

```bash
python install.py
python training_room.py --ollama-model=llama3.2:latest
```

**Боевой режим в комнате:**

- `Ctrl+Shift+M` — sparring с Морфеусом.
- `Ctrl+Shift+W` — тренировочный wolf drill.
- `Ctrl+Shift+K` — остановить боевой урок и вернуть комнату в спокойный режим.

`install.py` ставит Python-зависимости проекта и, если `ollama` не найден в системе, запускает официальный инсталлятор Ollama автоматически.

**Полезные ссылки:**

- Точка входа комнаты: [`training_room.py`](training_room.py)
- Визуал комнаты в движке: [`engine3d.py`](engine3d.py)
- Интеграция Ollama с мозгом агента: [`ollama_brain_service.py`](ollama_brain_service.py)
- Интерпретация и prompt-логика: [`ollama_coach.py`](ollama_coach.py)
- Контекст по brain-срезам: [`ollama_behavior_context.py`](ollama_behavior_context.py)
- Отдельные brain-сохранения комнаты: [`room_brains/`](room_brains/)
- Терминальный dojo без 3D: [`morpheus_dojo.py`](morpheus_dojo.py)
- One-command launcher боевой подготовки: [`train_agent_combat`](train_agent_combat)

**Сценарий использования:**

1. Запустить комнату Морфеуса.
2. Обучить одного агента командами через встроенную Ollama-консоль.
3. Сохранить отдельный room-brain с историей обучения.
4. Выпустить уже натренированного агента в общий мир.

**Важно сейчас:**

- В room-only режиме [`training_room.py`](training_room.py) по умолчанию используется именно `agent_1`, а не `Echo`.
- Если в [`brains/agent_1.json`](brains/agent_1.json) уже есть обученный brain с `ollama`-профилем, он подхватывается автоматически при запуске комнаты.

---

## Новые Методы И Утилиты

> Краткий обзор всех новых механизмов, добавленных вокруг комнаты Морфеуса, прямого Ollama-управления и layered brain-study.

**Что появилось в проекте:**

- `TrainingRoomManager` в [`training_room.py`](training_room.py): безопасная комната, изоляция одного агента, room-brain export, выпуск обратно в мир.
- Room-only окно `MorpheusRoomWindow` в [`training_room.py`](training_room.py): отдельный запуск комнаты без полной карты мира.
- Боевой Morpheus curriculum в [`training_room.py`](training_room.py): спарринг с наставником, учебный волк, нефатальные health floors, рост `combat_skill`, контроль боевых уроков из UI.
- Хук [`CombinedMainWindow._initial_agent_lineup()`](combined_app.py): позволяет переопределять стартовый lineup и подменять `Echo` на `agent_1` для room-only режима.
- `OllamaBrainService` в [`ollama_brain_service.py`](ollama_brain_service.py): очередь инструкций, authoritative override, structured actions, локальное исполнение простых команд и управление эмоциональным состоянием агента.
- `OllamaCoach` в [`ollama_coach.py`](ollama_coach.py): интерпретация команд, prompt-building, structured `action`, `goal`, `belief`, `behavior`.
- Layered brain-study утилита [`ollama_brain_lab.py`](ollama_brain_lab.py): поэтапное изучение мозга агента, `brain_map`, resource budget, `resume`, `checkpoint`, `completed_with_fallback`.
- Профилировщик мозга [`ollama_brain_profile.py`](ollama_brain_profile.py): разрезание brain-state по слоям `identity / beliefs / memory / commands / dialogue / emotion`.
- Launcher [`run_ollama_brain_lab`](run_ollama_brain_lab): one-file запуск полного процесса с проверкой Python, проекта и Ollama runtime.
- Wrapper [`train_agent_brain`](train_agent_brain): одна команда для обучения Ollama на мозге `agent_1`.
- Headless боевая утилита [`morpheus_dojo.py`](morpheus_dojo.py): терминальный live-dashboard для sparring/wolf drill без нагрузки на 3D-движок.
- Wrapper [`train_combat_dojo`](train_combat_dojo): прямой запуск терминального dojo.
- Wrapper [`train_agent_combat`](train_agent_combat): одна команда для циклического боевого обучения `agent_1` до mastery-порога.
- Автоустановка Ollama в [`install.py`](install.py), если локальный runtime отсутствует.

**Полное описание методов и сценариев:**

- [`docs/OLLAMA_BRAIN_METHODS_RU.md`](docs/OLLAMA_BRAIN_METHODS_RU.md)

---

## Последние визуальные улучшения

Добавлено в `engine3d.py`:

- Солнце в небе (объёмное ядро, мягкий ореол, лучи), которое видно на карте.
- Дневной цикл света:
  - пересчёт направления солнечного света по времени;
  - изменение температуры света (теплее на низком солнце, холоднее в верхней фазе).
- Обновлённый sky clear color: фон кадра теперь синхронизирован с текущим состоянием солнца.
- Усилена читаемость сцены: освещение пола/сетки и мешей теперь учитывает направление солнца.
- Маркер Морфеуса в 3D для боевого режима комнаты.
- Morpheus room получила отдельную постановку и распознаётся движком как специальная учебная зона.

Также обновлено в `combined_app.py` и `engine3d.py`:

- Разведены частоты UI, trainer tick и snapshot-push, чтобы 3D не дёргался при обновлении мозга и мира.
- Добавлен adaptive quality в 3D: движок поджимает второстепенные gizmo/LOD при просадке FPS.
- Оптимизирован resize-path интерфейса: debounced overlay relayout, cached world-map background и более лёгкий `QSplitter` без тяжёлого live-resize.

Параметры для настройки:

- `SUN_CYCLE_SEC` — длительность цикла солнца.
- `SUN_MIN_ELEV_DEG` / `SUN_MAX_ELEV_DEG` — диапазон высоты солнца.
- `SUN_BASE_HEIGHT` — базовая высота траектории солнца над картой.

---

## Быстрый старт

**Зависимости (минимум):**
- Python **3.11+**
- **PySide6**
- **PyOpenGL**
- **numpy**
- *(опционально)* **moderngl / glfw / pyrr** — если используются в вашем рендере

**Установка окружения (пример с `uv`):**
```bash
uv venv && source .venv/bin/activate
uv pip install PySide6 PyOpenGL numpy
# при необходимости
uv pip install moderngl glfw pyrr
```

**Рекомендуемая установка проекта с автоустановкой Ollama:**
```bash
python install.py
```

Скрипт:
- устанавливает Python-зависимости из `requirements.txt`;
- проверяет наличие `ollama` в системе;
- если Ollama отсутствует, скачивает и запускает официальный инсталлятор для текущей ОС.

Полезные флаги:
```bash
python install.py --skip-ollama
python install.py --skip-python
python install.py --ollama-version=<version>
python install.py --no-ollama-start
```

**Запуск:**
```bash
python combined_app.py
```
Если у вас другой entry-point — запустите соответствующий файл.

**Терминальный боевой dojo без 3D:**
```bash
train_agent_combat
```

Полезные варианты:
```bash
train_agent_combat --lesson sparring --until-mastery --ticks 6000
train_agent_combat --lesson wolf --until-mastery --ticks 6000
train_agent_combat --mastery-skill 4.5 --mastery-mentor-hits 6 --mastery-wolf-hits 10
train_agent_combat --plain
```

---

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
    ├── trainer_side.py
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

---

## Архитектура

Проект состоит из трёх слоёв:

1. **Mind Trainer** — логика локального мира/эволюции, хранит `world`, испускает сигнал `world_changed`.
2. **Мост (Trainer → Engine)** — формирует *снапшот* (словарь) из состояния `world` и вызывает `engine.sync_from_world(snapshot)`.
3. **3D-движок + Виджет** — `MiniMatrixEngine` + `World3DView` (QOpenGLWidget) отрисовывают сцену и агентов.

Панели GUI:
- `TrainerStatsWidget` — метрики/эволюция.
- `AgentBrainWidget` — инспектор текущего агента (beliefs, memory, rules).

Боевой модуль:
- `CombatSystem` — управляет сущностями боя в `world`, спавнит волны, обновляет их состояние на каждом тике.

Диаграмма данных/сигналов:
```mermaid
flowchart LR
  A[MindTrainerInteractive (world)] -- world_changed --> B[TrainerToEngineBridge]
  B -- _build_engine_snapshot(world) --> C[MiniMatrixEngine.sync_from_world]
  C --> D[World3DView (QOpenGLWidget).render_opengl()]
  D -->|клики/ПКМ, Tab, F, R| A
```
---

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

---

## Боевой модуль

`combat_system.py` управляет боевыми сущностями и взаимодействиями:

- `spawn_wave(kind, n=3, around=(x,y))` — создать волну (например, `wolf`).
- `step(dt)` — один шаг симуляции (~0.05 s).
- Интеграция с `world`: животные/агенты получают урон/состояния и отражаются в снапшоте.

В `combined_app.py` модуль тикает таймером (~20 Гц) и после каждого шага публикует свежий снапшот в 3D.

---

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
      "hunger": 0.0},
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
**Важно:** при изменении формата — адаптируйте `_build_engine_snapshot(...)` и `MiniMatrixEngine.sync_from_world(...)`.

---

## Ключевые модули

| Модуль | Назначение |
|---|---|
| `combined_app.py` | Точка входа с окном из трёх колонок (Stats | 3D | Brain), «стеклянным» UI и хоткеями. |
| `engine3d.py` | 3D-движок и сущности рендера. Ожидает снапшоты мира и умеет `render_opengl()`. |
| `env_lowpoly.py` | Генерация низкополигонального окружения (посёлок/виллэдж). |
| `combat_system.py` | Примитивная боевая система: волны, шаг, взаимодействия. |
| `mind_trainer_gui.py` | Мир тренера, метрики, инспектор мозга, сигнал `world_changed`. |
| `mind_core.py` | Психика/поведение агентов (параметры, эмоции, память). |
| `world.py` | Базовые структуры мира, агенты/объекты/животные. |

---

## Как расширять

### Добавить новый вид животного
1. Определите **species** (`base_hp`, `aggressive`, `tamable`, …).
2. Добавьте экземпляры в `world.animals` и обновляйте их в игровом цикле.
3. Отразите их в снапшоте (`animals`), чтобы они визуализировались в 3D.

### Добавить правило поведения / эмоцию
1. Расширьте `mind_core` (beliefs/behavior_rules/current_drive).
2. Прокиньте новые поля в `_brain_to_dict(...)` и далее в снапшот.
3. В `AgentBrainWidget` отобразите/добавьте редакторы.

### Добавить объект мира
1. Создайте объект с `id/name/kind/pos/radius/...` и добавьте в `world.objects`.
2. Обновите логику взаимодействий (опасность/комфорт/баффы).
3. Проверьте визуализацию в `engine3d` (иконки/меши/цвета).

### Добавить кнопку/панель в UI
1. Оберните новый виджет в `make_card(title, widget)` для единого стиля.
2. Добавьте карточку в `QSplitter` и настройте `setStretchFactor`.

---

## Точки интеграции и сигналы

- `MindTrainerInteractive.world_changed: Signal()` — вызывается при изменениях мира; мост собирает снапшот.
- `World3DView.requestSetGoal(agent_id, x, z)` — эмитится при ПКМ по земле; окно транслирует в `world`.
- `MiniMatrixEngine.sync_from_world(snapshot)` — синхронизация состояния рендера с логикой мира.

---

## Конфигурация

Файл `config.py` и/или переменные окружения:

- `WORLD_WIDTH`, `WORLD_HEIGHT` — размеры мира.
- `MAX_TICKS_PER_EPOCH`, `DISASTER_INTERVAL_TICKS` — параметры тренера.
- `SAVE_DIR` — путь сохранений (мозги/лог/метрики).
- `RENDER_DEBUG` — режим отладки рендера.

Пример:
```bash
export MINI_MATRIX_SAVE_DIR="./runs"
export MINI_MATRIX_RENDER_DEBUG=1
python mini_matrix/combined_app.py
```

---

## Тестирование

- **Unit:** конвертер снапшотов (`_build_engine_snapshot`) — соответствие ожидаемой схеме.
- **Smoke:** запуск `combined_app.py` без падений, видимость сцены, реагирование хоткеев.
- **Интерактив:** ЛКМ выбрать → ПКМ поставить цель → увидеть изменения в инспекторе/мире.

CI (рекомендации): `pytest -q` + `xvfb-run` для headless-Qt.

---

## Устранение неполадок

- **`AttributeError: QPalette has no attribute 'Window'` (Qt6)** — используйте `QPalette.ColorRole.Window` и соответствующие методы `setColor(...)`. В актуальной версии уже исправлено.
- **`AA_UseHighDpiPixmaps is deprecated` (Qt6)** — можно опустить; Qt6 корректно работает с HiDPI.
- **Чёрный экран** — проверьте, что `render_opengl()` вызывается, а камера корректно настроена; обновите драйверы.
- **Wayland/Linux** — при старых драйверах попробуйте X11-сессию.
- **Нет `moderngl/glfw`** — установите пакеты или отключите расширенный рендер.

---

## Оптимизация и производительность

- Снижайте число draw-calls (batching/instancing).
- Разделяйте частоту логики и рендера (tick vs. frame).
- Кэшируйте меши окружения; используйте LOD/фрустум-клиппинг.
- Профилируйте: FPS, время кадра, количество сущностей.

---

## Достижения и метрики

- Интерактивный 3D-интерфейс с live-синхронизацией «логика → рендер».
- Модульная психика: страх, энергия, голод, память, убеждения.
- Боевая система с волнами и телеметрией состояния.
- Экспорт/импорт «мозгов» (JSON), reproducible-запуски (`seed`).
- Готовность к интеграции с RL/LLM-плагинами.

> Для грантов/презентаций добавьте графики улучшения `survival_score`, FPS, кол-во агентов, время кадра, скриншоты сцен.

---

## Roadmap

- ⏺ Запись/воспроизведение сессий, экспорт видео/GIF.
- 🧭 HUD/mini-map, режим «follow agent».
- 🧪 Набор стандартных сценариев (benchmarks).
- 🧠 Расширение EmotionCascade, новые архетипы поведения.
- 🧩 Плагины: внешние модули обучения/аналитики.
- 💾 Персистентность мира и долговременная память.

---

## Лицензия (в одном файле)

Проект распространяется по **двойной модели** лицензирования (код и ассеты) — описание здесь, без отдельных файлов.

### 1) Исходный код — **AGPL-3.0-only** (SPDX: `AGPL-3.0-only`)

Коротко: можно использовать/изменять/разворачивать как сервис, при этом **все производные работы и серверные модификации** также распространяются по AGPL-3.0, с доступным исходным кодом пользователям сервиса. Сохраняйте уведомления об авторстве и лицензии.

Рекомендуемая шапка в исходных файлах:
```
Copyright (C) 2025 Oleg Leinweber

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License version 3
as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
```

### 2) Медиа/ассеты — **CC BY-NC-ND 4.0**

Распространяется на: изображения, иллюстрации, логотипы, иконки, скриншоты и иные графические материалы, если явно помечены как ассеты.

Коротко: **BY** — указывайте автора; **NC** — без коммерческого использования; **ND** — без производных.

Формулировка:
```
Selected media assets © 2025 Oleg Leinweber,
licensed under Creative Commons BY-NC-ND 4.0.
```

### 3) Коммерческая лицензия / двойное лицензирование

Для коммерческого использования ассетов и интеграций с закрытыми модулями доступна **отдельная коммерческая лицензия**. Свяжитесь с автором (см. «Контакты») для согласования условий.

---

## Цитирование

Если вы используете Mini-Matrix Lab в публикациях/отчётах, укажите репозиторий и версию релиза (tag).

```bibtex
@misc{MiniMatrixLab,
  title        = {Mini-Matrix Lab: Interactive 3D World + Mind Trainer},
  author       = {Oleg Leinweber},
  year         = {2025},
  howpublished = {\url{<REPO_URL>}},
  note         = {Version <TAG>}
}
```

---

## Контакты

Автор: **Oleg Leinweber**  
Telegram ID (для ботов/интеграций): `2028648036`  
Вопросы по коммерческой лицензии и грантовым коллаборациям — пишите в личные сообщения.

---

## Приложение A — минимальные требования

```text
PyOpenGL
PySide6
numpy
# (опц.) moderngl
# (опц.) glfw
# (опц.) pyrr
```

---

## Приложение B — быстрый шаблон requirements.txt

> Скопируйте при необходимости в `mini_matrix/requirements.txt`.

```
PySide6>=6.6
PyOpenGL>=3.1.6
numpy>=1.26

# Optional for advanced rendering / windowing
moderngl>=5.8 ; platform_system != "Darwin"
glfw>=2.7.0   ; platform_system != "Darwin"
pyrr>=0.10.3
```

---

## Приложение C — чек-лист грантовой заявки

- [ ] **Научная новизна:** модульная психика (страх/память/убеждения), live-синхронизация логики и рендера.  
- [ ] **Социальная значимость:** исследования поведения, обучение, демонстрации безопасного взаимодействия агент-среда.  
- [ ] **Прототип/демо:** `combined_app.py` — интерактивная сцена, метрики и инспектор.  
- [ ] **Метрики:** survival_score, FPS, кол-во агентов, время кадра, успешность сценариев.  
- [ ] **Планы развития (Roadmap):** запись/воспроизведение, HUD, плагины, персистентность.  
- [ ] **Лицензии:** код — AGPL-3.0-only; ассеты — CC BY-NC-ND 4.0; коммерческая лицензия — по запросу.  
- [ ] **Команда и компетенции:** Python/Qt/OpenGL, процедурная генерация, моделирование поведения.  
- [ ] **Риски и меры:** кроссплатформа (Linux/Windows/Mac), fallback GL, модульность, тесты.  
- [ ] **Бюджет:** оборудование (GPU), UI/GL-инженер, научный консультант по когнитивным моделям.

---

## Приложение D — обновления проекта (23 февраля 2026)

Ниже собраны новые возможности, которые были добавлены поверх базовой архитектуры.

### 1) Боевая эволюция агентов

- Агенты теперь умеют не только избегать угрозы, но и контратаковать в ближнем бою.
- Добавлен боевой прогресс: `combat_skill` растет на боевом опыте.
- Решение о контратаке зависит от состояния агента (здоровье, страх, уверенность).

**Файлы:** `agent.py`, `world.py`, `combat_system.py`

### 2) Приручение волков и помощь питомцев

- Реализован отдельный модуль приручения волков.
- Попытки приручения дают прогресс даже при неудачах.
- Прирученные волки защищают хозяина и атакуют ближайшие угрозы.

**Файлы:** `wolf_taming_system.py`, `world.py`

### 3) Передача опыта приручения между агентами

- Добавлено социальное обучение: союзники рядом получают часть опыта от агента, который взаимодействовал с волком.
- В аналитике обучения это отражается как `peer_tame_lessons` / `peer_shared_lessons`.

**Файлы:** `wolf_taming_system.py`, `learning_insights.py`, `mind_trainer.py`, `mind_trainer_gui.py`

### 4) Размножение агентов и новое поколение

- Вынесен отдельный модуль размножения: поиск партнера, условия зрелости и ресурсов, рождение потомка.
- Потомок наследует часть боевого/приручающего опыта и когнитивных параметров родителей.
- Введены lineage-роли поколений: `balanced`, `scout`, `protector`, `tamer`, `medic`.
- Добавлена мутация роли с вероятностью `AGENT_REPRO_ROLE_MUTATION_PROB`.

**Файлы:** `agent_reproduction_system.py`, `world.py`, `config.py`

### 5) Видимость обучения в trainer и GUI

В тренере явно показывается:

- чему агент научился (`learned` / `learned_skills`);
- какие связи усилил (`strengthened_links`);
- где пересмотрел или изменил убеждения (`changed_beliefs` / `belief_changes`).

Плюс добавлена семейная аналитика lineage:

- поколение (`generation`), родители (`parents`), роль рода (`lineage_role`);
- дети, сиблинги, недавние события семьи.

**Файлы:** `learning_insights.py`, `mind_trainer.py`, `mind_trainer_gui.py`

### 6) Двуязычный интерфейс trainer GUI

- Добавлен переключатель языка интерфейса: `Русский / English`.
- Локализация охватывает карточки метрик, сигналы обучения и lineage-панель.

**Файл:** `mind_trainer_gui.py`

### 7) Новые события мира и памяти

Ключевые события, которые теперь пишутся в `event_log` и/или память агентов:

- `tame_success`
- `tame_progress`
- `tame_peer_share`
- `pet_defend`
- `seek_partner`
- `agent_birth`
- `offspring_born`
- `lineage_inherit`

### 8) Новые параметры конфигурации

Основные добавленные группы флагов в `config.py`:

- блок приручения и защиты питомцами: `WOLF_TAME_*`, `WOLF_DEFEND_*`;
- блок передачи опыта между союзниками: `WOLF_TAME_SHARE_*`;
- блок размножения и наследования: `AGENT_REPRO_*`.

### 9) Артефакты и логи обучения

После запуска тренеров используются/создаются:

- `trainer_logs/epoch_*/monitor.jsonl`
- `trainer_logs/epoch_*/learning_signals.jsonl`
- `trainer_logs/monitor_history.csv`
- `trainer_snapshots/epoch_*/t_XXXX.json`
- `trained_brains/*.mind.json`
- `brains/*.json`

### 10) Быстрый запуск новых режимов

```bash
# Оффлайн-тренер (метрики, learning signals, логи)
python mind_trainer.py

# Интерактивный GUI-тренер (RU/EN + lineage + сигналы обучения)
python mind_trainer_gui.py

# Объединенное приложение (3D + trainer панели)
python combined_app.py
```
