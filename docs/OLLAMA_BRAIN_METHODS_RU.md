# Новые Методы И Утилиты CSEN

> Краткая карта новых механизмов, добавленных для комнаты Морфеуса, прямого управления агентом через Ollama и поэтапного изучения мозга агента.

## Что это добавляет проекту

Новый контур делает три вещи:

1. Даёт отдельную безопасную комнату Морфеуса для controlled training одного агента.
2. Подключает Ollama не сбоку, а прямо к мозгу агента.
3. Добавляет layered brain-study pipeline, который постепенно строит `brain_map` и учит интерпретировать команды оператора.

Комната теперь по умолчанию работает с `agent_1`, а его brain автоматически подхватывается из [`brains/agent_1.json`](../brains/agent_1.json).

---

## Комната Морфеуса

Файл: [`training_room.py`](../training_room.py)

### Основные публичные методы `TrainingRoomManager`

- `attach_world(world, preferred_agent_id=None, announce=False)`
  Подключает комнату к миру, создаёт room-zone, выбирает лабораторного агента и помещает его внутрь.

- `assign_agent(world, agent_id, announce=True)`
  Назначает конкретного агента лабораторным и переносит его в комнату.

- `release_agent(world, announce=True)`
  Выпускает обученного агента обратно в общий мир и сохраняет room-brain.

- `maintain_world(world)`
  Поддерживает комнату в безопасном состоянии: выталкивает опасности, животных и других агентов, стабилизирует лабораторного агента.

- `bounds_for(world)`
  Возвращает геометрию комнаты.

- `room_center(world)`
  Возвращает центр комнаты.

- `clamp_point_for_agent(world, agent_id, x, y)`
  Ограничивает движение лабораторного агента только пространством комнаты.

- `last_room_brain_path()`
  Возвращает путь к последнему сохранённому room-brain файлу.

### Что ещё добавлено

- Room-only окно `MorpheusRoomWindow`.
- Старт комнаты без полной карты мира.
- Автовыбор `agent_1` как основного лабораторного агента.
- Отдельный room-brain export в [`room_brains/`](../room_brains/).

---

## Переопределение Стартового Lineup

Файл: [`combined_app.py`](../combined_app.py)

### Новый хук

- `CombinedMainWindow._initial_agent_lineup()`
  Возвращает стартовый lineup агентов для окна. В базовом приложении оставляет `Echo`, `Nova` и `A0`, а в room-only режиме переопределяется так, чтобы первым шёл `agent_1`.

Это позволило не ломать обычный showcase-мир и при этом запускать комнату Морфеуса на другом составе агентов.

---

## Прямая Интеграция Ollama С Мозгом

Файл: [`ollama_brain_service.py`](../ollama_brain_service.py)

### Основные публичные методы `OllamaBrainService`

- `attach_world(world, training_room=None)`
  Привязывает сервис к миру и комнате.

- `configure_brain(brain, ...)`
  Включает Ollama-контур для конкретного мозга агента и задаёт параметры работы.

- `queue_instruction(brain, text, tick=None)`
  Ставит новую операторскую инструкцию прямо в очередь мозга.

- `force_request(brain)`
  Принудительно запускает обработку очередной инструкции.

- `before_world_tick(world)`
  Вызывается до шага мира. Готовит активные команды, authoritative override, локальные планы и запросы к модели.

- `after_world_tick(world)`
  Вызывается после шага мира. Завершает применение команд, пишет feedback и журнал действий.

### Какие механики появились

- Authoritative operator override.
- Очередь локального плана для составных команд.
- Structured actions: движение к ориентиру, поворот, ожидание, остановка, заметка в память, изменение скорости, эмоциональная настройка.
- Локальное исполнение простых команд без сетевого запроса к модели.
- Журнал `YOU / OLLAMA / AGENT / SYSTEM / MODEL / RECALL / PLAN`.

---

## Интерпретация Команд И Prompt-Логика

Файл: [`ollama_coach.py`](../ollama_coach.py)

### Основные публичные методы `OllamaCoach`

- `describe()`
  Возвращает краткое описание активной модели и host.

- `refresh_model()`
  Обновляет модель и runtime-конфигурацию.

- `request_advice(snapshot)`
  Получает training snapshot, строит prompt, вызывает Ollama и возвращает структурированный `CoachAdvice`.

### Что улучшено

- Нормализация `goal`, `belief`, `behavior`, `action`.
- Поддержка structured `action` вместо одного только `goal`.
- Локальное распознавание базовых операторских команд.
- Подмешивание retrieval-контекста из brain-срезов и прошлых удачных команд.

---

## Layered Brain Study

Файлы:

- [`ollama_brain_lab.py`](../ollama_brain_lab.py)
- [`ollama_brain_profile.py`](../ollama_brain_profile.py)

### Что делает `ollama_brain_profile.py`

- Разбивает brain-state на слои:
  - `identity`
  - `beliefs`
  - `memory`
  - `commands`
  - `dialogue`
  - `emotion`
- Формирует payload и stage-набор для обучения.
- Готовит компактные локальные сводки для resource-aware режима.

### Что делает `ollama_brain_lab.py`

- Запускает пошаговое изучение мозга агента.
- Формирует промежуточную `brain_map`.
- Пишет `progress`, `checkpoint` и итоговый `brain_profile`.
- Поддерживает `resume`, `fresh-start`, `paused_resource`, `completed_with_fallback`.
- Разделяет ресурсы по слоям и budget-unit’ам, чтобы не упираться в длинные таймауты.

### Ключевые режимы

- `--resume`
  Продолжить с последнего checkpoint.

- `--fresh-start`
  Начать обучение заново.

- `--low-resource`
  Включить бюджетный режим с локальными digest-этапами и компактными layer-synthesis вызовами.

- `--max-units-per-run`
  Ограничить число unit-ов за один запуск.

- `--max-request-timeout`
  Жёстко ограничить время одного запроса к Ollama.

### Что появляется на выходе

- [`ollama_brain_reports/*.brain_progress.json`](../ollama_brain_reports/)
- [`ollama_brain_reports/*.brain_checkpoint.json`](../ollama_brain_reports/)
- [`ollama_brain_reports/*.brain_profile.json`](../ollama_brain_reports/)

---

## Командные Утилиты

### `run_ollama_brain_lab`

Файл: [`run_ollama_brain_lab`](../run_ollama_brain_lab)

Назначение:

- Находит Python.
- Проверяет проектное окружение.
- При необходимости вызывает [`install.py`](../install.py).
- Проверяет наличие Ollama.
- Поднимает `ollama serve`, если нужен локальный runtime.
- После этого запускает [`ollama_brain_lab.py`](../ollama_brain_lab.py).

Примеры:

```bash
./run_ollama_brain_lab a1
./run_ollama_brain_lab --brain-file room_brains/agent_1.room_ollama.json
./run_ollama_brain_lab --brain-id agent_1 --model llama3.2:latest
```

### `train_agent_brain`

Файл: [`train_agent_brain`](../train_agent_brain)

Назначение:

- Один запуск для brain-study по умолчанию на [`brains/agent_1.json`](../brains/agent_1.json).
- Автоматически включает `resume`.
- По умолчанию использует low-resource режим.

Примеры:

```bash
train_agent_brain
train_agent_brain --fresh-start
train_agent_brain --model llama3.2:latest
train_agent_brain --max-request-timeout 45 --max-units-per-run 4 --cooldown-sec 2
```

---

## Автоустановка Ollama

Файл: [`install.py`](../install.py)

### Что добавлено

- Проверка наличия `ollama` в системе.
- Автоматический запуск официального инсталлятора Ollama, если runtime отсутствует.
- Поддержка совместной установки Python-зависимостей проекта и Ollama.

Это позволяет запускать room-only режим и brain-lab без ручной подготовки машины.

---

## Практический Сценарий

1. Запустить комнату Морфеуса через [`training_room.py`](../training_room.py).
2. Обучать `agent_1` через встроенную Ollama-консоль.
3. Сохранять room-brain в [`room_brains/`](../room_brains/).
4. При необходимости отдельно прогонять layered brain-study через `train_agent_brain`.
5. Использовать получившийся `brain_profile` и встроенную Ollama-карту мозга для более точного исполнения команд и управления эмоциями агента.
