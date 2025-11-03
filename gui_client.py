import sys
import requests
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QGraphicsLineItem, QGraphicsRectItem
)
from PySide6.QtCore import QTimer, QRectF, Qt, QPointF
from PySide6.QtGui import QColor, QBrush, QPen, QFont, QPainter


SERVER_URL = "http://localhost:8000"
POLL_MS = 500          # как часто опрашиваем /state
AGENT_RADIUS = 3.0     # базовый радиус кружка агента
HP_BAR_WIDTH = 16.0    # ширина полоски HP
HP_BAR_HEIGHT = 3.0    # высота полоски HP
VEL_ARROW_SCALE = 4.0  # масштаб стрелки скорости


def color_for_agent(hp: float, fear: float) -> QColor:
    """
    Подбираем цвет агента:
    - красный, если низкое HP
    - оранжевый, если паника
    - синий обычно
    """
    if hp < 30:
        return QColor(255, 0, 0)          # тяжело ранен
    if fear > 0.6:
        return QColor(255, 150, 0)        # паника
    return QColor(0, 150, 255)            # норм


class WorldView(QGraphicsView):
    """
    2D-карта мира:
    - границы мира
    - опасные и безопасные зоны с подписями
    - агенты (кружок, стрелка направления движения, hp-бар, подпись)
    """

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing, True)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # реальные размеры мира из снапшота
        self.world_w = 100.0
        self.world_h = 100.0

        # чтобы подписи читались
        self.label_font = QFont()
        self.label_font.setPointSizeF(7.5)

    def _draw_world_bounds(self):
        """
        Рисуем тёмный фон и белую рамку мира.
        """
        border_rect = self.scene.addRect(
            QRectF(0, 0, self.world_w, self.world_h),
            QPen(Qt.white),
            QBrush(Qt.black)
        )
        border_rect.setZValue(-100)

    def _draw_objects(self, objects_data):
        """
        Рисуем зоны окружения.
        - hazard: красная полупрозрачная окружность
        - safe: зелёная полупрозрачная окружность
        - neutral: сероватая
        Добавляем текстовую подпись с именем объекта.
        """
        for obj in objects_data:
            ox = obj["pos"]["x"]
            oy = obj["pos"]["y"]
            r = obj["radius"]
            kind = obj["kind"]
            name = obj.get("name", "")

            if kind == "hazard":
                fill = QColor(255, 0, 0, 70)
                outline = QColor(255, 0, 0)
                pen = QPen(outline, 0.5, Qt.DashLine)
            elif kind == "safe":
                fill = QColor(0, 255, 0, 60)
                outline = QColor(0, 200, 0)
                pen = QPen(outline, 0.5, Qt.SolidLine)
            else:
                fill = QColor(200, 200, 200, 40)
                outline = QColor(200, 200, 200)
                pen = QPen(outline, 0.5, Qt.DotLine)

            zone_item = QGraphicsEllipseItem(
                QRectF(ox - r, oy - r, r * 2, r * 2)
            )
            zone_item.setBrush(QBrush(fill))
            zone_item.setPen(pen)
            zone_item.setZValue(-10)
            self.scene.addItem(zone_item)

            # подпись объекта (имя)
            label_item = self.scene.addText(name, self.label_font)
            label_item.setDefaultTextColor(outline)
            # чутка сдвинем подпись вверх над центром
            label_item.setPos(ox - r, oy - r - 10)
            label_item.setZValue(-5)

    def _draw_agent(self, agent):
        """
        Рисуем одного агента:
        - сам кружок (цвет по состоянию)
        - стрелка направления движения
        - hp-бар
        - подпись (имя, hp, fear)
        """
        ax = float(agent["pos"]["x"])
        ay = float(agent["pos"]["y"])
        hp = float(agent["health"])
        fear = float(agent["fear"])
        name = agent["name"]

        vx = float(agent.get("vel", {}).get("x", 0.0))
        vy = float(agent.get("vel", {}).get("y", 0.0))

        agent_color = color_for_agent(hp, fear)

        # кружок агента
        body_item = QGraphicsEllipseItem(
            QRectF(ax - AGENT_RADIUS, ay - AGENT_RADIUS,
                   AGENT_RADIUS * 2, AGENT_RADIUS * 2)
        )
        body_item.setBrush(QBrush(agent_color))
        body_item.setPen(QPen(Qt.white, 0.5))
        body_item.setZValue(10)
        self.scene.addItem(body_item)

        # стрелка направления скорости (если есть движение)
        speed_len = (vx ** 2 + vy ** 2) ** 0.5
        if speed_len > 0.01:
            line_item = QGraphicsLineItem(
                ax, ay,
                ax + vx * VEL_ARROW_SCALE,
                ay + vy * VEL_ARROW_SCALE
            )
            line_item.setPen(QPen(QColor(255, 255, 0), 0.6))
            line_item.setZValue(11)
            self.scene.addItem(line_item)

        # полоска HP над агентом
        # нормализуем hp 0..100 к ширине HP_BAR_WIDTH
        hp_norm = max(0.0, min(1.0, hp / 100.0))
        bar_w = HP_BAR_WIDTH * hp_norm
        bar_x = ax - HP_BAR_WIDTH / 2.0
        bar_y = ay - AGENT_RADIUS - HP_BAR_HEIGHT - 2.0

        # фон hp-бара (серый)
        hp_bg = QGraphicsRectItem(
            QRectF(bar_x, bar_y, HP_BAR_WIDTH, HP_BAR_HEIGHT)
        )
        hp_bg.setBrush(QBrush(QColor(40, 40, 40)))
        hp_bg.setPen(QPen(Qt.white, 0.3))
        hp_bg.setZValue(12)
        self.scene.addItem(hp_bg)

        # актуальное HP (зел/желт/красн)
        if hp > 60:
            hp_col = QColor(0, 255, 0)
        elif hp > 30:
            hp_col = QColor(255, 200, 0)
        else:
            hp_col = QColor(255, 0, 0)

        hp_fg = QGraphicsRectItem(
            QRectF(bar_x, bar_y, bar_w, HP_BAR_HEIGHT)
        )
        hp_fg.setBrush(QBrush(hp_col))
        hp_fg.setPen(QPen(Qt.transparent))
        hp_fg.setZValue(13)
        self.scene.addItem(hp_fg)

        # подпись агента (имя, hp, fear)
        label_text = f"{name}\nHP:{hp:.0f} F:{fear:.2f}"
        label_item = self.scene.addText(label_text, self.label_font)
        label_item.setDefaultTextColor(Qt.white)
        label_item.setPos(ax + AGENT_RADIUS + 3, ay + AGENT_RADIUS + 3)
        label_item.setZValue(20)
        self.scene.addItem(label_item)

    def _draw_agents(self, agents_data):
        for agent in agents_data:
            self._draw_agent(agent)

    def update_world(self, snapshot: dict):
        """
        Полное обновление сцены:
        - очищаем
        - читаем габариты мира
        - рисуем всё заново
        - делаем fitInView, чтобы мир целиком влезал
        """
        self.scene.clear()

        # размеры мира (нужны и для рамки, и для fitInView)
        world_info = snapshot.get("world", {})
        self.world_w = float(world_info.get("width", 100))
        self.world_h = float(world_info.get("height", 100))

        # фон / границы
        self._draw_world_bounds()

        # окружение (зоны)
        objects_data = snapshot.get("objects", [])
        self._draw_objects(objects_data)

        # агенты
        agents_data = snapshot.get("agents", [])
        self._draw_agents(agents_data)

        # авто-масштаб, чтобы всегда видеть весь мир
        # (оставляем небольшой отступ)
        view_rect = QRectF(
            -5,
            -5,
            self.world_w + 10,
            self.world_h + 10
        )
        self.setSceneRect(view_rect)
        self.fitInView(view_rect, Qt.KeepAspectRatio)


class MainWindow(QWidget):
    """
    Главное окно наблюдателя:
    - слева карта мира (WorldView)
    - справа чат + статус симуляции (tick)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-Matrix Viewer")

        self.world_view = WorldView()

        # чат / лог общения агентов
        self.chat_box = QTextEdit()
        self.chat_box.setReadOnly(True)
        self.chat_box.setPlaceholderText("Здесь будет диалог агентов...")

        # информация о тике симуляции
        self.tick_label = QLabel("tick: ?")

        # заголовки справа и слева
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Мир"))
        left_col.addWidget(self.world_view, stretch=1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Чат"))
        right_col.addWidget(self.chat_box, stretch=1)
        right_col.addWidget(QLabel("Симуляция"))
        right_col.addWidget(self.tick_label)

        root = QHBoxLayout()
        root.addLayout(left_col, stretch=3)
        root.addLayout(right_col, stretch=2)
        self.setLayout(root)

        # таймер опроса бэкенда
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_state)
        self.timer.start(POLL_MS)

    def refresh_state(self):
        """
        Дёргаем /state, обновляем карту, чат и tick-лейбл.
        """
        try:
            r = requests.get(f"{SERVER_URL}/state", timeout=0.4)
            snap = r.json()
        except Exception as e:
            # сервер недоступен — покажем сообщение и не упадём
            self.chat_box.setPlainText(f"[нет связи с сервером]\n{e}")
            self.tick_label.setText("tick: (нет связи)")
            return

        # карта
        self.world_view.update_world(snap)

        # чат
        chat_lines = snap.get("chat", [])
        self.chat_box.setPlainText("\n".join(chat_lines))

        # tick
        tick_val = snap.get("tick", "?")
        self.tick_label.setText(f"tick: {tick_val}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1000, 650)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
