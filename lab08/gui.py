import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QLineEdit, QPushButton, 
                             QLabel, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Импортируем симуляцию из первого файла
from logic import FlowCalulations

class MplCanvas(FigureCanvas):
    """Класс для интеграции графиков matplotlib в PyQt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Моделирование пуассоновского потока")
        self.resize(1000, 600)

        # Основной виджет и слой (горизонтальный: слева панель управления, справа график)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Левая панель параметров
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=1)

        # Форма ввода параметров
        form_layout = QFormLayout()
        
        self.input_N = QLineEdit("10000")
        self.input_T = QLineEdit("5.0")
        self.input_lambda = QLineEdit("2.0")

        form_layout.addRow(QLabel("Число экспериментов (N):"), self.input_N)
        form_layout.addRow(QLabel("Интервал времени (T):"), self.input_T)
        form_layout.addRow(QLabel("Интенсивность (lambda):"), self.input_lambda)
        
        left_panel.addLayout(form_layout)

        # Кнопка запуска
        self.btn_run = QPushButton("Начать симуляцию")
        self.btn_run.setStyleSheet("font-weight: bold; height: 35px;")
        self.btn_run.clicked.connect(self.run_simulation)
        left_panel.addWidget(self.btn_run)

        # Текстовое поле для вывода числовых характеристик
        left_panel.addWidget(QLabel("Характеристики:"))
        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        left_panel.addWidget(self.text_output)

        # Правая панель с графиком
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        main_layout.addWidget(self.canvas, stretch=2)

    def run_simulation(self):
        # 1. Считывание и валидация данных
        try:
            N = int(self.input_N.text())
            T = float(self.input_T.text())
            intensity = float(self.input_lambda.text())

            if N <= 0 or T <= 0 or intensity <= 0:
                raise ValueError("Все значения должны быть больше нуля.")
        except ValueError as e:
            QMessageBox.critical(self, "Ошибка ввода", f"Некорректные параметры: {e}")
            return

        # 2. Выполнение симуляции
        calculator = FlowCalulations(N=N, T=T, intensity=intensity)
        frequencies = calculator.simulate()
        stats = calculator.get_statistics(frequencies)

        if not stats:
            QMessageBox.warning(self, "Ошибка", "Не удалось рассчитать статистику.")
            return

        # 3. Вывод текстовых результатов
        self.display_statistics(stats)

        # 4. Построение гистограммы распределения
        self.plot_distribution(stats)

    def display_statistics(self, stats: dict):
        self.text_output.clear()
        info = (
            f"ЭМПИРИЧЕСКИЕ ХАРАКТЕРИСТИКИ:\n"
            f"Среднее число заявок: {stats['empirical_mean']:.4f}\n"
            f"Дисперсия: {stats['empirical_variance']:.4f}\n\n"
            f"ТЕОРЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:\n"
            f"Математическое ожидание: {stats['theoretical_mean']:.4f}\n"
            f"Дисперсия: {stats['theoretical_variance']:.4f}\n\n"
            f"Разность средних: {abs(stats['empirical_mean'] - stats['theoretical_mean']):.4f}\n"
            f"Разность дисперсий: {abs(stats['empirical_variance'] - stats['theoretical_variance']):.4f}"
        )
        self.text_output.setText(info)

    def plot_distribution(self, stats: dict):
        # Очистка предыдущего графика
        self.canvas.axes.cla()

        counts = stats["counts"]
        empirical_probs = stats["empirical_probs"]
        theoretical_probs = stats["theoretical_probs"]

        # Построение эмпирического распределения (столбчатая диаграмма)
        self.canvas.axes.bar(counts, empirical_probs, alpha=0.6, color='#3498db', 
                             edgecolor='black', label='Эмпирическое')

        # Построение теоретического распределения Пуассона (линия с точками)
        self.canvas.axes.plot(counts, theoretical_probs, 'o-', color='#e74c3c', 
                             linewidth=2, markersize=6, label='Теоретическое (Пуассон)')

        # Оформление графика
        self.canvas.axes.set_title("Сравнение эмпирического и теоретического распределений")
        self.canvas.axes.set_xlabel("Количество заявок за интервал T")
        self.canvas.axes.set_ylabel("Вероятность")
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)

        # Обновление холста
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())