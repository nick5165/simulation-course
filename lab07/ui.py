import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from config import STATES, COLORS
from model import MarkovModel
from logger import CSVLogger

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Марковская модель погоды (CTMC)")
        self.geometry("1100x650")
        
        self.model = MarkovModel()
        self.logger = CSVLogger()
        self.is_running = False
        
        self.setup_ui()
        self.update_matrix_from_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === ЛЕВАЯ ПАНЕЛЬ ===
        self.left_frame = ctk.CTkFrame(self, width=300)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(self.left_frame, text="Матрица интенсивностей", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10))
        
        self.matrix_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.matrix_frame.pack(pady=10)
        
        self.entries = []
        labels = ["Ясно", "Облачно", "Пасмурно"]
        
        for i in range(3):
            row_entries = []
            ctk.CTkLabel(self.matrix_frame, text=labels[i]).grid(row=i+1, column=0, padx=5, pady=5)
            ctk.CTkLabel(self.matrix_frame, text=labels[i]).grid(row=0, column=i+1, padx=5, pady=5)
            for j in range(3):
                entry = ctk.CTkEntry(self.matrix_frame, width=50, justify="center")
                entry.grid(row=i+1, column=j+1, padx=5, pady=5)
                
                if i == j:
                    entry.insert(0, "0.0")
                    entry.configure(state="disabled", text_color="#ff6666")
                else:
                    default_vals = [[0, 0.2, 0.1], [0.3, 0, 0.4], [0.1, 0.5, 0]]
                    entry.insert(0, str(default_vals[i][j]))
                    entry.bind("<KeyRelease>", self.update_matrix_from_ui)
                row_entries.append(entry)
            self.entries.append(row_entries)

        self.speed_slider = ctk.CTkSlider(self.left_frame, from_=0.1, to=2.0, number_of_steps=19)
        self.speed_slider.set(0.5)
        ctk.CTkLabel(self.left_frame, text="Скорость симуляции").pack(pady=(20, 0))
        self.speed_slider.pack(pady=5)

        # Фрейм для кнопок управления
        self.buttons_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.buttons_frame.pack(pady=(20, 10))

        self.btn_start = ctk.CTkButton(self.buttons_frame, width=100, text="Старт", command=self.toggle_simulation, fg_color="green", hover_color="darkgreen")
        self.btn_start.grid(row=0, column=0, padx=5)

        self.btn_reset = ctk.CTkButton(self.buttons_frame, width=100, text="Сброс", command=self.reset_simulation, fg_color="gray", hover_color="darkgray")
        self.btn_reset.grid(row=0, column=1, padx=5)
        
        ctk.CTkLabel(self.left_frame, text="Данные автоматически\nпишутся в CSV", text_color="gray").pack(pady=10)

        # === ПРАВАЯ ПАНЕЛЬ ===
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.weather_frame = ctk.CTkFrame(self.right_frame, height=150)
        self.weather_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.day_label = ctk.CTkLabel(self.weather_frame, text="День: 0", font=ctk.CTkFont(size=24, weight="bold"))
        self.day_label.pack(pady=(20, 5))
        
        self.weather_label = ctk.CTkLabel(self.weather_frame, text="Ожидание...", font=ctk.CTkFont(size=36, weight="bold"))
        self.weather_label.pack(pady=10)

        self.plot_frame = ctk.CTkFrame(self.right_frame)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def update_matrix_from_ui(self, event=None):
        try:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        val = float(self.entries[i][j].get())
                        if val < 0: raise ValueError
                        self.model.q_matrix[i][j] = val
            
            diagonals = self.model.calculate_diagonals()
            
            for i in range(3):
                self.entries[i][i].configure(state="normal")
                self.entries[i][i].delete(0, "end")
                self.entries[i][i].insert(0, f"{diagonals[i]:.2f}")
                self.entries[i][i].configure(state="disabled")
            
            self.model.calculate_theoretical_pi()
            
            if not self.is_running:
                self.draw_plot()
        except ValueError:
            pass

    def toggle_simulation(self):
        if not self.is_running:
            self.is_running = True
            self.btn_start.configure(text="Пауза", fg_color="orange", hover_color="darkorange")
            self.btn_reset.configure(state="disabled") # Блокируем сброс во время работы
            
            for row in self.entries:
                for entry in row: entry.configure(state="disabled")
            
            if self.model.current_time == 0:
                self.logger.setup_file() 
            
            self.simulate_loop()
        else:
            self.is_running = False
            self.btn_start.configure(text="Продолжить", fg_color="green", hover_color="darkgreen")
            self.btn_reset.configure(state="normal") # Разрешаем сброс на паузе
            
            for i in range(3):
                for j in range(3):
                    if i != j: self.entries[i][j].configure(state="normal")

    def reset_simulation(self):
        """Полностью сбрасывает симуляцию на 0-й день"""
        self.is_running = False
        self.btn_start.configure(text="Старт", fg_color="green", hover_color="darkgreen")
        
        # Разблокируем ввод
        for i in range(3):
            for j in range(3):
                if i != j: self.entries[i][j].configure(state="normal")
        
        # Сброс модели и UI
        self.model.reset_state()
        self.logger.setup_file() # Пересоздаст пустой CSV с заголовками
        self.day_label.configure(text="День: 0")
        self.weather_label.configure(text="Ожидание...", text_color="white")
        self.draw_plot()

    def simulate_loop(self):
        if not self.is_running:
            return

        start, end, state, duration = self.model.do_simulation_step()
        
        self.logger.log_step(start, end, state, duration)

        # Теперь показываем день и сколько дней длилась эта погода!
        self.day_label.configure(text=f"День: {int(self.model.current_time)} (прошло: {duration:.1f} дн.)")
        self.weather_label.configure(
            text=STATES[self.model.current_state], 
            text_color=COLORS[self.model.current_state]
        )
        self.draw_plot()

        delay = int((2.1 - self.speed_slider.get()) * 1000)
        self.after(delay, self.simulate_loop)

    def draw_plot(self):
        self.ax.clear()
        
        labels = ['Ясно', 'Облачно', 'Пасмурно']
        x = np.arange(len(labels))
        width = 0.35
        
        empirical_pi = self.model.get_empirical_pi()

        self.ax.bar(x - width/2, self.model.theoretical_pi, width, label='Теоретическое', color='#1f77b4')
        self.ax.bar(x + width/2, empirical_pi, width, label='Эмпирическое (Наблюдаемое)', color='#ff7f0e')

        self.ax.set_ylabel('Вероятность', color='white')
        self.ax.set_title('Сравнение распределений', color='white')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, color='white')
        self.ax.legend()
        self.ax.set_ylim(0, 1.0)
        
        self.canvas.draw()