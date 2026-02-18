import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp

@jax.jit
def calculate_phisic(state, dt, params):
    x_old = state[0]
    y_old = state[1]
    v_x_old = state[2]
    v_y_old = state[3]

    k = params[0]
    m = params[1]
    g = params[2]

    v_full = jnp.sqrt(v_x_old ** 2 + v_y_old ** 2)

    ax = -(k / m) * v_full * v_x_old
    ay = -g - (k / m) * v_full * v_y_old

    x_new = x_old + v_x_old * dt
    y_new = y_old + v_y_old * dt
    v_x_new = v_x_old + ax * dt
    v_y_new = v_y_old + ay * dt

    return jnp.array([x_new, y_new, v_x_new, v_y_new])

def run_simulation(v0, angle_deg, h0, dt, k, m, g):
    angle_rad = angle_deg * math.pi / 180
    vx_start = v0 * math.cos(angle_rad)
    vy_start = v0 * math.sin(angle_rad)
    
    current_state = jnp.array([0.0, h0, vx_start, vy_start])
    params = jnp.array([k, m, g])
    
    history_x = [0.0]
    history_y = [h0]
    
    max_steps = 500000
    step = 0
    
    while current_state[1] >= 0 and step < max_steps:
        current_state = calculate_phisic(current_state, dt, params)
        history_x.append(current_state[0].item())
        history_y.append(current_state[1].item())
        step += 1
        
        if abs(current_state[0].item()) > 5000:
            break

    return history_x, history_y, current_state

def calculate_statistic(history_x, history_y, last_state):
    if not history_x: return 0,0,0
    flight_range = history_x[-1]
    max_height = max(history_y)
    vx_end = last_state[2].item()
    vy_end = last_state[3].item()
    v_end = math.sqrt(vx_end ** 2 + vy_end **2)
    return flight_range, max_height, v_end

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование полёта")
        self.root.geometry("1100x700")

        self.var_mass = tk.DoubleVar(value=10.0)
        self.var_angle = tk.DoubleVar(value=45.0)
        self.var_v0 = tk.DoubleVar(value=70.0)
        self.var_radius = tk.DoubleVar(value=0.05)
        self.var_c = tk.DoubleVar(value=0.47)
        self.var_rho = tk.DoubleVar(value=1.225)
        self.var_g = tk.DoubleVar(value=9.81)

        self.create_layout()

    def create_layout(self):
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(control_frame, text="Параметры", font=("Arial", 12, "bold")).pack(pady=10)

        group1 = ttk.LabelFrame(control_frame, text="Основные", padding=10)
        group1.pack(fill=tk.X, pady=5)
        
        self.create_input(group1, "Масса (кг):", self.var_mass)
        self.create_input(group1, "Угол (град):", self.var_angle)
        self.create_input(group1, "Скорость (м/с):", self.var_v0)

        group2 = ttk.LabelFrame(control_frame, text="Дополнительные", padding=10)
        group2.pack(fill=tk.X, pady=5)
        
        self.create_input(group2, "Радиус (м):", self.var_radius)
        self.create_input(group2, "Коэф. C:", self.var_c)
        self.create_input(group2, "Плотность:", self.var_rho)
        self.create_input(group2, "G:", self.var_g)

        btn_run = ttk.Button(control_frame, text="Старт", command=self.run_simulation_ui)
        btn_run.pack(pady=20, fill=tk.X)

        self.text_output = tk.Text(control_frame, height=15, width=40, font=("Consolas", 9))
        self.text_output.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots()
        self.ax.grid(True)
        self.ax.set_xlabel("X (м)")
        self.ax.set_ylabel("Y (м)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_input(self, parent, label_text, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=variable, width=10).pack(side=tk.RIGHT)

    def run_simulation_ui(self):
        try:
            m = self.var_mass.get()
            angle = self.var_angle.get()
            v0 = self.var_v0.get()
            r = self.var_radius.get()
            c = self.var_c.get()
            rho = self.var_rho.get()
            g = self.var_g.get()

            if m <= 0 or r <= 0 or v0 < 0 or rho < 0:
                raise ValueError("Некорректные значения")
            if not (0 <= angle <= 90):
                raise ValueError("Угол от 0 до 90")

        except Exception as e:
            messagebox.showerror("Ошибка", f"{e}")
            return

        k_val = 0.5 * c * rho * math.pi * (r ** 2)
        steps = [1, 0.1, 0.01, 0.001, 0.0001]
        
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("Дальность (м)")
        self.ax.set_ylabel("Высота (м)")
        
        self.text_output.delete("1.0", tk.END)
        header = f"{'dt':<8}| {'L (м)':<10}| {'H (м)':<10}| {'V (м/с)':<10}\n" + "-"*42 + "\n"
        self.text_output.insert(tk.END, header)

        global_max_x, global_max_y = 0, 0

        for step in steps:
            xs, ys, last_state = run_simulation(v0, angle, 0, step, k_val, m, g)
            dist, height, v_fin = calculate_statistic(xs, ys, last_state)
            
            row = f"{step:<8}| {dist:<10.1f}| {height:<10.1f}| {v_fin:<10.1f}\n"
            self.text_output.insert(tk.END, row)
            self.text_output.see(tk.END)

            if dist > 0:
                global_max_x = max(global_max_x, dist)
            global_max_y = max(global_max_y, height)
            
            self.ax.set_xlim(left=0, right=max(10, global_max_x * 1.1))
            self.ax.set_ylim(bottom=0, top=max(10, global_max_y * 1.1))

            line, = self.ax.plot([], [], label=f"dt={step}")
            self.ax.legend()
            
            skip_frame = max(1, len(xs) // 100)
            
            for i in range(0, len(xs), skip_frame):
                line.set_data(xs[:i], ys[:i])
                self.canvas.draw()
                self.root.update() 
            
            line.set_data(xs, ys)
            self.canvas.draw()
            self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()