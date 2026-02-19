import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import math
import os

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование полета")
        self.root.geometry("1200x800")

        self.results = {} 

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

        group1 = ttk.LabelFrame(control_frame, text="Физика", padding=10)
        group1.pack(fill=tk.X, pady=5)
        self.create_input(group1, "Масса (кг):", self.var_mass)
        self.create_input(group1, "Угол (°):", self.var_angle)
        self.create_input(group1, "Скорость (м/с):", self.var_v0)
        self.create_input(group1, "Радиус (м):", self.var_radius)
        self.create_input(group1, "Коэф. C:", self.var_c)
        self.create_input(group1, "Плотность воздуха:", self.var_rho)
        self.create_input(group1, "G (м/с²):", self.var_g)

        self.btn_run = ttk.Button(control_frame, text="Запустить симуляцию", command=self.run_simulation_ui)
        self.btn_run.pack(pady=10, fill=tk.X)

        self.btn_report = ttk.Button(control_frame, text="Сгенерировать отчет (.md)", command=self.generate_markdown_report, state=tk.DISABLED)
        self.btn_report.pack(pady=5, fill=tk.X)

        self.text_output = tk.Text(control_frame, height=12, width=45, font=("Consolas", 9))
        self.text_output.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.grid(True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_input(self, parent, label_text, variable):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=20).pack(side=tk.LEFT)
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
        except Exception:
            messagebox.showerror("Ошибка", "Проверьте введенные данные")
            return

        self.btn_run.config(state=tk.DISABLED)
        self.results = {} 
        
        k_const = 0.5 * c * rho * math.pi * (r ** 2)
        steps = [1, 0.1, 0.01, 0.001, 0.0001]
        
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("Дистанция (м)")
        self.ax.set_ylabel("Высота (м)")
        
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert(tk.END, f"{'dt':<8} | {'L (м)':<8} | {'H (м)':<8} | {'V (м/с)':<8}\n" + "-"*40 + "\n")

        colors = ['red', 'orange', 'green', 'blue', 'purple']

        for idx, dt in enumerate(steps):
            angle_rad = math.radians(angle)
            x, y = 0.0, 0.0
            vx = v0 * math.cos(angle_rad)
            vy = v0 * math.sin(angle_rad)
            
            history_x = [x]
            history_y = [y]
            max_h = 0.0
            
            line, = self.ax.plot([], [], label=f"dt={dt}", color=colors[idx])
            self.ax.legend()

            step_count = 0
            while True:
                prev_x, prev_y = x, y
                prev_vx, prev_vy = vx, vy
                
                v_full = math.sqrt(vx**2 + vy**2)
                ax_acc = -(k_const / m) * v_full * vx
                ay_acc = -g - (k_const / m) * v_full * vy
                
                x += vx * dt
                y += vy * dt
                vx += ax_acc * dt
                vy += ay_acc * dt
                
                if y < 0:
                    fraction = prev_y / (prev_y - y)
                    x = prev_x + (x - prev_x) * fraction
                    vx = prev_vx + (vx - prev_vx) * fraction
                    vy = prev_vy + (vy - prev_vy) * fraction
                    y = 0.0
                    history_x.append(x)
                    history_y.append(y)
                    break

                history_x.append(x)
                history_y.append(y)
                if y > max_h: max_h = y
                
                step_count += 1
                if dt > 0.001 or step_count % 150 == 0:
                    line.set_data(history_x, history_y)
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.canvas.draw_idle()
                    self.root.update()

            v_final = math.sqrt(vx**2 + vy**2)
            dist = history_x[-1]
            
            self.results[dt] = {
                'dist': round(dist, 4),
                'height': round(max_h, 4),
                'v_end': round(v_final, 4)
            }
            
            self.text_output.insert(tk.END, f"{dt:<8} | {dist:<8.2f} | {max_h:<8.2f} | {v_final:<8.2f}\n")
            self.text_output.see(tk.END)
            
            line.set_data(history_x, history_y)
            self.canvas.draw()

        self.btn_run.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.NORMAL)
        messagebox.showinfo("Готово", "Симуляция завершена")

    def generate_markdown_report(self):
        if not self.results:
            return

        steps = [1, 0.1, 0.01, 0.001, 0.0001]
        
        header = "| Шаг моделирования, с | " + " | ".join(map(str, steps)) + " |"
        separator = "| :--- | " + " :--- | " * len(steps)
        row_dist = "| Дальность полёта, м | " + " | ".join([str(self.results[s]['dist']) for s in steps]) + " |"
        row_height = "| Максимальная высота, м | " + " | ".join([str(self.results[s]['height']) for s in steps]) + " |"
        row_speed = "| Скорость в конечной точке, м/с | " + " | ".join([str(self.results[s]['v_end']) for s in steps]) + " |"

        table = "\n".join([header, separator, row_dist, row_height, row_speed])

        conclusion = (
            "### Вывод\n\n"
            "Первый шаг времени ($dt=1$) слишком большой, из-за чего первая линия получается угловатой. "
            "Последнее значение шага ($dt=0.0001$) очень маленькое, расчет требует значительно больше времени, "
            "при этом таких видимых изменений на графике практически не наблюдается по сравнению с предыдущим шагом."
        )

        report_content = f"# Отчет о моделировании\n\n{table}\n\n{conclusion}"

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            report_path = os.path.join(current_dir, "report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            messagebox.showinfo("Успех", f"Отчет создан в:\n{report_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()