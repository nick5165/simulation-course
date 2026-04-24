import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.stats import norm
import logic

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Имитационное моделирование СВ")
        self.geometry("1000x800")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.create_discrete_tab()
        self.create_normal_tab()

    def create_discrete_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Дискретная СВ")
        
        # Левая панель управления
        ctrl = ttk.Frame(tab)
        ctrl.pack(side='left', fill='y', padx=10, pady=10)
        
        ttk.Label(ctrl, text="Введите 5 вероятностей:").pack()
        self.p_entries = []
        for i in range(5):
            ent = ttk.Entry(ctrl, width=10)
            ent.insert(0, str(0.2))
            ent.pack(pady=2)
            self.p_entries.append(ent)
            
        ttk.Label(ctrl, text="Размер выборки (N):").pack(pady=(10,0))
        self.n_disc = ttk.Entry(ctrl, width=10)
        self.n_disc.insert(0, "1000")
        self.n_disc.pack()
        
        ttk.Button(ctrl, text="Запустить", command=self.run_discrete).pack(pady=20)
        
        self.res_disc = tk.Text(ctrl, width=30, height=15)
        self.res_disc.pack()

        # Правая панель для графика
        self.fig_disc, self.ax_disc = plt.subplots(figsize=(5, 4))
        self.canvas_disc = FigureCanvasTkAgg(self.fig_disc, master=tab)
        self.canvas_disc.get_tk_widget().pack(side='right', expand=True, fill='both')

    def create_normal_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Нормальная СВ")
        
        ctrl = ttk.Frame(tab)
        ctrl.pack(side='left', fill='y', padx=10, pady=10)
        
        ttk.Label(ctrl, text="Среднее (mu):").pack()
        self.mu_ent = ttk.Entry(ctrl, width=10)
        self.mu_ent.insert(0, "0")
        self.mu_ent.pack()
        
        ttk.Label(ctrl, text="Дисперсия (sigma^2):").pack()
        self.var_ent = ttk.Entry(ctrl, width=10)
        self.var_ent.insert(0, "1")
        self.var_ent.pack()
        
        ttk.Label(ctrl, text="Размер выборки (N):").pack()
        self.n_norm = ttk.Entry(ctrl, width=10)
        self.n_norm.insert(0, "1000")
        self.n_norm.pack()
        
        ttk.Button(ctrl, text="Запустить", command=self.run_normal).pack(pady=20)
        
        self.res_norm = tk.Text(ctrl, width=30, height=15)
        self.res_norm.pack()

        self.fig_norm, self.ax_norm = plt.subplots(figsize=(5, 4))
        self.canvas_norm = FigureCanvasTkAgg(self.fig_norm, master=tab)
        self.canvas_norm.get_tk_widget().pack(side='right', expand=True, fill='both')

    def run_discrete(self):
        try:
            raw_probs = [float(e.get()) for e in self.p_entries]
            n = int(self.n_disc.get())
            
            probs, msg = logic.normalize_probabilities(raw_probs)
            if msg:
                messagebox.showwarning("Нормировка", msg)
                for i, p in enumerate(probs):
                    self.p_entries[i].delete(0, tk.END)
                    self.p_entries[i].insert(0, f"{p:.4f}")
            
            res = logic.discrete_simulation(probs, n)
            
            # Отрисовка
            self.ax_disc.clear()
            x = np.arange(1, 6)
            self.ax_disc.bar(x - 0.2, probs, width=0.4, label='Теор.', alpha=0.6)
            self.ax_disc.bar(x + 0.2, res['emp_probs'], width=0.4, label='Эмп.', alpha=0.6)
            self.ax_disc.set_title(f"Дискретное распределение (N={n})")
            self.ax_disc.legend()
            self.canvas_disc.draw()
            
            # Вывод текста
            self.res_disc.delete(1.0, tk.END)
            out = (f"Среднее: {res['mean']:.4f}\n"
                   f"Погрешность M: {res['mean_err']:.2%}\n"
                   f"Дисперсия: {res['var']:.4f}\n"
                   f"Погрешность D: {res['var_err']:.2%}\n"
                   f"Хи-квадрат: {res['chi_stat']:.4f}\n"
                   f"Крит. знач: {res['critical_chi']:.4f}\n"
                   f"Гипотеза: {'Принята' if res['is_accepted'] else 'Отвергнута'}")
            self.res_disc.insert(tk.END, out)
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def run_normal(self):
        try:
            mu = float(self.mu_ent.get())
            sigma = np.sqrt(float(self.var_ent.get()))
            n = int(self.n_norm.get())
            
            res = logic.normal_simulation(mu, sigma, n)
            
            # Отрисовка
            self.ax_norm.clear()
            self.ax_norm.hist(res['samples'], bins=30, density=True, alpha=0.6, color='skyblue', label='Гистограмма')
            
            x_plot = np.linspace(min(res['samples']), max(res['samples']), 100)
            self.ax_norm.plot(x_plot, norm.pdf(x_plot, mu, sigma), 'r-', lw=2, label='Плотность')
            self.ax_norm.set_title(f"Нормальное распределение (N={n})")
            self.ax_norm.legend()
            self.canvas_norm.draw()
            
            # Вывод текста
            self.res_norm.delete(1.0, tk.END)
            out = (f"Ср. знач: {res['mean']:.4f}\n"
                   f"Выб. дисп: {res['var']:.4f}\n"
                   f"Хи-квадрат: {res['chi_stat']:.4f}\n"
                   f"Крит. знач: {res['critical_chi']:.4f}\n"
                   f"Гипотеза: {'Принята' if res['chi_stat'] < res['critical_chi'] else 'Отвергнута'}")
            self.res_norm.insert(tk.END, out)
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))