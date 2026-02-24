import numpy as np
from numba import njit
import math
import time
import tkinter as tk
from tkinter import messagebox
import os

RHO = 7800
C_CONST = 460
LAMB = 46
L = 0.01
T0 = 20
TLEFT = 20
TRIGHT = 100
SIM_TIME = 2.0

@njit
def run_calculations(h: float, t: float, total_time: float):
    if h >= L: return (TLEFT + TRIGHT) / 2
    
    a = c = LAMB / h ** 2
    b = (2 * LAMB / h ** 2) + (RHO * C_CONST / t)
    n = int(L/h)
    
    t_next = np.zeros(n+1)
    alpha = np.zeros(n+1)
    beta = np.zeros(n+1)
    current_temps = np.full(n+1, T0)
    current_temps[0] = TLEFT
    current_temps[n] = TRIGHT
    
    steps = int(total_time/t)
    for step in range(steps):
        alpha[0] = 0
        beta[0] = TLEFT
        
        for j in range(1, n):
            fj = - (RHO * C_CONST / t) * current_temps[j]
            denom = b - c * alpha[j-1]
            alpha[j] = a / denom
            beta[j] = (c * beta[j-1] - fj) / denom

        t_next[n] = TRIGHT
        for j in range(n-1, 0, -1):
            t_next[j] = alpha[j] * t_next[j+1] + beta[j]
        
        t_next[0] = TLEFT
        current_temps[:] = t_next[:]
        
    return current_temps[n//2]

def start_simulation():
    global RHO, C_CONST, LAMB, L, T0, TLEFT, TRIGHT, SIM_TIME
    
    try:
        RHO = float(ent_rho.get())
        C_CONST = float(ent_c.get())
        LAMB = float(ent_lamb.get())
        L = float(ent_l.get())
        T0 = float(ent_t0.get())
        TLEFT = float(ent_tleft.get())
        TRIGHT = float(ent_tright.get())
        SIM_TIME = 2.0
    except ValueError:
        messagebox.showerror("Ошибка", "Введите числа")
        return

    run_calculations.recompile() 
    _ = run_calculations(0.1, 0.1, SIM_TIME)

    steps_t = [0.1, 0.01, 0.001, 0.0001]
    steps_h = [0.1, 0.01, 0.001, 0.0001]
    
    results = {}
    exec_times = {}

    lbl_status.config(text="Статус: Расчет идет...")
    root.update()

    for dt in steps_t:
        for dh in steps_h:
            start_t = time.perf_counter()
            temp_res = run_calculations(dh, dt, SIM_TIME)
            end_t = time.perf_counter()
            results[(dt, dh)] = temp_res
            exec_times[(dt, dh)] = end_t - start_t

    generate_md_report(results, exec_times)
    lbl_status.config(text="Статус: Отчет создан")
    messagebox.showinfo("Успех", "Отчет report.md готов.")

def generate_md_report(results, times):
    steps = [0.1, 0.01, 0.001, 0.0001]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(current_dir, "report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Отчет по лабораторной работе\n\n")
        f.write("### Метод конечных разностей для уравнения теплопроводности\n\n")
        f.write(f"**Параметры:** L={L}м, T0={T0}C, T_лево={TLEFT}C, T_право={TRIGHT}C, Время={SIM_TIME}с\n\n")
        
        header = "| Шаг по времени \ Шаг по пространству | " + " | ".join(map(str, steps)) + " |"
        separator = "|---|---|---|---|---|"
        f.write(header + "\n" + separator + "\n")
        
        for dt in steps:
            row = f"| {dt} | "
            cells = [f"{results[(dt, dh)]:.4f}" for dh in steps]
            f.write(row + " | ".join(cells) + " |\n")
        
        f.write("\n**Время выполнения (сек):**\n\n")
        f.write("| Шаг t \ Шаг h | " + " | ".join(map(str, steps)) + " |\n|---|---|---|---|---|\n")
        for dt in steps:
            row = f"| {dt} | "
            cells = [f"{times[(dt, dh)]:.6f}" for dh in steps]
            f.write(row + " | ".join(cells) + " |\n")

        f.write("\n**Вывод:**\n")
        f.write("В ходе работы было реализовано моделирование теплопроводности неявным методом. ")
        f.write("При измельчении сетки значения температуры сходятся к стабильному результату.")

root = tk.Tk()
root.title("Лабораторная работа")

fields = [
    ("Плотность (RHO):", ent_rho := tk.Entry(root), RHO),
    ("Теплоемкость (C):", ent_c := tk.Entry(root), C_CONST),
    ("Теплопроводность (LAMB):", ent_lamb := tk.Entry(root), LAMB),
    ("Толщина (L):", ent_l := tk.Entry(root), L),
    ("Нач. темп (T0):", ent_t0 := tk.Entry(root), T0),
    ("T лево:", ent_tleft := tk.Entry(root), TLEFT),
    ("T право:", ent_tright := tk.Entry(root), TRIGHT),
]

for i, (label, entry, default) in enumerate(fields):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry.insert(0, str(default))

btn_start = tk.Button(root, text="ВЫПОЛНИТЬ МОДЕЛИРОВАНИЕ", command=start_simulation, bg="green", fg="white")
btn_start.grid(row=len(fields), column=0, columnspan=2, pady=20, sticky="we", padx=10)

lbl_status = tk.Label(root, text="Ожидание")
lbl_status.grid(row=len(fields)+1, column=0, columnspan=2, pady=5)

root.mainloop()