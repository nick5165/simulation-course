import tkinter as tk
from sim_state import SimulationState, Cloud
from simulation import SimulationEngine
from gui import ForestFireGUI

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Лесной пожар")
    state = SimulationState()
    state.clouds.append(Cloud(20, 20, 7))
    engine = SimulationEngine(state)
    app = ForestFireGUI(root, engine, state)
    root.mainloop()