import tkinter as tk
from cell import State
from sim_state import Cloud
import random

class ForestFireGUI:
    def __init__(self, root, engine, state):
        self.root = root
        self.engine, self.state = engine, state
        self.running = False
        self.cell_size = 14
        self.selected_tool = tk.IntVar(value=State.FOREST.value)
        self.dragged_cloud = None
        
        self.canvas = tk.Canvas(root, width=state.width*self.cell_size, height=state.height*self.cell_size, bg="#111")
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.paint); self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.start_drag); self.canvas.bind("<B3-Motion>", self.drag)

        self.panel = tk.Frame(root, padx=10); self.panel.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(self.panel, text="ИНСТРУМЕНТ", font='bold').pack()
        for t, v in [("Лес", State.FOREST.value), ("Вода", State.WATER.value), ("Земля", State.BARE_EARTH.value), ("ОГОНЬ!", State.FIRE.value)]:
            tk.Radiobutton(self.panel, text=t, variable=self.selected_tool, value=v).pack(anchor="w")

        tk.Label(self.panel, text="\nВЕТЕР", font='bold').pack()
        wf = tk.Frame(self.panel); wf.pack()
        dirs = [("↖",-1,-1),("↑",0,-1),("↗",1,-1),("←",-1,0),("●",0,0),("→",1,0),("↙",-1,1),("↓",0,1),("↘",1,1)]
        for i, (t, dx, dy) in enumerate(dirs):
            tk.Button(wf, text=t, width=3, command=lambda x=dx, y=dy: self.set_w(x,y)).grid(row=i//3, column=i%3)

        tk.Button(self.panel, text="ПУСК/ПАУЗА", bg="orange", command=self.toggle, height=2).pack(fill=tk.X, pady=10)
        tk.Button(self.panel, text="+ ОБЛАКО", command=self.add_c).pack(fill=tk.X)
        
        self.sl_l = tk.Scale(self.panel, from_=0, to=0.5, resolution=0.01, label="Шанс молнии", orient="horizontal", command=self.set_l)
        self.sl_l.set(state.cloud_lightning_chance); self.sl_l.pack()
        self.sl_w = tk.Scale(self.panel, from_=0, to=5, label="Сила ветра", orient="horizontal"); self.sl_w.set(state.wind_speed); self.sl_w.pack()
        self.sl_h = tk.Scale(self.panel, from_=0, to=1, resolution=0.1, label="Влажность", orient="horizontal"); self.sl_h.set(state.global_humidity); self.sl_h.pack()

        self.draw()

    def set_l(self, v): self.state.cloud_lightning_chance = float(v)
    def set_w(self, dx, dy): self.state.wind_dx, self.state.wind_dy = dx, dy
    
    def paint(self, e):
        x, y = e.x//self.cell_size, e.y//self.cell_size
        if 0<=x<self.state.width and 0<=y<self.state.height:
            tool = State(self.selected_tool.get())
            current_cell = self.engine.matrix[y][x]
            
            # ПРАВИЛО: Нельзя поджечь воду или голую землю кистью!
            if tool == State.FIRE and current_cell.state != State.FOREST:
                return # Игнорируем нажатие

            current_cell.state = tool
            current_cell.age = self.state.fire_duration if tool == State.FIRE else 0
            self.draw()

    def start_drag(self, e):
        ex, ey = e.x/self.cell_size, e.y/self.cell_size
        self.dragged_cloud = next((c for c in self.state.clouds if ((c.x-ex)**2+(c.y-ey)**2)**0.5 < c.radius), None)

    def drag(self, e):
        if self.dragged_cloud:
            self.dragged_cloud.x, self.dragged_cloud.y = e.x/self.cell_size, e.y/self.cell_size
            self.draw()

    def toggle(self):
        self.running = not self.running
        if self.running: self.loop()

    def loop(self):
        if self.running:
            self.state.wind_speed, self.state.global_humidity = self.sl_w.get(), self.sl_h.get()
            hits = self.engine.step(); self.draw(hits)
            self.root.after(100, self.loop)

    def add_c(self):
        self.state.clouds.append(Cloud(random.randint(0, self.state.width), random.randint(0, self.state.height)))
        self.draw()

    def draw(self, hits=None):
        self.canvas.delete("all")
        for y in range(self.state.height):
            for x in range(self.state.width):
                c = self.engine.matrix[y][x]
                if c.state == State.WATER: color = "#1e90ff"
                elif c.state == State.FOREST:
                    g = int(max(40, 200 - c.age * 0.5)); color = f"#00{g:02x}00"
                elif c.state == State.FIRE: color = "#ff4500"
                elif c.state == State.ASHES: color = "#666"
                else: color = "#5C4033"
                self.canvas.create_rectangle(x*self.cell_size, y*self.cell_size, (x+1)*self.cell_size, (y+1)*self.cell_size, fill=color, outline="")
        
        if hits:
            for hx, hy in hits:
                self.canvas.create_line(hx*self.cell_size, 0, hx*self.cell_size, hy*self.cell_size, fill="yellow", width=2)
        
        for cl in self.state.clouds:
            opacity = int((cl.water_reserve / 100) * 255)
            color = f"#{max(50, opacity):02x}{max(50, opacity):02x}{max(50, opacity):02x}"
            self.canvas.create_oval((cl.x-cl.radius)*self.cell_size, (cl.y-cl.radius)*self.cell_size, (cl.x+cl.radius)*self.cell_size, (cl.y+cl.radius)*self.cell_size, outline=color, width=2, dash=(4,4))