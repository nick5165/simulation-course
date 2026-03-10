import random, copy
from cell import Cell, State
import rules

class SimulationEngine:
    def __init__(self, state):
        self.state = state
        self.matrix = self._init_map()

    def _init_map(self):
        matrix = []
        for y in range(self.state.height):
            row = []
            for x in range(self.state.width):
                r = random.random()
                if r < 0.1: s = State.WATER
                elif r < 0.8: s = State.FOREST
                else: s = State.BARE_EARTH
                row.append(Cell(s, age=random.randint(0, 100)))
            matrix.append(row)
        return matrix

    def step(self):
        self._move_and_evaporate_clouds()
        self._update_humidity()
        new_matrix = copy.deepcopy(self.matrix)
        
        for y in range(self.state.height):
            for x in range(self.state.width):
                cell = self.matrix[y][x]
                new_c = new_matrix[y][x]
                
                if cell.state == State.FOREST:
                    new_c.age += 1
                    if new_c.age > self.state.max_forest_age:
                        new_c.state = State.BARE_EARTH
                    else:
                        self._check_fire(x, y, cell, new_c)
                elif cell.state == State.FIRE:
                    new_c.age -= 1
                    if new_c.age <= 0:
                        new_c.state = State.ASHES
                        new_c.age = self.state.ashes_duration
                elif cell.state == State.ASHES:
                    new_c.age -= 1
                    if new_c.age <= 0:
                        new_c.state = State.BARE_EARTH
                elif cell.state == State.BARE_EARTH:
                    self._process_regrowth(x, y, new_c)
        
        strikes = self._lightning(new_matrix)
        self.matrix = new_matrix
        return strikes

    def _process_regrowth(self, x, y, new_cell):
        # Эффект семян: считаем лесных соседей
        forest_neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.state.width and 0 <= ny < self.state.height:
                    if self.matrix[ny][nx].state == State.FOREST:
                        forest_neighbors += 1
        
        base_chance = 0.001 
        humidity_bonus = self.matrix[y][x].humidity * 0.02 
        seed_bonus = forest_neighbors * 0.005 
        total_chance = base_chance + humidity_bonus + seed_bonus

        if random.random() < total_chance:
            new_cell.state = State.FOREST
            new_cell.age = 0

    def _check_fire(self, x, y, cell, new_c):
        risk = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.state.width and 0 <= ny < self.state.height:
                    if self.matrix[ny][nx].state == State.FIRE:
                        wm = rules.get_wind_multiplier(self.state.wind_dx, self.state.wind_dy, self.state.wind_speed, dx, dy)
                        am = rules.get_age_multiplier(cell.age)
                        risk += rules.calculate_ignition_chance(2.5, am, wm, cell.humidity)
        
        if random.random() * 100 < risk:
            new_c.state = State.FIRE
            new_c.age = self.state.fire_duration

    def _move_and_evaporate_clouds(self):
        active = []
        for c in self.state.clouds:
            c.x = (c.x + self.state.wind_dx * 0.3) % self.state.width
            c.y = (c.y + self.state.wind_dy * 0.3) % self.state.height
            c.water_reserve -= 0.4
            if c.water_reserve > 0: active.append(c)
        self.state.clouds = active

    def _update_humidity(self):
        for y in range(self.state.height):
            for x in range(self.state.width):
                self.matrix[y][x].humidity = self.state.global_humidity
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0<=nx<self.state.width and 0<=ny<self.state.height:
                        if self.matrix[ny][nx].state == State.WATER:
                            self.matrix[y][x].humidity = min(1.0, self.matrix[y][x].humidity + 0.4)
                for c in self.state.clouds:
                    if ((x-c.x)**2 + (y-c.y)**2)**0.5 < c.radius:
                        self.matrix[y][x].humidity = 1.0

    def _lightning(self, matrix):
        hits = []
        for c in self.state.clouds:
            if random.random() < self.state.cloud_lightning_chance:
                lx, ly = int(c.x + random.uniform(-c.radius, c.radius)), int(c.y + random.uniform(-c.radius, c.radius))
                if 0 <= lx < self.state.width and 0 <= ly < self.state.height:
                    target = matrix[ly][lx]
                    if target.state == State.FOREST:
                        target.state = State.FIRE
                        target.age = self.state.fire_duration
                        hits.append((lx, ly))
                        c.water_reserve -= 15
        return hits