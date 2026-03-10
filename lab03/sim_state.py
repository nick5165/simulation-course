from dataclasses import dataclass, field
from typing import List

@dataclass
class Cloud:
    x: float
    y: float
    radius: float = 6.0
    water_reserve: float = 100.0 

@dataclass
class SimulationState:
    width: int = 60
    height: int = 40
    wind_speed: float = 1.0
    wind_dx: int = 1
    wind_dy: int = 0
    global_humidity: float = 0.4
    clouds: List[Cloud] = field(default_factory=list)
    
    cloud_lightning_chance: float = 0.1 
    
    max_forest_age: int = 2000
    fire_duration: int = 12
    ashes_duration: int = 30