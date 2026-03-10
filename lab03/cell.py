from enum import Enum
from dataclasses import dataclass

class State(Enum):
    WATER = 0
    BARE_EARTH = 1
    FOREST = 2
    FIRE = 3
    ASHES = 4

@dataclass
class Cell:
    state: State
    age: int = 0
    humidity: float = 0.0