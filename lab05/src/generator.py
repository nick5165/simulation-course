class MultiplicateConGenerator:
    def __init__(self):
        self.M = 1 << 63
        self.beta = (1 << 32) + 3
        self.x = self.beta

    def generate(self) -> float:
        remains = self.x * self.beta
        self.x = remains % self.M
        return self.x / self.M