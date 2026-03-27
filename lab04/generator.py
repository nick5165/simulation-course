class MultiplicateConGenerator:
    def __init__(self):
        self.M = 1 << 63
        self.beta = (1 << 32) + 3
        self.x = self.beta

    def generate(self) -> float:
        remains = self.x * self.beta
        self.x = remains % self.M
        return self.x / self.M

class FibGenerator:
    def __init__(self):
        self.sample = []
        self.other_gen = MultiplicateConGenerator()
        for _ in range(55):
            self.sample.append(self.other_gen.generate())
    
    def generate(self):
        s = self.sample[-55] + self.sample[-24]
        self.sample.append(s % 1.0)
        self.sample.pop(0)
        return self.sample[-1]