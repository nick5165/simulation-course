class EventModeler:
    def __init__(self, generator, events_list):
        self.generator = generator
        self.events = events_list
        self.normalize()
        self.sort_descending()

    def normalize(self):
        total_sum = sum(prob for name, prob in self.events)
        
        if abs(total_sum - 1.0) > 0.0001:
            for i in range(len(self.events)):
                name, prob = self.events[i]
                self.events[i] = (name, prob / total_sum)

    def sort_descending(self):
        n = len(self.events)
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if self.events[j][1] < self.events[j + 1][1]:
                    temp = self.events[j]
                    self.events[j] = self.events[j + 1]
                    self.events[j + 1] = temp

    def get_random_event(self) -> str:
        random_value = self.generator.generate()
        
        for name, prob in self.events:
            random_value -= prob
            if random_value < 0:
                return name
                
        return self.events[-1][0]