import math

class MultiplicateConGenerator:
    def __init__(self):
        self.M = 1 << 63
        self.beta = (1 << 32) + 3
        self.x = self.beta

    def generate(self) -> float:
        remains = self.x * self.beta
        self.x = remains % self.M
        val = self.x / self.M
        if val <= 0.0:
            val = 1e-15
        elif val >= 1.0:
            val = 1.0 - 1e-15
        return val

class Server:
    def __init__(self):
        self.is_busy = False
        self.time_when_free = float('inf')

    def assign_task(self, current_time, processing_time):
        self.is_busy = True
        self.time_when_free = current_time + processing_time

    def release(self):
        self.is_busy = False
        self.time_when_free = float('inf')

class QueuingSystem:
    def __init__(self, input_intensity: float, output_intensity: float, total_time: float, num_servers: int, queue_capacity: int):
        self.input_intensity = input_intensity
        self.output_intensity = output_intensity
        self.total_time = total_time
        self.num_servers = num_servers
        self.queue_capacity = queue_capacity
        
        self.servers = [Server() for _ in range(num_servers)]
        self.queue_size = 0
        
        self.generator = MultiplicateConGenerator()
        self.time_in_state = {i: 0.0 for i in range(num_servers + queue_capacity + 1)}
        
        self.processed_count = 0
        self.lost_count = 0
        self.total_arrivals = 0

    def _generate_time(self, intensity):
        return -math.log(self.generator.generate()) / intensity

    def simulate(self):
        t_simulation = 0.0
        prev_time = 0.0
        
        t_arrival = self._generate_time(self.input_intensity)
        
        while t_simulation < self.total_time:
            next_departure_time = float('inf')
            departing_server = None
            
            for server in self.servers:
                if server.time_when_free < next_departure_time:
                    next_departure_time = server.time_when_free
                    departing_server = server
            
            if t_arrival < next_departure_time:
                event_time = t_arrival
                is_arrival = True
            else:
                event_time = next_departure_time
                is_arrival = False

            if event_time > self.total_time:
                current_state = sum(1 for s in self.servers if s.is_busy) + self.queue_size
                self.time_in_state[current_state] += (self.total_time - prev_time)
                break
            
            current_state = sum(1 for s in self.servers if s.is_busy) + self.queue_size
            self.time_in_state[current_state] += (event_time - prev_time)
            
            prev_time = event_time
            t_simulation = event_time
            
            if is_arrival:
                self.total_arrivals += 1
                free_server = next((s for s in self.servers if not s.is_busy), None)
                
                if free_server:
                    processing_time = self._generate_time(self.output_intensity)
                    free_server.assign_task(t_simulation, processing_time)
                elif self.queue_size < self.queue_capacity:
                    self.queue_size += 1
                else:
                    self.lost_count += 1
                    
                t_arrival = t_simulation + self._generate_time(self.input_intensity)
            else:
                self.processed_count += 1
                
                if self.queue_size > 0:
                    self.queue_size -= 1
                    processing_time = self._generate_time(self.output_intensity)
                    departing_server.assign_task(t_simulation, processing_time)
                else:
                    departing_server.release()

    def get_statistics(self):
        total_time_recorded = sum(self.time_in_state.values()) or self.total_time
        probabilities = {state: time / total_time_recorded for state, time in self.time_in_state.items()}
        loss_probability = self.lost_count / self.total_arrivals if self.total_arrivals > 0 else 0.0
        
        return {
            "processed_count": self.processed_count,
            "lost_count": self.lost_count,
            "total_arrivals": self.total_arrivals,
            "probabilities": probabilities,
            "loss_probability": loss_probability
        }

if __name__ == "__main__":
    try:
        input_intensity = float(input("Входящий поток (лямбда): "))
        output_intensity = float(input("Интенсивность обслуживания на 1 прибор (мю): "))
        total_time = float(input("Общее время моделирования: "))
        num_servers = int(input("Количество приборов (каналов): "))
        queue_capacity = int(input("Вместимость очереди: "))
        
        if any(val <= 0 for val in [input_intensity, output_intensity, total_time, num_servers]) or queue_capacity < 0:
            print("Отрицательные или нулевые значения (кроме очереди) недопустимы.")
        else:
            system = QueuingSystem(input_intensity, output_intensity, total_time, num_servers, queue_capacity)
            system.simulate()
            stats = system.get_statistics()
            
            print("\n=== Статистика ===")
            print(f"Всего поступило заявок: {stats['total_arrivals']}")
            print(f"Успешно обработано:     {stats['processed_count']}")
            print(f"Потеряно (отказано):    {stats['lost_count']}")
            
            print("\n=== Вероятности состояний ===")
            for state, prob in stats['probabilities'].items():
                print(f"P{state} (заявок в системе: {state}): {prob:.4f}")
                
            print(f"\nВероятность отказа: {stats['loss_probability']:.4f}")
            
    except ValueError:
        print("Ошибка ввода.")