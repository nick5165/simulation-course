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
    
class QueuingSystem:
    def __init__(self, 
                 input_intensity: float, 
                 output_intensity: float, 
                 total_time: float):
        
        self.input_intensity = input_intensity
        self.output_intensity = output_intensity
        self.total_time = total_time

    def simulate(self):
        generator = MultiplicateConGenerator()
        
        t_simulation = 0.0
        prev_time = 0.0
        
        # Планируем первое событие прихода заявки
        t_arrival = -math.log(generator.generate()) / self.input_intensity
        t_departure = float('inf')
        
        # Состояние системы: 0 (свободна) или 1 (занята)
        state = 0
        
        # Время нахождения системы в состояниях 0 и 1
        time_in_state = {0: 0.0, 1: 0.0}
        
        processed_count = 0
        lost_count = 0
        total_arrivals = 0
        
        while t_simulation < self.total_time:
            # Определение времени ближайшего события
            if t_arrival < t_departure:
                event_time = t_arrival
                is_arrival = True
            else:
                event_time = t_departure
                is_arrival = False

            # Выход за пределы интервала моделирования
            if event_time > self.total_time:
                time_in_state[state] += (self.total_time - prev_time)
                break
            
            # Накопление времени в текущем состоянии
            time_in_state[state] += (event_time - prev_time)
            prev_time = event_time
            t_simulation = event_time
            
            if is_arrival:
                total_arrivals += 1
                if state == 0:
                    # Обработчик свободен, заявка принимается
                    state = 1
                    t_departure = t_simulation - math.log(generator.generate()) / self.output_intensity
                else:
                    # Обработчик занят, очереди нет -> заявка теряется
                    lost_count += 1
                
                # Планируем следующее прибытие
                t_arrival = t_simulation - math.log(generator.generate()) / self.input_intensity
            else:
                # Обслуживание завершено, обработчик освобождается
                state = 0
                processed_count += 1
                t_departure = float('inf')
                    
        total_time_recorded = sum(time_in_state.values())
        if total_time_recorded == 0:
            total_time_recorded = self.total_time
            
        # Наблюдаемые вероятности состояний системы
        p0 = time_in_state[0] / total_time_recorded
        p1 = time_in_state[1] / total_time_recorded
        
        # Наблюдаемая вероятность отказа (отношение потерянных к общему числу пришедших)
        loss_probability = lost_count / total_arrivals if total_arrivals > 0 else 0.0
        
        return {
            "processed_count": processed_count,
            "lost_count": lost_count,
            "total_arrivals": total_arrivals,
            "p0": p0,
            "p1": p1,
            "loss_probability": loss_probability
        }

if __name__ == "__main__":
    print("Моделирование СМО M/M/1/0 (без очереди)")
    try:
        input_intensity = float(input("Введите интенсивность входящего потока (лямбда): "))
        output_intensity = float(input("Введите интенсивность обслуживания (мю): "))
        total_time = float(input("Введите общее время моделирования: "))
        
        if input_intensity <= 0 or output_intensity <= 0 or total_time <= 0:
            print("Параметры должны быть положительными числами.")
        else:
            system = QueuingSystem(input_intensity, output_intensity, total_time)
            stats = system.simulate()
            
            print("\n=== Наблюдаемая статистика ===")
            print(f"Всего поступило заявок: {stats['total_arrivals']}")
            print(f"Успешно обработано:     {stats['processed_count']}")
            print(f"Потеряно (отказано):    {stats['lost_count']}")
            
            print("\n=== Наблюдаемые вероятности ===")
            print(f"Вероятность того, что система свободна (P0): {stats['p0']:.4f}")
            print(f"Вероятность того, что система занята (P1):   {stats['p1']:.4f}")
            print(f"Наблюдаемая вероятность отказа (P_отказа):   {stats['loss_probability']:.4f}")
            
    except ValueError:
        print("Ошибка ввода. Убедитесь, что ввели числовые значения.")