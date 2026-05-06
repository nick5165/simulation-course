import numpy as np

class MarkovModel:
    def __init__(self):
        self.q_matrix = np.zeros((3, 3))
        self.theoretical_pi = np.array([1/3, 1/3, 1/3])
        self.current_state = 0
        self.current_time = 0.0
        self.time_in_states = np.zeros(3)

    def calculate_diagonals(self):
        """Рассчитывает диагональные элементы и возвращает их."""
        diagonals = []
        for i in range(3):
            row_sum = sum(self.q_matrix[i][j] for j in range(3) if i != j)
            self.q_matrix[i][i] = -row_sum
            diagonals.append(-row_sum)
        return diagonals

    def calculate_theoretical_pi(self):
        """Расчет стационарного распределения."""
        eigvals, eigvecs = np.linalg.eig(self.q_matrix.T)
        zero_eig_idx = np.argmin(np.abs(eigvals))
        pi = eigvecs[:, zero_eig_idx].real
        
        if np.sum(pi) == 0 or np.any(pi < -1e-5):
            self.theoretical_pi = np.array([1/3, 1/3, 1/3])
        else:
            self.theoretical_pi = pi / np.sum(pi)

    def reset_state(self):
        """Сброс состояния при новом запуске."""
        self.current_time = 0.0
        self.time_in_states = np.zeros(3)
        self.current_state = np.random.choice([0, 1, 2])

    def do_simulation_step(self):
        """Один шаг симуляции. Возвращает данные для логгера и UI."""
        q_ii = self.q_matrix[self.current_state][self.current_state]
        
        if abs(q_ii) < 1e-6:
            time_held = 1.0 
            next_state = self.current_state
        else:
            time_held = np.random.exponential(scale=1.0/abs(q_ii))
            probs = self.q_matrix[self.current_state].copy()
            probs[self.current_state] = 0
            probs = probs / np.sum(probs)
            next_state = np.random.choice([0, 1, 2], p=probs)

        # Сохраняем значения для логов перед обновлением состояния
        start_time = self.current_time
        end_time = start_time + time_held
        old_state = self.current_state

        # Обновляем статистику модели
        self.time_in_states[self.current_state] += time_held
        self.current_time = end_time
        self.current_state = next_state

        return start_time, end_time, old_state, time_held

    def get_empirical_pi(self):
        """Возвращает текущие наблюдаемые вероятности."""
        total_time = np.sum(self.time_in_states)
        if total_time > 0:
            return self.time_in_states / total_time
        return np.zeros(3)