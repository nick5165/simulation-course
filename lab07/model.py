# model.py
import numpy as np

class MarkovModel:
    def __init__(self):
        self.q_matrix = np.zeros((3, 3))
        self.p_matrix = np.zeros((3, 3))
        # 1/3 - это лишь фолбэк до старта расчетов
        self.theoretical_pi = np.array([1/3, 1/3, 1/3]) 
        
        self.current_state = 0
        self.current_time = 0 
        self.time_in_states = np.zeros(3)

    def calculate_diagonals(self):
        diagonals = []
        for i in range(3):
            row_sum = sum(self.q_matrix[i][j] for j in range(3) if i != j)
            self.q_matrix[i][i] = -row_sum
            diagonals.append(-row_sum)
        return diagonals

    def calculate_theoretical_pi(self):
        # Настоящий расчет вероятностей через собственные векторы
        eigvals, eigvecs = np.linalg.eig(self.q_matrix.T)
        zero_eig_idx = np.argmin(np.abs(eigvals))
        pi = eigvecs[:, zero_eig_idx].real
        
        if np.sum(pi) == 0 or np.any(pi < -1e-5):
            self.theoretical_pi = np.array([1/3, 1/3, 1/3])
        else:
            self.theoretical_pi = pi / np.sum(pi)

    def _calculate_discrete_p_matrix(self):
        P = np.eye(3)
        term = np.eye(3)
        for i in range(1, 40):
            term = np.dot(term, self.q_matrix) / i
            P = P + term
            if np.max(np.abs(term)) < 1e-8:
                break
        P = np.maximum(P, 0)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1 
        self.p_matrix = P / row_sums[:, np.newaxis]

    def reset_state(self, initial_state_str):
        self.current_time = 0
        self.time_in_states = np.zeros(3)
        
        if initial_state_str == "Ясно":
            self.current_state = 0
        elif initial_state_str == "Облачно":
            self.current_state = 1
        elif initial_state_str == "Пасмурно":
            self.current_state = 2
        else:
            self.current_state = np.random.choice([0, 1, 2])

    def do_simulation_step(self):
        self._calculate_discrete_p_matrix()
        probs = self.p_matrix[self.current_state]
        next_state = np.random.choice([0, 1, 2], p=probs)
        old_state = self.current_state
        
        self.time_in_states[self.current_state] += 1
        self.current_time += 1
        self.current_state = next_state

        return self.current_time, old_state

    def get_empirical_pi(self):
        total_time = np.sum(self.time_in_states)
        if total_time > 0:
            return self.time_in_states / total_time
        return np.zeros(3)

    def get_hypothesis_test(self):
        """ Возвращает (значение хи-квадрат, Принята ли гипотеза) """
        total_time = np.sum(self.time_in_states)
        
        # Для Пирсона нужно чтобы ожидаемое число наблюдений было >= 5
        # В среднем это наступает примерно после 15-20 дней симуляции
        expected = total_time * self.theoretical_pi
        if total_time < 15 or np.any(expected < 3):
            return None, False

        observed = self.time_in_states
        chi_square = np.sum((observed - expected)**2 / expected)
        
        # Степени свободы = 3 состояния - 1 = 2. Уровень значимости = 0.05. Критическое значение = 5.991
        is_accepted = chi_square < 5.991 
        return chi_square, is_accepted