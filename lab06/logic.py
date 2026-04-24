import numpy as np
import math
from scipy import stats
from typing import List

class MultiplicateConGenerator:
    def __init__(self):
        self.M = 1 << 63
        self.beta = (1 << 32) + 3
        self.x = self.beta

    def generate(self) -> float:
        remains = self.x * self.beta
        self.x = remains % self.M
        return self.x / self.M

class NormalDistributionGenerator:
    def __init__(self, mu: float, std_dev: float, uniform_gen: MultiplicateConGenerator):
        self.k = 32
        self.mu = mu
        self.std_dev = std_dev
        self.uniform_gen = uniform_gen

    def generate_single(self) -> float:
        sum_u = 0.0
        for _ in range(self.k):
            # Используем ваш метод генерации равномерного числа
            sum_u += self.uniform_gen.generate()
        
        # Центрирование и нормирование по ЦПТ
        z = (sum_u - self.k / 2) / math.sqrt(self.k / 12)
        return self.mu + z * self.std_dev

    def generate_sample(self, n: int) -> List[float]:
        return [self.generate_single() for _ in range(n)]

# Создаем глобальный экземпляр базового генератора, чтобы последовательность была непрерывной
base_gen = MultiplicateConGenerator()

def normalize_probabilities(probs):
    total = sum(probs)
    is_normalized = np.isclose(total, 1.0, atol=1e-5)
    new_probs = probs
    message = ""
    if not is_normalized:
        new_probs = [p / total for p in probs]
        message = f"Внимание! Вероятности нормированы.\nНовые значения: {[round(p, 4) for p in new_probs]}"
    return new_probs, message

def discrete_simulation(probs, n_samples):
    """Моделирование дискретной величины вручную."""
    # 1. Построение кумулятивных вероятностей
    cum_probs = np.cumsum(probs)
    
    samples = []
    for _ in range(n_samples):
        u = base_gen.generate()
        # Определяем, в какой интервал попало число
        for i, cp in enumerate(cum_probs):
            if u < cp:
                samples.append(i + 1)
                break
    
    samples = np.array(samples)
    counts = np.bincount(samples, minlength=6)[1:]
    emp_probs = counts / n_samples
    
    # Теоретические показатели
    values = np.arange(1, 6)
    theoretical_mean = sum(v * p for v, p in zip(values, probs))
    theoretical_var = sum((v**2) * p for v, p in zip(values, probs)) - theoretical_mean**2
    
    # Выборочные показатели
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1) if n_samples > 1 else 0
    
    # Погрешности
    mean_err = abs(sample_mean - theoretical_mean) / (abs(theoretical_mean) + 1e-9)
    var_err = abs(sample_var - theoretical_var) / (abs(theoretical_var) + 1e-9)
    
    # Хи-квадрат
    expected_counts = np.array(probs) * n_samples
    expected_counts = expected_counts * (n_samples / np.sum(expected_counts)) # Исправление точности
    
    chi_stat, _ = stats.chisquare(counts, f_exp=expected_counts)
    critical_chi = stats.chi2.ppf(0.95, df=len(probs)-1)
    
    return {
        'samples': samples,
        'emp_probs': emp_probs,
        'mean': sample_mean,
        'var': sample_var,
        'mean_err': mean_err,
        'var_err': var_err,
        'chi_stat': chi_stat,
        'critical_chi': critical_chi,
        'is_accepted': chi_stat < critical_chi
    }

def normal_simulation(mu, var, n_samples):
    """Моделирование нормальной величины через ваш генератор."""
    std_dev = math.sqrt(var)
    norm_gen = NormalDistributionGenerator(mu, std_dev, base_gen)
    
    # Генерация выборки вашим методом
    samples = np.array(norm_gen.generate_sample(n_samples))
    
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1)
    
    # Группировка для Хи-квадрат
    counts, bin_edges = np.histogram(samples, bins=10)
    expected_probs = []
    for i in range(len(bin_edges)-1):
        if i == 0:
            p = stats.norm.cdf(bin_edges[i+1], mu, std_dev)
        elif i == len(bin_edges)-2:
            p = 1 - stats.norm.cdf(bin_edges[i], mu, std_dev)
        else:
            p = stats.norm.cdf(bin_edges[i+1], mu, std_dev) - stats.norm.cdf(bin_edges[i], mu, std_dev)
        expected_probs.append(p)
    
    expected_counts = np.array(expected_probs) * n_samples
    expected_counts = expected_counts * (n_samples / np.sum(expected_counts))
    
    mask = expected_counts > 0
    chi_stat, _ = stats.chisquare(counts[mask], f_exp=expected_counts[mask])
    df = np.sum(mask) - 1
    critical_chi = stats.chi2.ppf(0.95, df=max(1, df))
    
    return {
        'samples': samples,
        'mean': sample_mean,
        'var': sample_var,
        'chi_stat': chi_stat,
        'critical_chi': critical_chi
    }