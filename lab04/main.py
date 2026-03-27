import random
from typing import List, Tuple

from generator import MultiplicateConGenerator, FibGenerator

def generate_data() -> Tuple[List[float], List[float], List[float]]:
    mcg_gen = MultiplicateConGenerator()
    fib_gen = FibGenerator()

    data_mcg = []
    data_fib = []
    data_python = []

    for _ in range(100_000):
        data_mcg.append(mcg_gen.generate())
        data_fib.append(fib_gen.generate())
        data_python.append(random.random())

    return data_mcg, data_fib, data_python

def observed_values(sample: List[float]) -> Tuple[float, float]:
    observed_mean = sum(sample) / len(sample)
    sum_sq = 0

    for x in sample:
        sum_sq += (x - observed_mean) ** 2
    
    return observed_mean, sum_sq / (len(sample) - 1)

def main():
    data_mcg, data_fib, data_python = generate_data()

    mcg_mean, mcg_var = observed_values(data_mcg)
    fib_mean, fib_var = observed_values(data_fib)
    python_mean, python_var = observed_values(data_python)

    print(f"{'Источник':<25} {'Среднее':<15} {'Дисперсия':<15}")
    print("-" * 55)
    print(f"{'Теоретические':<25} {'0.50000':<15} {'0.08333':<15}")
    print("-" * 55)
    print(f"{'Мой МКГ':<25} {mcg_mean:<15.5f} {mcg_var:<15.5f}")
    print("-" * 55)
    print(f"{'Мой Фибоначчи':<25} {fib_mean:<15.5f} {fib_var:<15.5f}")
    print("-" * 55)
    print(f"{'Встроенный генератор':<25} {python_mean:<15.5f} {python_var:<15.5f}")


if __name__ == "__main__":
    main()