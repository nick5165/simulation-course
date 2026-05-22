import math

class MultiplicateConGenerator:
    def __init__(self):
        self.M = 1 << 63
        self.beta = (1 << 32) + 3
        self.x = self.beta

    def generate(self) -> float:
        remains = self.x * self.beta
        self.x = remains % self.M
        return self.x / self.M
    
class FlowCalulations:
    def __init__(self, N: int, T: float, intensity: float):
        self.N = N
        self.T = T
        self.intensity = intensity

    def simulate(self) -> dict:
        frequencies = {}
        generator = MultiplicateConGenerator()
        for _ in range(self.N):
            count = self.simulate_single_exp(generator)
            frequencies[count] = frequencies.get(count, 0) + 1
        return frequencies

    def simulate_single_exp(self, generator: MultiplicateConGenerator) -> int:
        t =0.0
        count = 0
        while True:
            alpha = generator.generate()
            dt = -math.log(alpha) / self.intensity
            t += dt
            if t< self.T:
                count += 1
            else:
                break

        return count
    
    def get_statistics(self, frequencies: dict) -> dict:
        total_runs = sum(frequencies.values())
        if total_runs == 0:
            return {}

        empirical_mean = sum(k * v for k, v in frequencies.items()) / total_runs
        empirical_variance = sum(v * (k - empirical_mean) ** 2 for k, v in frequencies.items()) / total_runs

        theoretical_mean = self.intensity * self.T
        theoretical_variance = self.intensity * self.T

        counts = sorted(frequencies.keys())
        empirical_probs = [frequencies[k] / total_runs for k in counts]
        
        theoretical_probs = []
        for k in counts:
            try:
                prob = (math.pow(theoretical_mean, k) * math.exp(-theoretical_mean)) / math.factorial(k)
            except OverflowError:
                prob = 0.0
            theoretical_probs.append(prob)

        return {
            "empirical_mean": empirical_mean,
            "empirical_variance": empirical_variance,
            "theoretical_mean": theoretical_mean,
            "theoretical_variance": theoretical_variance,
            "counts": counts,
            "empirical_probs": empirical_probs,
            "theoretical_probs": theoretical_probs
        }
