
from collections import defaultdict
import chaospy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from utils.oscillator import Oscillator

def load_reference(filename: str) -> tuple[float, float]:
     # TODO: load reference values for the mean and variance加载均值和方差的参考值
    with open(filename, 'r') as f:
        lines = f.readlines()
        mean = float(lines[0].strip())
        var = float(lines[1].strip())
    return mean, var

def simulate(
    # simulate the oscillator  -------模拟振荡器
    # with the given parameters and return
    t_grid: npt.NDArray,
    omega_distr: cp.Distribution,
    n_samples: int,
    model_kwargs: dict[str, float],
    init_cond: dict[str, float],
    rule="random",
    seed=36,
) -> npt.NDArray:
    # omega_distr: chaospy.Distribution,
    # 生成样本点
    samples = omega_distr.sample(n_samples, rule=rule,seed=seed)
    sample_solutions = np.zeros((n_samples, len(t_grid)))
    
    for i in range(n_samples):
        omega = float(samples[i])
        oscillator = Oscillator(**model_kwargs, omega=omega)
        solution = oscillator.solve(t_grid, init_cond)
        sample_solutions[i] = solution
    
    return sample_solutions

def compute_relative_errors(
    samples: npt.NDArray, mean_ref: float, var_ref: float
) -> tuple[float, float]:
    
    # TODO: compute the absolute errors of the mean and variance 
    mean_est = np.mean(samples[:, -1])
    var_est = np.var(samples[:, -1], ddof=1)
    mean_error = np.abs((mean_est - mean_ref) / mean_ref)
    var_error = np.abs((var_est - var_ref) / var_ref)
    return mean_error, var_error

if __name__ == "__main__":
    # define the parameters of the simulations参数设置
    
    t_grid = np.arange(0, 10.01, 0.01)
    omega_distr = cp.Uniform(0.95, 1.05)
    sample_sizes = [10, 100, 1000, 10000]
    model_kwargs = {"c": 0.5, "k": 2.0, "f": 0.5}
    init_cond = {"y0": 0.5, "y1": 0.0}

    mean_ref, var_ref = load_reference("oscillator_ref.txt")

    rules = ["random", "halton"]
    results = defaultdict(lambda: {"mean_errors": [], "var_errors": []})
    mc_trajectories = None

    for rule in rules:
        for n in sample_sizes:
            samples = simulate(t_grid, omega_distr, n, model_kwargs, init_cond, rule, seed=36)
            mean_error, var_error = compute_relative_errors(samples, mean_ref, var_ref)
            results[rule]["mean_errors"].append(mean_error)
            results[rule]["var_errors"].append(var_error)

            if rule == "random" and n == sample_sizes[-1]:
                mc_trajectories = samples[:10, :]

    # 绘制误差收敛图
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    
    # TODO: plot the results on the log-log scale绘制均值和方差的相对误差
    for rule in rules:
        label = "Monte Carlo" if rule == "random" else "QMC Halton"
        plt.loglog(sample_sizes, results[rule]["mean_errors"], 'o-', label=label)
    plt.xlabel("number of samples")
    plt.ylabel("Relative error in mean")
    plt.title("Convergence of Mean Estimate")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for rule in rules:
        label = "Monte Carlo" if rule == "random" else "QMC Halton"
        plt.loglog(sample_sizes, results[rule]["var_errors"], 'o-', label=label)
    plt.xlabel("number of samples")
    plt.ylabel("Relative error in variance")
    plt.title("Convergence of Variance Estimate")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("convergence.png")

    #  TODO: plot sampled trajectories绘制 10 条样本轨迹图
    plt.figure(figsize=(10, 8))
    
    for i in range(10):
        plt.plot(t_grid, mc_trajectories[i], alpha=0.7, linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.title("10 Solution Trajectories using MC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trajectory.png")

    plt.show()
