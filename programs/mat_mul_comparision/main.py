import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Deterministic: Standard matrix multiplication (triple loop)
def deterministic_matmul(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Probabilistic: Randomized sketching (approximate multiplication)
def probabilistic_matmul(A, B, s=10):
    # s: number of random samples (sketch size)
    n = A.shape[0]
    C = np.zeros((n, n))
    for _ in range(s):
        # Randomly sample a column from A and a row from B
        idx = np.random.randint(0, n)
        C += np.outer(A[:, idx], B[idx, :])
    C /= s
    return C

def run_benchmark(n):
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    # Deterministic
    start = time.time()
    deterministic_matmul(A, B)
    det_time = time.time() - start

    # Probabilistic
    start = time.time()
    probabilistic_matmul(A, B, s=10)
    prob_time = time.time() - start

    return n, det_time, prob_time

def main():
    sizes = list(range(10, 1001, 10))
    det_times = []
    prob_times = []
    size_order = []

    print("Starting matrix multiplication benchmarks...")
    print("Matrix sizes to test:", sizes)
    print("-" * 50)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_benchmark, n) for n in sizes]
        for future in as_completed(futures):
            n, det_time, prob_time = future.result()
            size_order.append(n)
            det_times.append((n, det_time))
            prob_times.append((n, prob_time))
            print(f"Completed size {n}x{n}:")
            print(f"  Deterministic: {det_time:.6f} seconds")
            print(f"  Probabilistic: {prob_time:.6f} seconds")
            print(f"  Speedup: {det_time/prob_time:.2f}x")
            print("-" * 50)

    # Sort results by matrix size
    det_times.sort()
    prob_times.sort()
    sizes_sorted = [n for n, _ in det_times]
    det_times_sorted = [t for _, t in det_times]
    prob_times_sorted = [t for _, t in prob_times]

    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_sorted, det_times_sorted, 'b-', label='Deterministic (Standard)')
    plt.plot(sizes_sorted, prob_times_sorted, 'r-', label='Probabilistic (Randomized Sketch)')
    plt.yscale('log')
    plt.xlabel('Matrix Size (n x n)')
    plt.ylabel('Time (seconds, log scale)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    plt.close()
    print("Plot saved as 'comparison.png'")

if __name__ == "__main__":
    main()