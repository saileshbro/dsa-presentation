import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os

# Create images directory if it doesn't exist
os.makedirs("programs/nqueens-lv/images", exist_ok=True)

def is_safe(board, row, col, n):
    for i in range(col):
        if board[row][i] == 1:
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    return True

def solve_nqueens_backtracking(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    cols = set()
    diag1 = set()
    diag2 = set()
    def backtrack(row):
        if row == n:
            return True
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            board[row][col] = 1
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            if backtrack(row + 1):
                return True
            board[row][col] = 0
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
        return False
    start_time = time.time()
    result = backtrack(0)
    elapsed_time = time.time() - start_time
    return board if result else None, elapsed_time

def solve_nqueens_las_vegas(n, max_attempts=1000):
    start_time = time.time()
    for attempt in range(max_attempts):
        board = [[0 for _ in range(n)] for _ in range(n)]
        cols = set()
        diag1 = set()
        diag2 = set()
        success = True
        for row in range(n):
            available = [col for col in range(n)
                         if col not in cols and (row - col) not in diag1 and (row + col) not in diag2]
            if not available:
                success = False
                break
            col = random.choice(available)
            board[row][col] = 1
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
        if success:
            elapsed_time = time.time() - start_time
            return board, elapsed_time
    elapsed_time = time.time() - start_time
    return None, float('nan')

def draw_chessboard(board, n):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            color = 'white' if (i+j) % 2 == 0 else 'gray'
            ax.add_patch(plt.Rectangle((j, n-i-1), 1, 1, color=color))
            if board[i][j] == 1:
                ax.text(j+0.5, n-i-0.5, '♕', fontsize=30, ha='center', va='center')
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.title(f"{n}-Queens Solution")
    plt.tight_layout()
    plt.savefig(f"programs/nqueens-lv/images/{n}queens-solution.png", dpi=300, bbox_inches='tight')
    plt.close()

def compare_performance():
    n_values = [4,6,8,10,12,14,16,18,20,24,28,32,40,48,64,80,96,112,128]
    backtracking_times = []
    las_vegas_times = []
    for n in n_values:
        # Reduce number of trials for large N
        if n <= 20:
            num_trials = 5
        elif n <= 64:
            num_trials = 3
        else:
            num_trials = 1
        bt_total = 0
        lv_total = 0
        for _ in range(num_trials):
            if n <= 20:
                _, bt_time = solve_nqueens_backtracking(n)
                bt_total += bt_time
            else:
                bt_total = float('nan')
            _, lv_time = solve_nqueens_las_vegas(n)
            lv_total += lv_time
        backtracking_times.append(bt_total / num_trials if bt_total != float('nan') else float('nan'))
        las_vegas_times.append(lv_total / num_trials)
        print(f"N={n}: Backtracking={backtracking_times[-1]:.4f}s, Las Vegas={las_vegas_times[-1]:.4f}s")
    plt.figure(figsize=(12, 7))
    # Only plot backtracking for N where it was measured
    valid_bt_n = [n for i, n in enumerate(n_values) if not np.isnan(backtracking_times[i])]
    valid_bt = [t for t in backtracking_times if not np.isnan(t)]
    if valid_bt_n:
        plt.plot(valid_bt_n, valid_bt, 'o-', label='Backtracking (N ≤ 20)', color='#1f77b4')
    plt.plot(n_values, las_vegas_times, 's-', label='Las Vegas', color='#ff7f0e')
    # Add a note to the plot
    if valid_bt_n and valid_bt_n[-1] < n_values[-1]:
        plt.annotate('Backtracking not shown for N > 20 (too slow)',
                     xy=(valid_bt_n[-1], valid_bt[-1]),
                     xytext=(valid_bt_n[-1]+10, valid_bt[-1]*2),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=12, color='#1f77b4')
    plt.title('Performance Comparison: N-Queens Problem')
    plt.xlabel('Board Size (N)')
    plt.ylabel('Average Solution Time (seconds)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("programs/nqueens-lv/images/performance-comparison.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    random.seed(42)
    # Use higher max_attempts for small N
    max_attempts = 1000 if 8 <= 20 else 100
    board, _ = solve_nqueens_las_vegas(8, max_attempts=1000)
    if board is not None:
        draw_chessboard(board, 8)
        print("8-Queens solution image generated")
    else:
        print("Warning: No solution found for 8-Queens with Las Vegas approach.")
    print("Comparing performance...")
    compare_performance()
    print("Performance comparison chart generated")
    print("All images have been saved to the programs/nqueens-lv/images/ directory")