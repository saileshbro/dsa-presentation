import sys
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import math # Import math for log in plotting

# Increase recursion limit - mostly for safety if you were to add recursive sorts later
# Not strictly necessary for the iterative implementations here.
# Be cautious setting this very high on production systems.
sys.setrecursionlimit(10000)

# ----- SORTING ALGORITHMS -----

def insertion_sort(arr):
    """Pure insertion sort with O(n²) complexity"""
    # Sorts in-place on the provided array (expected to be a copy)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr # Return the modified array

# --- CORRECTED Deterministic Quicksort ---
def deterministic_quicksort_iterative(arr):
    """Iterative implementation of deterministic quicksort with median-of-three pivot
       using Lomuto partition scheme."""
    # Sorts in-place on the provided array (expected to be a copy)

    stack = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()

        if low >= high:
            continue

        # --- Median-of-Three Pivot Selection and Swap to high ---
        # Choosing median of low, mid, high indices
        mid = low + (high - low) // 2

        # Sort elements at low, mid, high to put median at mid
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]: # Only need to compare arr[high] and arr[mid] now
            arr[mid], arr[high] = arr[high], arr[mid]

        # Swap the median element (now at arr[mid]) with arr[high]
        # This places the pivot at the end for the Lomuto partition
        arr[mid], arr[high] = arr[high], arr[mid]

        # Pivot is now arr[high]
        pivot = arr[high]

        # --- Lomuto Partition ---
        # i is the index of the last element <= pivot found so far
        i = low - 1

        # Iterate through elements from low up to high-1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        # Place pivot element at the correct position (i+1)
        # Elements from low to i are <= pivot
        # Elements from i+1 to high-1 are > pivot
        # The pivot (originally at high) should go at i+1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]

        # p is the partitioning index (final position of the pivot)
        p = i + 1

        # --- Push Subproblems ---
        # Push the left partition (elements from low to p-1)
        stack.append((low, p - 1))
        # Push the right partition (elements from p+1 to high)
        stack.append((p + 1, high))

    return arr # Return the modified array

def randomized_quicksort_iterative(arr):
    """Iterative implementation of randomized quicksort with optimizations"""
    # Sorts in-place on the provided array (expected to be a copy)

    # Use stack to avoid recursion
    stack = [(0, len(arr) - 1)]

    # Cutoff size for switching to insertion sort
    CUTOFF = 10

    while stack:
        low, high = stack.pop()

        # Base case: sub-array has 0 or 1 element
        if low >= high:
            continue

        # For very small subarrays, use insertion sort (optimization)
        # high - low + 1 is the size of the subarray
        if high - low + 1 < CUTOFF:
            # Insertion sort for this small slice from index low to high
            for i in range(low + 1, high + 1):
                key = arr[i]
                j = i - 1
                while j >= low and arr[j] > key:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            continue # Subarray sorted, move to next stack item

        # --- Randomized Pivot Selection and Partition ---
        # Choose a random element's index
        pivot_idx = random.randint(low, high)
        # Swap the random element to the high position
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        # The pivot value is now the element at the high position
        pivot = arr[high]

        # --- Lomuto Partition ---
        # i is the index of the last element <= pivot found so far
        i = low - 1

        # Iterate through elements from low up to high-1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        # Place pivot element at the correct position (i+1)
        # Elements from low to i are <= pivot
        # Elements from i+1 to high-1 are > pivot
        # The pivot (originally at high) should go at i+1
        arr[i + 1], arr[high] = arr[high], arr[i + 1]

        # p is the partitioning index (final position of the pivot)
        p = i + 1

        # --- Push Subproblems ---
        # Push larger partition first to limit stack depth (optimization)
        # This is important for correctness and performance of iterative Quicksort
        if high - p > p - low:
            stack.append((p + 1, high))
            stack.append((low, p - 1))
        else:
            stack.append((low, p - 1))
            stack.append((p + 1, high))

    return arr

# ----- BENCHMARKING FUNCTIONS -----

def benchmark_sort(sort_fn, arr_copy):
    """Time a sorting function on a pre-copied array"""
    # The sort_fn is expected to sort arr_copy in-place.
    start = time.perf_counter()
    sort_fn(arr_copy) # Pass the copy to the sorting function
    end = time.perf_counter()
    return end - start

def generate_test_cases(size, case_type):
    """Generate different test cases based on the case type"""
    if case_type == "random":
        # Ensure distinct elements for predictable randomized quicksort performance
        return random.sample(range(size * 10), size)
    elif case_type == "sorted":
        return list(range(size))
    elif case_type == "reversed":
        return list(range(size, 0, -1))
    elif case_type == "few_unique":
        # Generate with a limited range of values
        return [random.randint(0, max(10, size // 100)) for _ in range(size)] # Max value scales slightly with size
    else:
        raise ValueError(f"Unknown case type: {case_type}")

def bench_single(sort_name, sort_fn, size, case_type, num_runs):
    """Benchmark a specific sorting function"""
    total_time = 0
    print(f"  Benchmarking {sort_name.capitalize()} sort on {case_type} array of size {size:,} ({num_runs} runs)...")
    try:
        for i in range(num_runs):
            arr = generate_test_cases(size, case_type)
            # Benchmark function now expects a copy
            total_time += benchmark_sort(sort_fn, arr.copy())
            # Optional: Print progress for very long runs
            # if num_runs > 1 and i > 0 and i % max(1, num_runs // 5) == 0:
            #     print(f"    {sort_name} size {size:,}: Run {i+1}/{num_runs} complete...")

        avg_time = total_time / num_runs
        return (sort_name, size, case_type, avg_time)
    except Exception as e:
        print(f"  Error during benchmark for {sort_name} size {size:,}: {e}")
        # Return None or raise exception, depending on desired behavior
        # Returning None will cause it to be skipped in results
        return None


def run_benchmarks():
    """Run benchmarks focused on specific cases for different algorithms"""
    # Define array sizes to test.
    # We'll use different size sets for Quicksort vs Insertion sort
    quicksort_sizes = [100, 500, 1000, 2500, 5000, 10000, 20000, 30000, 50000, 75000, 100000]

    # Maximum size to run insertion sort on (since it's O(n²))
    insertion_sort_max_size = 25000

    # Focus on specific cases as requested
    # Setting up algorithms with their worst/best case scenarios:
    # - Deterministic Quicksort: worst case (sorted array)
    # - Insertion Sort: worst case (reversed array)
    # - Randomized Quicksort: average case (random array)
    case_mapping = {
        "deterministic": "sorted",
        "randomized": "random",
        "insertion": "reversed"
    }

    sort_functions = {
        "deterministic": deterministic_quicksort_iterative,
        "randomized": randomized_quicksort_iterative,
        "insertion": insertion_sort
    }

    results = {
        "deterministic": {},
        "randomized": {},
        "insertion": {}
    }

    futures = []
    print(f"Starting benchmarks with array sizes up to {quicksort_sizes[-1]:,}...")
    print("WARNING: Insertion sort is O(n^2) and will be very slow for larger sizes.")
    print("         Tasks taking excessively long might effectively hang or use extreme resources.")

    with ProcessPoolExecutor(max_workers=4) as executor: # Use up to 4 CPU cores
        for size in quicksort_sizes:
            # Adjust number of runs based on size to keep total time reasonable
            if size <= 7500:
                num_runs = 30
            elif size <= 10000:
                num_runs = 20
            elif size <= 25000:
                num_runs = 15
            elif size <= 50000:
                num_runs = 10
            elif size <= 75000:
                num_runs = 5
            elif size <= 100000:
                num_runs = 3
            else: # Very large sizes
                num_runs = 1 # Reduced runs for largest sizes

            # Submit tasks for all algorithms for the current size
            for sort_name, sort_fn in sort_functions.items():
                 # Use smaller number of runs for Insertion sort, especially for worst-case
                 current_num_runs = num_runs
                 if sort_name == "insertion":
                     # Reduce runs for insertion sort as it's O(n²)
                     if size >= 1000: current_num_runs = max(1, num_runs // 3)
                     if size >= 5000: current_num_runs = max(1, num_runs // 5)
                     if size >= 10000: current_num_runs = 1
                     # Skip insertion sort for sizes that are too large
                     if size > insertion_sort_max_size:
                         print(f"  Skipping Insertion Sort for size {size:,} due to O(n²) time complexity.")
                         continue
                     # Add a hard skip for sizes that are almost certainly too slow
                     if size > 500000: # Even 500k is likely hours, 1M is days
                         print(f"  Skipping Insertion Sort for size {size:,} due to extreme O(n^2) runtime.")
                         continue

                 case_type = case_mapping[sort_name]

                 futures.append(executor.submit(
                     bench_single, sort_name, sort_fn,
                     size, case_type, current_num_runs
                 ))


        # Process results as they complete
        completed = 0
        total_tasks = len(futures)
        print(f"\nSubmitted {total_tasks} benchmark tasks. Waiting for results...\n")
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    sort_name, size, case_type, avg_time = result
                    results[sort_name][size] = avg_time
                    completed += 1
                    # Print completion message (already done inside bench_single now)
                    # print(f"[{completed}/{total_tasks}] {sort_name.capitalize()} sort on {case_type} array of size {size:,}: {avg_time:.6f}s")
                else:
                     completed += 1 # Count None results as completed tasks
            except Exception as e:
                completed += 1 # Count failed task too
                print(f"[{completed}/{total_tasks}] An unhandled error occurred for one task: {e}")

    print(f"\nAll benchmark tasks attempted. Tested quicksort with sizes up to {quicksort_sizes[-1]:,} elements.")
    print(f"Insertion sort attempted with sizes up to {quicksort_sizes[-1]:,} elements (some large sizes might have been skipped, timed out, or failed).")


    # Combine all sizes actually tested for plotting
    all_sizes = sorted(set().union(*[set(d.keys()) for d in results.values()]))


    return all_sizes, results

# ----- VISUALIZATION FUNCTIONS -----

def plot_results(sizes, results):
    """Create a more polished plot comparing sorting algorithms"""
    plt.figure(figsize=(14, 10))  # Larger figure size

    # Set plot style for better aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')

    # Prepare data points, using NaN for missing sizes
    # Only include sizes for which data exists
    det_sizes = sorted(results.get("deterministic", {}).keys())
    det_times = [results["deterministic"][size] for size in det_sizes]

    rand_sizes = sorted(results.get("randomized", {}).keys())
    rand_times = [results["randomized"][size] for size in rand_sizes]

    insertion_sizes = sorted(results.get("insertion", {}).keys())
    insertion_times = [results["insertion"][size] for size in insertion_sizes]

    # Plot with enhanced styling
    if det_sizes:
        plt.plot(det_sizes, det_times, 'ro-', label='Deterministic Quicksort (worst-case: sorted array)',
                 linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1)
    if rand_sizes:
        plt.plot(rand_sizes, rand_times, 'bs-', label='Randomized Quicksort (random array)',
                 linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1)

    # Add insertion sort if data exists for any size
    if insertion_sizes:
        plt.plot(insertion_sizes, insertion_times, 'gD-', label='Insertion Sort (worst-case: reversed array)',
                 linewidth=3, markersize=10, markeredgecolor='black', markeredgewidth=1)

    # Get all unique sizes that were actually benchmarked for reference lines
    all_tested_sizes = sorted(set().union(results.get("deterministic", {}).keys(),
                                         results.get("randomized", {}).keys(),
                                         results.get("insertion", {}).keys()))

    # Add theoretical complexity lines for reference
    if all_tested_sizes and len(all_tested_sizes) > 1:
        reference_sizes = np.array(all_tested_sizes)

        # Add more points for smoother curves
        extended_sizes = np.logspace(np.log10(min(reference_sizes)), np.log10(max(reference_sizes)), 100)

        # Calculate reference scaling factors based on middle-sized array
        mid_index = len(reference_sizes) // 2
        mid_size = reference_sizes[mid_index]

        # Get actual times at the middle size (if available)
        mid_det_time = results.get("deterministic", {}).get(mid_size, None)
        mid_rand_time = results.get("randomized", {}).get(mid_size, None)
        mid_ins_time = results.get("insertion", {}).get(mid_size, None)

        # O(n²) complexity reference
        if mid_ins_time is not None:
            n2_factor = mid_ins_time / (mid_size**2) * 0.8  # Factor with visual adjustment
            n2_times = [size**2 * n2_factor for size in extended_sizes]
            plt.plot(extended_sizes, n2_times, 'k--', alpha=0.7, linewidth=2, label='O(n²) reference')

        # O(n log n) complexity reference
        if mid_rand_time is not None:
            nlogn_factor = mid_rand_time / (mid_size * np.log(mid_size)) * 0.8  # Factor with visual adjustment
            nlogn_times = [size * np.log(size) * nlogn_factor for size in extended_sizes]
            plt.plot(extended_sizes, nlogn_times, 'k:', alpha=0.7, linewidth=2, label='O(n log n) reference')

    # Enhanced styling
    plt.title('Sorting Algorithm Performance Comparison', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Array Size (n)', fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel('Time (seconds)', fontsize=16, fontweight='bold', labelpad=10)
    plt.grid(True, which="both", alpha=0.3, linestyle='--')

    # Create a more visually appealing legend
    legend = plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9,
                        shadow=True, borderpad=1, loc='upper left')
    legend.get_frame().set_facecolor('#f0f0f0')

    plt.xscale('log')
    plt.yscale('log')

    # Annotate key points if data is available
    if det_sizes and rand_sizes and insertion_sizes:
        # Find the largest common size for comparison annotations
        common_sizes = set(det_sizes) & set(rand_sizes) & set(insertion_sizes)
        if common_sizes:
            max_common = max(common_sizes)
            det_time = results["deterministic"][max_common]
            rand_time = results["randomized"][max_common]
            ins_time = results["insertion"][max_common]

            # Add annotations showing relative performance
            if ins_time > det_time:
                ratio = ins_time / det_time
                plt.annotate(f"~{ratio:.0f}x slower",
                            xy=(max_common, ins_time),
                            xytext=(max_common*0.8, ins_time*0.5),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                            fontsize=12)

    # Add a text box explaining the time complexities
    plt.figtext(0.02, 0.02,
               "Time Complexity Analysis:\n"
               "- Insertion Sort (reversed): O(n²) - Quadratic time\n"
               "- Deterministic Quicksort (sorted): O(n²) worst case, but median-of-three helps\n"
               "- Randomized Quicksort (random): O(n log n) average case",
               fontsize=12,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig('./sorting_algorithms_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print performance comparison table
    print("\nPerformance Comparison Table:")
    print("-" * 90) # Adjusted width
    print(f"{'Array Size':>12} | {'Deterministic QS (Worst)':>23} | {'Randomized QS (Random)':>23} | {'Insertion Sort (Worst)':>23} | {'QS Speedup (Det/Rand)':>22}") # Adjusted headers
    print("-" * 90) # Adjusted width

    # Get all unique sizes that were actually benchmarked
    all_tested_sizes = sorted(set().union(results.get("deterministic", {}).keys(), results.get("randomized", {}).keys(), results.get("insertion", {}).keys()))

    for size in all_tested_sizes:
        det_time = results.get("deterministic", {}).get(size, float('nan'))
        rand_time = results.get("randomized", {}).get(size, float('nan'))
        ins_time = results.get("insertion", {}).get(size, float('nan'))

        # Calculate speedups
        qs_speedup = float('nan')
        if not np.isnan(det_time) and not np.isnan(rand_time) and rand_time > 0:
             qs_speedup = det_time / rand_time

        # Format each time value, handling NaN cases
        det_str = f"{det_time:.6f}s" if not np.isnan(det_time) else "N/A"
        rand_str = f"{rand_time:.6f}s" if not np.isnan(rand_time) else "N/A"
        ins_str = f"{ins_time:.6f}s" if not np.isnan(ins_time) else "N/A"
        speedup_str = f"{qs_speedup:.2f}x" if not np.isnan(qs_speedup) else "N/A"

        print(f"{size:12,d} | {det_str:>23} | {rand_str:>23} | {ins_str:>23} | {speedup_str:>22}")


def test_correctness():
    """Test that all sorting implementations work correctly"""
    test_cases = [
        [],
        [1],
        [2, 1],
        [3, 1, 4, 1, 5, 9, 2, 6, 5],
        list(range(10, 0, -1)), # Reversed
        [5, 5, 5, 5, 5], # Few unique
        list(range(10)), # Already sorted
        list(range(10, 0, -1)), # Reversed
        random.sample(range(100), 50), # Random
        [random.randint(0, 5) for _ in range(20)], # Few unique
        [1, 2, 1, 2, 1, 2], # Repeated elements
    ]

    sort_functions = {
        "insertion": insertion_sort,
        "deterministic": deterministic_quicksort_iterative,
        "randomized": randomized_quicksort_iterative
    }

    all_passed = True
    print("Running correctness tests...")
    for i, arr in enumerate(test_cases):
        expected = sorted(arr)
        original_arr = arr.copy() # Keep a copy of the original test case

        for name, sort_fn in sort_functions.items():
            try:
                # Pass a copy to the sorting function so it doesn't modify original_arr
                result = sort_fn(original_arr.copy())
                assert result == expected, f"{name.capitalize()} sort failed for input {original_arr} -> Got {result}, Expected {expected}"
                # print(f"  ✅ {name.capitalize()} passed test case {i+1}") # Optional: print individual test success
            except AssertionError as e:
                print(f"  ❌ Correctness test failed: {e}")
                all_passed = False
            except Exception as e:
                 print(f"  ❌ An error occurred during correctness test for {name.capitalize()} with input {original_arr}: {e}")
                 all_passed = False

    print("-" * 30)
    if all_passed:
        print("✅ All sorting implementations passed correctness tests!")
    else:
        print("❌ Some correctness tests failed!")
    print("-" * 30)


# ----- MAIN EXECUTION -----

if __name__ == "__main__":
    # First test correctness
    test_correctness()

    # Then run benchmarks if tests passed (optional - remove if you want to benchmark even if tests fail)
    # if all_passed: # Added check
    print("\n" + "="*40 + "\nStarting Benchmarks\n" + "="*40)
    sizes, results = run_benchmarks()

    # Plot results
    print("\n" + "="*40 + "\nGenerating Plots\n" + "="*40)
    plot_results(sizes, results)

    print("\nBenchmark complete! Results have been plotted and printed.")
    # else: # Added else block
    #     print("\nSkipping benchmarks due to correctness test failures.")