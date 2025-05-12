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
    # Define array sizes to test. Insertion sort will be run on ALL these sizes.
    # BE PREPARED FOR INSERTION SORT TO BE EXTREMELY SLOW FOR SIZES > ~50,000.
    quicksort_sizes = [100, 1000, 10000, 50000, 100000, 250000, 500000, 1000000]

    # Focus on specific cases as requested
    # Deterministic Quicksort: worst case (sorted array)
    # Randomized Quicksort: average case (random array)
    # Insertion Sort: standard case (random array)
    case_mapping = {
        "deterministic": "sorted",
        "randomized": "random",
        "insertion": "random"
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
            if size <= 1000:
                num_runs = 10
            elif size <= 10000:
                num_runs = 5
            elif size <= 100000:
                num_runs = 3
            else: # Very large sizes
                num_runs = 1 # Reduced runs for largest sizes

            # Submit tasks for all algorithms for the current size
            for sort_name, sort_fn in sort_functions.items():
                 # Use smaller number of runs for Insertion sort at very large sizes
                 current_num_runs = num_runs
                 if sort_name == "insertion":
                     if size > 50000: current_num_runs = max(1, num_runs // 2)
                     if size > 100000: current_num_runs = 1
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
    """Create a plot comparing sorting algorithms"""
    plt.figure(figsize=(12, 8)) # Make figure slightly larger

    # Prepare data points, using NaN for missing sizes
    # Only include sizes for which data exists
    det_sizes = sorted(results.get("deterministic", {}).keys())
    det_times = [results["deterministic"][size] for size in det_sizes]

    rand_sizes = sorted(results.get("randomized", {}).keys())
    rand_times = [results["randomized"][size] for size in rand_sizes]

    insertion_sizes = sorted(results.get("insertion", {}).keys())
    insertion_times = [results["insertion"][size] for size in insertion_sizes]


    # Plot with more detailed markers and styling
    if det_sizes:
        plt.plot(det_sizes, det_times, 'ro-', label='Deterministic Quicksort (worst-case: sorted array)',
                 linewidth=2, markersize=8)
    if rand_sizes:
        plt.plot(rand_sizes, rand_times, 'bs-', label='Randomized Quicksort (random array)',
                 linewidth=2, markersize=8)

    # Add insertion sort if data exists for any size
    if insertion_sizes:
        plt.plot(insertion_sizes, insertion_times, 'gd-', label='Insertion Sort (random array, O(n²))',
                 linewidth=2, markersize=8)

    # Add theoretical complexity lines for reference on a log-log plot
    # Use sizes that actually have data for scaling
    if det_sizes and len(det_sizes) > 1:
         # For n^2 complexity (scaled to match a point on the deterministic data)
         valid_det_sizes = [s for s in det_sizes if s > 0]
         if valid_det_sizes: # Check if list is not empty
             # Use the largest valid size with data
             last_det_size = valid_det_sizes[-1]
             last_det_time = results["deterministic"][last_det_size]
             # Calculate factor based on the last data point
             if last_det_size > 0 and last_det_time > 0:
                 n2_factor = last_det_time / (last_det_size**2) * 0.5 # Adjust factor for better visual fit
                 n2_line = [size**2 * n2_factor for size in valid_det_sizes]
                 plt.plot(valid_det_sizes, n2_line, 'r--', alpha=0.5, label='O(n²) reference')


    if rand_sizes and len(rand_sizes) > 1:
        # For n log n complexity (scaled to match a point on the randomized data)
        valid_rand_sizes = [s for s in rand_sizes if s > 1] # log(1) is 0
        if valid_rand_sizes: # Check if list is not empty
            last_rand_size = valid_rand_sizes[-1]
            last_rand_time = results["randomized"][last_rand_size]
            if last_rand_size > 1 and last_rand_time > 0:
                 # Calculate factor based on the last data point
                 nlogn_factor = last_rand_time / (last_rand_size * math.log(last_rand_size)) * 0.8 # Adjust factor
                 nlogn_line = [size * math.log(size) * nlogn_factor for size in valid_rand_sizes]
                 plt.plot(valid_rand_sizes, nlogn_line, 'b--', alpha=0.5, label='O(n log n) reference')


    plt.title('Sorting Algorithm Performance Comparison', fontsize=16)
    plt.xlabel('Array Size (n)', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.grid(True, which="both", alpha=0.3) # Grid on both major and minor ticks
    plt.legend(fontsize=10)
    plt.xscale('log')
    plt.yscale('log')


    plt.tight_layout()
    plt.savefig('sorting_algorithms_benchmark.png', dpi=300)
    plt.show()

    # Print performance comparison table
    print("\nPerformance Comparison Table:")
    print("-" * 90) # Adjusted width
    print(f"{'Array Size':>12} | {'Deterministic QS (Sorted)':>23} | {'Randomized QS (Random)':>23} | {'Insertion Sort (Random)':>23} | {'QS Speedup (Det/Rand)':>22}") # Adjusted headers
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