import time
import functools
from lru_cache import LRUCache


def run_benchmark(iterations=100_000, capacity=10_000):
    """
    Benchmarks standard functools.lru_cache against the custom LRUCache.
    """
    print(f"Benchmarking with {iterations:,} operations and capacity {capacity:,}...\n")

    # --- Setup Standard functools.lru_cache ---
    @functools.lru_cache(maxsize=capacity)
    def standard_cache_func(key):
        return key * 2

    # --- Setup Custom LRUCache ---
    custom_cache = LRUCache(capacity)

    # ==========================
    # Test 1: Sequential Writes
    # ==========================
    start_time = time.time()
    for i in range(iterations):
        standard_cache_func(i)
    standard_write_time = time.time() - start_time

    start_time = time.time()
    for i in range(iterations):
        custom_cache.put(i, i * 2)
    custom_write_time = time.time() - start_time

    print("Sequential Writes (Misses):")
    print(f"  functools.lru_cache: {standard_write_time:.4f} seconds")
    print(f"  custom LRUCache:     {custom_write_time:.4f} seconds")
    print(f"  Ratio:               {custom_write_time / standard_write_time:.2f}x\n")

    # ==========================
    # Test 2: Sequential Reads (Hits)
    # ==========================
    # Reset tracking loops to iterate over recent elements
    keys_to_read = list(range(iterations - capacity, iterations))

    start_time = time.time()
    for key in keys_to_read:
        standard_cache_func(key)
    standard_read_time = time.time() - start_time

    start_time = time.time()
    for key in keys_to_read:
        custom_cache.get(key)
    custom_read_time = time.time() - start_time

    print("Sequential Reads (Hits):")
    print(f"  functools.lru_cache: {standard_read_time:.4f} seconds")
    print(f"  custom LRUCache:     {custom_read_time:.4f} seconds")
    print(f"  Ratio:               {custom_read_time / standard_read_time:.2f}x\n")

    # ==========================
    # Conclusion
    # ==========================
    print("Optimization Notes:")
    print(
        "- `functools.lru_cache` is implemented in pure C (as a deeply integrated Python built-in)."
    )
    print(
        "- Our custom `LRUCache` is implemented in pure Python but retains O(1) properties."
    )
    print(
        "- As expected, the C implementation is faster due to lower level execution speed, "
    )
    print(
        "  however the algorithmic complexity remains globally optimal across both implementations."
    )


if __name__ == "__main__":
    run_benchmark()
