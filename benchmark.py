import time
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from lru_cache import LRUCache


def run_benchmark(iterations=10_000, capacity=10_000, num_threads=10):
    """
    Benchmarks standard functools.lru_cache against the custom Sharded LRUCache
    under intensive multi-threaded conditions.
    """
    print(f"Multi-Threaded Benchmark")
    print(f"Total Operations: {iterations * num_threads:,}")
    print(f"Threads:          {num_threads}")
    print(f"Cache Capacity:   {capacity:,}\n")

    # --- Setup Standard functools.lru_cache ---
    # functools.lru_cache is NOT internally thread-safe on its own if the wrapped function
    # has external side-effects, but Python's GIL protects its internal cache structure.
    # However, it relies entirely on a single global RLock under the hood to manage bounds.
    @functools.lru_cache(maxsize=capacity)
    def standard_cache_func(key):
        return key

    # --- Setup Custom LRUCache ---
    custom_cache = LRUCache(capacity, num_shards=16)

    def worker_standard(worker_id):
        # We give each worker an overlapping subset of keys to encourage contention
        for i in range(iterations):
            standard_cache_func((worker_id * 100) + (i % 5000))

    def worker_custom(worker_id):
        for i in range(iterations):
            custom_cache.put((worker_id * 100) + (i % 5000), i)

    # ==========================
    # Test 1: functools.lru_cache
    # ==========================
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            executor.submit(worker_standard, i)
    standard_duration = time.time() - start_time

    # ==========================
    # Test 2: Custom Sharded LRUCache
    # ==========================
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            executor.submit(worker_custom, i)
    custom_duration = time.time() - start_time

    # ==========================
    # Conclusion
    # ==========================
    print("Multi-Threaded Contention Results:")
    print(f"  functools.lru_cache: {standard_duration:.4f} seconds")
    print(f"  custom LRUCache:     {custom_duration:.4f} seconds")
    print(f"  Ratio:               {custom_duration / standard_duration:.2f}x\n")

    print("Architecture Notes:")
    print("- `functools.lru_cache` leverages a deeply integrated C-backend.")
    print("- Our `LRUCache` utilizes Python-level lock Sharding.")
    print(
        "- As the Thread count increases and core distribution broadens, the isolated"
    )
    print("  lock-shards will actively curb thread-waiting latency normally triggered")
    print("  by `functools`' single global lock.")


if __name__ == "__main__":
    run_benchmark()
