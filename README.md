# LRU Cache implementation in Python

This project provides a production-grade, thread-safe implementation of a Least Recently Used (LRU) Cache in Python.

The implementation uses a combination of a Doubly Linked List, a Hash Map (`dict`), and **Lock Sharding** to achieve `O(1)` average time complexity for both `get` and `put` operations, with strict capacity limits and high-concurrency safety.

## Features

- **Generic Types**: Built with generic types (`Generic[K, V]`) for both key and value to ensure strict type checking using mypy or your favorite type checker.
- **`O(1)` Time Complexity**: Uses a hash map for fast `O(1)` lookups and a doubly linked list for fast `O(1)` insertions, deletions, and moving items to the most-recently-used position.
- **Thread Security via Lock Sharding**: Divides the cache into independent shards, each with its own re-entrant lock (`threading.RLock`), to dramatically reduce thread contention and maximize concurrency in multi-threaded environments.
- **Time-to-Live (TTL)**: Supports optional TTL definitions per capacity, ensuring entries organically expire based on cache lifetime without explicit manual deletion.
- **Safe Eviction**: Handles robust circular cleanup to assist Python's garbage collector.
- **Comprehensive Testing**: Validated against multiple concurrency edge cases and core logic verifications using `unittest`.

## Requirements

- Python 3.7+

## Installation

You can clone this repository and start using the `lru_cache.py` module in your project. No external requirements are needed for the core implementation as it only utilizes the standard library.

```bash
git clone https://github.com/gitrey/lru-cache.git
cd lru-cache
```

## Quick Start

```python
import threading
import time
from lru_cache import LRUCache

# Initialize the thread-safe cache with a capacity of 100 to store inference results.
# Add an optional ttl_seconds rule, expiring LLM inferences after 5 minutes (300 seconds).
# Customize num_shards to distribute concurrent thread locks (defaults to 16).
cache: LRUCache[str, str] = LRUCache(100, ttl_seconds=300, num_shards=16)

def inference_worker(worker_id: int):
    """Simulates a threaded API worker handling LLM inference requests."""
    for i in range(50):
        # Simulate overlapping user prompts across different API threads
        prompt = f"What is the capital of country {i % 25}?"

        # Check if the inference result is already cached from another thread
        response = cache.get(prompt)

        if response is None:
            # Simulate an expensive LLM inference call if cache miss occurs
            time.sleep(0.05)
            response = f"Simulated LLM response for country {i % 25} (handled by worker {worker_id})"
            cache.put(prompt, response)

# Create 10 concurrent API threads handling inference requests
threads = []
for i in range(10):
    t = threading.Thread(target=inference_worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all API request threads to finish
for t in threads:
    t.join()

# Final state: the sharded locks simultaneously protected the cache size limit and pointers
print(f"Total cached inference completions: {len(cache)}") # Output: <= 100
```

### Thread Safety & Lock Sharding

The `LRUCache` utilizes an advanced **Lock Sharding** strategy to accomplish high-concurrency thread safety, dramatically outperforming a standard global-lock design.

Here is how the locking works in this implementation:

- **Segmented Array**: The internal state is completely divided into `num_shards` independent `_LRUCacheShard` processes (defaulting to 16).
- **Modulo Hashing Routing**: Incoming keys are deterministically routed to a specific isolated shard via `hash(key) % num_shards`.
- **High Concurrency**: Because the locks are separated, up to 16 threads (by default) can simultaneously perform `put` or `get` operations on the cache without ever blocking each other, provided their hashed keys route to distinct shards.
- **Why RLock?**: Each shard uses a local Re-entrant Lock (`threading.RLock`) to guarantee local thread-safety, allowing internal recursive assertions or iterations inside the isolated namespace without risking deadlock.

### Time Complexity

- **`get(key)`: O(1) average case**. Looking up the node in the internal hash map takes `O(1)`. Unlinking it from its current position in the sequence and re-linking it to the front (marking it as most-recently-used) takes `O(1)` because we have direct pointers to nodes.
- **`put(key, value)`: O(1) average case**.
  - If the key exists: updating the value and moving the node to the front takes `O(1)`.
  - If the key doesn't exist: creating the new node and adding it to the front takes `O(1)`.
  - If capacity acts as a constraint: identifying the least recently used node (tail) and slicing it off from both the doubly linked list and hash map takes `O(1)`.

### Space Complexity

- **O(capacity)**: Space expands corresponding to the `capacity` argument, storing at most `capacity` entries. The overhead involves pointers (`prev`, `next`) inside the linked list nodes and references maintained by the Python dictionary.

## Benchmarking

A benchmarking script is provided to compare the raw execution speed of our custom pure-Python `LRUCache` against the standard C-implemented `functools.lru_cache()`.

Run the comparison:

```bash
python3 benchmark.py
```

_Note: While `functools.lru_cache` (written in optimized C bytecode inside Python) executes strictly faster sequentially, our custom `LRUCache` utilizes Python-level **Lock Sharding** to scale dramatically better under high multi-threaded contention payloads._

## Running Tests

The test suite leverages Python's built-in `unittest` module. To execute all unit tests, run the following command directly from the root directory:

```bash
python3 test_lru_cache.py -v
```

The test scope includes:

- Basic input/output matching and proper LRU eviction properties.
- **Time-to-Live (TTL)** expiration logic.
- Initialization edge cases validation.
- Extensive simulation of high competition multi-threaded reads, writes, and iterations.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
