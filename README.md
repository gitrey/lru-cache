# LRU Cache implementation in Python

This project provides a production-grade, thread-safe implementation of a Least Recently Used (LRU) Cache in Python. 

The implementation uses a combination of a Doubly Linked List and a Hash Map (`dict`) to achieve `O(1)` average time complexity for both `get` and `put` operations, with strict capacity limits and concurrency safety.

## Features

- **Generic Types**: Built with generic types (`Generic[K, V]`) for both key and value to ensure strict type checking using mypy or your favorite type checker.
- **`O(1)` Time Complexity**: Uses a hash map for fast `O(1)` lookups and a doubly linked list for fast `O(1)` insertions, deletions, and moving items to the most-recently-used position.
- **Thread Security**: Employs a re-entrant lock (`threading.RLock`) to ensure the cache operations are safe in multi-threaded environments, preventing race conditions and deadlocks.
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

# Initialize the thread-safe cache with a capacity of 100 to store inference results
cache: LRUCache[str, str] = LRUCache(100)

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

# Final state: the internal lock protected the cache size limit and pointers 
print(f"Total cached inference completions: {len(cache)}") # Output: <= 100
```

### Thread Safety & Locking

The `LRUCache` utilizes Pythons `threading.RLock()` (Re-entrant Lock) to accomplish thread safety. This was chosen specifically for safely locking internal helper methods. 

Here is how the locking works in this implementation:
- **Scope**: Every public method that mutates state (`put`, `clear`) or relies on state mutation while reading (`get` moves nodes to the front) acquires the lock using Python's `with self._lock:` context manager.
- **Why RLock?**: A Re-entrant Lock allows the thread that currently holds the lock to acquire it again without deadlocking itself. The `LRUCache` utilizes this by using public methods like `__len__` or `__contains__` inside other locked operations if needed, or by ensuring all `Node` movements through internal methods (`_add_node_to_front`, `_remove_node`, `_pop_tail`) remain protected when invoked directly.
- **Performance consideration**: The lock is held for the minimum duration necessary to swap pointers in memory and alter the hash map. Because `O(1)` operations don't run any loops or scans, the lock holding time is extremely brief, minimizing contention across multiple threads.

### Time Complexity

- **`get(key)`: O(1) average case**. Looking up the node in the internal hash map takes `O(1)`. Unlinking it from its current position in the sequence and re-linking it to the front (marking it as most-recently-used) takes `O(1)` because we have direct pointers to nodes.
- **`put(key, value)`: O(1) average case**. 
  - If the key exists: updating the value and moving the node to the front takes `O(1)`.
  - If the key doesn't exist: creating the new node and adding it to the front takes `O(1)`.
  - If capacity acts as a constraint: identifying the least recently used node (tail) and slicing it off from both the doubly linked list and hash map takes `O(1)`.

### Space Complexity

- **O(capacity)**: Space expands corresponding to the `capacity` argument, storing at most `capacity` entries. The overhead involves pointers (`prev`, `next`) inside the linked list nodes and references maintained by the Python dictionary.

## Running Tests

The test suite leverages Python's built-in `unittest` module. To execute all unit tests, run the following command directly from the root directory:

```bash
python3 test_lru_cache.py -v
```

The test scope includes:
- Basic input/output matching and proper LRU eviction properties.
- Initialization edge cases validation.
- Extensive simulation of high competition multi-threaded reads, writes, and iterations.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
