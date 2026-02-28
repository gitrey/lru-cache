import unittest
import threading
import time
from lru_cache import LRUCache


class TestLRUCache(unittest.TestCase):

    def test_lru_cache_initialization(self):
        with self.assertRaises(TypeError):
            LRUCache("invalid")
        with self.assertRaises(ValueError):
            LRUCache(0)
        with self.assertRaises(ValueError):
            LRUCache(-5)
        with self.assertRaises(ValueError):
            LRUCache(3, ttl_seconds=-1)

        cache = LRUCache(3, ttl_seconds=5)
        self.assertEqual(cache.capacity, 3)
        self.assertEqual(cache.ttl_seconds, 5)
        self.assertEqual(len(cache), 0)

    def test_lru_cache_basic_operations(self):
        # Strict sequential eviction tests should run against a single shard
        # because hash routing distributes entries unpredictably across multiple shards.
        cache = LRUCache(2, num_shards=1)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(cache.get(1), 1)

        # Needs eviction
        cache.put(3, 3)
        self.assertIsNone(cache.get(2))

        # Update and eviction
        cache.put(4, 4)
        self.assertIsNone(cache.get(1))
        self.assertEqual(cache.get(3), 3)
        self.assertEqual(cache.get(4), 4)

    def test_lru_cache_overwrite(self):
        cache = LRUCache(2, num_shards=1)
        cache.put(1, 1)
        cache.put(2, 2)
        cache.put(1, 10)  # 1 is updated and moved to front
        cache.put(3, 3)  # 2 should be evicted

        self.assertEqual(cache.get(1), 10)
        self.assertIsNone(cache.get(2))
        self.assertEqual(cache.get(3), 3)

    def test_lru_cache_clear(self):
        cache = LRUCache(3, num_shards=1)
        cache.put(1, 1)
        cache.put(2, 2)
        self.assertEqual(len(cache), 2)

        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(1))

    def test_lru_cache_ttl_expiration(self):
        # Extremely short TTL
        cache = LRUCache(3, ttl_seconds=0.01, num_shards=1)
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertEqual(cache.get("a"), 1)

        time.sleep(0.02)

        # Values should be expired
        self.assertIsNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))
        self.assertFalse("b" in cache)
        # Len calculates non-expired values
        self.assertEqual(len(cache), 0)

    def test_lru_cache_ttl_refresh_on_put(self):
        cache = LRUCache(5, ttl_seconds=0.02, num_shards=1)
        cache.put("a", 1)

        time.sleep(0.01)
        self.assertEqual(cache.get("a"), 1)

        # Overwrite item completely refreshes TTL timestamp
        cache.put("a", 2)
        time.sleep(0.01)

        # It's been 0.02 since creation, but only 0.01 since overwrite hook
        self.assertTrue("a" in cache)
        self.assertEqual(cache.get("a"), 2)
        self.assertEqual(len(cache), 1)

    def test_lru_cache_thread_safety(self):
        cache = LRUCache(1000)

        def worker_put(start, end):
            for i in range(start, end):
                cache.put(i, i)

        threads = []
        # 10 threads adding items 0 to 999
        for i in range(10):
            t = threading.Thread(target=worker_put, args=(i * 100, (i + 1) * 100))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.assertEqual(len(cache), 1000)

        # Test concurrent reads and writes leading to evictions
        cache2 = LRUCache(500)
        threads2 = []

        def worker_mixed():
            for i in range(1000):
                cache2.put(i, i)
                cache2.get(i - 100)

        for i in range(10):
            t = threading.Thread(target=worker_mixed)
            threads2.append(t)
            t.start()

        for t in threads2:
            t.join()

        # Capacity shouldn't be exceeded
        self.assertLessEqual(len(cache2), 500)

    def test_lru_cache_statistics(self):
        cache = LRUCache(10, num_shards=2)

        # Test 1 miss initially
        cache.get("a")
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 1)

        # Test 1 put followed by 1 hit
        cache.put("a", 1)
        cache.get("a")
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 1)

        # Test misses over different namespace
        cache.get("b")
        cache.get("c")
        self.assertEqual(cache.hits, 1)
        self.assertEqual(cache.misses, 3)

        # Test clear resets stats
        cache.clear()
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)

        # Test TTL boundaries trigger misses
        ttl_cache = LRUCache(10, ttl_seconds=0.01, num_shards=1)
        ttl_cache.put("expire_me", 100)
        time.sleep(0.02)  # Force expire

        ttl_cache.get("expire_me")
        self.assertEqual(ttl_cache.hits, 0)
        self.assertEqual(ttl_cache.misses, 1)

    def test_lru_cache_shard_metrics(self):
        cache = LRUCache(10, num_shards=4)

        # Artificially trigger multiple routes to distribute stats over the sub-dicts
        for i in range(100):
            cache.get(f"key_{i}")  # All misses initially

        metrics = cache.get_shard_metrics()

        self.assertEqual(len(metrics), 4)

        total_misses = sum(data["misses"] for data in metrics.values())
        total_hits = sum(data["hits"] for data in metrics.values())

        self.assertEqual(total_misses, 100)
        self.assertEqual(total_hits, 0)
        self.assertEqual(cache.misses, 100)


if __name__ == "__main__":
    unittest.main()
