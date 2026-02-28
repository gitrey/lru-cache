import threading
import time
from typing import TypeVar, Generic, Dict, Optional, cast

K = TypeVar("K")
V = TypeVar("V")


class Node(Generic[K, V]):
    """A node in the doubly linked list used by the LRU Cache.

    Attributes:
        key (K): The key associated with this node.
        value (V): The value associated with this node.
        created_at (float): The timestamp when this node's value was created or last updated.
        prev (Optional[Node[K, V]]): Pointer to the previous node in the list.
        next (Optional[Node[K, V]]): Pointer to the next node in the list.
    """

    __slots__ = ("key", "value", "created_at", "prev", "next")

    def __init__(self, key: K, value: V):
        """Initializes a new doubly linked list node.

        Args:
            key (K): The key to store.
            value (V): The value to store.
        """
        self.key: K = key
        self.value: V = value
        self.created_at: float = time.monotonic()
        self.prev: Optional["Node[K, V]"] = None
        self.next: Optional["Node[K, V]"] = None


class _LRUCacheShard(Generic[K, V]):
    """An internal thread-safe LRU Cache Shard.

    Handles a specific bounded capacity and owns its own independent re-entrant lock.
    """

    def __init__(self, capacity: int, ttl_seconds: Optional[float] = None):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

        self._capacity: int = capacity
        self._ttl_seconds: Optional[float] = ttl_seconds
        self._cache: Dict[K, Node[K, V]] = {}

        # Dummy head and tail nodes to simplify edge cases in doubly linked list operations
        self._head: Node[K, V] = Node(cast(K, None), cast(V, None))
        self._tail: Node[K, V] = Node(cast(K, None), cast(V, None))
        self._head.next = self._tail
        self._tail.prev = self._head

        self._lock = threading.RLock()

    def _is_expired(self, node: Node[K, V]) -> bool:
        if self._ttl_seconds is None:
            return False
        return (time.monotonic() - node.created_at) > self._ttl_seconds

    def _remove_node(self, node: Node[K, V]) -> None:
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node
        node.prev = None
        node.next = None

    def _add_node_to_front(self, node: Node[K, V]) -> None:
        node.prev = self._head
        node.next = self._head.next
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: Node[K, V]) -> None:
        self._remove_node(node)
        self._add_node_to_front(node)

    def _pop_tail(self) -> Optional[Node[K, V]]:
        tail_node = self._tail.prev
        if tail_node is self._head or tail_node is None:
            return None
        self._remove_node(tail_node)
        return tail_node

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            node = self._cache.get(key)
            if node is None:
                return None
            if self._is_expired(node):
                self._remove_node(node)
                del self._cache[key]
                return None
            self._move_to_front(node)
            return node.value

    def put(self, key: K, value: V) -> None:
        with self._lock:
            node = self._cache.get(key)
            if node is not None:
                node.value = value
                node.created_at = time.monotonic()
                self._move_to_front(node)
                return

            new_node = Node(key, value)
            self._cache[key] = new_node
            self._add_node_to_front(new_node)

            if len(self._cache) > self._capacity:
                lru_node = self._pop_tail()
                if lru_node is not None:
                    del self._cache[lru_node.key]

    def __len__(self) -> int:
        with self._lock:
            if self._ttl_seconds is None:
                return len(self._cache)
            return sum(1 for node in self._cache.values() if not self._is_expired(node))

    def __contains__(self, key: K) -> bool:
        with self._lock:
            node = self._cache.get(key)
            if node is None:
                return False
            return not self._is_expired(node)

    def clear(self) -> None:
        with self._lock:
            curr = self._head.next
            while curr and curr is not self._tail:
                nxt = curr.next
                curr.prev = None
                curr.next = None
                curr = nxt
            self._cache.clear()
            self._head.next = self._tail
            self._tail.prev = self._head


class LRUCache(Generic[K, V]):
    """A thread-safe, sharded Least Recently Used (LRU) Cache implementation.

    This cache uses multiple independent internal shards (segments) to distribute locking
    contention in highly concurrent environments. Keys are deterministically routed to
    shards using modulo hashing.

    Attributes:
        _capacity (int): The global maximum number of items the cache overall can hold.
        _ttl_seconds (Optional[float]): The maximum lifetime of an entry in seconds.
        _num_shards (int): The number of independent internal cache segments.
        _shards (list[_LRUCacheShard]): Array storing the separated cache processes.
    """

    def __init__(
        self, capacity: int, ttl_seconds: Optional[float] = None, num_shards: int = 16
    ):
        """Initializes the Sharded LRU Cache.

        Args:
            capacity (int): The maximum combined number of items the cache can hold.
            ttl_seconds (Optional[float], optional): The maximum time an item should remain valid.
            num_shards (int, optional): The number of independent locks/shards to allocate.
                Defaults to 16, typically sufficient for common multi-core concurrency.

        Raises:
            ValueError: If the capacity is less than or equal to 0, or if arguments are invalid.
            TypeError: If types supplied are incorrectly casted.
        """
        if not isinstance(capacity, int):
            raise TypeError("Capacity must be an integer.")
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("TTL must be a non-negative number.")
        if not isinstance(num_shards, int) or num_shards <= 0:
            raise ValueError("num_shards must be a positive integer.")

        self._capacity: int = capacity
        self._ttl_seconds: Optional[float] = ttl_seconds
        self._num_shards: int = num_shards

        # Calculate shard capacities. We distribute uniformly; any remainder
        # increases the capacity of the first few shards.
        base_shard_capacity = capacity // num_shards
        remainder = capacity % num_shards

        self._shards: list[_LRUCacheShard[K, V]] = []
        for i in range(num_shards):
            shard_cap = base_shard_capacity + (1 if i < remainder else 0)
            # Ensure every shard has at least capacity 1 to remain functional
            shard_cap = max(1, shard_cap)
            self._shards.append(
                _LRUCacheShard(capacity=shard_cap, ttl_seconds=ttl_seconds)
            )

    def _get_shard(self, key: K) -> _LRUCacheShard[K, V]:
        """Routes a key deterministically to its assigned shard.

        Args:
            key (K): The key logic to be evaluated.

        Returns:
            _LRUCacheShard[K, V]: The isolated shard instance owning this hashed namespace.
        """
        shard_index = hash(key) % self._num_shards
        return self._shards[shard_index]

    def get(self, key: K) -> Optional[V]:
        """Retrieves an item from the targeted shard.

        Args:
            key (K): The key to lookup.

        Returns:
            Optional[V]: The value associated with the key, or None if not found or expired.
        """
        return self._get_shard(key).get(key)

    def put(self, key: K, value: V) -> None:
        """Inserts or updates a key-value pair directly within its isolated shard lock.

        Adding or updating a node within its shard manages the local LRU sequence.

        Args:
            key (K): The key to insert or update.
            value (V): The value to associate with the key.
        """
        self._get_shard(key).put(key, value)

    @property
    def capacity(self) -> int:
        """int: The approximate global targeted capacity of the cache."""
        return self._capacity

    @property
    def ttl_seconds(self) -> Optional[float]:
        """Optional[float]: The Time-to-Live configured globally."""
        return self._ttl_seconds

    def __len__(self) -> int:
        """Calculates the combined number of unexpired elements universally resting in the shards.

        Returns:
            int: The aggregate number of active entries globally in the cache.
        """
        return sum(len(shard) for shard in self._shards)

    def __contains__(self, key: K) -> bool:
        """Checks if a key is present inside its specific shard map.

        Args:
            key (K): The key to evaluate.

        Returns:
            bool: True if the key exists globally without expiration.
        """
        return key in self._get_shard(key)

    def clear(self) -> None:
        """Completely clears all entries sequentially from all isolated cache shards."""
        for shard in self._shards:
            shard.clear()
