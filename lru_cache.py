import threading
import time
from typing import TypeVar, Generic, Dict, Optional, cast

K = TypeVar('K')
V = TypeVar('V')

class Node(Generic[K, V]):
    """A node in the doubly linked list used by the LRU Cache.
    
    Attributes:
        key (K): The key associated with this node.
        value (V): The value associated with this node.
        created_at (float): The timestamp when this node's value was created or last updated.
        prev (Optional[Node[K, V]]): Pointer to the previous node in the list.
        next (Optional[Node[K, V]]): Pointer to the next node in the list.
    """
    __slots__ = ('key', 'value', 'created_at', 'prev', 'next')
    
    def __init__(self, key: K, value: V):
        """Initializes a new doubly linked list node.
        
        Args:
            key (K): The key to store.
            value (V): The value to store.
        """
        self.key: K = key
        self.value: V = value
        self.created_at: float = time.monotonic()
        self.prev: Optional['Node[K, V]'] = None
        self.next: Optional['Node[K, V]'] = None

class LRUCache(Generic[K, V]):
    """A thread-safe Least Recently Used (LRU) Cache implementation.
    
    This cache uses a combination of a hash map and a doubly linked list
    to provide average O(1) time complexity for get and put operations.
    It supports capacity limits and optional Time-To-Live (TTL) expiration.
    
    Attributes:
        _capacity (int): The maximum number of items the cache can hold.
        _ttl_seconds (Optional[float]): The maximum lifetime of an entry in seconds.
        _cache (Dict[K, Node[K, V]]): Hash map storing key-to-node references.
        _head (Node[K, V]): Dummy head node representing the most recently used end.
        _tail (Node[K, V]): Dummy tail node representing the least recently used end.
        _lock (threading.RLock): Re-entrant lock for thread-safety.
    """
    
    def __init__(self, capacity: int, ttl_seconds: Optional[float] = None):
        """Initializes the LRU Cache.
        
        Args:
            capacity (int): The maximum number of items the cache can hold.
            ttl_seconds (Optional[float], optional): The maximum time in seconds an item 
                should remain valid in the cache. If None, items never expire 
                based on time. Defaults to None.
            
        Raises:
            ValueError: If the capacity is less than or equal to 0, or if ttl_seconds is negative.
            TypeError: If the capacity is not an integer.
        """
        if not isinstance(capacity, int):
            raise TypeError("Capacity must be an integer.")
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        if ttl_seconds is not None and ttl_seconds < 0:
            raise ValueError("TTL must be a non-negative number.")
            
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
        """Checks if a cache node has exceeded its TTL.
        
        Args:
            node (Node[K, V]): The cache node to inspect.
            
        Returns:
            bool: True if the node is expired, False otherwise or if TTL is disabled.
        """
        if self._ttl_seconds is None:
            return False
            
        return (time.monotonic() - node.created_at) > self._ttl_seconds

    def _remove_node(self, node: Node[K, V]) -> None:
        """Removes a node from the doubly linked list.
        
        Note:
            This is an internal method and must be called with the lock acquired.
            
        Args:
            node (Node[K, V]): The referenced node to splice out of the list.
        """
        prev_node = node.prev
        next_node = node.next
        if prev_node and next_node:
            prev_node.next = next_node
            next_node.prev = prev_node
            
        # Clear references to help garbage collection
        node.prev = None
        node.next = None

    def _add_node_to_front(self, node: Node[K, V]) -> None:
        """Adds a node right after the dummy head (most recently used position).
        
        Note:
            This is an internal method and must be called with the lock acquired.
            
        Args:
            node (Node[K, V]): The node to insert at the front.
        """
        node.prev = self._head
        node.next = self._head.next
        
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: Node[K, V]) -> None:
        """Moves an existing list node to the most recently used position.
        
        Note:
            This is an internal method and must be called with the lock acquired.
            
        Args:
            node (Node[K, V]): The referenced node to re-link to the front.
        """
        self._remove_node(node)
        self._add_node_to_front(node)

    def _pop_tail(self) -> Optional[Node[K, V]]:
        """Removes and returns the node right before the dummy tail.
        
        This represents the least recently used element in the cache, allowing 
        for eviction when the bounds are exceeded.
        
        Note:
            This is an internal method and must be called with the lock acquired.
            
        Returns:
            Optional[Node[K, V]]: The evicted node, or None if the cache is empty.
        """
        tail_node = self._tail.prev
        if tail_node is self._head or tail_node is None:
            return None
            
        self._remove_node(tail_node)
        return tail_node

    def get(self, key: K) -> Optional[V]:
        """Retrieves an item from the cache.
        
        If the item is found and valid (i.e., not expired), it becomes the most 
        recently used entry. If the entry is expired, it is treated as a miss
        and is actively deleted from the cache structure.
        
        Args:
            key (K): The key to lookup.
            
        Returns:
            Optional[V]: The value associated with the key, or None if not found or expired.
        """
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
        """Inserts or updates a key-value pair in the cache.
        
        Adding or updating a node makes it the most recently used entry and resets 
        its TTL countdown. If adding this node exceeds the maximum capacity, 
        the least recently used entry is evicted.
        
        Args:
            key (K): The key to insert or update.
            value (V): The value to associate with the key.
        """
        with self._lock:
            node = self._cache.get(key)
            if node is not None:
                # Update existing value and timestamp, then make most recently used
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

    @property
    def capacity(self) -> int:
        """int: The maximum capacity of the cache."""
        return self._capacity
        
    @property 
    def ttl_seconds(self) -> Optional[float]:
        """Optional[float]: The Time-to-Live configured for this cache."""
        return self._ttl_seconds

    def __len__(self) -> int:
        """Calculates the number of unexpired elements currently residing in the cache.
        
        Caution: This method actively iterates and filters expired entries to return an 
        accurate count of usable items. It does not evict them proactively during counting.
        
        Returns:
            int: The number of active, non-expired cache entries.
        """
        with self._lock:
            if self._ttl_seconds is None:
                return len(self._cache)
                
            # Filter our count by lazily analyzing timestamps without evicting them
            return sum(1 for node in self._cache.values() if not self._is_expired(node))
            
    def __contains__(self, key: K) -> bool:
        """Checks if a valid, non-expired key is present in the cache.
        
        Unlike `get()`, this method does not modify the most recently used order 
        or evict the key proactively upon encountering an expiration.
        
        Args:
            key (K): The key to evaluate.
            
        Returns:
            bool: True if the key exists and is non-expired, False otherwise.
        """
        with self._lock:
            node = self._cache.get(key)
            if node is None:
                return False
                
            return not self._is_expired(node)
            
    def clear(self) -> None:
        """Completely clears all entries from the cache structure.
        
        This resets both the hash map and doubly linked list, breaking internal 
        reference cycles promptly to assist the Garbage Collector.
        """
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
