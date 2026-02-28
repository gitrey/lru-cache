import threading
from typing import TypeVar, Generic, Dict, Optional, cast

K = TypeVar('K')
V = TypeVar('V')

class Node(Generic[K, V]):
    """
    A node in the doubly linked list used by the LRU Cache.
    """
    __slots__ = ('key', 'value', 'prev', 'next')
    
    def __init__(self, key: K, value: V):
        self.key: K = key
        self.value: V = value
        self.prev: Optional['Node[K, V]'] = None
        self.next: Optional['Node[K, V]'] = None

class LRUCache(Generic[K, V]):
    """
    A thread-safe Least Recently Used (LRU) Cache implementation using a hash map
    and a doubly linked list.
    
    Time Complexity:
    - get(key): O(1) average case. Hash map lookup is O(1) and moving a node to 
      the front of the linked list is O(1).
    - put(key, value): O(1) average case. Hash map insertion/update is O(1), and
      linked list node addition/movement is O(1). If eviction is needed, removing
      the tail node and deleting from the hash map is also O(1).
      
    Space Complexity:
    - O(capacity) where 'capacity' is the maximum number of items the cache can hold.
      The space is used by the hash map and the doubly linked list.
    """
    
    def __init__(self, capacity: int):
        """
        Initializes the LRU Cache with a maximum capacity.
        
        Args:
            capacity: The maximum number of items the cache can hold.
            
        Raises:
            ValueError: If the capacity is less than or equal to 0.
            TypeError: If the capacity is not an integer.
        """
        if not isinstance(capacity, int):
            raise TypeError("Capacity must be an integer.")
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
            
        self._capacity: int = capacity
        self._cache: Dict[K, Node[K, V]] = {}
        
        # Dummy head and tail nodes to simplify edge cases in doubly linked list operations
        # We use cast to satisfy the type checker for dummy nodes
        self._head: Node[K, V] = Node(cast(K, None), cast(V, None))
        self._tail: Node[K, V] = Node(cast(K, None), cast(V, None))
        self._head.next = self._tail
        self._tail.prev = self._head
        
        # Re-entrant lock for thread-safety, allowing internal calls
        self._lock = threading.RLock()

    def _remove_node(self, node: Node[K, V]) -> None:
        """
        Removes a node from the doubly linked list.
        Note: This is an internal method and must be called with the lock acquired.
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
        """
        Adds a node right after the dummy head (most recently used position).
        Note: This is an internal method and must be called with the lock acquired.
        """
        node.prev = self._head
        node.next = self._head.next
        
        if self._head.next:
            self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: Node[K, V]) -> None:
        """
        Moves an existing node to the front of the list.
        Note: This is an internal method and must be called with the lock acquired.
        """
        self._remove_node(node)
        self._add_node_to_front(node)

    def _pop_tail(self) -> Optional[Node[K, V]]:
        """
        Removes and returns the node right before the dummy tail (least recently used).
        Note: This is an internal method and must be called with the lock acquired.
        """
        tail_node = self._tail.prev
        if tail_node is self._head or tail_node is None:
            return None
            
        self._remove_node(tail_node)
        return tail_node

    def get(self, key: K) -> Optional[V]:
        """
        Retrieves the value associated with the given key if it exists in the cache.
        If the key exists, it is moved to the most recently used position.
        
        Args:
            key: The key to look up in the cache.
            
        Returns:
            The value associated with the key, or None if the key is not found.
        """
        with self._lock:
            node = self._cache.get(key)
            if node is None:
                return None
            
            self._move_to_front(node)
            return node.value

    def put(self, key: K, value: V) -> None:
        """
        Inserts or updates a key-value pair in the cache.
        If the cache exceeds its capacity, the least recently used item is evicted.
        
        Args:
            key: The key to insert or update.
            value: The value to associate with the key.
        """
        with self._lock:
            # Check if key already exists to update its value and move it
            node = self._cache.get(key)
            if node is not None:
                node.value = value
                self._move_to_front(node)
                return

            # Create a new node since it wasn't found
            new_node = Node(key, value)
            self._cache[key] = new_node
            self._add_node_to_front(new_node)
            
            # Check capacity and evict if necessary
            if len(self._cache) > self._capacity:
                lru_node = self._pop_tail()
                if lru_node is not None:
                    del self._cache[lru_node.key]

    @property
    def capacity(self) -> int:
        """Returns the maximum capacity of the cache."""
        return self._capacity

    def __len__(self) -> int:
        """Returns the current number of items in the cache."""
        with self._lock:
            return len(self._cache)
            
    def __contains__(self, key: K) -> bool:
        """
        Checks if the key is in the cache without updating its recently used status.
        """
        with self._lock:
            return key in self._cache
            
    def clear(self) -> None:
        """Clears all items from the cache."""
        with self._lock:
            # Explicitly break reference cycles to help GC
            curr = self._head.next
            while curr and curr is not self._tail:
                nxt = curr.next
                curr.prev = None
                curr.next = None
                curr = nxt
                
            self._cache.clear()
            self._head.next = self._tail
            self._tail.prev = self._head
