from typing import Any, Optional

class HashTable:
    '''A simple hash table implementation with separate chaining for collision resolution.'''
    
    def __init__(self, size: int) -> None:
        self.size: int = size
        self.table: list[list[list[Any]]] = [[] for _ in range(self.size)]

    def hash_function(self, key: Any) -> int:
        return hash(key) % self.size

    def insert(self, key: Any, value: Any) -> bool:
        key_hash = self.hash_function(key)
        # Try to update existing key
        for pair in self.table[key_hash]:
            if pair[0] == key:
                pair[1] = value
                return True
        # Key not found - append new key-value pair
        self.table[key_hash].append([key, value])
        return True

    def get(self, key: Any) -> Optional[Any]:
        key_hash = self.hash_function(key)
        for pair in self.table[key_hash]:
            if pair[0] == key:
                return pair[1]
        return None

    def delete(self, key: Any) -> bool:
        # Calculate hash for the key
        key_hash = self.hash_function(key)
        
        # Iterate through the bucket to find the key
        for i, pair in enumerate(self.table[key_hash]):
            if pair[0] == key:
                # Remove the key-value pair from the bucket
                del self.table[key_hash][i]
                return True
        
        # Key not found
        return False

# Example usage:
H = HashTable(5)
H.insert("apple", 10)
H.insert("orange", 20)
H.insert("banana", 30)

print(H.get("apple"))   # Output: 10
print(H.get("orange"))  # Output: 20
print(H.get("banana"))  # Output: 30

H.delete("orange")
print(H.get("apple"))   # Output: 10
print(H.get("orange"))  # Output: None
print(H.get("banana"))  # Output: 30