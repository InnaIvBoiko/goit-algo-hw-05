from typing import List, Tuple, Optional


def binary_search_upper_bound(arr: List[float], target: float) -> Tuple[int, Optional[float]]:
    """
    Binary search for sorted array of floating-point numbers.
    
    Args:
        arr: Sorted list of floating-point numbers
        target: Target value to search for
        
    Returns:
        Tuple containing:
        - Number of iterations needed
        - Upper bound (smallest element >= target) or None if not found
    """
    left = 0
    right = len(arr) - 1
    iterations = 0
    
    while left <= right:
        iterations += 1
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return (iterations, arr[mid])
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # After loop, left points to the insertion position
    if left < len(arr):
        upper_bound = arr[left]
    else:
        upper_bound = None
    
    return (iterations, upper_bound)

# Test the function
arr = [1.1, 2.3, 3.5, 4.7, 5.9, 7.2, 8.4, 9.6]
target = 4.0

iterations, upper_bound = binary_search_upper_bound(arr, target)
print(f"Iterations: {iterations}, Upper bound: {upper_bound}")

# Additional test cases
test_cases = [
    ([1.1, 2.3, 3.5, 4.7, 5.9], 3.5),  # Exact match
    ([1.1, 2.3, 3.5, 4.7, 5.9], 3.2),  # Between elements
    ([1.1, 2.3, 3.5, 4.7, 5.9], 0.5),  # Smaller than all
    ([1.1, 2.3, 3.5, 4.7, 5.9], 6.0),  # Larger than all
]

for i, (test_arr, test_target) in enumerate(test_cases):
    iterations, upper_bound = binary_search_upper_bound(test_arr, test_target)
    print(f"Test {i+1}: target={test_target}, iterations={iterations}, upper_bound={upper_bound}")
