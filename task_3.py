import timeit
import functools
from typing import Dict, List, Callable


def build_shift_table(pattern: str) -> Dict[str, int]:
    """
    Create shift table for Boyer-Moore algorithm.
    For each character in pattern, set shift equal to pattern length minus position.
    """
    table: Dict[str, int] = {}
    length = len(pattern)
    # For each character in substring set shift equal to substring length
    for index, char in enumerate(pattern[:-1]):
        table[char] = length - index - 1
    # If pattern is empty, nothing to set; otherwise set default for last char
    if pattern:
        table.setdefault(pattern[-1], length)
    return table


def boyer_moore_search(text: str, pattern: str) -> int:
    """
    Boyer-Moore string search algorithm implementation
    
    Args:
        text (str): Text to search in
        pattern (str): Pattern to search for
    
    Returns:
        int: Index of first occurrence, -1 if not found
    """
    if not pattern:
        return 0
    
    # Create shift table for pattern (substring)
    shift_table = build_shift_table(pattern)
    i = 0  # Initialize starting index for main text

    # Go through main text, comparing with substring
    while i <= len(text) - len(pattern):
        j = len(pattern) - 1  # Start from end of substring

        # Compare characters from end of substring to its beginning
        while j >= 0 and text[i + j] == pattern[j]:
            j -= 1  # Move towards beginning of substring

        # If entire substring matches, return its position in text
        if j < 0:
            return i  # Substring found

        # Shift index i based on shift table
        # This allows to "jump over" non-matching parts of text
        i += shift_table.get(text[i + len(pattern) - 1], len(pattern))

    # If substring not found, return -1
    return -1


def compute_lps(pattern: str) -> List[int]:
    """
    Compute Longest Proper Prefix which is also Suffix (LPS) array for KMP algorithm.
    This array is used to skip characters when mismatch occurs.
    """
    lps: List[int] = [0] * len(pattern)
    length: int = 0
    i: int = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps


def kmp_search(main_string: str, pattern: str) -> int:
    """
    Knuth-Morris-Pratt string search algorithm implementation
    
    Args:
        main_string (str): Text to search in
        pattern (str): Pattern to search for
    
    Returns:
        int: Index of first occurrence, -1 if not found
    """
    if not pattern:
        return 0
    
    M = len(pattern)
    N = len(main_string)

    # Compute LPS array for pattern
    lps = compute_lps(pattern)

    i = j = 0  # Index for main_string and pattern

    while i < N:
        if pattern[j] == main_string[i]:
            i += 1
            j += 1
        elif j != 0:
            # Mismatch after j matches, use LPS to skip characters
            j = lps[j - 1]
        else:
            # No match at current position, move to next character
            i += 1

        if j == M:
            return i - j  # Pattern found at index (i - j)

    return -1  # Pattern not found


def polynomial_hash(s: str, base: int = 256, modulus: int = 101) -> int:
    """
    Calculate polynomial hash of string s.
    Hash = (s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]*base^0) % modulus
    """
    n = len(s)
    hash_value = 0
    for i, char in enumerate(s):
        power_of_base = pow(base, n - i - 1, modulus)
        hash_value = (hash_value + ord(char) * power_of_base) % modulus
    return hash_value


def rabin_karp_search(main_string: str, substring: str) -> int:
    """
    Rabin-Karp string search algorithm implementation
    
    Args:
        main_string (str): Text to search in
        substring (str): Pattern to search for
    
    Returns:
        int: Index of first occurrence, -1 if not found
    """
    if not substring:
        return 0
    
    # Length of main string and search substring
    substring_length = len(substring)
    main_string_length = len(main_string)
    
    if substring_length > main_string_length:
        return -1
    
    # Base number for hashing and modulus
    base = 256 
    modulus = 101  
    
    # Hash values for search substring and current slice in main string
    substring_hash = polynomial_hash(substring, base, modulus)
    current_slice_hash = polynomial_hash(main_string[:substring_length], base, modulus)
    
    # Precomputed value for hash recalculation
    h_multiplier = pow(base, substring_length - 1) % modulus
    
    # Go through main string
    for i in range(main_string_length - substring_length + 1):
        # If hash values match, verify character by character
        if substring_hash == current_slice_hash:
            if main_string[i:i+substring_length] == substring:
                return i  # Substring found
        
        # Recalculate hash for next window (rolling hash)
        if i < main_string_length - substring_length:
            # Remove leftmost character and add rightmost character
            current_slice_hash = (current_slice_hash - ord(main_string[i]) * h_multiplier) % modulus
            current_slice_hash = (current_slice_hash * base + ord(main_string[i + substring_length])) % modulus
            if current_slice_hash < 0:
                current_slice_hash += modulus

    return -1  # Substring not found


def read_file_content(filename: str) -> str:
    """
    Read content from file
    
    Args:
        filename (str): Path to file
    
    Returns:
        str: File content
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(filename, 'r', encoding='cp1251') as file:
            return file.read()


def measure_algorithm_performance(algorithm: Callable[[str, str], int], text: str, pattern: str, iterations: int = 1000) -> float:
    """
    Measure algorithm performance using timeit
    
    Args:
        algorithm (Callable[[str, str], int]): Algorithm function to test
        text (str): Text to search in
        pattern (str): Pattern to search for
        iterations (int): Number of iterations for timing
    
    Returns:
        float: Average execution time
    """
    timer = timeit.Timer(functools.partial(algorithm, text, pattern))
    time_taken = timer.timeit(iterations) / iterations
    return time_taken


def run_performance_tests():
    """
    Run performance tests on both text files with existing and fictional patterns
    """
    # Load text files
    print("Loading text files...")
    article1 = read_file_content("article1.txt")
    article2 = read_file_content("article2.txt")
    
    # Define test patterns
    # Existing patterns (found in texts)
    existing_pattern1 = "алгоритм"  # Common word in article1
    existing_pattern2 = "структури даних"  # Common phrase in article2
    
    # Fictional patterns (not found in texts)
    fictional_pattern1 = "неіснуючий_рядок_123"
    fictional_pattern2 = "вигаданий_текст_456"
    
    algorithms: Dict[str, Callable[[str, str], int]] = {
        "Boyer-Moore": boyer_moore_search,
        "Knuth-Morris-Pratt": kmp_search,
        "Rabin-Karp": rabin_karp_search
    }
    
    test_cases = [
        ("Article 1", article1, existing_pattern1, "existing"),
        ("Article 1", article1, fictional_pattern1, "fictional"),
        ("Article 1", article1, existing_pattern2, "existing"),
        ("Article 1", article1, fictional_pattern2, "fictional"),
        ("Article 2", article2, existing_pattern1, "existing"),
        ("Article 2", article2, fictional_pattern1, "fictional"),
        ("Article 2", article2, existing_pattern2, "existing"),
        ("Article 2", article2, fictional_pattern2, "fictional")
    ]
    
    print("\nPerformance Test Results:")
    print("=" * 80)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for text_name, text, pattern, pattern_type in test_cases:
        print(f"\n{text_name} - {pattern_type.capitalize()} pattern: '{pattern}'")
        print("-" * 60)
        
        test_results: Dict[str, float] = {}
        
        for alg_name, algorithm in algorithms.items():
            # Verify algorithm works correctly
            result = algorithm(text, pattern)
            
            # Measure performance
            time_taken = measure_algorithm_performance(algorithm, text, pattern)
            test_results[alg_name] = time_taken
            
            print(f"{alg_name:20}: {time_taken:.8f}s (found at index: {result})")
        
        # Find fastest algorithm for this test case
        fastest_alg: str = min(test_results, key=test_results.__getitem__)
        print(f"{'Fastest algorithm:':20} {fastest_alg}")
        
        # Store results for overall analysis
        key = f"{text_name}_{pattern_type}"
        results[key] = test_results
    
    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)
    
    # Calculate average performance for each algorithm
    overall_performance: Dict[str, List[float]] = {alg: [] for alg in algorithms.keys()}
    
    for test_result in results.values():
        for alg, time_val in test_result.items():
            overall_performance[alg].append(time_val)
    
    avg_performance = {alg: sum(times) / len(times) 
                      for alg, times in overall_performance.items()}
    
    print("\nAverage performance across all tests:")
    for alg, avg_time in sorted(avg_performance.items(), key=lambda x: x[1]):
        print(f"{alg:20}: {avg_time:.8f}s")
    
    fastest_overall = min(avg_performance, key=avg_performance.__getitem__)
    print(f"\nFastest algorithm overall: {fastest_overall}")
    
    # Analysis by text
    print("\nPerformance by text:")
    
    # Article 1 analysis
    article1_times: Dict[str, float] = {alg: (results["Article 1_existing"][alg] + results["Article 1_fictional"][alg]) / 2
                     for alg in algorithms.keys()}
    fastest_article1: str = min(article1_times, key=article1_times.__getitem__)
    print(f"Article 1 fastest: {fastest_article1}")
    
    # Article 2 analysis
    article2_times: Dict[str, float] = {alg: (results["Article 2_existing"][alg] + results["Article 2_fictional"][alg]) / 2
                     for alg in algorithms.keys()}
    fastest_article2: str = min(article2_times, key=article2_times.__getitem__)
    print(f"Article 2 fastest: {fastest_article2}")

if __name__ == "__main__":

    run_performance_tests()
