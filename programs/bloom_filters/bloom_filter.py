import math
import mmh3  # MurmurHash3 library
from bitarray import bitarray

class BloomFilter:
    """
    Simple Bloom Filter implementation using MurmurHash3 and bitarray

    Install dependencies:
    pip install mmh3 bitarray
    """

    def __init__(self, capacity, error_rate=0.001):
        """
        Initialize a Bloom Filter

        Args:
            capacity: Expected number of elements to be inserted
            error_rate: Desired false positive probability
        """
        self.capacity = capacity
        self.error_rate = error_rate

        # Calculate optimal size and number of hash functions
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)

        # Initialize bit array
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

        print(f"Bloom Filter created with:")
        print(f"  - {self.size} bits ({self.size/8/1024:.2f} KB)")
        print(f"  - {self.hash_count} hash functions")
        print(f"  - Expected false positive rate: {self.error_rate}")

    def _calculate_size(self, n, p):
        """Calculate optimal bit array size"""
        size = -n * math.log(p) / (math.log(2) ** 2)
        return int(size)

    def _calculate_hash_count(self, m, n):
        """Calculate optimal number of hash functions"""
        k = (m / n) * math.log(2)
        return int(k)

    def _get_hash_positions(self, item):
        """Get all hash positions for an item"""
        positions = []
        for seed in range(self.hash_count):
            # Use different seed values to get different hash functions
            position = mmh3.hash(str(item), seed) % self.size
            positions.append(abs(position))
        return positions

    def add(self, item):
        """Add an item to the bloom filter"""
        for position in self._get_hash_positions(item):
            self.bit_array[position] = 1

    def contains(self, item):
        """Check if an item might be in the bloom filter"""
        for position in self._get_hash_positions(item):
            if not self.bit_array[position]:
                return False
        return True

    def estimate_current_error_rate(self, items_count=None):
        """Estimate the current false positive rate based on fill ratio"""
        if items_count is None:
            items_count = self.capacity

        # Calculate the probability of a random bit being 0
        p_zero = (1 - 1/self.size) ** (self.hash_count * items_count)

        # False positive probability is (1 - p_zero)^k
        fp_prob = (1 - p_zero) ** self.hash_count
        return fp_prob

if __name__ == "__main__":
    # Create a Bloom filter for 10,000 elements with 1% error rate
    bloom = BloomFilter(capacity=10000, error_rate=0.01)

    # Add some elements
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.org/page1",
        "https://example.org/page2",
    ]

    for url in urls:
        bloom.add(url)

    # Test for membership
    test_urls = [
        "https://example.com/page1",  # Should be in the set
        "https://example.com/page4",  # Should not be in the set
        "https://example.org/page1",  # Should be in the set
        "https://example.net/page1",  # Should not be in the set
    ]

    for url in test_urls:
        result = bloom.contains(url)
        print(f"URL: {url} - In bloom filter: {result}")

    # Add many elements and check false positive rate
    for i in range(10000):
        bloom.add(f"item{i}")

    print(f"Current estimated false positive rate: {bloom.estimate_current_error_rate():.4f}")