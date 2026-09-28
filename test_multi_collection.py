import sys
from unittest.mock import MagicMock
sys.modules['pymongo'] = MagicMock()
sys.modules['pymongo.collection'] = MagicMock()

import os
import psutil
import time

# Use small limit for test
os.environ["HOT_CACHE_MAX_MEMORY_MB"] = "800.0"
os.environ["HOT_CACHE_MAX_ITEMS"] = "500000"

from vector_store_v2 import VectorStore, _cache_manager
import numpy as np

# Mock mongodb
class MockCursor:
    def __init__(self, count):
        self.count = count
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx >= self.count:
            raise StopIteration
        self.idx += 1
        return {"embedding": [0.1]*768, "text": "test", "metadata": {"chunk_id": f"c{self.idx}"}}

class MockCol:
    def __init__(self, count):
        self.count = count
    def count_documents(self, query):
        return self.count
    def find(self, query):
        return MockCursor(self.count)
    def estimated_document_count(self):
        return self.count

class MockClient:
    def close(self): pass

def mock_connect(self):
    self._client = MockClient()
    # 250k vectors per collection
    return MockCol(250000)

VectorStore.connect = mock_connect

print("=== Starting Test ===")
initial_rss = psutil.Process().memory_info().rss / 1024 / 1024
print(f"Initial RSS: {initial_rss:.2f} MB")

stores = []
for i in range(5):
    print(f"\n--- Loading Collection {i} ---")
    s = VectorStore("mock", "mock", f"col_{i}")
    stores.append(s)
    # trigger rebuild
    s._rebuild_hot_cache()
    
    rss = psutil.Process().memory_info().rss / 1024 / 1024
    
    cache = _cache_manager.get_cache(s.collection_name)
    if cache:
        print(f"Store {i} loaded. Count={cache['count']}, Global Items={_cache_manager.global_item_count}, Global Bytes={_cache_manager.global_memory_bytes}, RSS={rss:.2f} MB")
    else:
        print(f"Store {i} DENIED. Global Items={_cache_manager.global_item_count}, Global Bytes={_cache_manager.global_memory_bytes}, RSS={rss:.2f} MB")

print("\n=== Testing Eviction explicitly ===")
time.sleep(1)
# Add a large collection that forces eviction
class GiantCol:
    def count_documents(self, q): return 400000
    def find(self, q): return MockCursor(400000)
    
VectorStore.connect = lambda self: GiantCol()
s_giant = VectorStore("mock", "mock", "col_giant")
s_giant._rebuild_hot_cache()
rss = psutil.Process().memory_info().rss / 1024 / 1024
cache = _cache_manager.get_cache("col_giant")
print(f"Giant store loaded: Count={cache['count'] if cache else 0}, Global Items={_cache_manager.global_item_count}, Global Bytes={_cache_manager.global_memory_bytes}, RSS={rss:.2f} MB")
print(f"Active caches: {list(_cache_manager.caches.keys())}")

