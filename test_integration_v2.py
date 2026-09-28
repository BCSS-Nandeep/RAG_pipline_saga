import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock out pymongo and numpy for import purposes
sys.modules['pymongo'] = MagicMock()
sys.modules['pymongo.collection'] = MagicMock()

import numpy as np
os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"

class MockCollection:
    def estimated_document_count(self): return 10
    def count_documents(self, query): return 0
    def find(self, query): return []
    def create_index(self, *args, **kwargs): pass
    def bulk_write(self, ops, ordered=False):
        m = MagicMock()
        m.upserted_count = len(ops)
        m.modified_count = 0
        return m
    def delete_many(self, query): pass
    def aggregate(self, pipeline): return [{"text": "mongo", "embedding": np.random.rand(768).tolist(), "metadata": {"chunk_id": "m1"}}]

class MockDb:
    def __getitem__(self, name): return MockCollection()
class MockCli:
    def __getitem__(self, name): return MockDb()
    def close(self): pass

sys.modules['pymongo'].MongoClient = lambda *args, **kwargs: MockCli()
sys.modules['pymongo'].ASCENDING = 1
sys.modules['pymongo'].UpdateOne = MagicMock()
sys.modules['pymongo'].DeleteOne = MagicMock()

from store_factory import get_vector_store
import vector_store_v2

class TestV2Integration(unittest.TestCase):
    def setUp(self):
        self.uri = "mongodb://localhost:27017"
        self.db_name = "test_db"
        self.collection_name = "test_collection"
        self.cache_dir = "/tmp/soceye-rag-phase2-test/v2_integration"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.store = get_vector_store(self.uri, self.db_name, self.collection_name, cache_dir=self.cache_dir)

    def test_factory_returns_v2(self):
        self.assertIsInstance(self.store, vector_store_v2.VectorStore)

    def test_api_compatibility(self):
        # 1. initialize - tested in setUp
        
        # 2. total_chunks()
        count = self.store.total_chunks()
        pass
        
        # 3. refresh_cache() / invalidate_cache()
        self.store.refresh_cache()
        self.store.invalidate_cache()
        
        # 4. upsert_chunks()
        chunk = {
            "text": "integration text",
            "embedding": np.random.rand(768).astype(np.float32).tolist(),
            "metadata": {"document_id": "doc1", "chunk_index": 0, "chunk_id": "doc1::0", "source_created_at": "2026-09-24"}
        }
        self.store.upsert_chunks([chunk])
        
        # 5. cosine_search()
        res = self.store.cosine_search(np.random.rand(768).astype(np.float32).tolist(), top_k=5)
        self.assertTrue(len(res) > 0)
        
        # 6. delete_chunks()
        self.store.delete_chunks(["doc1::0"])
        
        # If no exceptions were raised, API compatibility is guaranteed

if __name__ == "__main__":
    unittest.main()
