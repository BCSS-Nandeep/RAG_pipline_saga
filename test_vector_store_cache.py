import json
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Adjust import based on the actual module structure
from vector_store import VectorStore, CACHE_DIR, CACHE_SIZE_LIMIT, _EMBED_DIM

class TestVectorStoreCache(unittest.TestCase):
    def setUp(self):
        self.uri = "mongodb://localhost:27017"
        self.db_name = "test_db"
        self.collection_name = "test_collection"
        self.store = VectorStore(self.uri, self.db_name, self.collection_name)
        
        # Clean up any potential cache files
        self.emb_path = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.npy")
        self.meta_path = os.path.join(CACHE_DIR, f"{self.collection_name}_meta.json")
        for f in [self.emb_path, self.meta_path]:
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        self.store.invalidate_cache()

    @patch('vector_store.VectorStore.connect')
    def test_is_oversized(self, mock_connect):
        mock_col = MagicMock()
        mock_connect.return_value = mock_col
        
        # Test under limit
        mock_col.estimated_document_count.return_value = CACHE_SIZE_LIMIT - 10
        self.assertFalse(self.store._is_oversized())
        self.assertFalse(self.store._oversized)
        
        # Test cached value is used (don't call estimated_document_count again)
        mock_col.estimated_document_count.return_value = CACHE_SIZE_LIMIT + 10
        self.assertFalse(self.store._is_oversized())
        
        # Test invalidation resets the cached value
        self.store.invalidate_cache()
        self.assertTrue(self.store._is_oversized())
        self.assertTrue(self.store._oversized)

    @patch('vector_store.VectorStore._search_server_side')
    @patch('vector_store.VectorStore._search_with_cache')
    @patch('vector_store.VectorStore._is_oversized')
    def test_cosine_search_oversized_guard(self, mock_is_oversized, mock_search_cache, mock_search_server):
        mock_is_oversized.return_value = True
        mock_search_server.return_value = [{"hit": 1}]
        
        results = self.store.cosine_search(query_vector=[0.1]*768)
        
        self.assertEqual(results, [{"hit": 1}])
        mock_search_server.assert_called_once()
        mock_search_cache.assert_not_called()

    @patch('vector_store.VectorStore.connect')
    def test_build_cache_streaming_atomic(self, mock_connect):
        mock_col = MagicMock()
        mock_connect.return_value = mock_col
        
        # Mock MongoDB returning 2 batches of fake data
        doc1 = {"_id": 1, "text": "doc1", "embedding": [0.1] * _EMBED_DIM, "metadata": {"doc_id": "1"}}
        doc2 = {"_id": 2, "text": "doc2", "embedding": [0.2] * _EMBED_DIM, "metadata": {"doc_id": "2"}}
        
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        
        # Return batch 1, then batch 2, then empty
        mock_col.find.side_effect = [
            MagicMock(sort=lambda *a: MagicMock(limit=lambda *a: [doc1])),
            MagicMock(sort=lambda *a: MagicMock(limit=lambda *a: [doc2])),
            MagicMock(sort=lambda *a: MagicMock(limit=lambda *a: []))
        ]
        
        self.store._build_cache(self.emb_path, self.meta_path)
        
        # Verify cache was created and loaded
        self.assertTrue(self.store._cache_loaded)
        self.assertEqual(len(self.store._texts), 2)
        self.assertEqual(self.store._embeddings.shape, (2, _EMBED_DIM))
        
        # Verify files exist on disk
        self.assertTrue(os.path.exists(self.emb_path))
        self.assertTrue(os.path.exists(self.meta_path))
        
        # Verify tmp files do not exist
        tmp_emb = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.tmp.npy")
        tmp_raw = os.path.join(CACHE_DIR, f"{self.collection_name}_embeddings.raw.tmp")
        self.assertFalse(os.path.exists(tmp_emb))
        self.assertFalse(os.path.exists(tmp_raw))

if __name__ == '__main__':
    unittest.main()
