import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# Mock out pymongo and numpy for import purposes
sys.modules['pymongo'] = MagicMock()
sys.modules['pymongo.collection'] = MagicMock()

import numpy as np

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
    def aggregate(self, pipeline): return []

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
import intent
import vector_store_v2
import vector_store

class TestDateRouting(unittest.TestCase):
    
    def test_date_extraction(self):
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        
        self.assertEqual(intent.extract_dates("what happened today?"), (today, today))
        self.assertEqual(intent.extract_dates("what happened yesterday?"), (yesterday, yesterday))
        self.assertEqual(intent.extract_dates("alerts on 2026-09-10"), ("2026-09-10", "2026-09-10"))
        self.assertEqual(intent.extract_dates("alerts from 2026-09-10 to 2026-09-12"), ("2026-09-10", "2026-09-12"))
        
        # Unparseable
        self.assertEqual(intent.extract_dates("random query with no date"), (None, None))
        
    @patch.object(vector_store_v2.VectorStore, '_search_hot_cache')
    @patch.object(vector_store_v2.VectorStore, '_search_historical')
    @patch.object(vector_store_v2.VectorStore, '_search_server_side')
    def test_routing_today(self, mock_mongo, mock_hist, mock_hot):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        store = get_vector_store("uri", "db", "col")
        
        # Return 5 hits from hot cache
        mock_hot.return_value = [{"score": 1.0, "metadata": {"chunk_id": f"hot{i}"}} for i in range(5)]
        
        res = store.cosine_search([0.1]*768, top_k=5, query_text="what happened today?")
        mock_hot.assert_called_once()
        mock_hist.assert_not_called()
        mock_mongo.assert_not_called()
        self.assertEqual(len(res), 5)
        
    @patch.object(vector_store_v2.VectorStore, '_search_hot_cache')
    @patch.object(vector_store_v2.VectorStore, '_search_historical')
    @patch.object(vector_store_v2.VectorStore, '_search_server_side')
    def test_routing_historical(self, mock_mongo, mock_hist, mock_hot):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        store = get_vector_store("uri", "db", "col")
        
        mock_hist.return_value = [{"score": 1.0, "metadata": {"chunk_id": f"hist{i}"}} for i in range(5)]
        
        res = store.cosine_search([0.1]*768, top_k=5, query_text="what happened on 2026-09-10")
        mock_hot.assert_not_called()
        mock_hist.assert_called_once()
        # Ensure it passed the exact date to historical
        self.assertEqual(mock_hist.call_args[0][2], "2026-09-10")
        self.assertEqual(len(res), 5)

    @patch.object(vector_store_v2.VectorStore, '_search_hot_cache')
    @patch.object(vector_store_v2.VectorStore, '_search_historical')
    @patch.object(vector_store_v2.VectorStore, '_search_server_side')
    def test_routing_multi_day(self, mock_mongo, mock_hist, mock_hot):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        store = get_vector_store("uri", "db", "col")
        
        # Give 2 hits per day
        mock_hist.return_value = [{"score": 1.0, "metadata": {"chunk_id": "h"}}] 
        
        res = store.cosine_search([0.1]*768, top_k=5, query_text="what happened from 2026-09-10 to 2026-09-12")
        self.assertEqual(mock_hist.call_count, 3) # 10, 11, 12
        
    @patch.object(vector_store_v2.VectorStore, '_search_hot_cache')
    @patch.object(vector_store_v2.VectorStore, '_search_historical')
    @patch.object(vector_store_v2.VectorStore, '_search_server_side')
    @patch.object(vector_store_v2.VectorStore, '_get_all_historical_dates')
    def test_routing_no_date_waterfall(self, mock_get_dates, mock_mongo, mock_hist, mock_hot):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        store = get_vector_store("uri", "db", "col")
        mock_get_dates.return_value = ["2026-09-10"]
        
        mock_hot.return_value = [{"score": 0.9, "metadata": {"chunk_id": "hot1"}}]
        mock_hist.return_value = [{"score": 0.8, "metadata": {"chunk_id": "hist1"}}]
        mock_mongo.return_value = [{"score": 0.7, "metadata": {"chunk_id": "mongo1"}}]
        
        res = store.cosine_search([0.1]*768, top_k=5, query_text="general query")
        
        mock_hot.assert_called_once()
        mock_hist.assert_called_once()
        mock_mongo.assert_called_once()
        
        self.assertEqual(len(res), 3)
        
    @patch.object(vector_store.VectorStore, '_search_server_side')
    def test_phase1_regression(self, mock_mongo):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "false"
        store = get_vector_store("uri", "db", "col")
        
        self.assertIsInstance(store, vector_store.VectorStore)
        
        # Should not break Phase 1 if dates exist in text
        mock_mongo.return_value = []
        store.cosine_search([0.1]*768, top_k=5, query_text="what happened today?")
        
if __name__ == '__main__':
    unittest.main()
