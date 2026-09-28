import unittest
from datetime import datetime, timezone
from db import init_db
from source_store import SourceStore

class TestSourceStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.store = SourceStore()
        cls.store.delete_test_documents()

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_test_documents()

    def test_1_insert_and_retrieve(self):
        doc = {
            "_id": "pg_test_001",
            "created_at": "2026-09-28T10:00:00+00:00",
            "title": "Test alert",
            "description": "This is a PostgreSQL migration test",
            "nested": {
                "location": "Hyderabad",
                "severity": "high"
            }
        }
        
        # Insert
        written = self.store.upsert_documents([doc], "alerts")
        self.assertEqual(written, 1)
        
        # Retrieve by ID
        fetched = self.store.get_document("pg_test_001")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Test alert")
        self.assertEqual(fetched["nested"]["location"], "Hyderabad")
        # JSON structure preservation
        self.assertIn("_id", fetched)
        
        # Retrieve by collection
        batch = self.store.fetch_batch("alerts", limit=10)
        self.assertTrue(any(d["_id"] == "pg_test_001" for d in batch))

    def test_2_batch_insert_and_upsert(self):
        docs = [
            {
                "_id": "pg_test_002",
                "created_at": datetime(2026, 9, 28, 11, 0, 0, tzinfo=timezone.utc),
                "title": "Test event 1"
            },
            {
                "_id": "pg_test_003",
                "created_at": datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc),
                "title": "Test event 2"
            }
        ]
        
        written = self.store.upsert_documents(docs, "events")
        self.assertEqual(written, 2)
        
        # Upsert
        docs[0]["title"] = "Test event 1 UPDATED"
        written = self.store.upsert_documents([docs[0]], "events")
        self.assertEqual(written, 1)
        
        fetched = self.store.get_document("pg_test_002")
        self.assertEqual(fetched["title"], "Test event 1 UPDATED")

    def test_3_incremental_retrieval(self):
        # Fetch after_id
        batch = self.store.fetch_batch("events", limit=10, after_id="pg_test_002")
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0]["_id"], "pg_test_003")
        
        # Fetch since
        since_dt = datetime(2026, 9, 28, 11, 30, 0, tzinfo=timezone.utc)
        batch2 = self.store.fetch_batch("events", limit=10, since=since_dt)
        self.assertEqual(len(batch2), 1)
        self.assertEqual(batch2[0]["_id"], "pg_test_003")

    def test_4_collection_filtering(self):
        alerts_count = self.store.count_documents("alerts")
        events_count = self.store.count_documents("events")
        
        alerts_batch = self.store.fetch_batch("alerts")
        events_batch = self.store.fetch_batch("events")
        
        self.assertTrue(all(d["_id"] != "pg_test_002" for d in alerts_batch))
        self.assertTrue(any(d["_id"] == "pg_test_002" for d in events_batch))

if __name__ == "__main__":
    unittest.main()
