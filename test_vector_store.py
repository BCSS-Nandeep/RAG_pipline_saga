import unittest
import math
from vector_store import VectorStore
from db import init_db

class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.store = VectorStore()
        # Ensure cleanup before tests
        cls._cleanup()

    @classmethod
    def tearDownClass(cls):
        cls._cleanup()

    @classmethod
    def _cleanup(cls):
        with cls.store._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vector_embeddings WHERE document_id LIKE 'pg_migration_test_%'")
            conn.commit()

    def test_1_single_upsert(self):
        v = [0.1] * 768
        chunks = [{
            "text": "Single test chunk",
            "embedding": v,
            "metadata": {
                "document_id": "pg_migration_test_doc_001",
                "chunk_index": 0,
                "source_collection": "test_col"
            }
        }]
        written = self.store.upsert_chunks(chunks)
        self.assertEqual(written, 1)

    def test_2_batch_upsert_and_duplicate(self):
        v1 = [0.2] * 768
        v2 = [0.3] * 768
        chunks = [
            {
                "text": "Batch chunk 1",
                "embedding": v1,
                "metadata": {"document_id": "pg_migration_test_doc_002", "chunk_index": 0, "source_collection": "test_col_2"}
            },
            {
                "text": "Batch chunk 2",
                "embedding": v2,
                "metadata": {"document_id": "pg_migration_test_doc_002", "chunk_index": 1, "source_collection": "test_col_2"}
            }
        ]
        written = self.store.upsert_chunks(chunks)
        self.assertEqual(written, 2)
        
        # Duplicate upsert (should update, not fail)
        written_dup = self.store.upsert_chunks(chunks)
        self.assertEqual(written_dup, 2)
        
    def test_3_invalid_dimension(self):
        v_invalid = [0.1] * 10
        chunks = [{
            "text": "Invalid chunk",
            "embedding": v_invalid,
            "metadata": {
                "document_id": "pg_migration_test_doc_003",
                "chunk_index": 0,
                "source_collection": "test_col"
            }
        }]
        with self.assertRaises(ValueError):
            self.store.upsert_chunks(chunks)
            
        with self.assertRaises(ValueError):
            self.store.cosine_search(query_vector=v_invalid)

    def test_4_cosine_search_top_k(self):
        # Insert specific vectors for searching
        v_target = [0.0] * 768; v_target[0] = 1.0
        v_other = [0.0] * 768; v_other[0] = -1.0
        
        self.store.upsert_chunks([
            {"text": "Target text", "embedding": v_target, "metadata": {"document_id": "pg_migration_test_doc_004", "chunk_index": 0, "source_collection": "search_col", "extra": "data"}},
            {"text": "Other text", "embedding": v_other, "metadata": {"document_id": "pg_migration_test_doc_005", "chunk_index": 0, "source_collection": "search_col"}},
        ])
        
        results = self.store.cosine_search(query_vector=v_target, top_k=1, source_collection="search_col")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "Target text")
        self.assertEqual(results[0]["metadata"]["document_id"], "pg_migration_test_doc_004")
        self.assertEqual(results[0]["metadata"]["extra"], "data") # Metadata preservation

    def test_5_document_id_filter(self):
        v = [1.0] + [0.0]*767
        # Ensure we find doc_005 even though doc_004 is closer to query if not filtered
        results = self.store.cosine_search(query_vector=v, top_k=5, source_collection="search_col", doc_ids=["pg_migration_test_doc_005"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["document_id"], "pg_migration_test_doc_005")

    def test_6_empty_results(self):
        v = [1.0] + [0.0]*767
        results = self.store.cosine_search(query_vector=v, top_k=5, source_collection="non_existent_col")
        self.assertEqual(len(results), 0)
        
        # Empty query vector
        results2 = self.store.cosine_search(query_vector=[])
        self.assertEqual(len(results2), 0)

if __name__ == "__main__":
    unittest.main()
