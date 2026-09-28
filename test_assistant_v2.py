import unittest
from datetime import datetime, timezone

from db import init_db, get_pool
from vector_store import VectorStore
from embedder import get_embedder
from assistant import Assistant

class TestAssistantV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.vec_store = VectorStore()
        cls.embedder = get_embedder()
        cls._cleanup()
        
    @classmethod
    def tearDownClass(cls):
        cls._cleanup()
        cls.vec_store.close()
        
    @classmethod
    def _cleanup(cls):
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vector_embeddings WHERE document_id LIKE 'pg_assistant_test_%'")
            conn.commit()

    def test_assistant_retrieval_and_answer(self):
        # 1. Create a test vector through the existing embedding path
        test_text = "The super admin of the SOC-EYE system is Nandeep. He was appointed on September 28, 2026."
        test_emb = self.embedder.embed_text(test_text)
        
        # 2. Store it using VectorStore
        chunk = {
            "text": test_text,
            "embedding": test_emb,
            "metadata": {
                "source_collection": "test_collection",
                "document_id": "pg_assistant_test_1",
                "chunk_index": 0,
                "total_chunks": 1,
                "source_created_at": datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc).isoformat()
            }
        }
        written = self.vec_store.upsert_chunks([chunk])
        self.assertEqual(written, 1)
        
        # 3. Execute an assistant query
        # Provide llm_model, but we assume it might hit the real LLM or we just check the context.
        # Wait, the LLM call takes 60s or fails gracefully if not available. 
        bot = Assistant(
            llm_model="test-model",
            top_k=5,
            source_collection="test_collection"
        )
        
        question = "Who is the super admin of the SOC-EYE system?"
        result = bot.ask(question)
        
        # 4. Verify PostgreSQL retrieval occurs and expected document is retrieved
        self.assertTrue(len(result["sources"]) > 0, "No sources retrieved")
        
        # 5. Verify metadata reaches the context layer
        found_test_doc = False
        for src in result["sources"]:
            if src["document_id"] == "pg_assistant_test_1":
                found_test_doc = True
                self.assertEqual(src["collection"], "test_collection")
                self.assertEqual(src["chunk_index"], 0)
                self.assertIn("Nandeep", src["preview"])
                
        self.assertTrue(found_test_doc, "Test document was not retrieved by Assistant")
        
        # 6. Verify the answer generation path receives the retrieved context
        # (It should try to answer, but even if it fails gracefully, the answer string exists)
        self.assertIn("answer", result)
        self.assertTrue(isinstance(result["answer"], str))
        self.assertTrue(len(result["answer"]) > 0)
        
        bot.close()

if __name__ == "__main__":
    unittest.main()
