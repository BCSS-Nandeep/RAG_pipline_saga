import unittest
from datetime import datetime, timezone
import os

from db import init_db, get_pool
from source_store import SourceStore
from processor import PostgresStreamProcessor, DocumentConverter
from chunker import TokenAwareChunker
from embedder import get_embedder
from vector_store import VectorStore

class TestIntegrationV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.src_store = SourceStore()
        # DB_NAME and URI are no longer used for postgres connection, but we pass them
        cls.vec_store = VectorStore("dummy", "dummy", "vector_embeddings")
        cls.embedder = get_embedder()
        
        cls._cleanup()
        
    @classmethod
    def tearDownClass(cls):
        cls._cleanup()
        cls.vec_store.close()
        
    @classmethod
    def _cleanup(cls):
        cls.src_store.delete_test_documents()
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM vector_embeddings WHERE document_id LIKE 'pg_test_%'")
            conn.commit()

    def test_full_pipeline(self):
        # 1. Insert test documents
        docs = [
            {
                "_id": "pg_test_int_1",
                "created_at": datetime(2026, 9, 28, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
                "title": "First integration test document",
                "content": "This is a detailed paragraph to ensure that chunking actually happens. " * 10
            },
            {
                "_id": "pg_test_int_2",
                "created_at": datetime(2026, 9, 28, 10, 5, 0, tzinfo=timezone.utc).isoformat(),
                "title": "Second integration test document",
                "content": "Another distinct document to verify batching and pagination logic across multiple files. " * 10
            }
        ]
        
        written = self.src_store.upsert_documents(docs, "test_alerts")
        self.assertEqual(written, 2)
        
        # 2. Setup processors
        streamer = PostgresStreamProcessor("test_alerts", batch_size=1)
        converter = DocumentConverter()
        chunker = TokenAwareChunker(min_tokens=20, max_tokens=50, overlap=5)
        
        # Verify collection count
        self.assertEqual(streamer.count_documents(), 2)
        
        # 3. Simulate Pipeline Ingestion Loop
        batch_num = 1
        after_id = None
        docs_processed = 0
        chunks_stored = 0
        
        while True:
            batch = streamer.fetch_batch(batch_num, after_id=after_id)
            if not batch:
                break
                
            self.assertEqual(len(batch), 1)  # Since batch_size=1
            
            all_chunks = []
            all_texts = []
            
            for doc in batch:
                doc_id = str(doc.get("_id", ""))
                
                # Check document structure preservation
                self.assertIn("title", doc)
                self.assertIn("_id", doc)
                self.assertTrue(doc_id.startswith("pg_test_int_"))
                
                text = converter.convert(doc)
                
                doc_chunks = chunker.chunk_document(text, "test_alerts", doc_id)
                self.assertTrue(len(doc_chunks) > 1) # Ensure chunking occurred
                
                for chunk in doc_chunks:
                    chunk._source_created_at = doc.get("created_at")
                    all_chunks.append(chunk)
                    all_texts.append(chunk.text)
            
            embeddings = self.embedder.embed_texts(all_texts)
            self.assertEqual(len(embeddings), len(all_texts))
            self.assertEqual(len(embeddings[0]), 768)
            
            batch_to_store = []
            for chunk, emb in zip(all_chunks, embeddings):
                batch_to_store.append({
                    "text": chunk.text,
                    "embedding": emb,
                    "metadata": {
                        "source_collection": chunk.metadata.source_collection,
                        "document_id": chunk.metadata.document_id,
                        "chunk_index": chunk.metadata.chunk_index,
                        "total_chunks": chunk.metadata.total_chunks,
                        "source_created_at": getattr(chunk, '_source_created_at', None),
                    },
                })
                
            written = self.vec_store.upsert_chunks(batch_to_store)
            chunks_stored += written
            docs_processed += 1
            
            after_id = str(batch[-1].get("_id", ""))
            batch_num += 1
            
        self.assertEqual(docs_processed, 2)
        self.assertTrue(chunks_stored > 2)
        
        # 4. Verify storage results
        # Use VectorStore search to check metadata preservation
        query_emb = self.embedder.embed_texts(["integration test document"])[0]
        results = self.vec_store.cosine_search(query_emb, top_k=10, source_collection="test_alerts")
        
        self.assertTrue(len(results) > 0)
        
        # Check metadata
        for res in results:
            self.assertEqual(res["metadata"]["source_collection"], "test_alerts")
            self.assertTrue(res["metadata"]["document_id"].startswith("pg_test_int_"))

if __name__ == "__main__":
    unittest.main()
