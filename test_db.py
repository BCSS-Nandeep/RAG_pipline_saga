import logging
from db import init_db, check_postgres_connection, close_pool, get_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    print("Testing PostgreSQL connection layer...")
    try:
        init_db()
        success = check_postgres_connection()
        if success:
            print("\nSUCCESS: Connection acquired, SELECT 1 succeeded, and native schema is available.")
            
            # Test vector insertion and retrieval
            pool = get_pool()
            import math
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    print("Testing vector insert...")
                    
                    # Create 3 dummy vectors
                    v1 = [0.1] * 768; v1[0] = 1.0; norm1 = math.sqrt(sum(x*x for x in v1))
                    v2 = [0.1] * 768; v2[0] = 0.5; norm2 = math.sqrt(sum(x*x for x in v2))
                    v3 = [0.1] * 768; v3[0] = -1.0; norm3 = math.sqrt(sum(x*x for x in v3))
                    
                    cur.execute("""
                        INSERT INTO vector_embeddings (document_id, chunk_index, text, embedding, embedding_norm, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s),
                               (%s, %s, %s, %s, %s, %s),
                               (%s, %s, %s, %s, %s, %s)
                        RETURNING id;
                    """, (
                        "doc_A", 0, "Hello vector 1", v1, norm1, '{"source_collection": "alerts"}',
                        "doc_A", 1, "Hello vector 2", v2, norm2, '{"source_collection": "alerts"}',
                        "doc_B", 0, "Hello vector 3", v3, norm3, '{"source_collection": "news"}'
                    ))
                    
                    print("Testing vector query with cosine similarity...")
                    query_vec = v1  # Should match v1 best
                    query_norm = norm1
                    
                    cur.execute("""
                        SELECT document_id, chunk_index, text, 
                               cosine_similarity(%s::DOUBLE PRECISION[], %s::DOUBLE PRECISION, embedding, embedding_norm) as score
                        FROM vector_embeddings
                        WHERE document_id IN ('doc_A', 'doc_B')
                          AND metadata->>'source_collection' = 'alerts'
                        ORDER BY score DESC
                        LIMIT 2;
                    """, (query_vec, query_norm))
                    
                    results = cur.fetchall()
                    print("\nResults:")
                    for r in results:
                        print(f"Doc: {r[0]}, Chunk: {r[1]}, Text: {r[2]}, Score: {r[3]:.4f}")
                        
                    print("\nCleaning up test rows...")
                    cur.execute("DELETE FROM vector_embeddings WHERE document_id IN ('doc_A', 'doc_B')")
                conn.commit()
                print("Vector test complete and cleaned up.")
                
        else:
            print("\nFAILED: Health check did not pass.")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        close_pool()
