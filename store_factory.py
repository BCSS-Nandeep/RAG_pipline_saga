import os
import logging

logger = logging.getLogger(__name__)

def get_vector_store(uri: str, db_name: str, collection_name: str, cache_dir=None):
    phase2_enabled = os.getenv("PHASE2_HOT_CACHE_ENABLED", "false").lower() == "true"
    
    if phase2_enabled:
        try:
            from vector_store_v2 import VectorStore as VectorStoreV2
            kwargs = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            return VectorStoreV2(uri, db_name, collection_name, **kwargs)
        except Exception as e:
            logger.error("Phase 2 initialization failed: %s", e)
            # Step 6: Startup Safety - Do not silently fall back to Phase 1. Fail explicitly.
            raise e
    else:
        from vector_store import VectorStore
        # Phase 1 initialization
        if cache_dir:
            return VectorStore(uri, db_name, collection_name, cache_dir)
        return VectorStore(uri, db_name, collection_name)
