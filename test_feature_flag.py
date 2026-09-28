import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock out pymongo and numpy for import purposes
sys.modules['pymongo'] = MagicMock()
sys.modules['pymongo.collection'] = MagicMock()
sys.modules['numpy'] = MagicMock()

import store_factory

class TestFeatureFlag(unittest.TestCase):
    def setUp(self):
        # Reset flag before each test
        if "PHASE2_HOT_CACHE_ENABLED" in os.environ:
            del os.environ["PHASE2_HOT_CACHE_ENABLED"]
            
    def test_flag_absent_defaults_to_phase1(self):
        store = store_factory.get_vector_store("uri", "db", "col")
        self.assertEqual(store.__class__.__name__, "VectorStore")
        self.assertEqual(store.__module__, "vector_store")

    def test_explicit_false_uses_phase1(self):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "false"
        store = store_factory.get_vector_store("uri", "db", "col")
        self.assertEqual(store.__class__.__name__, "VectorStore")
        self.assertEqual(store.__module__, "vector_store")

    def test_explicit_true_uses_phase2(self):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        store = store_factory.get_vector_store("uri", "db", "col")
        self.assertEqual(store.__class__.__name__, "VectorStore")
        # In V2, the class is actually VectorStore too, but inside vector_store_v2 module
        self.assertEqual(store.__module__, "vector_store_v2")

    def test_phase2_init_failure_does_not_fallback(self):
        os.environ["PHASE2_HOT_CACHE_ENABLED"] = "true"
        # We will mock vector_store_v2.VectorStore to raise an error
        import vector_store_v2
        original = vector_store_v2.VectorStore
        vector_store_v2.VectorStore = MagicMock(side_effect=Exception("Simulated Phase 2 Crash"))
        
        try:
            with self.assertRaises(Exception) as context:
                store_factory.get_vector_store("uri", "db", "col")
            self.assertTrue("Simulated Phase 2 Crash" in str(context.exception))
        finally:
            vector_store_v2.VectorStore = original

if __name__ == "__main__":
    unittest.main()
