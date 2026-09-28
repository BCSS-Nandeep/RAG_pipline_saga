import sys
from unittest.mock import MagicMock
sys.modules['pymongo'] = MagicMock()
sys.modules['pymongo.collection'] = MagicMock()
import unittest
import test_vector_store_cache
import test_feature_flag
unittest.main(module=None, argv=['unittest', 'test_vector_store_cache', 'test_feature_flag'])
