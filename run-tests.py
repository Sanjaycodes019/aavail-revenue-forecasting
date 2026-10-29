#!/usr/bin/env python
"""
run the whole unit test suite with one command:

    python run-tests.py

all tests run in test mode (models/test-*.joblib, logs/test-*.log), so production
models and logs are never touched. Exits with a non-zero code if any test fails.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(os.path.join(ROOT, "unittests"), pattern="*Tests.py",
                                                top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
