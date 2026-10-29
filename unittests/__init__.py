import os
import sys

## make the project modules importable when tests are run from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def production_snapshot():
    """modification times of every production model and log file"""

    snap = {}
    for sub in ("models", "logs"):
        folder = os.path.join(ROOT, sub)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if not f.startswith("test-"):
                    snap[os.path.join(sub, f)] = os.path.getmtime(os.path.join(folder, f))
    return snap
