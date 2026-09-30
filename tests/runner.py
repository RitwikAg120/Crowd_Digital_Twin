"""
Minimal test runner so the checks run without pytest (plain `python
tests/test_x.py` from the project folder); pytest also collects them.
"""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def run(namespace: dict):
    tests = [(n, f) for n, f in namespace.items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        t = time.perf_counter()
        try:
            note = fn()
            print(f"PASS {name}" + (f" — {note}" if note else "") + f"  ({time.perf_counter() - t:.1f} s)")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failed}/{len(tests)} checks passed")
    sys.exit(1 if failed else 0)
