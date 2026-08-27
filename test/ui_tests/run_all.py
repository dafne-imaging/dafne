#!/usr/bin/env python3
"""
Run every UI test script in this directory, each in its own process (the Dafne
interface is not designed to be created and torn down repeatedly inside a single
process). Usage:

    python test/ui_tests/run_all.py [pattern ...]

Optional patterns filter the scripts by substring (e.g. "viewer" runs only the
viewer tests). Exits with a nonzero code if any script fails.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
TIMEOUT = 600  # seconds per script


def main():
    patterns = sys.argv[1:]
    scripts = sorted(TEST_DIR.glob('test_*.py'))
    if patterns:
        scripts = [s for s in scripts if any(p in s.name for p in patterns)]
    if not scripts:
        print('No test scripts matched.')
        sys.exit(1)

    results = {}
    for script in scripts:
        print('=' * 70)
        print('Running', script.name)
        print('=' * 70, flush=True)
        start = time.time()
        try:
            proc = subprocess.run([sys.executable, '-u', str(script)], cwd=str(TEST_DIR),
                                  timeout=TIMEOUT)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            print('TIMEOUT after {} s'.format(TIMEOUT))
            ok = False
        results[script.name] = (ok, time.time() - start)

    print()
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    failed = 0
    for name, (ok, elapsed) in results.items():
        print('{:<40} {:>8} {:>8.1f}s'.format(name, 'PASS' if ok else 'FAIL', elapsed))
        if not ok:
            failed += 1
    print('-' * 70)
    print('{} of {} scripts passed'.format(len(results) - failed, len(results)))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
