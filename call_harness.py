"""Runs one function-call test: python call_harness.py program.py function_name < arguments

The arguments arrive one JSON value per line, and the return value is printed as JSON on the last
line. LiveCodeBench starter code uses typing names such as List without importing them, so they
are made available here, as LiveCodeBench's own harness does.
"""

import json
import runpy
import sys
import typing

program = runpy.run_path(sys.argv[1], {name: getattr(typing, name) for name in typing.__all__})
target = program["Solution"]() if "Solution" in program else None
function = getattr(target, sys.argv[2]) if target else program[sys.argv[2]]
result = function(*[json.loads(line) for line in sys.stdin.read().splitlines()])
print()
print(json.dumps(list(result) if isinstance(result, tuple) else result))
