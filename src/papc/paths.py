"""Filesystem roots shared by every runner.

Benchmarks live under ``$PAPC_DATA_ROOT/benchmarks`` and private run rows under
``$PAPC_DATA_ROOT/papc``. The root defaults to ``~/data``.
"""

import os

DATA_ROOT = os.environ.get("PAPC_DATA_ROOT") or os.path.expanduser("~/data")
