from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a process-independent NumPy seed from stable text labels."""
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)

