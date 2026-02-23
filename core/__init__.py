"""
Core module for curve digitization functionality.

Includes OpenAI API client and image processing capabilities.

IMPORTANT – SSL bootstrap
~~~~~~~~~~~~~~~~~~~~~~~~~
Corporate proxies / SSL-inspection environments set ``SSL_CERT_FILE`` to a
path that may not exist on every machine.  The fix below runs **before**
any networking code is imported, so the invalid variable never causes
``[Errno 2] No such file or directory``.

DO NOT REMOVE THIS BLOCK – it is the permanent fix for the SSL error that
appears after every ``git pull``.
"""

import os as _os

# ── Fix 1: remove bogus SSL_CERT_FILE ────────────────────────────────────────
_ssl_cert = _os.environ.get("SSL_CERT_FILE")
if _ssl_cert and not _os.path.exists(_ssl_cert):
    del _os.environ["SSL_CERT_FILE"]

# ── Fix 2: inject Windows / system CA store via truststore (if installed) ────
try:
    import truststore as _ts  # noqa: F401
    _ts.inject_into_ssl()
except Exception:          # ImportError, RuntimeError, etc.
    pass

# ── Public API ───────────────────────────────────────────────────────────────
from .openai_client import OpenAIClient   # noqa: E402
from .image_processor import CurveDigitizer  # noqa: E402

__all__ = ["OpenAIClient", "CurveDigitizer"]
