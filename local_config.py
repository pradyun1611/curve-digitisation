"""
Local Configuration - SSL Certificate Fix

This file handles local environment fixes that shouldn't be committed to git.
It removes invalid SSL_CERT_FILE environment variable if it points to a non-existent file.
It also injects the Windows certificate store into Python's SSL for corporate proxy support.

This file is gitignored and won't be overwritten by git pulls.
"""

import os

# Fix 1: Remove invalid SSL_CERT_FILE if it points to a non-existent file
# This is needed because some tools set SSL_CERT_FILE to a placeholder path
_ssl_cert_file = os.environ.get('SSL_CERT_FILE')
if _ssl_cert_file and not os.path.exists(_ssl_cert_file):
    del os.environ['SSL_CERT_FILE']
    print(f"[local_config] Removed invalid SSL_CERT_FILE: {_ssl_cert_file}")

# Fix 2: Use Windows certificate store for SSL verification (corporate proxy support)
# This injects system CA certs so Python trusts whatever Windows trusts
try:
    import truststore
    truststore.inject_into_ssl()
    print("[local_config] Injected Windows certificate store into SSL")
except ImportError:
    print("[local_config] Warning: truststore not installed, run: pip install truststore")
except Exception as e:
    print(f"[local_config] Warning: Could not inject truststore: {e}")
