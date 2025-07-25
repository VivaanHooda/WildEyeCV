
import ssl
import os

# Store original function
_original_create_default_https_context = ssl.create_default_context

def _create_unverified_context():
    """Create an unverified SSL context for downloads"""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# Only apply workaround if certificates fail
try:
    ssl.create_default_context().check_hostname = True
except Exception:
    ssl._create_default_https_context = _create_unverified_context
