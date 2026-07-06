
"""Minimal fcntl stub for Windows compatibility."""
import os

LOCK_SH = 1
LOCK_EX = 2
LOCK_UN = 8

def flock(fp, operation):
    """No-op on Windows - file locking not available."""
    pass
