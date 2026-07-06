"""Minimal fcntl stub for Windows. On Linux, the real fcntl is used instead."""
LOCK_SH = 1
LOCK_EX = 2
LOCK_UN = 8
F_GETFD = 1
F_SETFD = 2
FD_CLOEXEC = 1

def flock(fp, operation):
    """No-op on Windows."""
    pass

def fcntl(fd, cmd, arg=0):
    return 0

def lockf(fd, cmd, len=0, start=0, whence=0):
    return None
