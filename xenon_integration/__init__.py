"""
XENON Integration Layer for C-ACT (OPTIONAL — not used by default).

Adapters to interface C-ACT with non-XENON agent bases.
Current CactMemory directly wraps XENON DecomposedMemory.
Use these adapters when porting C-ACT to a different agent.
"""

from .xenon_adapter import XenonAdapter
from .adg_parser import ADGParser
from .fam_parser import FAMParser
from .executor_wrapper import ExecutorWrapper
