"""
XENON Integration Layer for C-ACT.

Adapters to interface C-ACT's decision-time admission layer
with XENON's knowledge structures:
  - Adaptive Dependency Graph (ADG): dependency corrections
  - Failure-aware Action Memory (FAM): action corrections / remedies
"""

from .xenon_adapter import XenonAdapter
from .adg_parser import ADGParser
from .fam_parser import FAMParser
from .executor_wrapper import ExecutorWrapper
