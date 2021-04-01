"""
Error correction benchmarking module
"""

from .circuits.base import LatticeError, TopologicalLattice, TopologicalQubit
from .circuits.xxzz import XXZZQubit
from .fitters.xxzz import XXZZGraphDecoderBase as XXZZDecoder
from .fitters.xzzx import XZZXGraphDecoderBase as XZZXDecoder

# will add more later
