from enum import Enum
from typing import NamedTuple
from dataclasses import dataclass
from typing import List, Optional, Dict

class Signal(Enum):
    """Possible trading signal directions"""
    CALL = "call"
    PUT = "put"
    HOLD = "hold"


class Candle(NamedTuple):
    """Light-weight immutable OHLCV candle representation"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class TradeResult:
    """Utility container used by the strategy & executor to keep track of the
    outcome of a completed trade so that statistics, draw-down and safety
    layers can be updated consistently across the whole code-base."""
    timestamp: float
    signal: str
    profit: float
    success: bool