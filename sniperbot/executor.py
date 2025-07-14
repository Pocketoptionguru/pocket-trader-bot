import logging
from collections import deque
from typing import Dict, List, Optional

from utils import Candle
from strategy import EnhancedStrategyManager
from filters import VolatilityFilter, AdaptiveConfidence, SmartPause
from logger import LiveLogger
from mlfilter import MLSignalFilter
from config import BALANCED_1MIN_CONFIG

__all__ = [
    'EnhancedMiniSwingExecutor',
    'vol_filter', 'confidence_filter', 'smart_pause', 'live_logger', 'ml_filter',
]


# -----------------------------------------------------------
#               GLOBAL SINGLETON MODULE INSTANCES
# -----------------------------------------------------------
vol_filter = VolatilityFilter()
confidence_filter = AdaptiveConfidence()
smart_pause = SmartPause()
live_logger = LiveLogger()
ml_filter = MLSignalFilter()  # no model by default


class EnhancedMiniSwingExecutor:
    """High-level wrapper that feeds candles into the strategy manager, keeps
    short-term history and exposes performance/statistics."""

    def __init__(self, config: Dict = None, debug: bool = False):
        self.strategy_manager = EnhancedStrategyManager(config or BALANCED_1MIN_CONFIG, debug)
        self.signal_history = deque(maxlen=200)
        self.stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
        }

    # ------------------------------------------------------------------
    def process_candles(self, candles: List[Candle]) -> Optional[Dict]:
        if not candles:
            return None
        # feed candle to volatility filter
        vol_filter.update({
            'open': candles[-1].open,
            'high': candles[-1].high,
            'low': candles[-1].low,
            'close': candles[-1].close,
            'timestamp': candles[-1].timestamp,
        })
        signal = self.strategy_manager.process_new_candle(candles)
        if signal:
            self.signal_history.append(signal)
            self.stats['total_signals'] += 1
            logging.info(f"EnhancedMiniSwingExecutor: {signal['signal'].upper()} @ {signal['confidence']:.1%} – {signal['reason']}")
        return signal

    # ------------------------------------------------------------------
    def update_trade_result(self, signal_dir: str, profit: float, success: bool):
        self.strategy_manager.update_trade_result(signal_dir, profit, success)
        if success:
            self.stats['successful_signals'] += 1
        else:
            self.stats['failed_signals'] += 1

    def get_performance_stats(self):
        return self.stats.copy()

    def get_strategy_status(self):
        return self.strategy_manager.get_status()


# Convenience function mirroring sniper.py original helper

def run_enhanced_mini_swing_strategy(candles: List[Candle]):
    executor = EnhancedMiniSwingExecutor()
    result = executor.process_candles(candles)
    return result['signal'] if result else None