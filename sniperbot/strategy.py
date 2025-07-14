import time
import math
import logging
import statistics
from typing import List, Dict, Optional

from utils import Candle, TradeResult


class EnhancedMiniSwingStrategy:
    """
    Enhanced Mini Swing Strategy for 1-minute expiry trades.
    Focuses on early trend continuations and trend alignment filtering.
    """

    def __init__(self, config: Dict = None, debug: bool = False):
        # --- DEFAULT CONFIGURATION ---
        self.config = {
            'momentum_candles': 3,
            'body_strength_threshold': 0.6,
            'trend_candles': 6,
            'trend_threshold': 0.7,
            'breakout_periods': 8,
            'consolidation_threshold': 0.3,
            'cooldown_seconds': 20,
            'min_confidence': 0.7,
            'max_trades_per_session': 80,
            'max_consecutive_losses': 3,
            'max_drawdown_percent': 20,
            'candle_timing_tolerance': 0.5,
            'trend_alignment_candles': 15,
            'swing_continuation_threshold': 0.65,
            'spike_rejection_ratio': 2.0,
            'trend_strength_weight': 0.3,
            'pullback_tolerance': 0.4,
            'min_swing_candles': 2,
        }

        if config:
            self.config.update(config)

        self.debug = debug

        # --- STATE ---
        self.last_signal_time: float = 0
        self.trade_history: List[TradeResult] = []
        self.consecutive_losses: int = 0
        self.peak_balance: float = 0
        self.current_balance: float = 0
        self.session_trades: int = 0
        self.is_stopped: bool = False
        self.total_signals: int = 0
        self.successful_trades: int = 0

    # ---------------------------------------------------------------------
    #                       CANDLE-LEVEL HELPERS
    # ---------------------------------------------------------------------
    def calculate_candle_strength(self, candle: Candle) -> float:
        if candle.high == candle.low:
            return 0.0
        body = abs(candle.close - candle.open)
        return body / (candle.high - candle.low)

    def is_bullish_candle(self, candle: Candle) -> bool:
        return candle.close > candle.open

    def is_bearish_candle(self, candle: Candle) -> bool:
        return candle.close < candle.open

    def detect_spike(self, candles: List[Candle]) -> bool:
        if len(candles) < 2:
            return False
        body_curr = abs(candles[-1].close - candles[-1].open)
        body_prev = abs(candles[-2].close - candles[-2].open)
        if body_prev > 0 and body_curr / body_prev > self.config['spike_rejection_ratio']:
            if self.debug:
                print("Spike detected – ignoring last candle")
            return True
        return False

    # ---------------------------------------------------------------------
    #                            TREND UTILS
    # ---------------------------------------------------------------------
    def get_slow_trend(self, candles: List[Candle]) -> str:
        if len(candles) < self.config['trend_alignment_candles']:
            return 'neutral'
        subset = candles[-self.config['trend_alignment_candles']:]
        prices = [c.close for c in subset]
        first_avg = sum(prices[:len(prices)//3]) / (len(prices)//3)
        last_avg = sum(prices[-len(prices)//3:]) / (len(prices)//3)
        price_change = (last_avg - first_avg) / first_avg
        bull_count = sum(1 for c in subset if self.is_bullish_candle(c))
        bear_count = sum(1 for c in subset if self.is_bearish_candle(c))
        bull_ratio = bull_count / len(subset)
        bear_ratio = bear_count / len(subset)

        if price_change > 0.001 and bull_ratio >= (0.5 - self.config['pullback_tolerance']):
            return 'bullish'
        if price_change < -0.001 and bear_ratio >= (0.5 - self.config['pullback_tolerance']):
            return 'bearish'
        return 'neutral'

    # ---------------------------------------------------------------------
    #                       MOMENTUM / TREND / BREAKOUT
    # ---------------------------------------------------------------------
    def detect_mini_swing_momentum(self, candles: List[Candle]):
        if len(candles) < self.config['momentum_candles'] + 1:
            return {'detected': False}
        if self.detect_spike(candles):
            return {'detected': False}
        recent = candles[-self.config['momentum_candles']:]
        bull_score = bear_score = strength_sum = 0.0
        consec_bull = consec_bear = max_bull = max_bear = 0
        for idx, c in enumerate(recent):
            strength = self.calculate_candle_strength(c)
            strength_sum += strength
            weight = (idx + 1) / len(recent)
            if strength >= self.config['body_strength_threshold']:
                if self.is_bullish_candle(c):
                    bull_score += weight
                    consec_bull += 1; consec_bear = 0
                    max_bull = max(max_bull, consec_bull)
                elif self.is_bearish_candle(c):
                    bear_score += weight
                    consec_bear += 1; consec_bull = 0
                    max_bear = max(max_bear, consec_bear)
        avg_strength = strength_sum / len(recent)
        if bull_score >= 2 and max_bull >= self.config['min_swing_candles']:
            return {'detected': True, 'direction': 'bullish', 'strength': avg_strength, 'count': bull_score, 'consecutive': max_bull}
        if bear_score >= 2 and max_bear >= self.config['min_swing_candles']:
            return {'detected': True, 'direction': 'bearish', 'strength': avg_strength, 'count': bear_score, 'consecutive': max_bear}
        return {'detected': False, 'strength': avg_strength}

    def detect_trend_continuation(self, candles: List[Candle]):
        if len(candles) < self.config['trend_candles']:
            return {'detected': False}
        recent = candles[-self.config['trend_candles']:]
        bulls = sum(1 for c in recent if self.is_bullish_candle(c))
        bears = sum(1 for c in recent if self.is_bearish_candle(c))
        req = int(len(recent) * self.config['trend_threshold'])
        if bulls >= req:
            last_two = recent[-2:]
            sustained = all(self.is_bullish_candle(c) for c in last_two)
            progression = self.calculate_candle_strength(last_two[-1]) >= self.calculate_candle_strength(last_two[-2]) * 0.8
            return {'detected': True, 'direction': 'bullish', 'confidence': bulls/len(recent), 'sustained': sustained, 'strength_progression': progression}
        if bears >= req:
            last_two = recent[-2:]
            sustained = all(self.is_bearish_candle(c) for c in last_two)
            progression = self.calculate_candle_strength(last_two[-1]) >= self.calculate_candle_strength(last_two[-2]) * 0.8
            return {'detected': True, 'direction': 'bearish', 'confidence': bears/len(recent), 'sustained': sustained, 'strength_progression': progression}
        return {'detected': False}

    def detect_breakout(self, candles: List[Candle]):
        if len(candles) < self.config['breakout_periods'] + 1:
            return {'detected': False}
        past = candles[-(self.config['breakout_periods'] + 1):-1]
        current = candles[-1]
        high = max(c.high for c in past)
        low = min(c.low for c in past)
        rng = high - low
        if rng == 0:
            return {'detected': False}
        consolidation_pct = rng / ((high + low)/2)
        if consolidation_pct > self.config['consolidation_threshold']:
            return {'detected': False}
        cur_strength = self.calculate_candle_strength(current)
        if current.close > high and self.is_bullish_candle(current) and cur_strength >= self.config['body_strength_threshold']:
            return {'detected': True, 'direction': 'bullish', 'confidence': cur_strength, 'breakout_strength': (current.close - high)/rng}
        if current.close < low and self.is_bearish_candle(current) and cur_strength >= self.config['body_strength_threshold']:
            return {'detected': True, 'direction': 'bearish', 'confidence': cur_strength, 'breakout_strength': (low - current.close)/rng}
        return {'detected': False}

    # ---------------------------------------------------------------------
    #                           PUBLIC INTERFACE
    # ---------------------------------------------------------------------
    def generate_signal(self, candle_data: List[Candle]):
        if self.is_stopped or (time.time() - self.last_signal_time) < self.config['cooldown_seconds']:
            return None
        min_req = max(self.config['momentum_candles'], self.config['trend_candles'], self.config['breakout_periods'], self.config['trend_alignment_candles'])
        if len(candle_data) < min_req:
            return None

        reasons, call_conf, put_conf = [], 0.0, 0.0
        slow_trend = self.get_slow_trend(candle_data)

        # 1. Momentum
        momentum = self.detect_mini_swing_momentum(candle_data)
        if momentum.get('detected'):
            weight = momentum['strength'] * 0.35
            if momentum.get('consecutive', 0) >= 2:
                weight *= 1.15
            if momentum['direction'] == 'bullish':
                call_conf += weight
            else:
                put_conf += weight
            reasons.append("mini swing momentum")

        # 2. Trend continuation
        trend = self.detect_trend_continuation(candle_data)
        if trend.get('detected'):
            weight = trend['confidence'] * 0.3
            if trend.get('sustained'):  # slight boost
                weight *= 1.1
            if trend.get('strength_progression'):
                weight *= 1.05
            if trend['direction'] == 'bullish':
                call_conf += weight
            else:
                put_conf += weight
            reasons.append("trend continuation")

        # 3. Breakout
        breakout = self.detect_breakout(candle_data)
        if breakout.get('detected'):
            weight = breakout['confidence'] * 0.2
            if breakout['direction'] == 'bullish':
                call_conf += weight
            else:
                put_conf += weight
            reasons.append("breakout pattern")

        # Trend alignment bonus & final decision
        preliminary_dir = None
        base_conf = 0
        if call_conf > put_conf:
            preliminary_dir = 'call'; base_conf = call_conf
        elif put_conf > call_conf:
            preliminary_dir = 'put'; base_conf = put_conf

        trend_bonus = 0.0
        if preliminary_dir:
            if (preliminary_dir == 'call' and slow_trend == 'bullish') or (preliminary_dir == 'put' and slow_trend == 'bearish'):
                trend_bonus = self.config['trend_strength_weight']
                reasons.append(f"trend aligned ({slow_trend})")
            elif slow_trend != 'neutral':
                return None  # mis-aligned trend, ignore signal

        final_conf = min(base_conf + trend_bonus, 1.0)
        if preliminary_dir and final_conf >= self.config['min_confidence'] and len(reasons) >= 2:
            self.last_signal_time = time.time()
            self.total_signals += 1
            return {
                'signal': preliminary_dir,
                'confidence': final_conf,
                'timestamp': time.time(),
                'reason': ' + '.join(reasons),
                'slow_trend': slow_trend,
                'call_conf': call_conf,
                'put_conf': put_conf,
                'trend_aligned': trend_bonus > 0,
            }
        return None

    # --------------------  TRADE RESULT / STATISTICS ---------------------
    def update_trade_result(self, signal: str, profit: float, success: bool):
        self.trade_history.append(TradeResult(time.time(), signal, profit, success))
        self.current_balance += profit
        self.peak_balance = max(self.peak_balance, self.current_balance)
        if success:
            self.consecutive_losses = 0
            self.successful_trades += 1
        else:
            self.consecutive_losses += 1

    def get_statistics(self):
        win_rate = (self.successful_trades / self.total_signals * 100) if self.total_signals else 0
        return {
            'total_signals': self.total_signals,
            'successful_trades': self.successful_trades,
            'win_rate': win_rate,
            'consecutive_losses': self.consecutive_losses,
            'session_trades': self.session_trades,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'is_stopped': self.is_stopped,
        }

    # ------------------------------ MISC -------------------------------
    def reset_session(self):
        self.session_trades = 0
        self.consecutive_losses = 0
        self.is_stopped = False
        self.last_signal_time = 0

    def force_stop(self):
        self.is_stopped = True

    def resume(self):
        self.is_stopped = False
        self.consecutive_losses = 0


class EnhancedStrategyManager:
    """Helper that throttles candle processing and delegates to the strategy"""

    def __init__(self, strategy_config: Dict = None, debug: bool = False):
        self.strategy = EnhancedMiniSwingStrategy(strategy_config, debug)
        self.last_processed_candle = 0

    def process_new_candle(self, candle_data: List[Candle]):
        if not candle_data:
            return None
        latest_ts = candle_data[-1].timestamp
        tolerance = self.strategy.config.get('candle_timing_tolerance', 0.5)
        if latest_ts <= self.last_processed_candle + tolerance:
            return None
        self.last_processed_candle = latest_ts
        return self.strategy.generate_signal(candle_data)

    def update_trade_result(self, signal: str, profit: float, success: bool):
        self.strategy.update_trade_result(signal, profit, success)

    def get_status(self):
        stats = self.strategy.get_statistics()
        return {
            'active': not self.strategy.is_stopped,
            'cooldown_remaining': max(0, self.strategy.config['cooldown_seconds'] - (time.time() - self.strategy.last_signal_time)),
            'trades_today': stats['session_trades'],
            'win_rate': f"{stats['win_rate']:.1f}%",
            'consecutive_losses': stats['consecutive_losses'],
            'balance_change': stats['current_balance'],
        }


# Legacy aliases for backward compatibility
UltraFastStrategy = EnhancedMiniSwingStrategy
StrategyManager = EnhancedStrategyManager