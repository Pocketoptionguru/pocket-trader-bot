from typing import List, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class TradingStrategies:
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        a = np.convolve(prices, weights, mode='full')[:len(prices)]
        a[:period] = a[period]
        return a[-1]

    @staticmethod
    def momentum_breakout(candles: List[Candle]) -> Optional[str]:
        """
        Fast breakout on new highs/lows, RSI and volume confirmation.
        Returns: 'call', 'put', or None
        """
        if len(candles) < 7:
            return None
        last = candles[-1]
        prev_5_highs = [c.high for c in candles[-6:-1]]
        prev_5_lows = [c.low for c in candles[-6:-1]]
        prices = [c.close for c in candles]
        rsi = TradingStrategies.calculate_rsi(prices)
        volumes = [c.volume for c in candles]
        avg_vol = np.mean(volumes[-6:-1])
        if (last.close > max(prev_5_highs) and rsi > 55 and last.volume > avg_vol * 1.1 and last.close > last.open):
            return "call"
        if (last.close < min(prev_5_lows) and rsi < 45 and last.volume > avg_vol * 1.1 and last.close < last.open):
            return "put"
        return None

    @staticmethod
    def one_minute_reversal(candles: List[Candle]) -> Optional[str]:
        """
        Looks for a strong engulfing reversal after a run, 1m chart.
        Returns: 'call', 'put', or None
        """
        if len(candles) < 4:
            return None
        c2, c1, c0 = candles[-3], candles[-2], candles[-1]
        # Bullish Engulfing after a red candle streak
        if (c2.close < c2.open and c1.close < c1.open and c0.close > c0.open and
            c0.open < c1.close and c0.close > c1.open and
            c0.close - c0.open > (c1.open - c1.close) * 1.2):
            return "call"
        # Bearish Engulfing after a green candle streak
        if (c2.close > c2.open and c1.close > c1.open and c0.close < c0.open and
            c0.open > c1.close and c0.close < c1.open and
            c0.open - c0.close > (c1.close - c1.open) * 1.2):
            return "put"
        return None

    @staticmethod
    def rapid_ma_cross(candles: List[Candle]) -> Optional[str]:
        """
        Trades when short EMA crosses long EMA, checks for fast execution.
        Returns: 'call', 'put', or None
        """
        if len(candles) < 25:
            return None
        prices = [c.close for c in candles]
        ema_5_prev = TradingStrategies.calculate_ema(prices[:-1], 5)
        ema_21_prev = TradingStrategies.calculate_ema(prices[:-1], 21)
        ema_5_now = TradingStrategies.calculate_ema(prices, 5)
        ema_21_now = TradingStrategies.calculate_ema(prices, 21)
        # Fast cross up
        if ema_5_prev < ema_21_prev and ema_5_now > ema_21_now:
            return "call"
        # Fast cross down
        if ema_5_prev > ema_21_prev and ema_5_now < ema_21_now:
            return "put"
        return None

    @staticmethod
    def impulse_spike(candles: List[Candle]) -> Optional[str]:
        """
        Detects a big candle ("rocket" candle) after congestion for fast scalp.
        Returns: 'call', 'put', or None
        """
        if len(candles) < 8:
            return None
        last = candles[-1]
        prev_bodies = [abs(c.close - c.open) for c in candles[-7:-1]]
        avg_body = np.mean(prev_bodies)
        # Up rocket
        if (last.close > last.open and (last.close - last.open) > avg_body * 2.2):
            return "call"
        # Down rocket
        if (last.close < last.open and (last.open - last.close) > avg_body * 2.2):
            return "put"
        return None