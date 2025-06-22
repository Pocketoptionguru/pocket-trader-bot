import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

class HighFrequencyStrategies:
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if np.any(gains[-period:]) else 0
        avg_loss = np.mean(losses[-period:]) if np.any(losses[-period:]) else 0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return np.mean(prices[-period:])

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
    def aggressive_momentum_scalper(candles: List[Candle]) -> Optional[str]:
        """
        High-frequency momentum strategy - trades on strong directional moves
        Entry: When current candle shows strong momentum vs previous 3 candles
        """
        if len(candles) < 5:
            return None
        
        c0 = candles[-1]  # Current candle
        c1 = candles[-2]  # Previous candle
        
        # Calculate momentum indicators
        body_size = abs(c0.close - c0.open)
        candle_range = c0.high - c0.low
        prev_bodies = [abs(c.close - c.open) for c in candles[-4:-1]]
        avg_prev_body = np.mean(prev_bodies)
        
        # Strong bullish momentum
        if (c0.close > c0.open and  # Bullish candle
            body_size > avg_prev_body * 1.2 and  # Larger body than recent average
            c0.close > c1.high and  # Breaking previous high
            body_size > candle_range * 0.6):  # Strong body vs wick ratio
            return "call"
        
        # Strong bearish momentum
        if (c0.close < c0.open and  # Bearish candle
            body_size > avg_prev_body * 1.2 and  # Larger body than recent average
            c0.close < c1.low and  # Breaking previous low
            body_size > candle_range * 0.6):  # Strong body vs wick ratio
            return "put"
        
        return None

    @staticmethod
    def rapid_rsi_extremes(candles: List[Candle]) -> Optional[str]:
        """
        Fast RSI strategy with lower thresholds for more frequent trades
        Entry: RSI extremes with confirmation from price action
        """
        if len(candles) < 8:
            return None
        
        prices = [c.close for c in candles]
        rsi = HighFrequencyStrategies.calculate_rsi(prices, 7)  # Faster RSI
        c0 = candles[-1]
        c1 = candles[-2]
        
        # Price trend confirmation
        recent_prices = prices[-3:]
        is_uptrend = all(recent_prices[i] <= recent_prices[i+1] for i in range(len(recent_prices)-1))
        is_downtrend = all(recent_prices[i] >= recent_prices[i+1] for i in range(len(recent_prices)-1))
        
        # Oversold with bullish reversal signs
        if (rsi < 25 and  # Lower RSI threshold
            c0.close > c0.open and  # Current candle bullish
            c0.low < c1.low and  # Made lower low
            c0.close > c1.close):  # But closed higher
            return "call"
        
        # Overbought with bearish reversal signs
        if (rsi > 75 and  # Lower RSI threshold
            c0.close < c0.open and  # Current candle bearish
            c0.high > c1.high and  # Made higher high
            c0.close < c1.close):  # But closed lower
            return "put"
        
        return None

    @staticmethod
    def dual_ema_crossover_aggressive(candles: List[Candle]) -> Optional[str]:
        """
        Fast EMA crossover with immediate entry on cross confirmation
        Entry: 5 EMA crossing 13 EMA with momentum confirmation
        """
        if len(candles) < 15:
            return None
        
        closes = [c.close for c in candles]
        ema5 = HighFrequencyStrategies.calculate_ema(closes, 5)
        ema13 = HighFrequencyStrategies.calculate_ema(closes, 13)
        
        # Previous EMA values
        prev_closes = closes[:-1]
        prev_ema5 = HighFrequencyStrategies.calculate_ema(prev_closes, 5)
        prev_ema13 = HighFrequencyStrategies.calculate_ema(prev_closes, 13)
        
        c0 = candles[-1]
        
        # Bullish crossover with momentum
        if (prev_ema5 <= prev_ema13 and  # Was below or equal
            ema5 > ema13 and  # Now crossed above
            c0.close > c0.open and  # Current candle bullish
            c0.close > ema5):  # Price above faster EMA
            return "call"
        
        # Bearish crossover with momentum
        if (prev_ema5 >= prev_ema13 and  # Was above or equal
            ema5 < ema13 and  # Now crossed below
            c0.close < c0.open and  # Current candle bearish
            c0.close < ema5):  # Price below faster EMA
            return "put"
        
        return None

    @staticmethod
    def volume_price_breakout(candles: List[Candle]) -> Optional[str]:
        """
        Volume-confirmed price breakouts for high probability entries
        Entry: Price breaking key levels with above-average volume
        """
        if len(candles) < 8:
            return None
        
        c0 = candles[-1]
        
        # Calculate volume average (use mock volume if not available)
        volumes = [max(c.volume, 1.0) for c in candles[-7:-1]]
        avg_volume = np.mean(volumes)
        current_volume = max(c0.volume, 1.0)
        
        # Calculate recent highs and lows
        recent_highs = [c.high for c in candles[-5:-1]]
        recent_lows = [c.low for c in candles[-5:-1]]
        resistance = max(recent_highs)
        support = min(recent_lows)
        
        # Bullish breakout
        if (c0.close > resistance and  # Breaking resistance
            c0.close > c0.open and  # Bullish candle
            current_volume > avg_volume * 1.1 and  # Above average volume
            c0.high == max([c.high for c in candles[-5:]])):  # New recent high
            return "call"
        
        # Bearish breakdown
        if (c0.close < support and  # Breaking support
            c0.close < c0.open and  # Bearish candle
            current_volume > avg_volume * 1.1 and  # Above average volume
            c0.low == min([c.low for c in candles[-5:]])):  # New recent low
            return "put"
        
        return None

    @staticmethod
    def triple_confirmation_scalper(candles: List[Candle]) -> Optional[str]:
        """
        Three-factor confirmation system for high-confidence entries
        Entry: RSI + EMA + Price Action all align
        """
        if len(candles) < 10:
            return None
        
        closes = [c.close for c in candles]
        rsi = HighFrequencyStrategies.calculate_rsi(closes, 9)
        ema8 = HighFrequencyStrategies.calculate_ema(closes, 8)
        
        c0 = candles[-1]
        c1 = candles[-2]
        
        # Calculate price momentum
        price_momentum = (c0.close - candles[-3].close) / candles[-3].close * 100
        
        # Bullish confluence
        bullish_rsi = 20 < rsi < 60  # Not overbought, room to move up
        bullish_ema = c0.close > ema8  # Price above EMA
        bullish_momentum = price_momentum > -0.5  # Not falling too fast
        bullish_candle = c0.close > c0.open  # Current candle bullish
        
        if (bullish_rsi and bullish_ema and bullish_momentum and bullish_candle and
            c0.close > c1.high):  # Breaking previous high
            return "call"
        
        # Bearish confluence
        bearish_rsi = 40 < rsi < 80  # Not oversold, room to move down
        bearish_ema = c0.close < ema8  # Price below EMA
        bearish_momentum = price_momentum < 0.5  # Not rising too fast
        bearish_candle = c0.close < c0.open  # Current candle bearish
        
        if (bearish_rsi and bearish_ema and bearish_momentum and bearish_candle and
            c0.close < c1.low):  # Breaking previous low
            return "put"
        
        return None

# Updated strategy mapping for your bot
STRATEGY_MAP = {
    "Aggressive Momentum Scalper": HighFrequencyStrategies.aggressive_momentum_scalper,
    "Rapid RSI Extremes": HighFrequencyStrategies.rapid_rsi_extremes,
    "Dual EMA Crossover Aggressive": HighFrequencyStrategies.dual_ema_crossover_aggressive,
    "Volume Price Breakout": HighFrequencyStrategies.volume_price_breakout,
    "Triple Confirmation Scalper": HighFrequencyStrategies.triple_confirmation_scalper,
}