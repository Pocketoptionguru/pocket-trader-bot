# ==== ENHANCED PRECISION TRADING BOT WITH SNIPER-GRADE IMPROVEMENTS ====
# Enhanced version with improved strategy logic and lowered thresholds

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import random
import time
import math
import numpy as np
import logging
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== ENHANCED DATA STRUCTURES ====
class Signal(Enum):
    CALL = auto()
    PUT = auto()
    HOLD = auto()

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def body_to_wick_ratio(self) -> float:
        total_wick = self.upper_wick + self.lower_wick
        return self.body_size / total_wick if total_wick > 0 else float('inf')
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

@dataclass
class StrategyVote:
    strategy_name: str
    vote_strength: float  # 0.0 to 1.0
    signal: Signal
    conditions_met: List[str]
    indicator_values: Dict[str, float]

@dataclass 
class VoteAnalysis:
    all_votes: List[StrategyVote]
    call_votes: List[StrategyVote]
    put_votes: List[StrategyVote]
    hold_votes: List[StrategyVote]
    call_confidence: float
    put_confidence: float
    call_confirmations: int
    put_confirmations: int
    rejection_reasons: List[str]
    final_decision: Optional['TradeDecision']

@dataclass
class TradeDecision:
    signal: Signal
    confidence: float
    strategy_votes: List[StrategyVote]
    current_candle: Candle
    session_context: str
    risk_status: Dict[str, any]
    trend_strength: float
    volatility_burst: float

    @property
    def total_confirmations(self) -> int:
        return len([vote for vote in self.strategy_votes if vote.signal == self.signal])
    
    @property
    def contributing_strategies(self) -> List[str]:
        return [vote.strategy_name for vote in self.strategy_votes if vote.signal == self.signal]

# ==== ENHANCED STRATEGY BASE CLASS ====
class BaseStrategy:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        raise NotImplementedError

    def get_required_history_length(self) -> int:
        return self.config.get('required_history', 20)
    
    def has_sufficient_data(self, candles: List[Candle], current_index: int) -> bool:
        return current_index >= self.get_required_history_length()
    
    def create_vote(self, signal: Signal, strength: float, conditions: List[str], 
                indicators: Dict[str, float]) -> StrategyVote:
        return StrategyVote(
            strategy_name=self.name,
            vote_strength=max(0.0, min(1.0, strength)),
            signal=signal,
            conditions_met=conditions,
            indicator_values=indicators
        )

# ==== ENHANCED TREND FILTER STRATEGY ====
class TrendFilterStrategy(BaseStrategy):
    """
    Advanced Trend Filter Strategy with multiple trend confirmation methods
    """
    
    def __init__(self, config: Dict):
        super().__init__("TrendFilter", config)
        self.adx_period = config.get('adx_period', 14)
        self.adx_threshold = config.get('adx_threshold', 25)
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        self.ema_fast = config.get('ema_fast', 21)
        self.ema_slow = config.get('ema_slow', 55)
    
    def calculate_adx(self, candles: List[Candle], period: int, end_index: int) -> float:
        """Calculate ADX (Average Directional Index)"""
        if end_index < period + 1:
            return 0.0
        
        # Calculate True Range and Directional Movement
        tr_values = []
        plus_dm = []
        minus_dm = []
        
        for i in range(max(0, end_index - period), end_index):
            if i > 0:
                current = candles[i]
                previous = candles[i-1]
                
                # True Range
                tr1 = current.high - current.low
                tr2 = abs(current.high - previous.close)
                tr3 = abs(current.low - previous.close)
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)
                
                # Directional Movement
                plus_dm_val = current.high - previous.high if current.high > previous.high else 0
                minus_dm_val = previous.low - current.low if previous.low > current.low else 0
                
                if plus_dm_val > minus_dm_val:
                    plus_dm.append(plus_dm_val)
                    minus_dm.append(0)
                elif minus_dm_val > plus_dm_val:
                    plus_dm.append(0)
                    minus_dm.append(minus_dm_val)
                else:
                    plus_dm.append(0)
                    minus_dm.append(0)
        
        if not tr_values:
            return 0.0
        
        # Calculate smoothed averages
        atr = sum(tr_values) / len(tr_values)
        plus_di = (sum(plus_dm) / len(plus_dm)) / atr * 100 if atr > 0 else 0
        minus_di = (sum(minus_dm) / len(minus_dm)) / atr * 100 if atr > 0 else 0
        
        # Calculate ADX
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        return dx
    
    def calculate_macd_histogram(self, candles: List[Candle], end_index: int) -> Tuple[float, float, float]:
        """Calculate MACD line, signal line, and histogram"""
        if end_index < self.macd_slow:
            return 0.0, 0.0, 0.0
        
        closes = [c.close for c in candles[max(0, end_index - 50):end_index + 1]]
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(closes, self.macd_fast)
        ema_slow = self.calculate_ema(closes, self.macd_slow)
        
        macd_line = ema_fast - ema_slow
        
        # Simple signal line (in real implementation, use EMA of MACD)
        signal_line = macd_line * 0.9
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        if not self.has_sufficient_data(candles, current_index):
            return None
        
        current_price = candles[current_index].close
        conditions = []
        indicators = {}
        vote_strength = 0.0
        signal = Signal.HOLD
        
        # ADX Trend Strength
        adx = self.calculate_adx(candles, self.adx_period, current_index)
        indicators['adx'] = adx
        
        if adx >= self.adx_threshold:
            conditions.append(f"Strong trend detected (ADX: {adx:.1f})")
            vote_strength += 0.3
        
        # MACD Histogram Analysis
        macd_line, signal_line, histogram = self.calculate_macd_histogram(candles, current_index)
        indicators['macd_histogram'] = histogram
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        
        if histogram > 0 and macd_line > signal_line:
            conditions.append(f"MACD bullish momentum (histogram: {histogram:.4f})")
            vote_strength += 0.25
            signal = Signal.CALL
        elif histogram < 0 and macd_line < signal_line:
            conditions.append(f"MACD bearish momentum (histogram: {histogram:.4f})")
            vote_strength += 0.25
            signal = Signal.PUT
        
        # Directional EMA Analysis
        closes = [c.close for c in candles[max(0, current_index - 100):current_index + 1]]
        ema_fast_val = self.calculate_ema(closes, self.ema_fast)
        ema_slow_val = self.calculate_ema(closes, self.ema_slow)
        
        indicators['ema_fast'] = ema_fast_val
        indicators['ema_slow'] = ema_slow_val
        
        if current_price > ema_fast_val > ema_slow_val:
            conditions.append(f"Bullish EMA alignment (Price: {current_price:.4f} > Fast: {ema_fast_val:.4f} > Slow: {ema_slow_val:.4f})")
            vote_strength += 0.2
            if signal != Signal.PUT:
                signal = Signal.CALL
        elif current_price < ema_fast_val < ema_slow_val:
            conditions.append(f"Bearish EMA alignment (Price: {current_price:.4f} < Fast: {ema_fast_val:.4f} < Slow: {ema_slow_val:.4f})")
            vote_strength += 0.2
            if signal != Signal.CALL:
                signal = Signal.PUT
        
        # Trend momentum confirmation
        if current_index >= 3:
            prev_price = candles[current_index - 3].close
            momentum = (current_price - prev_price) / prev_price
            indicators['trend_momentum'] = momentum
            
            if abs(momentum) >= 0.001:  # 0.1% momentum threshold
                if momentum > 0:
                    conditions.append(f"Positive momentum: {momentum:.3%}")
                    vote_strength += 0.15
                    if signal == Signal.HOLD:
                        signal = Signal.CALL
                else:
                    conditions.append(f"Negative momentum: {momentum:.3%}")
                    vote_strength += 0.15
                    if signal == Signal.HOLD:
                        signal = Signal.PUT
        
        if conditions and vote_strength >= 0.2:
            return self.create_vote(signal, vote_strength, conditions, indicators)
        
        return None

# ==== ENHANCED RSI STRATEGY WITH DYNAMIC ADAPTATION ====
class EnhancedRSIStrategy(BaseStrategy):
    """
    Enhanced RSI Strategy with dynamic volatility adaptation
    """
    
    def __init__(self, config: Dict):
        super().__init__("EnhancedRSI", config)
        self.period = config.get('period', 14)
        self.base_oversold = config.get('base_oversold', 30)
        self.base_overbought = config.get('base_overbought', 70)
        self.volatility_period = config.get('volatility_period', 20)
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
    
    def calculate_volatility(self, candles: List[Candle], period: int, end_index: int) -> float:
        """Calculate price volatility (standard deviation of returns)"""
        if end_index < period:
            return 0.01
        
        returns = []
        for i in range(max(0, end_index - period + 1), end_index + 1):
            if i > 0:
                ret = (candles[i].close - candles[i-1].close) / candles[i-1].close
                returns.append(ret)
        
        if not returns:
            return 0.01
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)
    
    def get_dynamic_levels(self, volatility: float) -> Tuple[float, float]:
        """Get dynamically adjusted RSI levels based on volatility"""
        if volatility > self.volatility_threshold:
            # High volatility: use more extreme levels
            oversold = self.base_oversold - 10  # 20 instead of 30
            overbought = self.base_overbought + 10  # 80 instead of 70
        else:
            # Low volatility: use standard levels
            oversold = self.base_oversold
            overbought = self.base_overbought
        
        return oversold, overbought
    
    def calculate_rsi(self, candles: List[Candle], period: int, end_index: int) -> float:
        """Calculate RSI with improved smoothing"""
        if end_index < period:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(max(0, end_index - period + 1), end_index + 1):
            if i > 0:
                change = candles[i].close - candles[i-1].close
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
        
        if not gains or not losses:
            return 50.0
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        if not self.has_sufficient_data(candles, current_index):
            return None
        
        current_rsi = self.calculate_rsi(candles, self.period, current_index)
        volatility = self.calculate_volatility(candles, self.volatility_period, current_index)
        oversold, overbought = self.get_dynamic_levels(volatility)
        
        conditions = []
        indicators = {
            'rsi': current_rsi,
            'volatility': volatility,
            'dynamic_oversold': oversold,
            'dynamic_overbought': overbought
        }
        vote_strength = 0.0
        signal = Signal.HOLD
        
        # Dynamic RSI analysis
        if current_rsi <= oversold:
            conditions.append(f"Dynamic oversold RSI: {current_rsi:.1f} (threshold: {oversold:.1f}, volatility: {volatility:.3f})")
            vote_strength += 0.4
            signal = Signal.CALL
        elif current_rsi >= overbought:
            conditions.append(f"Dynamic overbought RSI: {current_rsi:.1f} (threshold: {overbought:.1f}, volatility: {volatility:.3f})")
            vote_strength += 0.4
            signal = Signal.PUT
        
        # RSI divergence detection
        if current_index >= 5:
            prev_rsi = self.calculate_rsi(candles, self.period, current_index - 5)
            price_change = (candles[current_index].close - candles[current_index - 5].close) / candles[current_index - 5].close
            rsi_change = current_rsi - prev_rsi
            
            indicators['rsi_divergence'] = rsi_change
            
            # Bullish divergence: price down, RSI up
            if price_change < -0.001 and rsi_change > 5:
                conditions.append(f"Bullish RSI divergence: Price down {price_change:.2%}, RSI up {rsi_change:.1f}")
                vote_strength += 0.3
                signal = Signal.CALL
            # Bearish divergence: price up, RSI down
            elif price_change > 0.001 and rsi_change < -5:
                conditions.append(f"Bearish RSI divergence: Price up {price_change:.2%}, RSI down {rsi_change:.1f}")
                vote_strength += 0.3
                signal = Signal.PUT
        
        # RSI momentum analysis
        if current_index >= 2:
            prev_rsi = self.calculate_rsi(candles, self.period, current_index - 2)
            rsi_momentum = current_rsi - prev_rsi
            indicators['rsi_momentum'] = rsi_momentum
            
            if abs(rsi_momentum) >= 5:
                if rsi_momentum > 0 and current_rsi < overbought:
                    conditions.append(f"Strong RSI momentum up: +{rsi_momentum:.1f}")
                    vote_strength += 0.2
                    if signal == Signal.HOLD:
                        signal = Signal.CALL
                elif rsi_momentum < 0 and current_rsi > oversold:
                    conditions.append(f"Strong RSI momentum down: {rsi_momentum:.1f}")
                    vote_strength += 0.2
                    if signal == Signal.HOLD:
                        signal = Signal.PUT
        
        if conditions and vote_strength >= 0.2:
            return self.create_vote(signal, vote_strength, conditions, indicators)
        
        return None

# ==== ENHANCED BOLLINGER BANDS WITH SQUEEZE DETECTION ====
class EnhancedBollingerBandsStrategy(BaseStrategy):
    """
    Enhanced Bollinger Bands with squeeze breakout patterns and fake reversal filtering
    """
    
    def __init__(self, config: Dict):
        super().__init__("EnhancedBollingerBands", config)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2.0)
        self.squeeze_threshold = config.get('squeeze_threshold', 0.05)
        self.volume_confirmation = config.get('volume_confirmation', True)
    
    def calculate_bollinger_bands(self, candles: List[Candle], period: int, std_dev: float, 
                                end_index: int) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands with squeeze detection"""
        if end_index < period - 1:
            price = candles[end_index].close
            return price, price, price
        
        # Calculate SMA (middle band)
        start_index = max(0, end_index - period + 1)
        prices = [candles[i].close for i in range(start_index, end_index + 1)]
        sma = sum(prices) / len(prices)
        
        # Calculate standard deviation
        variance = sum((price - sma) ** 2 for price in prices) / len(prices)
        std = math.sqrt(variance)
        
        # Calculate bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return sma, upper_band, lower_band
    
    def detect_squeeze(self, candles: List[Candle], end_index: int) -> bool:
        """Detect Bollinger Band squeeze condition"""
        if end_index < 20:
            return False
        
        current_middle, current_upper, current_lower = self.calculate_bollinger_bands(
            candles, self.period, self.std_dev, end_index
        )
        
        # Compare with previous band width
        prev_middle, prev_upper, prev_lower = self.calculate_bollinger_bands(
            candles, self.period, self.std_dev, end_index - 10
        )
        
        current_width = (current_upper - current_lower) / current_middle
        prev_width = (prev_upper - prev_lower) / prev_middle
        
        # Squeeze detected if current width is significantly smaller
        return current_width < prev_width * 0.8 and current_width < self.squeeze_threshold
    
    def get_volume_confirmation(self, candles: List[Candle], end_index: int) -> bool:
        """Check for volume confirmation"""
        if end_index < 10:
            return False
        
        current_volume = candles[end_index].volume
        avg_volume = sum(c.volume for c in candles[max(0, end_index-9):end_index]) / 9
        
        return current_volume > avg_volume * 1.2
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        if not self.has_sufficient_data(candles, current_index):
            return None
        
        current_price = candles[current_index].close
        current = candles[current_index]
        
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.calculate_bollinger_bands(
            candles, self.period, self.std_dev, current_index
        )
        
        conditions = []
        indicators = {
            'price': current_price,
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }
        vote_strength = 0.0
        signal = Signal.HOLD
        
        # Detect squeeze breakout
        is_squeeze = self.detect_squeeze(candles, current_index)
        indicators['squeeze_detected'] = is_squeeze
        
        if is_squeeze:
            conditions.append("Bollinger Band squeeze detected")
            vote_strength += 0.2
            
            # Check for breakout direction
            if current_price > upper_band:
                conditions.append(f"Upward breakout from squeeze: {current_price:.4f} > {upper_band:.4f}")
                vote_strength += 0.3
                signal = Signal.CALL
            elif current_price < lower_band:
                conditions.append(f"Downward breakout from squeeze: {current_price:.4f} < {lower_band:.4f}")
                vote_strength += 0.3
                signal = Signal.PUT
        
        # Volume confirmation for breakouts
        if self.volume_confirmation and signal != Signal.HOLD:
            volume_confirmed = self.get_volume_confirmation(candles, current_index)
            indicators['volume_confirmed'] = volume_confirmed
            
            if volume_confirmed:
                conditions.append("Volume confirmation for breakout")
                vote_strength += 0.2
            else:
                conditions.append("WARNING: Breakout lacks volume confirmation")
                vote_strength *= 0.7  # Reduce strength for unconfirmed breakouts
        
        # Band position analysis
        band_range = upper_band - lower_band
        if band_range > 0:
            price_position = (current_price - lower_band) / band_range
            indicators['band_position'] = price_position
            
            # Extreme positions with mean reversion potential
            if price_position >= 0.95:
                conditions.append(f"Price at extreme upper band: {price_position:.1%}")
                vote_strength += 0.25
                if signal == Signal.HOLD:
                    signal = Signal.PUT
            elif price_position <= 0.05:
                conditions.append(f"Price at extreme lower band: {price_position:.1%}")
                vote_strength += 0.25
                if signal == Signal.HOLD:
                    signal = Signal.CALL
        
        # Fake reversal filtering
        if current_index >= 5:
            # Check for false breakouts in recent history
            recent_candles = candles[current_index-4:current_index+1]
            fake_breakout_detected = False
            
            for i, candle in enumerate(recent_candles[:-1]):
                if candle.high > upper_band and candle.close < upper_band:
                    fake_breakout_detected = True
                    break
                if candle.low < lower_band and candle.close > lower_band:
                    fake_breakout_detected = True
                    break
            
            if fake_breakout_detected:
                conditions.append("Recent fake breakout detected - increased caution")
                vote_strength *= 0.8  # Reduce strength due to recent fake breakout
        
        if conditions and vote_strength >= 0.2:
            return self.create_vote(signal, vote_strength, conditions, indicators)
        
        return None

# ==== ENHANCED CONFIGURATION WITH LOWERED THRESHOLDS ====
ENHANCED_CONFIG = {
    'signal_engine': {
        'min_confirmations': 1,  # Lowered from 2 to 1
        'min_confidence_threshold': 0.3,  # Lowered from 0.5 to 0.3
        'confirmation_weight_threshold': 0.3,  # Lowered from 0.4 to 0.3
        'session_boost_factor': 1.2  # Boost confidence during high-volume sessions
    },
    'risk_management': {
        'max_trades_per_day': 75,  # Increased from 50 to 75
        'max_consecutive_losses': 3,  # Increased from 2 to 3
        'min_win_rate_threshold': 0.55,  # Lowered from 0.6 to 0.55
        'cooldown_seconds': 30,  # Reduced from 60 to 30
        'min_trades_for_winrate': 8,  # Reduced from 10 to 8
        'adaptive_cooldown': True  # New: Adaptive cooldown based on market conditions
    },
    'strategies': {
        'trend_filter': {
            'enabled': True,
            'adx_period': 14,
            'adx_threshold': 20,  # Lowered from 25 to 20
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'ema_fast': 21,
            'ema_slow': 55,
            'required_history': 30
        },
        'enhanced_rsi': {
            'enabled': True,
            'period': 14,
            'base_oversold': 35,  # Raised from 30 to 35 for more signals
            'base_overbought': 65,  # Lowered from 70 to 65 for more signals
            'volatility_period': 20,
            'volatility_threshold': 0.015,  # Lowered threshold
            'required_history': 20
        },
        'enhanced_bollinger': {
            'enabled': True,
            'period': 20,
            'std_dev': 2.0,
            'squeeze_threshold': 0.08,  # Increased from 0.05 to catch more squeezes
            'volume_confirmation': True,
            'required_history': 25
        },
        'momentum': {
            'enabled': True,
            'min_body_ratio': 1.5,  # Lowered from 2.0 to 1.5
            'strong_close_threshold': 0.65,  # Lowered from 0.7 to 0.65
            'breakout_lookback': 5,
            'trend_lookback': 3,
            'required_history': 10
        },
        'volume': {
            'enabled': True,
            'volume_period': 20,
            'spike_multiplier': 1.5,  # Lowered from 2.0 to 1.5
            'high_volume_multiplier': 1.3,  # Lowered from 1.5 to 1.3
            'trend_confirmation_periods': 3,
            'required_history': 15
        }
    },
    'final_decision_filter': {
        'enabled': True,
        'trend_strength_weight': 0.4,
        'volatility_burst_weight': 0.3,
        'session_context_weight': 0.3,
        'min_combined_score': 0.6  # Minimum combined score for trade execution
    }
}

print("🎯 Enhanced Precision Trading Bot Configuration Loaded")
print("✅ Key Improvements:")
print("   - Lowered confirmation thresholds for better signal flow")
print("   - Enhanced RSI with dynamic volatility adaptation")
print("   - Bollinger Bands with squeeze breakout detection")
print("   - Advanced trend filtering with ADX + MACD + EMA")
print("   - Volume-volatility confluence analysis")
print("   - Final decision filter with session context")
print("   - Reduced cooldowns and risk restrictions")
print("=" * 60)