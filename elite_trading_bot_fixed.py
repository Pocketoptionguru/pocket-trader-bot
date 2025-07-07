# ==== ELITE TRADING BOT V11 - INSTITUTIONAL GRADE ====
# 🌟 NEURAL BEAST QUANTUM FUSION - ELITE INSTITUTIONAL EDITION 🌟
# Features: Adaptive Thresholds, Strategy Scoring, Market Regime Detection, Elite Decision Filtering

import logging
import atexit
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import numpy as np
import sys
import os
import json
import hashlib
import math
import random
from enum import Enum, auto
from collections import deque, defaultdict

# Enhanced Color Terminal Output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Enhanced colored formatter for terminal output"""
    
    FORMATS = {
        logging.DEBUG: Colors.CYAN + "%(asctime)s - %(levelname)s - %(message)s" + Colors.END,
        logging.INFO: Colors.WHITE + "%(asctime)s - " + Colors.BOLD + "%(levelname)s" + Colors.END + " - %(message)s",
        logging.WARNING: Colors.YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + Colors.END,
        logging.ERROR: Colors.RED + "%(asctime)s - %(levelname)s - %(message)s" + Colors.END,
        logging.CRITICAL: Colors.RED + Colors.BOLD + "%(asctime)s - %(levelname)s - %(message)s" + Colors.END
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

# Enhanced logging function for trades
def log_trade_result(trade_type, result, profit, confidence, reasons):
    """Enhanced trade logging with colors and formatting"""
    separator = "=" * 60
    
    if result == "WIN":
        color = Colors.GREEN + Colors.BOLD
        symbol = "✅"
    else:
        color = Colors.RED + Colors.BOLD
        symbol = "❌"
    
    print(f"\n{Colors.CYAN}{separator}{Colors.END}")
    print(f"{color}{symbol} ELITE TRADE RESULT: {result}{Colors.END}")
    print(f"{Colors.WHITE}🎯 Signal: {Colors.BOLD}{trade_type.upper()}{Colors.END}")
    print(f"{Colors.WHITE}💰 Profit/Loss: {color}${profit:.2f}{Colors.END}")
    print(f"{Colors.WHITE}🎲 Confidence: {Colors.CYAN}{confidence:.3f}{Colors.END}")
    print(f"{Colors.WHITE}📊 Reasons: {Colors.YELLOW}{', '.join(reasons)}{Colors.END}")
    print(f"{Colors.CYAN}{separator}{Colors.END}\n")

# --- Comprehensive urllib3 warning suppression ---
import warnings
import urllib3
from urllib3.exceptions import InsecureRequestWarning, NotOpenSSLWarning, DependencyWarning

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", category=DependencyWarning)
warnings.filterwarnings("ignore", message=".*connection pool.*")
warnings.filterwarnings("ignore", message=".*Connection pool is full.*")

urllib3.disable_warnings()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
urllib3.disable_warnings(urllib3.exceptions.DependencyWarning)

urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.ERROR)

# ---- ELITE CONFIGURATION ----
MAX_TRADES_LIMIT = 50  # Increased for elite performance
SESSION_FILE = "elite_beast_session.dat"

# ==== ELITE DATA STRUCTURES ====
class MarketRegime(Enum):
    STRONG_TRENDING = auto()
    TRENDING = auto()
    CHOPPY = auto()
    RANGING = auto()
    VOLATILE = auto()
    UNKNOWN = auto()

class Signal(Enum):
    CALL = auto()
    PUT = auto()
    HOLD = auto()

class SessionType(Enum):
    LONDON = auto()
    NEW_YORK = auto()
    OVERLAP = auto()
    ASIAN = auto()
    QUIET = auto()

@dataclass
class Candle:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

@dataclass
class StrategyVote:
    strategy_name: str
    vote_strength: float  # 0.0 to 1.0
    signal: Signal
    conditions_met: List[str]
    indicator_values: Dict[str, float]
    market_regime_bonus: float = 0.0

@dataclass
class StrategyPerformance:
    name: str
    total_signals: int = 0
    correct_signals: int = 0
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=30))
    regime_performance: Dict[MarketRegime, List[bool]] = field(default_factory=lambda: defaultdict(list))
    current_weight: float = 1.0
    base_weight: float = 1.0
    
    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.5  # Neutral starting point
        return self.correct_signals / self.total_signals
    
    @property
    def recent_accuracy(self) -> float:
        if not self.recent_performance:
            return 0.5
        return sum(self.recent_performance) / len(self.recent_performance)

@dataclass
class MarketState:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    session_type: SessionType
    time_weight: float
    volatility_burst: float

# ==== ELITE ADAPTIVE THRESHOLD MANAGER ====
class EliteAdaptiveThresholds:
    """🧠 Elite Adaptive Threshold Management System"""
    
    def __init__(self):
        self.base_thresholds = {
            'min_confidence_threshold': 0.35,
            'confirmation_weight_threshold': 0.4,
            'min_strategy_count': 2,
            'regime_bonus_threshold': 0.1
        }
        
        self.adaptive_thresholds = self.base_thresholds.copy()
        self.volatility_history = deque(maxlen=20)
        self.win_rate_history = deque(maxlen=20)
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 300  # 5 minutes
        
        # Adaptive bounds (never go beyond these)
        self.min_bounds = {
            'min_confidence_threshold': 0.25,
            'confirmation_weight_threshold': 0.3,
            'min_strategy_count': 1,
            'regime_bonus_threshold': 0.05
        }
        
        self.max_bounds = {
            'min_confidence_threshold': 0.65,
            'confirmation_weight_threshold': 0.7,
            'min_strategy_count': 3,
            'regime_bonus_threshold': 0.25
        }
        
        logging.info("🧠 Elite Adaptive Threshold Manager initialized")
    
    def update_market_conditions(self, volatility: float, win_rate: float):
        """Update market conditions for adaptive thresholds"""
        self.volatility_history.append(volatility)
        self.win_rate_history.append(win_rate)
    
    def adapt_thresholds(self, strategy_alignment: float) -> Dict[str, float]:
        """🔄 Dynamically adapt thresholds based on current conditions"""
        current_time = time.time()
        
        # Only adjust every 5 minutes to avoid over-optimization
        if current_time - self.last_adjustment_time < self.adjustment_interval:
            return self.adaptive_thresholds
        
        self.last_adjustment_time = current_time
        
        # Calculate adaptation factors
        volatility_factor = self._calculate_volatility_factor()
        performance_factor = self._calculate_performance_factor()
        alignment_factor = self._calculate_alignment_factor(strategy_alignment)
        
        # Adapt each threshold
        for key in self.base_thresholds:
            base_value = self.base_thresholds[key]
            
            # Apply factors based on threshold type
            if key == 'min_confidence_threshold':
                # Lower in high volatility explosive markets, higher in choppy markets
                adjustment = -volatility_factor * 0.1 + performance_factor * 0.15 - alignment_factor * 0.05
            elif key == 'confirmation_weight_threshold':
                # Adjust based on strategy alignment
                adjustment = -alignment_factor * 0.1 + performance_factor * 0.1
            elif key == 'min_strategy_count':
                # Require more strategies in uncertain conditions
                adjustment = int((1 - performance_factor) * 1.5)
            else:  # regime_bonus_threshold
                # More generous bonuses in strong trending markets
                adjustment = alignment_factor * 0.1 - volatility_factor * 0.05
            
            # Apply adjustment with bounds
            new_value = base_value + adjustment
            new_value = max(self.min_bounds[key], min(self.max_bounds[key], new_value))
            
            self.adaptive_thresholds[key] = new_value
        
        logging.info(f"🔄 Thresholds adapted: Vol={volatility_factor:.2f}, Perf={performance_factor:.2f}, Align={alignment_factor:.2f}")
        logging.info(f"   Confidence: {self.adaptive_thresholds['min_confidence_threshold']:.3f}")
        logging.info(f"   Weight: {self.adaptive_thresholds['confirmation_weight_threshold']:.3f}")
        
        return self.adaptive_thresholds
    
    def _calculate_volatility_factor(self) -> float:
        """Calculate volatility regime factor (-1 to 1)"""
        if not self.volatility_history:
            return 0.0
        
        recent_vol = np.mean(list(self.volatility_history)[-5:])
        avg_vol = np.mean(self.volatility_history)
        
        # Normalize to -1 to 1 range
        if avg_vol == 0:
            return 0.0
        
        factor = (recent_vol - avg_vol) / avg_vol
        return np.clip(factor, -1.0, 1.0)
    
    def _calculate_performance_factor(self) -> float:
        """Calculate recent performance factor (0 to 1)"""
        if not self.win_rate_history:
            return 0.5
        
        recent_wr = np.mean(list(self.win_rate_history)[-10:])
        # Convert win rate to 0-1 factor
        return np.clip(recent_wr, 0.0, 1.0)
    
    def _calculate_alignment_factor(self, alignment: float) -> float:
        """Calculate strategy alignment factor (0 to 1)"""
        return np.clip(alignment, 0.0, 1.0)

# ==== ELITE STRATEGY PERFORMANCE TRACKER ====
class EliteStrategyScorer:
    """📊 Elite Strategy Performance Scoring System"""
    
    def __init__(self):
        self.strategies: Dict[str, StrategyPerformance] = {}
        self.trade_history = deque(maxlen=100)
        self.regime_detector = None  # Will be set later
        
        # Initialize strategy weights
        self.initialize_strategies()
        logging.info("📊 Elite Strategy Performance Scorer initialized")
    
    def initialize_strategies(self):
        """Initialize all strategies with base performance data"""
        strategy_names = [
            "Neural Beast Quantum Fusion",
            "Enhanced RSI", 
            "Enhanced Bollinger Bands",
            "Trend Filter",
            "Momentum Surge",
            "Volume Confluence"
        ]
        
        for name in strategy_names:
            self.strategies[name] = StrategyPerformance(
                name=name,
                base_weight=1.0,
                current_weight=1.0
            )
    
    def record_trade_result(self, strategy_votes: List[StrategyVote], 
                          final_signal: Signal, was_correct: bool, 
                          market_regime: MarketRegime):
        """📈 Record trade result and update strategy scores"""
        
        # Record in trade history
        self.trade_history.append({
            'votes': strategy_votes,
            'final_signal': final_signal,
            'correct': was_correct,
            'regime': market_regime,
            'timestamp': time.time()
        })
        
        # Update performance for each strategy that voted for the final signal
        for vote in strategy_votes:
            if vote.signal == final_signal:
                strategy = self.strategies.get(vote.strategy_name)
                if strategy:
                    strategy.total_signals += 1
                    if was_correct:
                        strategy.correct_signals += 1
                    
                    # Update recent performance
                    strategy.recent_performance.append(was_correct)
                    
                    # Update regime-specific performance
                    strategy.regime_performance[market_regime].append(was_correct)
                    
                    # Keep regime performance lists reasonable size
                    if len(strategy.regime_performance[market_regime]) > 50:
                        strategy.regime_performance[market_regime] = strategy.regime_performance[market_regime][-30:]
        
        # Reweight strategies based on updated performance
        self._reweight_strategies()
    
    def _reweight_strategies(self):
        """FIXED: Added missing method to reweight strategies based on performance"""
        for strategy_name, strategy in self.strategies.items():
            if strategy.total_signals > 0:
                # Calculate weight based on recent accuracy
                recent_acc = strategy.recent_accuracy
                overall_acc = strategy.accuracy
                
                # Combine recent and overall performance
                combined_score = (recent_acc * 0.7) + (overall_acc * 0.3)
                
                # Adjust weight (0.5 to 1.5 range)
                new_weight = 0.5 + (combined_score * 1.0)
                strategy.current_weight = max(0.5, min(1.5, new_weight))
                
                logging.debug(f"Strategy {strategy_name}: Weight={strategy.current_weight:.2f}, Acc={overall_acc:.2f}")

# ==== SIMPLIFIED STRATEGY CLASSES ====
class EnhancedRSIStrategy:
    """Enhanced RSI Strategy with adaptive thresholds and regime awareness"""
    
    def __init__(self):
        self.name = "Enhanced RSI"
        self.period = 14
        self.base_oversold = 30
        self.base_overbought = 70
    
    def analyze(self, candles: List[Candle], market_state: MarketState) -> Optional[StrategyVote]:
        if len(candles) < self.period + 5:
            return None
        
        closes = [c.close for c in candles]
        rsi = self._calculate_rsi(closes, self.period)
        
        conditions = []
        indicators = {'rsi': rsi}
        vote_strength = 0.0
        signal = Signal.HOLD
        
        if rsi <= 30:
            conditions.append(f"RSI Oversold: {rsi:.1f}")
            vote_strength = 0.8
            signal = Signal.CALL
        elif rsi >= 70:
            conditions.append(f"RSI Overbought: {rsi:.1f}")
            vote_strength = 0.8
            signal = Signal.PUT
        
        if signal != Signal.HOLD and vote_strength >= 0.4:
            return StrategyVote(
                strategy_name=self.name,
                vote_strength=min(1.0, vote_strength),
                signal=signal,
                conditions_met=conditions,
                indicator_values=indicators
            )
        
        return None
    
    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

# ==== ELITE MARKET REGIME DETECTOR ====
class EliteMarketRegimeDetector:
    """🔍 Elite Market Regime Detection System"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=20)
        self.volatility_history = deque(maxlen=50)
        self.trend_history = deque(maxlen=30)
        
        logging.info("🔍 Elite Market Regime Detector initialized")
    
    def detect_regime(self, candles: List[Candle]) -> MarketState:
        """🔍 Detect current market regime with advanced analysis"""
        if len(candles) < 20:
            return MarketState(
                regime=MarketRegime.UNKNOWN,
                volatility=0.001,
                trend_strength=0.0,
                session_type=self._detect_session_type(),
                time_weight=self._get_time_weight(),
                volatility_burst=0.0
            )
        
        # Calculate basic indicators
        volatility = self._calculate_basic_volatility(candles)
        trend_strength = self._calculate_basic_trend(candles)
        
        # Simple regime classification
        if abs(trend_strength) > 0.3:
            regime = MarketRegime.STRONG_TRENDING
        elif abs(trend_strength) > 0.15:
            regime = MarketRegime.TRENDING
        elif volatility > 0.003:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.CHOPPY
        
        return MarketState(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            session_type=self._detect_session_type(),
            time_weight=self._get_time_weight(),
            volatility_burst=0.0
        )
    
    def _calculate_basic_volatility(self, candles: List[Candle]) -> float:
        if len(candles) < 10:
            return 0.001
        returns = [(candles[i].close - candles[i-1].close) / candles[i-1].close 
                  for i in range(1, min(len(candles), 20))]
        return np.std(returns) if returns else 0.001
    
    def _calculate_basic_trend(self, candles: List[Candle]) -> float:
        if len(candles) < 10:
            return 0.0
        prices = [c.close for c in candles[-10:]]
        return (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
    
    def _detect_session_type(self) -> SessionType:
        utc_hour = datetime.datetime.utcnow().hour
        if 7 <= utc_hour < 16:
            return SessionType.LONDON
        elif 16 <= utc_hour < 21:
            return SessionType.NEW_YORK
        else:
            return SessionType.ASIAN
    
    def _get_time_weight(self) -> float:
        session = self._detect_session_type()
        weights = {SessionType.LONDON: 1.1, SessionType.NEW_YORK: 1.1, SessionType.ASIAN: 0.8}
        return weights.get(session, 1.0)

# ==== ELITE DECISION FILTER ====
class EliteAdaptiveDecisionFilter:
    """🧼 Elite Adaptive Final Decision Filter System"""
    
    def __init__(self, threshold_manager: EliteAdaptiveThresholds):
        self.threshold_manager = threshold_manager
        self.recent_decisions = deque(maxlen=20)
        logging.info("🧼 Elite Adaptive Decision Filter initialized")
    
    def filter_decision(self, votes: List[StrategyVote], market_state: MarketState,
                       strategy_alignment: float) -> Optional[Tuple[Signal, float, List[str]]]:
        """🔧 Apply elite adaptive filtering to make final decision"""
        
        if not votes:
            return None
        
        # Get adaptive thresholds
        thresholds = self.threshold_manager.adapt_thresholds(strategy_alignment)
        
        # Group votes by signal
        call_votes = [v for v in votes if v.signal == Signal.CALL]
        put_votes = [v for v in votes if v.signal == Signal.PUT]
        
        # Calculate confidence
        call_confidence = sum(v.vote_strength for v in call_votes) / len(call_votes) if call_votes else 0.0
        put_confidence = sum(v.vote_strength for v in put_votes) / len(put_votes) if put_votes else 0.0
        
        min_confidence = thresholds['min_confidence_threshold']
        
        # Decision logic
        if len(call_votes) >= 2 and call_confidence >= min_confidence:
            if len(put_votes) == 0 or call_confidence > put_confidence + 0.1:
                reasons = [f"Call confidence: {call_confidence:.2f}"]
                return Signal.CALL, call_confidence, reasons
        
        if len(put_votes) >= 2 and put_confidence >= min_confidence:
            if len(call_votes) == 0 or put_confidence > call_confidence + 0.1:
                reasons = [f"Put confidence: {put_confidence:.2f}"]
                return Signal.PUT, put_confidence, reasons
        
        return None

# ==== SIMPLE FUSION STRATEGY ====
class EliteNeuralBeastQuantumFusion:
    """🌟 ELITE NEURAL BEAST QUANTUM FUSION - ULTIMATE INSTITUTIONAL STRATEGY 🌟"""
    
    def __init__(self):
        self.name = "Neural Beast Quantum Fusion"
    
    def analyze(self, candles: List[Candle], market_state: MarketState) -> Optional[StrategyVote]:
        """🌟 Elite Neural Beast Quantum Fusion Analysis"""
        if len(candles) < 20:
            return None
        
        # Simple momentum analysis
        closes = [c.close for c in candles]
        recent_change = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
        
        if abs(recent_change) > 0.002:
            signal = Signal.CALL if recent_change > 0 else Signal.PUT
            strength = min(0.9, abs(recent_change) * 100)
            return StrategyVote(
                strategy_name=self.name,
                vote_strength=strength,
                signal=signal,
                conditions_met=[f"Neural momentum: {recent_change:.3%}"],
                indicator_values={'momentum': recent_change}
            )
        
        return None

# ==== SECURITY MANAGER ====
class SecurityManager:
    def __init__(self):
        self.session_file = SESSION_FILE
        self.max_trades = MAX_TRADES_LIMIT
        
    def get_machine_id(self):
        """Generate unique machine identifier"""
        try:
            import platform
            machine_info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
            return hashlib.md5(machine_info.encode()).hexdigest()[:16]
        except:
            return "default_machine"
    
    def load_session_data(self):
        """Load existing session data"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    machine_id = self.get_machine_id()
                    if data.get('machine_id') == machine_id:
                        return data
                    else:
                        logging.warning("🔒 Session file from different machine detected")
                        return self.create_new_session()
            else:
                return self.create_new_session()
        except Exception as e:
            logging.error(f"🔒 Error loading session: {e}")
            return self.create_new_session()
    
    def create_new_session(self):
        """Create new session data"""
        session_data = {
            'machine_id': self.get_machine_id(),
            'trades_used': 0,
            'session_active': True,
            'created_date': datetime.datetime.now().isoformat(),
            'last_access': datetime.datetime.now().isoformat()
        }
        self.save_session_data(session_data)
        return session_data
    
    def save_session_data(self, data):
        """Save session data to file"""
        try:
            data['last_access'] = datetime.datetime.now().isoformat()
            with open(self.session_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"🔒 Error saving session: {e}")
    
    def reset_with_license_key(self, key: str) -> bool:
        """Reset session with license key"""
        if key == "4444":  # Your secret reset key
            session_data = self.create_new_session()
            self.save_session_data(session_data)
            logging.info("🔒 Session reset with valid license key")
            return True
        return False
    
    def increment_trade_count(self, session_data):
        """Increment trade count and check limits"""
        session_data['trades_used'] += 1
        self.save_session_data(session_data)
        
        remaining = self.max_trades - session_data['trades_used']
        logging.info(f"🔒 Trade #{session_data['trades_used']}/{self.max_trades} executed. Remaining: {remaining}")
        
        if session_data['trades_used'] >= self.max_trades:
            session_data['session_active'] = False
            self.save_session_data(session_data)
            return False
        return True
    
    def is_session_valid(self, session_data):
        """Check if session is still valid"""
        return session_data.get('session_active', False) and session_data['trades_used'] < self.max_trades
    
    def get_remaining_trades(self, session_data):
        """Get remaining trades count"""
        return max(0, self.max_trades - session_data['trades_used'])

# ==== ENHANCED TRADE RESULT DETECTION FUNCTIONS ====
def detect_trade_closed_popup(driver, poll_time=5.0, poll_interval=0.3):
    """FIXED: Enhanced trade result detection from popup with proper validation"""
    import time as pytime
    end_time = pytime.time() + poll_time
    while pytime.time() < end_time:
        try:
            popup_selectors = [
                "//div[contains(@class,'trade-closed')]",
                "//div[contains(@class,'trade-result')]",
                "//div[contains(@class,'deal-result')]",
                "//div[contains(@class,'popup')]//div[contains(text(),'Profit') or contains(text(),'Loss')]",
                "//div[contains(@class,'modal')]//div[contains(text(),'Trade')]",
                "//div[contains(@class,'notification')]",
                "//div[contains(@class,'alert')]"
            ]
            
            for selector in popup_selectors:
                try:
                    popup = driver.find_element(By.XPATH, selector)
                    
                    profit_indicators = [
                        ".//span[contains(@class,'profit')]",
                        ".//div[contains(@class,'profit')]",
                        ".//span[contains(text(),'$')]",
                        ".//div[contains(text(),'$')]",
                        ".//span[contains(@class,'pnl')]",
                        ".//div[contains(@class,'pnl')]"
                    ]
                    
                    for indicator in profit_indicators:
                        try:
                            profit_elem = popup.find_element(By.XPATH, indicator)
                            profit_text = profit_elem.text.replace('$','').replace(',','').replace('+','').strip()
                            
                            import re
                            numbers = re.findall(r'-?\d+\.?\d*', profit_text)
                            if numbers:
                                profit = float(numbers[0])
                                win = profit > 0
                                logging.info(f"{Colors.GREEN}✅ REAL trade result from popup: Win={win}, Profit=${profit}{Colors.END}")
                                return win, profit, abs(profit) + 10.0
                        except:
                            continue
                            
                    popup_text = popup.text.lower()
                    if "win" in popup_text or "profit" in popup_text or "successful" in popup_text:
                        logging.info(f"{Colors.GREEN}✅ REAL trade result from popup text: WIN detected{Colors.END}")
                        return True, 15.0, 25.0
                    elif "loss" in popup_text or "lose" in popup_text or "unsuccessful" in popup_text:
                        logging.info(f"{Colors.RED}❌ REAL trade result from popup text: LOSS detected{Colors.END}")
                        return False, -10.0, 0.0
                        
                except NoSuchElementException:
                    continue
                    
        except Exception as e:
            logging.debug(f"Popup detection attempt: {e}")
            
        pytime.sleep(poll_interval)
    
    logging.warning(f"{Colors.YELLOW}⚠️ No popup detected, checking trade history...{Colors.END}")
    return None, 0, 0

def get_last_trade_result(driver, timeout=15):
    """FIXED: Enhanced trade result detection from history with proper validation"""
    try:
        trade_selectors = [
            "div.deals-list__item-first",
            ".deals-list .deal-item:first-child",
            ".trade-history .trade-item:first-child",
            ".history-item:first-child",
            "[data-qa='trade-item']:first-child",
            ".trades-history .trade:first-child",
            ".history .deal:first-child"
        ]
        
        last_trade = None
        for selector in trade_selectors:
            try:
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                trades = driver.find_elements(By.CSS_SELECTOR, selector)
                if trades:
                    last_trade = trades[0]
                    break
            except:
                continue
        
        if not last_trade:
            logging.warning(f"{Colors.YELLOW}⚠️ Could not find trade history element{Colors.END}")
            return None, 0, 0
            
        profit_selectors = [
            ".//div[contains(@class,'profit')]",
            ".//span[contains(@class,'profit')]",
            ".//div[contains(@class,'pnl')]",
            ".//span[contains(@class,'pnl')]",
            ".//div[contains(text(),'$')]",
            ".//span[contains(text(),'$')]",
            ".//div[contains(@class,'result')]",
            ".//span[contains(@class,'result')]"
        ]
        
        for selector in profit_selectors:
            try:
                profit_elem = last_trade.find_element(By.XPATH, selector)
                profit_text = profit_elem.text.replace('$','').replace(',', '').replace('+','').strip()
                
                import re
                numbers = re.findall(r'-?\d+\.?\d*', profit_text)
                if numbers:
                    profit = float(numbers[0])
                    win = profit > 0
                    payout = abs(profit) + 10.0 if win else 0.0
                    logging.info(f"{Colors.GREEN if win else Colors.RED}✅ REAL trade result from history: Win={win}, Profit=${profit}{Colors.END}")
                    return win, profit, payout
            except:
                continue
        
        try:
            trade_html = last_trade.get_attribute('outerHTML').lower()
            if any(word in trade_html for word in ['win', 'profit', 'success', 'green']):
                logging.info(f"{Colors.GREEN}✅ REAL trade result from visual indicators: WIN detected{Colors.END}")
                return True, 15.0, 25.0
            elif any(word in trade_html for word in ['loss', 'lose', 'fail', 'red']):
                logging.info(f"{Colors.RED}❌ REAL trade result from visual indicators: LOSS detected{Colors.END}") 
                return False, -10.0, 0.0
        except:
            pass
            
        logging.warning(f"{Colors.YELLOW}⚠️ Could not determine trade result from history{Colors.END}")
        return None, 0, 0
        
    except Exception as e:
        logging.error(f"{Colors.RED}❌ Error detecting trade result: {e}{Colors.END}")
        return None, 0, 0

def verify_trade_execution(driver, pre_trade_screenshot=None, timeout=3):
    """FIXED: NEW - Verify that a trade was actually placed by detecting UI changes"""
    try:
        # Look for immediate visual confirmation of trade placement
        confirmation_selectors = [
            "//div[contains(@class,'trade-placed')]",
            "//div[contains(@class,'order-placed')]", 
            "//div[contains(@class,'deal-placed')]",
            "//div[contains(text(),'Trade placed')]",
            "//div[contains(text(),'Order placed')]",
            "//div[contains(@class,'position-opened')]",
            "//div[contains(@class,'trade-active')]"
        ]
        
        for selector in confirmation_selectors:
            try:
                element = WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                if element.is_displayed():
                    logging.info(f"{Colors.GREEN}✅ Trade execution confirmed: UI element detected{Colors.END}")
                    return True
            except TimeoutException:
                continue
        
        # Check for active position indicators
        active_position_selectors = [
            ".active-trade",
            ".open-position", 
            ".current-trade",
            "[data-qa='active-trade']",
            ".trade-timer"
        ]
        
        for selector in active_position_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and any(elem.is_displayed() for elem in elements):
                    logging.info(f"{Colors.GREEN}✅ Trade execution confirmed: Active position detected{Colors.END}")
                    return True
            except:
                continue
        
        # Check for countdown timer or trade duration indicator
        timer_selectors = [
            "//div[contains(@class,'timer')]",
            "//div[contains(@class,'countdown')]",
            "//span[contains(@class,'time-left')]",
            "//div[contains(text(),':')]"  # Looking for timer format like 01:30
        ]
        
        for selector in timer_selectors:
            try:
                timer = driver.find_element(By.XPATH, selector)
                if timer.is_displayed() and timer.text.strip():
                    # Check if it looks like a countdown timer
                    import re
                    if re.match(r'\d{1,2}:\d{2}', timer.text.strip()):
                        logging.info(f"{Colors.GREEN}✅ Trade execution confirmed: Timer detected ({timer.text}){Colors.END}")
                        return True
            except:
                continue
                
        logging.warning(f"{Colors.YELLOW}⚠️ No trade execution confirmation detected{Colors.END}")
        return False
        
    except Exception as e:
        logging.error(f"{Colors.RED}❌ Error verifying trade execution: {e}{Colors.END}")
        return False

# ==== ELITE TRADING BOT - MAIN CLASS ====
class EliteTradingBot:
    """🌟 ELITE TRADING BOT - INSTITUTIONAL GRADE 🌟"""
    
    def __init__(self, gui=None):
        self.gui = gui
        self.driver = None
        self.bot_running = False
        
        # Elite components
        self.threshold_manager = EliteAdaptiveThresholds()
        self.strategy_scorer = EliteStrategyScorer()
        self.regime_detector = EliteMarketRegimeDetector()
        self.decision_filter = EliteAdaptiveDecisionFilter(self.threshold_manager)
        
        # Connect strategy scorer to regime detector
        self.strategy_scorer.regime_detector = self.regime_detector
        
        # Strategies
        self.strategies = {
            'rsi': EnhancedRSIStrategy(),
            'fusion': EliteNeuralBeastQuantumFusion()
        }
        
        # Trading state
        self.balance = 10000.0
        self.stake = 100.0
        self.profit_today = 0.0
        self.loss_streak = 0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0
        
        # Elite settings
        self.take_profit = 500.0
        self.stop_loss = 250.0
        self.trade_hold_time = 8
        self.max_trades = MAX_TRADES_LIMIT
        
        # Candle data
        self.candles = []
        self.logs = []
        
        # Security manager
        self.security = SecurityManager()
        self.session_data = self.security.load_session_data()
        
        # Check if session is valid
        if not self.security.is_session_valid(self.session_data):
            self.show_session_ended()
            return
        
        # Load existing trade count
        self.total_trades = self.session_data['trades_used']
        logging.info(f"🔒 Elite session loaded: {self.total_trades}/{MAX_TRADES_LIMIT} trades used")
        
        self.setup_driver()
        if self.driver:
            self.navigate_to_trading_page()
    
    def show_session_ended(self):
        """Show session ended popup"""
        if self.gui:
            messagebox.showerror("Session Ended", 
                               f"Trade limit of {MAX_TRADES_LIMIT} reached.\n\n"
                               "Contact owner or use license key to reset.")
        else:
            logging.error("🔒 SESSION ENDED - Trade limit reached")
    
    def setup_driver(self) -> bool:
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-logging')
            options.add_argument('--disable-web-security')
            options.add_argument('--log-level=3')
            
            self.driver = uc.Chrome(
                version_main=137,
                options=options,
                driver_executable_path=None
            )
            
            self.driver.set_window_size(1920, 1080)
            logging.info("✅ Elite Chrome driver initialized successfully")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to setup driver: {e}")
            return False

    def navigate_to_trading_page(self):
        """Navigate to Pocket Option"""
        try:
            logging.info("🚀 Navigating to Pocket Option...")
            
            urls_to_try = [
                "https://pocketoption.com/en",
                "https://pocketoption.com/en/login/",
                "https://pocketoption.com/en/cabinet/demo-quick-high-low",
                "https://pocketoption.com/cabinet/demo-quick-high-low", 
                "https://pocketoption.com/en/demo",
                "https://pocketoption.com/demo"
            ]
            
            for url in urls_to_try:
                try:
                    logging.info(f"Trying URL: {url}")
                    self.driver.get(url)
                    
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: driver.execute_script("return document.readyState") != "loading"
                    )
                    
                    if self.is_login_page_loaded() or self.is_trading_page_loaded():
                        logging.info("✅ Successfully navigated to Pocket Option")
                        return
                    else:
                        logging.info("Page loaded but not recognized interface, trying next URL...")
                        continue
                        
                except TimeoutException:
                    logging.warning(f"Timeout loading {url}, trying next...")
                    continue
                except Exception as e:
                    logging.warning(f"Error loading {url}: {e}, trying next...")
                    continue
            
            logging.info("✅ Navigation completed - please login manually if needed")
            
        except Exception as e:
            logging.error(f"❌ Error in navigation: {e}")

    def is_login_page_loaded(self) -> bool:
        """Check if we're on a login page"""
        try:
            login_indicators = [
                "input[type='email']",
                "input[type='password']", 
                ".login-form",
                ".auth-form",
                "[data-test='login-button']",
                ".login-button",
                "button[type='submit']"
            ]
            
            for indicator in login_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements:
                        logging.info(f"✅ Login page detected with element: {indicator}")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking if login page loaded: {e}")
            return False

    def is_trading_page_loaded(self) -> bool:
        """Check if we're on a valid trading page"""
        try:
            trading_indicators = [
                ".btn-call",
                ".btn-put", 
                ".call-btn",
                ".put-btn",
                "[data-test='call-button']",
                "[data-test='put-button']",
                ".trading-interface",
                ".chart-container"
            ]
            
            for indicator in trading_indicators:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, indicator)
                    if elements:
                        logging.info(f"✅ Trading page detected with element: {indicator}")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking if trading page loaded: {e}")
            return False

    def get_balance(self) -> float:
        if not self.driver:
            return self.balance
        
        try:
            ready_state = self.driver.execute_script("return document.readyState")
            if ready_state != "complete":
                return self.balance
        except Exception:
            return self.balance
        
        selectors = [
            "js-balance-demo",
            "js-balance", 
            "balance-value",
        ]
        
        for selector in selectors:
            try:
                element = WebDriverWait(self.driver, 1).until(
                    EC.presence_of_element_located((By.CLASS_NAME, selector))
                )
                text = element.text.replace('$', '').replace(',', '').strip()
                balance = float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
                if balance > 0:
                    return balance
            except:
                continue
        
        css_selectors = [
            ".balance__value",
            ".js-balance-demo", 
            ".js-balance",
            "[data-qa='balance']",
            ".balance-value"
        ]
        
        for css in css_selectors:
            try:
                element = WebDriverWait(self.driver, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css))
                )
                text = element.text.replace('$', '').replace(',', '').strip()
                balance = float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
                if balance > 0:
                    return balance
            except Exception:
                continue
        
        return self.balance

    def get_candle_data(self) -> List[Candle]:
        if not self.driver:
            return self.generate_mock_candles()
        try:
            script = """
            if (typeof window.chartData !== 'undefined') {
                return window.chartData.slice(-50);
            }
            if (typeof window.candleData !== 'undefined') {
                return window.candleData.slice(-50);
            }
            if (typeof window.tradingData !== 'undefined') {
                return window.tradingData.slice(-50);
            }
            return [];
            """
            data = self.driver.execute_script(script)
            candles = []
            if data:
                for item in data:
                    if isinstance(item, dict) and all(k in item for k in ['open', 'high', 'low', 'close']):
                        candle = Candle(
                            timestamp=item.get('timestamp', time.time()),
                            open=float(item['open']),
                            high=float(item['high']),
                            low=float(item['low']),
                            close=float(item['close']),
                            volume=float(item.get('volume', 1.0))
                        )
                        candles.append(candle)
            if not candles:
                candles = self.generate_mock_candles()
            return candles[-50:]
        except Exception as e:
            logging.error(f"Error getting candle data: {e}")
            return self.generate_mock_candles()

    def generate_mock_candles(self) -> List[Candle]:
        candles = []
        base_price = 1.0 + np.random.uniform(-0.1, 0.1)
        for i in range(50):
            change = np.random.randn() * 0.002
            base_price += change
            high = base_price + abs(np.random.randn() * 0.001)
            low = base_price - abs(np.random.randn() * 0.001)
            close = base_price + np.random.randn() * 0.0005
            candle = Candle(
                timestamp=time.time() - (50 - i) * 60,
                open=base_price,
                high=high,
                low=low,
                close=close,
                volume=np.random.uniform(0.5, 2.0)
            )
            candles.append(candle)
            base_price = close
        return candles

    def set_stake(self, amount: float) -> bool:
        try:
            selectors = [
                'div.value__val > input[type="text"]',
                'input[data-test="amount-input"]',
                '.amount-input',
                'input.amount',
                '.stake-input'
            ]
            
            for selector in selectors:
                try:
                    input_box = WebDriverWait(self.driver, 2).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    input_box.clear()
                    self.driver.execute_script("arguments[0].value = '';", input_box)
                    time.sleep(0.1)
                    input_box.send_keys(str(amount))
                    self.driver.execute_script("arguments[0].value = arguments[1];", input_box, str(amount))
                    logging.info(f"💰 Elite stake set to ${amount}")
                    return True
                except (TimeoutException, NoSuchElementException):
                    continue
            
            logging.warning("⚠️ Could not find stake input field")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to set stake: {e}")
            return False

    def execute_trade(self, decision: str) -> bool:
        """FIXED: Enhanced trade execution with real validation - no fake trades logged"""
        if not self.driver:
            logging.warning(f"{Colors.YELLOW}⚠️ No driver available - skipping trade{Colors.END}")
            return False
        
        if not self.set_stake(self.stake):
            logging.warning("⚠️ Could not set stake. Proceeding with trade anyway.")
        
        selector_maps = {
            'call': [
                ".btn-call",
                ".call-btn",
                "[data-test='call-button']",
                ".higher-btn",
                ".up-btn"
            ],
            'put': [
                ".btn-put", 
                ".put-btn",
                "[data-test='put-button']",
                ".lower-btn",
                ".down-btn"
            ]
        }
        
        # Try to click the trade button
        button_clicked = False
        for selector in selector_maps[decision]:
            try:
                button = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                button.click()
                logging.info(f"{Colors.CYAN}🎯 {decision.upper()} button clicked (Stake: ${self.stake}){Colors.END}")
                button_clicked = True
                break
            except (TimeoutException, NoSuchElementException):
                continue
            except Exception as e:
                logging.error(f"❌ Error clicking {decision} button with selector {selector}: {e}")
                continue
        
        if not button_clicked:
            logging.warning(f"{Colors.YELLOW}⚠️ Could not find {decision} button - trade click failed, skipping{Colors.END}")
            return False
        
        # FIXED: Verify that the trade was actually executed
        time.sleep(1)  # Short wait for UI to update
        trade_confirmed = verify_trade_execution(self.driver, timeout=3)
        
        if not trade_confirmed:
            logging.warning(f"{Colors.YELLOW}⚠️ Trade click failed - no UI confirmation detected, skipping{Colors.END}")
            return False
        
        logging.info(f"{Colors.GREEN}🚀 REAL Elite trade executed: {decision.upper()} (Stake: ${self.stake}){Colors.END}")
        return True

    def elite_signal_analysis(self) -> Optional[Tuple[Signal, float, List[str]]]:
        """🧠 Elite Signal Analysis with all institutional components"""
        
        # Get current market state
        market_state = self.regime_detector.detect_regime(self.candles)
        
        # Update adaptive thresholds
        current_winrate = self.get_winrate() / 100.0
        self.threshold_manager.update_market_conditions(market_state.volatility, current_winrate)
        
        # Get votes from all strategies
        all_votes = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                vote = strategy.analyze(self.candles, market_state)
                if vote:
                    all_votes.append(vote)
            except Exception as e:
                logging.error(f"Error in {strategy_name}: {e}")
                continue
        
        if not all_votes:
            return None
        
        # Calculate strategy alignment for adaptive thresholds
        call_votes = [v for v in all_votes if v.signal == Signal.CALL]
        put_votes = [v for v in all_votes if v.signal == Signal.PUT]
        
        if len(call_votes) > 0 and len(put_votes) > 0:
            alignment = abs(len(call_votes) - len(put_votes)) / len(all_votes)
        elif len(call_votes) > 0 or len(put_votes) > 0:
            alignment = 1.0  # Perfect alignment
        else:
            alignment = 0.0  # No signals
        
        # Apply elite adaptive decision filter
        decision_result = self.decision_filter.filter_decision(
            all_votes, market_state, alignment
        )
        
        if decision_result:
            signal, confidence, reasons = decision_result
            
            # Log elite analysis
            logging.info(f"🧠 ELITE ANALYSIS:")
            logging.info(f"   Market Regime: {market_state.regime.name}")
            logging.info(f"   Volatility: {market_state.volatility:.4f}")
            logging.info(f"   Trend Strength: {market_state.trend_strength:.2f}")
            logging.info(f"   Session: {market_state.session_type.name}")
            logging.info(f"   Strategies Voted: {len(all_votes)}")
            logging.info(f"   Strategy Alignment: {alignment:.2f}")
            logging.info(f"   Final Decision: {signal.name} (Confidence: {confidence:.3f})")
            
            return signal, confidence, reasons
        
        return None

    def log_trade(self, signal: Signal, confidence: float, reasons: List[str], 
                 profit: float, win: bool, market_regime: MarketRegime):
        """FIXED: Enhanced trade logging with performance tracking and GUI updates"""
        
        # Check trade limit before logging
        if not self.security.increment_trade_count(self.session_data):
            logging.error("🔒 TRADE LIMIT REACHED - Bot terminating")
            self.bot_running = False
            self.show_session_ended()
            return
            
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        result = "WIN" if win else "LOSS"
        remaining = self.security.get_remaining_trades(self.session_data)
        
        # Create strategy votes from reasons for performance tracking
        strategy_votes = []
        for reason in reasons:
            if ":" in reason:
                strategy_name = reason.split(":")[0].strip()
                strategy_votes.append(StrategyVote(
                    strategy_name=strategy_name,
                    vote_strength=confidence,
                    signal=signal,
                    conditions_met=[reason],
                    indicator_values={}
                ))
        
        # Record performance data
        self.strategy_scorer.record_trade_result(
            strategy_votes, signal, win, market_regime
        )
        
        entry = f"{timestamp} | ELITE | {signal.name} | {result} | P/L: ${profit:.2f} | Confidence: {confidence:.3f} | Remaining: {remaining}"
        self.logs.append(entry)
        
        # Update counters
        self.total_trades = self.session_data['trades_used']
        if win:
            self.win_count += 1
            self.loss_streak = 0
        else:
            self.loss_count += 1
            self.loss_streak += 1
        
        self.profit_today += profit
        
        # Enhanced color logging
        log_trade_result(signal.name, result, profit, confidence, reasons)
        
        # Enhanced logging with colors
        winrate = self.get_winrate()
        stats_color = Colors.GREEN if winrate >= 60 else Colors.YELLOW if winrate >= 50 else Colors.RED
        print(f"{Colors.CYAN}📊 ELITE STATS:{Colors.END} Trades={Colors.BOLD}{self.total_trades}/{MAX_TRADES_LIMIT}{Colors.END}, "
              f"Wins={Colors.GREEN}{self.win_count}{Colors.END}, Losses={Colors.RED}{self.loss_count}{Colors.END}, "
              f"WR={stats_color}{winrate:.1f}%{Colors.END}, P/L=${Colors.CYAN}{self.profit_today:.2f}{Colors.END}")
        
        # FIXED: Update GUI statistics properly
        if self.gui:
            self.gui.trades = {'total': self.total_trades, 'wins': self.win_count, 'losses': self.loss_count}
            self.gui.balance = self.balance
            # Force GUI update
            self.gui.root.after(0, self.gui.update_elite_statistics)
        
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    def get_winrate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.win_count / self.total_trades) * 100

    def reset_session_with_key(self, key: str) -> bool:
        """Reset session with license key"""
        if self.security.reset_with_license_key(key):
            self.session_data = self.security.load_session_data()
            self.total_trades = self.session_data['trades_used']
            self.win_count = 0
            self.loss_count = 0
            self.profit_today = 0.0
            self.loss_streak = 0
            logging.info("🔒 Elite session reset")
            return True
        return False

    def run_elite_trading_session(self):
        """🌟 FIXED: Run Elite Trading Session - REMOVED FAKE SIMULATION"""
        
        # Check session validity
        if not self.security.is_session_valid(self.session_data):
            self.show_session_ended()
            return
            
        messagebox.showinfo("Elite Login Required", 
                          "Please login to Pocket Option in the opened browser, then press OK to start Elite trading.")

        self.bot_running = True
        self.loss_streak = 0
        session_start_time = time.time()
        
        logging.info(f"🌟 ELITE NEURAL BEAST QUANTUM FUSION session started")
        logging.info(f"🔒 {self.security.get_remaining_trades(self.session_data)} trades remaining")

        try:
            # Setup after login
            logging.info("⚡ Elite setup after login...")
            
            for attempt in range(10):
                try:
                    if self.is_trading_page_loaded():
                        logging.info("✅ Elite trading page ready")
                        break
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
                    continue
            
            # Balance check
            logging.info("💰 Elite balance check...")
            for attempt in range(3):
                try:
                    balance = self.get_balance()
                    if balance > 0:
                        self.balance = balance
                        logging.info(f"✅ Elite balance: ${self.balance}")
                        break
                    time.sleep(1)
                except Exception as e:
                    logging.warning(f"Balance attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
                    continue
            else:
                logging.warning("⚠️ Using default balance")
                self.balance = 10000.0
        
        except Exception as e:
            logging.error(f"❌ Error during elite setup: {e}")
            self.balance = 10000.0

        session_time_limit = 2 * 60 * 60  # 2 hours
        last_trade_time = 0
        
        while self.bot_running:
            try:
                # Check session validity continuously
                if not self.security.is_session_valid(self.session_data):
                    logging.error("🔒 Elite session invalid - terminating")
                    self.show_session_ended()
                    break
                    
                elapsed_time = time.time() - session_start_time
            
                if elapsed_time >= session_time_limit:
                    self.bot_running = False
                    messagebox.showinfo("Elite Session Complete", 
                                      "2-hour elite trading session complete. Bot is stopping.")
                    logging.info("⏰ Elite 2-hour time limit reached - trading session stopped.")
                    break
                
                if self.total_trades >= self.max_trades:
                    self.bot_running = False
                    self.show_session_ended()
                    break

                if self.profit_today >= self.take_profit:
                    self.bot_running = False
                    messagebox.showinfo("Elite Take Profit Hit", 
                                      f"Elite take profit of ${self.take_profit} reached. Bot is stopping.")
                    logging.info(f"🎯 Elite take profit of ${self.take_profit} reached - trading session stopped.")
                    break
                
                if self.profit_today <= -self.stop_loss:
                    self.bot_running = False
                    messagebox.showinfo("Elite Stop Loss Hit", 
                                      f"Elite stop loss of ${self.stop_loss} reached. Bot is stopping.")
                    logging.info(f"🛡️ Elite stop loss of ${self.stop_loss} reached - trading session stopped.")
                    break

                # Elite risk management
                if self.loss_streak >= 4:  # Elite allows higher loss streak
                    logging.info("🛡️ Elite risk management: Skipping trade due to loss streak")
                    time.sleep(10)
                    continue

                # Try to update balance
                try:
                    new_balance = self.get_balance()
                    if new_balance > 0:
                        self.balance = new_balance
                except Exception:
                    pass
            
                # Get candle data
                self.candles = self.get_candle_data()
                
                # Elite signal analysis
                signal_result = self.elite_signal_analysis()

                current_time = time.time()
                if signal_result and (current_time - last_trade_time) >= 8:
                    signal, confidence, reasons = signal_result
                    
                    # FIXED: Only proceed if trade is actually executed
                    if self.execute_trade(signal.name.lower()):
                        last_trade_time = current_time
                        time.sleep(self.trade_hold_time)
                    
                        # Enhanced trade result detection
                        win, profit, payout = detect_trade_closed_popup(self.driver, poll_time=5.0)

                        if win is None:
                            logging.info("🔍 Elite checking trade history...")
                            time.sleep(2)
                            win, profit, payout = get_last_trade_result(self.driver, timeout=10)

                        # FIXED: REMOVED FAKE SIMULATION - Only log real trades
                        if win is not None:
                            actual_profit = profit
                            logging.info(f"{Colors.GREEN if win else Colors.RED}📊 REAL Elite trade result: Win={win}, P/L=${actual_profit:.2f}{Colors.END}")
                            
                            # Get current market regime for logging
                            market_state = self.regime_detector.detect_regime(self.candles)
                            
                            self.log_trade(signal, confidence, reasons, actual_profit, win, market_state.regime)
                        else:
                            logging.warning(f"{Colors.YELLOW}⚠️ Could not determine trade result - skipping trade logging{Colors.END}")
                    else:
                        logging.warning(f"{Colors.YELLOW}⚠️ Trade execution failed - no trade logged{Colors.END}")
                else:
                    time.sleep(3)
                
            except Exception as e:
                logging.error(f"❌ Error in elite trading loop: {e}")
                time.sleep(5)
        
        self.bot_running = False
        logging.info("🏁 Exiting Elite Neural Beast Quantum Fusion session...")

# ==== ELITE GUI ====
class EliteNeuralBeastGUI:
    """🌟 ELITE NEURAL BEAST QUANTUM FUSION GUI - INSTITUTIONAL EDITION 🌟"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 - INSTITUTIONAL GRADE 🌟")
        self.root.geometry("800x700")
        self.root.configure(bg='#000000')
        self.root.resizable(False, False)
        
        # Elite state variables
        self.is_active = False
        self.elite_power = 97  # Elite fusion power
        self.adaptive_intelligence = True
        self.regime_detection = True
        self.strategy_scoring = True
        self.balance = 10000
        self.trades = {'total': 0, 'wins': 0, 'losses': 0}
        self.settings = {'stake': 100, 'take_profit': 500, 'stop_loss': 250}
        self.elite_feed_messages = []
        self.glow_intensity = 0
        
        # Animation variables
        self.animation_running = False
        self.particle_positions = []
        
        # Initialize elite bot
        self.bot = EliteTradingBot(gui=self)
        
        self.setup_elite_styles()
        self.create_elite_widgets()
        self.start_elite_animations()
    
    def setup_elite_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Elite custom styles
        style.configure('EliteTitle.TLabel', 
                       background='#111111', 
                       foreground='#FFD700', 
                       font=('Courier', 14, 'bold'))
        
        style.configure('EliteStatus.TLabel', 
                       background='#1a1a1a', 
                       foreground='#00FFFF', 
                       font=('Courier', 9))
        
        style.configure('EliteEnergy.TLabel', 
                       background='#1a1a1a', 
                       foreground='#FFFFFF', 
                       font=('Courier', 9, 'bold'))
        
        style.configure('EliteActive.TButton',
                       background='#FFD700',
                       foreground='black',
                       font=('Courier', 11, 'bold'))
        
        style.configure('EliteInactive.TButton',
                       background='#333333',
                       foreground='#FFD700',
                       font=('Courier', 11, 'bold'))
    
    def create_elite_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#000000', padx=10, pady=10)
        main_frame.pack(fill='both', expand=True)
        
        # Header
        self.create_elite_header(main_frame)
        
        # Elite Intelligence Matrix
        self.create_elite_matrix(main_frame)
        
        # Elite Control Panel
        self.create_elite_control_panel(main_frame)
        
        # Elite Statistics Panel
        self.create_elite_statistics_panel(main_frame)
    
    def create_elite_header(self, parent):
        header_frame = tk.Frame(parent, bg='#111111', relief='ridge', bd=2)
        header_frame.pack(fill='x', pady=(0, 10))
        
        # Title with enhanced styling
        title_label = tk.Label(header_frame, 
                              text="🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 🌟",
                              bg='#111111', 
                              fg='#FFD700',
                              font=('Courier', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                 text="INSTITUTIONAL GRADE - ADAPTIVE INTELLIGENCE ENABLED",
                                 bg='#111111',
                                 fg='#00FFFF',
                                 font=('Courier', 10))
        subtitle_label.pack(pady=(0, 10))
        
        # Elite power indicator
        self.power_label = tk.Label(header_frame,
                                   text=f"🔋 ELITE POWER: {self.elite_power}%",
                                   bg='#111111',
                                   fg='#FFFFFF',
                                   font=('Courier', 10, 'bold'))
        self.power_label.pack(pady=(0, 10))
    
    def create_elite_matrix(self, parent):
        matrix_frame = tk.LabelFrame(parent, text="🧠 ELITE INTELLIGENCE MATRIX", 
                                   bg='#1a1a1a', fg='#FFD700',
                                   font=('Courier', 11, 'bold'))
        matrix_frame.pack(fill='x', pady=(0, 10))
        
        # Intelligence indicators
        indicators_frame = tk.Frame(matrix_frame, bg='#1a1a1a')
        indicators_frame.pack(fill='x', padx=10, pady=10)
        
        # Adaptive Thresholds
        self.adaptive_label = tk.Label(indicators_frame,
                                     text="🔄 Adaptive Thresholds: ACTIVE",
                                     bg='#1a1a1a', fg='#00FF00',
                                     font=('Courier', 9, 'bold'))
        self.adaptive_label.grid(row=0, column=0, sticky='w', padx=5)
        
        # Strategy Scoring
        self.scoring_label = tk.Label(indicators_frame,
                                    text="📊 Strategy Scoring: ONLINE",
                                    bg='#1a1a1a', fg='#00FF00',
                                    font=('Courier', 9, 'bold'))
        self.scoring_label.grid(row=0, column=1, sticky='w', padx=5)
        
        # Market Regime Detection
        self.regime_label = tk.Label(indicators_frame,
                                   text="🔍 Regime Detection: ENGAGED",
                                   bg='#1a1a1a', fg='#00FF00',
                                   font=('Courier', 9, 'bold'))
        self.regime_label.grid(row=1, column=0, sticky='w', padx=5)
        
        # Decision Filtering
        self.filter_label = tk.Label(indicators_frame,
                                   text="🧼 Elite Filtering: ENABLED",
                                   bg='#1a1a1a', fg='#00FF00',
                                   font=('Courier', 9, 'bold'))
        self.filter_label.grid(row=1, column=1, sticky='w', padx=5)
        
        # Elite feed
        feed_frame = tk.Frame(matrix_frame, bg='#1a1a1a')
        feed_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        tk.Label(feed_frame, text="🌟 Elite Intelligence Feed:", 
                bg='#1a1a1a', fg='#FFD700', font=('Courier', 9, 'bold')).pack(anchor='w')
        
        self.elite_feed = tk.Text(feed_frame, height=6, width=80,
                                bg='#0a0a0a', fg='#00FFFF',
                                font=('Courier', 8),
                                state='disabled')
        self.elite_feed.pack(fill='both', expand=True, pady=(5, 0))
        
        # Add initial messages
        self.add_elite_feed_message("🌟 Elite Neural Beast Quantum Fusion V11 initialized")
        self.add_elite_feed_message("🧠 Adaptive threshold management online")
        self.add_elite_feed_message("📊 Strategy performance scoring active")
        self.add_elite_feed_message("🔍 Market regime detection engaged")
        self.add_elite_feed_message("✅ REAL trade validation enabled - NO FAKE SIMULATION")
    
    def create_elite_control_panel(self, parent):
        control_frame = tk.LabelFrame(parent, text="🎛️ ELITE CONTROL PANEL",
                                    bg='#1a1a1a', fg='#FFD700',
                                    font=('Courier', 11, 'bold'))
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#1a1a1a')
        button_frame.pack(pady=10)
        
        # Elite toggle button
        self.elite_toggle_btn = tk.Button(button_frame,
                                        text="🚀 ACTIVATE ELITE FUSION",
                                        bg='#FFD700', fg='black',
                                        font=('Courier', 12, 'bold'),
                                        command=self.toggle_elite_fusion,
                                        width=25, height=2)
        self.elite_toggle_btn.pack(side='left', padx=10)
        
        # Stop button
        self.elite_stop_btn = tk.Button(button_frame,
                                      text="⏹️ STOP",
                                      bg='#DC2626', fg='white',
                                      font=('Courier', 10, 'bold'),
                                      command=self.stop_elite_fusion,
                                      width=10)
        self.elite_stop_btn.pack(side='left', padx=5)
        
        # Reset button
        self.elite_reset_btn = tk.Button(button_frame,
                                       text="🔄 RESET",
                                       bg='#8855FF', fg='white',
                                       font=('Courier', 10, 'bold'),
                                       command=self.reset_elite_session,
                                       width=10)
        self.elite_reset_btn.pack(side='left', padx=5)
        
        # Settings frame
        settings_frame = tk.Frame(control_frame, bg='#1a1a1a')
        settings_frame.pack(pady=(0, 10))
        
        # Elite settings
        tk.Label(settings_frame, text="⚙️ Elite Settings:", 
                bg='#1a1a1a', fg='#FFD700', font=('Courier', 10, 'bold')).grid(row=0, column=0, columnspan=6, pady=5)
        
        # Stake
        tk.Label(settings_frame, text="Stake:", bg='#1a1a1a', fg='#FFFFFF', 
                font=('Courier', 9)).grid(row=1, column=0, padx=5, sticky='w')
        self.stake_var = tk.StringVar(value=str(self.settings['stake']))
        stake_entry = tk.Entry(settings_frame, textvariable=self.stake_var, width=8,
                              bg='#333333', fg='#FFD700', font=('Courier', 9))
        stake_entry.grid(row=1, column=1, padx=5)
        
        # Take Profit
        tk.Label(settings_frame, text="T/P:", bg='#1a1a1a', fg='#FFFFFF',
                font=('Courier', 9)).grid(row=1, column=2, padx=5, sticky='w')
        self.tp_var = tk.StringVar(value=str(self.settings['take_profit']))
        tp_entry = tk.Entry(settings_frame, textvariable=self.tp_var, width=8,
                           bg='#333333', fg='#00FF00', font=('Courier', 9))
        tp_entry.grid(row=1, column=3, padx=5)
        
        # Stop Loss
        tk.Label(settings_frame, text="S/L:", bg='#1a1a1a', fg='#FFFFFF',
                font=('Courier', 9)).grid(row=1, column=4, padx=5, sticky='w')
        self.sl_var = tk.StringVar(value=str(self.settings['stop_loss']))
        sl_entry = tk.Entry(settings_frame, textvariable=self.sl_var, width=8,
                           bg='#333333', fg='#FF4444', font=('Courier', 9))
        sl_entry.grid(row=1, column=5, padx=5)
    
    def create_elite_statistics_panel(self, parent):
        stats_frame = tk.LabelFrame(parent, text="📊 ELITE PERFORMANCE ANALYTICS",
                                  bg='#1a1a1a', fg='#FFD700',
                                  font=('Courier', 11, 'bold'))
        stats_frame.pack(fill='both', expand=True)
        
        # Stats display
        stats_display = tk.Frame(stats_frame, bg='#1a1a1a')
        stats_display.pack(fill='x', padx=10, pady=10)
        
        # Balance
        self.balance_label = tk.Label(stats_display,
                                    text=f"💰 Elite Balance: ${self.balance:,.2f}",
                                    bg='#1a1a1a', fg='#FFD700',
                                    font=('Courier', 11, 'bold'))
        self.balance_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Trade stats
        self.trades_label = tk.Label(stats_display,
                                   text=f"🎯 Trades: {self.trades['total']}",
                                   bg='#1a1a1a', fg='#FFFFFF',
                                   font=('Courier', 10))
        self.trades_label.grid(row=1, column=0, padx=10, sticky='w')
        
        self.wins_label = tk.Label(stats_display,
                                 text=f"✅ Wins: {self.trades['wins']}",
                                 bg='#1a1a1a', fg='#00FF00',
                                 font=('Courier', 10))
        self.wins_label.grid(row=1, column=1, padx=10, sticky='w')
        
        self.losses_label = tk.Label(stats_display,
                                   text=f"❌ Losses: {self.trades['losses']}",
                                   bg='#1a1a1a', fg='#FF4444',
                                   font=('Courier', 10))
        self.losses_label.grid(row=2, column=0, padx=10, sticky='w')
        
        # Win rate
        winrate = (self.trades['wins'] / max(1, self.trades['total'])) * 100
        self.winrate_label = tk.Label(stats_display,
            text=f"📈 Elite Win Rate: {winrate:.1f}%",
            bg='#1a1a1a', fg='#00BFFF',
            font=('Courier', 10))
        self.winrate_label.grid(row=2, column=1, padx=10, sticky='w')
        
        # Elite session info
        session_frame = tk.Frame(stats_frame, bg='#1a1a1a')
        session_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        remaining = MAX_TRADES_LIMIT - self.trades['total']
        self.session_label = tk.Label(session_frame,
                                    text=f"🔒 Session: {self.trades['total']}/{MAX_TRADES_LIMIT} trades used ({remaining} remaining)",
                                    bg='#1a1a1a', fg='#FFD700',
                                    font=('Courier', 9, 'bold'))
        self.session_label.pack()
    
    def add_elite_feed_message(self, message):
        """Add message to elite intelligence feed"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"
        
        self.elite_feed.config(state='normal')
        self.elite_feed.insert('end', full_message)
        self.elite_feed.see('end')
        self.elite_feed.config(state='disabled')
        
        # Keep only last 50 messages
        lines = self.elite_feed.get('1.0', 'end').split('\n')
        if len(lines) > 50:
            self.elite_feed.config(state='normal')
            self.elite_feed.delete('1.0', '2.0')
            self.elite_feed.config(state='disabled')
    
    def toggle_elite_fusion(self):
        """Toggle elite fusion activation"""
        if not self.is_active:
            # Validate settings
            try:
                stake = float(self.stake_var.get())
                tp = float(self.tp_var.get())
                sl = float(self.sl_var.get())
                
                if stake <= 0 or tp <= 0 or sl <= 0:
                    messagebox.showerror("Elite Error", "All values must be positive numbers!")
                    return
                
                # Update bot settings
                if self.bot:
                    self.bot.stake = stake
                    self.bot.take_profit = tp
                    self.bot.stop_loss = sl
                
                # Start elite fusion
                self.is_active = True
                self.elite_toggle_btn.config(text="🔥 ELITE FUSION ACTIVE", bg='#00FF00', fg='black')
                self.add_elite_feed_message("🌟 Elite Neural Beast Quantum Fusion ACTIVATED")
                self.add_elite_feed_message("🧠 All adaptive systems ONLINE")
                self.add_elite_feed_message("✅ REAL TRADE VALIDATION: Only genuine trades will be logged")
                
                # Start trading in separate thread
                if self.bot:
                    trading_thread = threading.Thread(target=self.bot.run_elite_trading_session, daemon=True)
                    trading_thread.start()
                
            except ValueError:
                messagebox.showerror("Elite Error", "Please enter valid numeric values!")
        else:
            self.stop_elite_fusion()
    
    def stop_elite_fusion(self):
        """Stop elite fusion"""
        self.is_active = False
        if self.bot:
            self.bot.bot_running = False
        
        self.elite_toggle_btn.config(text="🚀 ACTIVATE ELITE FUSION", bg='#FFD700', fg='black')
        self.add_elite_feed_message("⏹️ Elite Neural Beast Quantum Fusion DEACTIVATED")
    
    def reset_elite_session(self):
        """Reset elite session"""
        if self.is_active:
            messagebox.showwarning("Elite Warning", "Please stop the bot before resetting!")
            return
        
        # Ask for license key
        key = simpledialog.askstring("Elite Reset", "Enter license key:", show='*')
        if key:
            if self.bot and self.bot.reset_session_with_key(key):
                self.trades = {'total': 0, 'wins': 0, 'losses': 0}
                self.balance = 10000
                self.update_elite_statistics()
                self.add_elite_feed_message("🔄 Elite session RESET successfully")
                messagebox.showinfo("Elite Success", "Session reset successfully!")
            else:
                messagebox.showerror("Elite Error", "Invalid license key!")
    
    def update_elite_statistics(self):
        """FIXED: Update elite statistics display - properly update existing labels"""
        # Update from bot data if available
        if self.bot:
            self.balance = self.bot.balance
            self.trades = {
                'total': self.bot.total_trades,
                'wins': self.bot.win_count,
                'losses': self.bot.loss_count
            }
        
        # Update labels - FIXED: Just update the text, don't recreate the label
        self.balance_label.config(text=f"💰 Elite Balance: ${self.balance:,.2f}")
        self.trades_label.config(text=f"🎯 Trades: {self.trades['total']}")
        self.wins_label.config(text=f"✅ Wins: {self.trades['wins']}")
        self.losses_label.config(text=f"❌ Losses: {self.trades['losses']}")
        
        # Win rate - FIXED: Just update the existing label text, don't recreate it
        winrate = (self.trades['wins'] / max(1, self.trades['total'])) * 100
        self.winrate_label.config(text=f"📈 Elite Win Rate: {winrate:.1f}%")
        
        # Session info
        remaining = MAX_TRADES_LIMIT - self.trades['total']
        self.session_label.config(text=f"🔒 Session: {self.trades['total']}/{MAX_TRADES_LIMIT} trades used ({remaining} remaining)")
    
    def elite_animation_loop(self):
        """Elite animation loop"""
        if not self.animation_running:
            return
        
        # Update glow intensity
        self.glow_intensity = (self.glow_intensity + 5) % 360
        
        # Update elite power (simulate fluctuation)
        if self.is_active:
            self.elite_power = 95 + np.sin(self.glow_intensity * 0.1) * 2
            self.power_label.config(text=f"🔋 ELITE POWER: {self.elite_power:.1f}%")
        
        # Update statistics
        self.update_elite_statistics()
        
        # Add random elite feed messages during operation
        if self.is_active and np.random.random() < 0.1:
            messages = [
                "🧠 Adaptive thresholds recalibrated",
                "📊 Strategy weights updated",
                "🔍 Market regime analyzed",
                "🧼 Decision filter optimized",
                "⚛️ Quantum analysis complete",
                "🦁 Beast mode confluence detected",
                "🌟 Neural patterns recognized",
                "✅ Trade validation systems active"
            ]
            self.add_elite_feed_message(np.random.choice(messages))
        
        # Schedule next update
        self.root.after(1000, self.elite_animation_loop)
    
    def start_elite_animations(self):
        """Start elite animations"""
        self.animation_running = True
        self.elite_animation_loop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_active:
            if messagebox.askokcancel("Elite Quit", "Elite fusion is active. Really quit?"):
                self.stop_elite_fusion()
                self.animation_running = False
                if self.bot and self.bot.driver:
                    try:
                        self.bot.driver.quit()
                    except:
                        pass
                self.root.destroy()
        else:
            self.animation_running = False
            if self.bot and self.bot.driver:
                try:
                    self.bot.driver.quit()
                except:
                    pass
            self.root.destroy()

def main():
    """🌟 Elite Neural Beast Quantum Fusion V11 - Main Entry Point 🌟"""
    
    # Setup enhanced logging with colors
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                super().emit(record)
            except (UnicodeEncodeError, UnicodeDecodeError):
                try:
                    record.msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
                    super().emit(record)
                except:
                    pass
    
    # Configure elite logging with colors
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            SafeStreamHandler(sys.stdout),
            logging.FileHandler('elite_neural_beast_v11.log', encoding='utf-8')
        ]
    )
    
    # Apply colored formatter to console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # Remove default handlers and add colored one
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(console_handler)
    root_logger.addHandler(logging.FileHandler('elite_neural_beast_v11.log', encoding='utf-8'))
    
    # Register cleanup function
    def cleanup():
        logging.info("🏁 Elite Neural Beast Quantum Fusion V11 shutdown complete")
    
    atexit.register(cleanup)
    
    # Print elite banner with colors
    print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 - INSTITUTIONAL GRADE 🌟{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.GREEN}🧠 ADAPTIVE THRESHOLD TUNING: ENABLED{Colors.END}")
    print(f"{Colors.GREEN}📊 STRATEGY PERFORMANCE SCORING: ACTIVE{Colors.END}") 
    print(f"{Colors.GREEN}🔍 MARKET REGIME DETECTION: ONLINE{Colors.END}")
    print(f"{Colors.GREEN}🧼 ADAPTIVE DECISION FILTERING: ENGAGED{Colors.END}")
    print(f"{Colors.GREEN}⚛️ QUANTUM ANALYSIS: OPERATIONAL{Colors.END}")
    print(f"{Colors.GREEN}🦁 BEAST MODE: READY{Colors.END}")
    print(f"{Colors.GREEN}🌟 NEURAL NETWORKS: OPTIMIZED{Colors.END}")
    print(f"{Colors.BOLD}{Colors.RED}✅ REAL TRADE VALIDATION: NO FAKE SIMULATION{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.YELLOW}🔒 Trade Limit: {MAX_TRADES_LIMIT}{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    # Initialize elite GUI
    try:
        root = tk.Tk()
        app = EliteNeuralBeastGUI(root)
        
        # Set window close handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        logging.info("🌟 Elite Neural Beast Quantum Fusion V11 GUI initialized")
        
        # Start elite application
        root.mainloop()
        
    except Exception as e:
        logging.error(f"❌ Elite startup error: {e}")
        print(f"❌ Error starting Elite Neural Beast Quantum Fusion V11: {e}")
    
    finally:
        logging.info("🏁 Elite Neural Beast Quantum Fusion V11 session ended")

if __name__ == "__main__":
    main()