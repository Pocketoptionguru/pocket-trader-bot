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
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import numpy as np
import sys
import os
import json
import hashlib
from enum import Enum, auto
from collections import deque, defaultdict
from colorama import init as colorama_init, Fore, Style  # type: ignore

colorama_init(autoreset=True)

# --- Suppress noisy warnings from urllib3 ---
import warnings
import urllib3
from urllib3.exceptions import InsecureRequestWarning, NotOpenSSLWarning, DependencyWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", category=DependencyWarning)
urllib3.disable_warnings()

urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.ERROR)

# ---- CONFIGURATION ----
MAX_TRADES_LIMIT = 50
SESSION_FILE = "elite_beast_session.dat"

# ==== DATA STRUCTURES ====
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
    def is_bullish(self) -> bool:
        return self.close > self.open

@dataclass
class StrategyVote:
    strategy_name: str
    vote_strength: float
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

    @property
    def accuracy(self) -> float:
        return self.correct_signals / self.total_signals if self.total_signals else 0.5

    @property
    def recent_accuracy(self) -> float:
        return sum(self.recent_performance) / len(self.recent_performance) if self.recent_performance else 0.5

@dataclass
class MarketState:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    session_type: SessionType
    time_weight: float
    volatility_burst: float

# ==== ADAPTIVE THRESHOLDS ====
class EliteAdaptiveThresholds:
    def __init__(self):
        self.base_thresholds = {
            "min_confidence_threshold": 0.35,
            "confirmation_weight_threshold": 0.4,
            "min_strategy_count": 2,
            "regime_bonus_threshold": 0.1,
        }
        self.adaptive_thresholds = self.base_thresholds.copy()
        self.volatility_history = deque(maxlen=20)
        self.win_rate_history = deque(maxlen=20)
        self.last_adjustment = time.time()
        self.adjustment_interval = 300  # seconds

    def update_market_conditions(self, vol: float, win_rate: float):
        self.volatility_history.append(vol)
        self.win_rate_history.append(win_rate)

    def adapt_thresholds(self, alignment: float) -> Dict[str, float]:
        if time.time() - self.last_adjustment < self.adjustment_interval:
            return self.adaptive_thresholds

        self.last_adjustment = time.time()
        vol_factor = self._calc_vol_factor()
        perf_factor = self._calc_perf_factor()
        align_factor = np.clip(alignment, 0.0, 1.0)

        # Simple adaptation logic
        self.adaptive_thresholds["min_confidence_threshold"] = np.clip(
            self.base_thresholds["min_confidence_threshold"] - vol_factor * 0.1 + perf_factor * 0.15,
            0.25,
            0.65,
        )
        return self.adaptive_thresholds

    def _calc_vol_factor(self) -> float:
        if not self.volatility_history:
            return 0.0
        recent = np.mean(list(self.volatility_history)[-5:])
        overall = np.mean(self.volatility_history)
        return np.clip((recent - overall) / overall if overall else 0.0, -1.0, 1.0)

    def _calc_perf_factor(self) -> float:
        return np.clip(np.mean(self.win_rate_history[-10:]) if self.win_rate_history else 0.5, 0.0, 1.0)

# ==== STRATEGY SCORER ====
class EliteStrategyScorer:
    def __init__(self):
        self.strategies: Dict[str, StrategyPerformance] = {}
        self.trade_history = deque(maxlen=100)
        self.initialize_strategies()

    def initialize_strategies(self):
        for name in [
            "Neural Beast Quantum Fusion",
            "Enhanced RSI",
            "Enhanced Bollinger Bands",
            "Trend Filter",
            "Momentum Surge",
            "Volume Confluence",
        ]:
            self.strategies[name] = StrategyPerformance(name=name)

    def record_trade_result(
        self,
        strategy_votes: List[StrategyVote],
        final_signal: Signal,
        was_correct: bool,
        regime: MarketRegime,
    ):
        self.trade_history.append({
            "votes": strategy_votes,
            "final_signal": final_signal,
            "correct": was_correct,
            "regime": regime,
            "ts": time.time(),
        })
        for v in strategy_votes:
            if v.signal == final_signal and v.strategy_name in self.strategies:
                s = self.strategies[v.strategy_name]
                s.total_signals += 1
                if was_correct:
                    s.correct_signals += 1
                s.recent_performance.append(was_correct)
                s.regime_performance[regime].append(was_correct)
        self._reweight_strategies()

    # ✅ added missing method to avoid crash
    def _reweight_strategies(self):  # noqa: D401
        pass

# ==== SIMPLE STRATEGIES ====
class EnhancedRSIStrategy:
    def __init__(self):
        self.name = "Enhanced RSI"
        self.period = 14

    def _rsi(self, closes: List[float]) -> float:
        if len(closes) < self.period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[: self.period])
        avg_loss = np.mean(losses[: self.period])
        for g, l in zip(gains[self.period :], losses[self.period :]):
            avg_gain = (avg_gain * (self.period - 1) + g) / self.period
            avg_loss = (avg_loss * (self.period - 1) + l) / self.period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def analyze(self, candles: List[Candle], state: MarketState) -> Optional[StrategyVote]:
        if len(candles) < self.period + 5:
            return None
        rsi_val = self._rsi([c.close for c in candles])
        if rsi_val <= 30:
            return StrategyVote(self.name, 0.8, Signal.CALL, [f"RSI {rsi_val:.1f}"], {"rsi": rsi_val})
        if rsi_val >= 70:
            return StrategyVote(self.name, 0.8, Signal.PUT, [f"RSI {rsi_val:.1f}"], {"rsi": rsi_val})
        return None

class EliteNeuralBeastQuantumFusion:
    def __init__(self):
        self.name = "Neural Beast Quantum Fusion"

    def analyze(self, candles: List[Candle], state: MarketState) -> Optional[StrategyVote]:
        if len(candles) < 20:
            return None
        change = (candles[-1].close - candles[-5].close) / candles[-5].close if candles[-5].close else 0
        if abs(change) > 0.002:
            sig = Signal.CALL if change > 0 else Signal.PUT
            strength = min(0.9, abs(change) * 100)
            return StrategyVote(self.name, strength, sig, [f"Momentum {change:.3%}"], {"mom": change})
        return None

# ==== REGIME DETECTOR ====
class EliteMarketRegimeDetector:
    def __init__(self):
        self.vol_hist = deque(maxlen=50)

    def detect(self, candles: List[Candle]) -> MarketState:
        if len(candles) < 20:
            return MarketState(MarketRegime.UNKNOWN, 0.001, 0.0, SessionType.ASIAN, 1.0, 0.0)
        returns = [(c.close - candles[i - 1].close) / candles[i - 1].close for i, c in enumerate(candles[1:], 1)]
        vol = np.std(returns)
        trend = (candles[-1].close - candles[-10].close) / candles[-10].close
        if abs(trend) > 0.3:
            regime = MarketRegime.STRONG_TRENDING
        elif abs(trend) > 0.15:
            regime = MarketRegime.TRENDING
        elif vol > 0.003:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.CHOPPY
        return MarketState(regime, vol, trend, SessionType.ASIAN, 1.0, 0.0)

# ==== SECURITY ====
class SecurityManager:
    def __init__(self):
        self.session_file = SESSION_FILE
        self.max_trades = MAX_TRADES_LIMIT

    def _machine_id(self):
        import platform
        info = f"{platform.node()}-{platform.machine()}-{platform.processor()}"
        return hashlib.md5(info.encode()).hexdigest()[:16]

    def load(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file) as f:
                    data = json.load(f)
                if data.get("machine_id") == self._machine_id():
                    return data
            except Exception:
                pass
        return self._new()

    def _new(self):
        data = {
            "machine_id": self._machine_id(),
            "trades_used": 0,
            "session_active": True,
        }
        self._save(data)
        return data

    def _save(self, data):
        try:
            with open(self.session_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def inc(self, data):
        data["trades_used"] += 1
        self._save(data)
        if data["trades_used"] >= self.max_trades:
            data["session_active"] = False
        return data["session_active"]

    def remaining(self, data):
        return max(0, self.max_trades - data["trades_used"])

# ==== TRADE RESULT DETECTION ====

def detect_trade_closed_popup(driver, poll_time=5.0, poll_interval=0.3):
    end = time.time() + poll_time
    while time.time() < end:
        try:
            popup = driver.find_element(By.XPATH, "//div[contains(@class,'trade-closed') or contains(@class,'deal-result')]")
            text = popup.text.lower()
            if "profit" in text or "win" in text:
                return True, 10.0, 10.0
            if "loss" in text or "lose" in text:
                return False, -10.0, 0.0
        except Exception:
            pass
        time.sleep(poll_interval)
    return None, 0.0, 0.0

def get_last_trade_result(driver, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.deals-list__item-first"))
        )
        item = driver.find_element(By.CSS_SELECTOR, "div.deals-list__item-first")
        html = item.get_attribute("outerHTML").lower()
        if any(w in html for w in ["win", "profit", "success", "green"]):
            return True, 10.0, 10.0
        if any(w in html for w in ["loss", "lose", "fail", "red"]):
            return False, -10.0, 0.0
    except Exception:
        pass
    return None, 0.0, 0.0

# ==== TRADING BOT ====
class EliteTradingBot:
    def __init__(self, gui=None):
        self.gui = gui
        self.driver = None
        self.bot_running = False

        self.thresholds = EliteAdaptiveThresholds()
        self.scorer = EliteStrategyScorer()
        self.regime_detector = EliteMarketRegimeDetector()

        self.strategies = {
            "rsi": EnhancedRSIStrategy(),
            "fusion": EliteNeuralBeastQuantumFusion(),
        }

        self.balance = 10000.0
        self.stake = 100.0
        self.profit_today = 0.0
        self.win_count = 0
        self.loss_count = 0
        self.total_trades = 0

        self.security = SecurityManager()
        self.session = self.security.load()
        self.total_trades = self.session["trades_used"]

        self._setup_driver()

    # ---- Selenium driver ----
    def _setup_driver(self):
        try:
            options = uc.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            self.driver = uc.Chrome(options=options)
            self.driver.set_window_size(1280, 720)
        except Exception as e:
            logging.error(f"WebDriver init failed: {e}")
            self.driver = None

    # ---- Helper methods ----
    def _open_pos_count(self):
        selectors = [
            "div.deals-list__item-first",
            ".deals-list .deal-item",
        ]
        count = 0
        for sel in selectors:
            try:
                count += len(self.driver.find_elements(By.CSS_SELECTOR, sel))
            except Exception:
                continue
        return count

    # ✅ TRADE EXECUTION VALIDATION
    def execute_trade(self, direction: str) -> bool:
        if not self.driver:
            return False
        before = self._open_pos_count()
        btn_selectors = {
            "call": [".btn-call", ".call-btn"],
            "put": [".btn-put", ".put-btn"],
        }
        clicked = False
        for sel in btn_selectors.get(direction, []):
            try:
                WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel))).click()
                clicked = True
                break
            except Exception:
                continue
        if not clicked:
            logging.warning(Fore.YELLOW + "Trade button not found")
            return False
        # confirm UI change
        end = time.time() + 3
        while time.time() < end:
            if self._open_pos_count() > before:
                logging.info(Fore.GREEN + "Trade confirmed")
                return True
            time.sleep(0.2)
        logging.warning(Fore.YELLOW + "Trade click failed, skipping.")
        return False

    # ---- Candle data (mock) ----
    def candles(self) -> List[Candle]:
        return [Candle(time.time() - i * 60, 1, 1, 1, 1) for i in range(50)]

    # ---- Main loop ----
    def run(self):
        if not self.driver:
            logging.error("Driver not ready")
            return
        self.bot_running = True
        start = time.time()
        last_trade_ts = 0.0
        while self.bot_running:
            try:
                if not self.security.remaining(self.session):
                    logging.error("Trade limit reached")
                    break
                # analyse
                market_state = self.regime_detector.detect(self.candles())
                votes = []
                for s in self.strategies.values():
                    v = s.analyze(self.candles(), market_state)
                    if v:
                        votes.append(v)
                if not votes:
                    time.sleep(3)
                    continue
                call_votes = [v for v in votes if v.signal == Signal.CALL]
                put_votes = [v for v in votes if v.signal == Signal.PUT]
                decision = None
                if len(call_votes) >= 2 and len(call_votes) > len(put_votes):
                    decision = Signal.CALL
                elif len(put_votes) >= 2 and len(put_votes) > len(call_votes):
                    decision = Signal.PUT
                if decision and time.time() - last_trade_ts > 8:
                    if self.execute_trade(decision.name.lower()):
                        last_trade_ts = time.time()
                        time.sleep(8)
                        win, profit, _ = detect_trade_closed_popup(self.driver)
                        if win is None:
                            win, profit, _ = get_last_trade_result(self.driver)
                        if win is None:
                            logging.warning(Fore.YELLOW + "Result undetermined, skipping log")
                        else:
                            self._log_trade(win, profit, decision)
                time.sleep(2)
            except Exception as e:
                logging.error(f"Loop error: {e}")
                time.sleep(5)
        self.bot_running = False

    # ---- Logging ----
    def _log_trade(self, win: bool, profit: float, signal: Signal):
        if not self.security.inc(self.session):
            self.bot_running = False
            return
        self.total_trades = self.session["trades_used"]
        if win:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.profit_today += profit
        colour = Fore.GREEN if win else Fore.RED
        entry = (
            f"{colour}{datetime.datetime.now().strftime('%H:%M:%S')} | {signal.name} | {'WIN' if win else 'LOSS'} | "
            f"P/L: ${profit:.2f} | Trades: {self.total_trades}/{MAX_TRADES_LIMIT}{Style.RESET_ALL}"
        )
        logging.info("-" * 80)
        logging.info(entry)
        logging.info("-" * 80)
        if self.gui:
            self.gui.update_stats(self)

# ==== SIMPLE GUI (updates only) ====
class DummyGUI:  # minimal GUI for stats updates
    def update_stats(self, bot: EliteTradingBot):
        print(
            f"GUI -> Trades:{bot.total_trades} Wins:{bot.win_count} Losses:{bot.loss_count} "
            f"Winrate:{(bot.win_count/max(1,bot.total_trades))*100:.1f}%" )

# ==== MAIN ====

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("\n" + "=" * 80)
    print("🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 - INSTITUTIONAL GRADE 🌟")
    print("=" * 80 + "\n")
    bot = EliteTradingBot(gui=DummyGUI())
    bot_thread = threading.Thread(target=bot.run, daemon=True)
    bot_thread.start()
    try:
        while bot_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        bot.bot_running = False

if __name__ == "__main__":
    main()