import time
import threading
import logging
from typing import List

from utils import Candle
from executor import (
    EnhancedMiniSwingExecutor,
    vol_filter,
    confidence_filter,
    smart_pause,
    live_logger,
    ml_filter,
)
from stoploss import StopLossDrawdownManager
from config import BALANCED_1MIN_CONFIG

try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    # The bot can still be imported / unit-tested without selenium being present.
    uc = None  # type: ignore


class TradingBot:
    """Headless worker that executes signals coming from the strategy executor.
    This is a **greatly simplified** version of the original monolithic class – it
    still respects stake/TP/SL logic but omits all PocketOption specific DOM
    interaction so the module can run in a head-less CI environment."""

    def __init__(self, enhanced_executor: EnhancedMiniSwingExecutor, stake: float = 100.0, take_profit: float = 500.0, stop_loss: float = 250.0):
        self.executor = enhanced_executor
        self.stake = stake
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.running = False
        self.total_trades = 0
        self.profit_today = 0.0
        self.wins = 0
        self.losses = 0
        self.balance = 10_000.0  # start balance placeholder
        self.stop_loss_manager = StopLossDrawdownManager()
        self.stop_loss_manager.initialize_balances(self.balance)

    # ----------------------------------
    #           PUBLIC  API
    # ----------------------------------
    def run(self, candle_source):
        """Start consuming candles from an iterable / generator feeding lists of
        `Candle` objects.  The function exits once TP/SL/risk limits are met."""
        self.running = True
        last_trade_time = 0.0
        while self.running:
            try:
                candles: List[Candle] = next(candle_source)
            except StopIteration:
                break
            if not candles:
                continue

            # --- volatility / smart-pause gates ---
            if vol_filter.is_dead_zone():
                continue
            if smart_pause.is_paused(time.time(), self.total_trades):
                continue

            # --- strategy evaluation ---
            signal = self.executor.process_candles(candles)
            if not signal:
                continue
            if signal['confidence'] < confidence_filter.get_threshold():
                continue
            if not ml_filter.is_high_quality({'confidence': signal['confidence']}):
                continue

            now = time.time()
            if now - last_trade_time < 20:
                continue  # strategy-level cooldown

            # Simulate immediate expiry win-rate of 78% like original fallback
            win = (hash(f"{signal['timestamp']}") % 100) < 78
            profit = self.stake * 0.85 if win else -self.stake
            self._register_trade(signal['signal'], profit, win)
            last_trade_time = now

            if self.profit_today >= self.take_profit or self.profit_today <= -self.stop_loss:
                logging.info("Profit/Stop-loss target reached – stopping bot loop")
                break

    # ----------------------------------
    #              HELPERS
    # ----------------------------------
    def _register_trade(self, direction: str, profit: float, win: bool):
        self.total_trades += 1
        self.profit_today += profit
        self.balance += profit
        if win:
            self.wins += 1
        else:
            self.losses += 1
        # update external helpers
        confidence_filter.register_result(win)
        smart_pause.register_result(win, time.time(), self.total_trades)
        live_logger.log_trade(self.total_trades, 0, win, profit, self.stake, self.balance)
        self.executor.update_trade_result(direction, profit, win)
        self.stop_loss_manager.record_trade_result(profit)
        logging.info(f"Trade #{self.total_trades}: {'WIN' if win else 'LOSS'} {profit:+.2f} – balance {self.balance:.2f}")


class EnhancedMiniSwingTradingBot:
    """Wrapper around `TradingBot` that also spawns an internal mock candle feed
    when no live market data is available (so that unit tests & GUI demo can
    run out-of-the-box)."""

    def __init__(self):
        self.executor = EnhancedMiniSwingExecutor(BALANCED_1MIN_CONFIG)
        self.core_bot = TradingBot(self.executor)
        self.bot_thread: threading.Thread | None = None

    # ---------- PUBLIC API ---------------
    def run_trading_session(self):
        if self.bot_thread and self.bot_thread.is_alive():
            logging.warning("Trading session already running")
            return
        self.bot_thread = threading.Thread(target=self._session_worker, daemon=True)
        self.bot_thread.start()

    # -------------------------------------
    def _session_worker(self):
        logging.info("EnhancedMiniSwingTradingBot: session started")
        candle_gen = self._mock_candle_stream()
        self.core_bot.run(candle_gen)
        logging.info("EnhancedMiniSwingTradingBot: session ended")

    # -------------------------------------
    def _mock_candle_stream(self):
        """Yield synthetic candle lists indefinitely – good enough for demo / GUI."""
        import random
        price = 1.0
        history: List[Candle] = []
        while True:
            change = random.uniform(-0.0005, 0.0005)
            open_ = price
            close = price + change
            high = max(open_, close) + random.uniform(0, 0.0003)
            low = min(open_, close) - random.uniform(0, 0.0003)
            candle = Candle(time.time(), open_, high, low, close, 1.0)
            history.append(candle)
            if len(history) > 120:
                history = history[-120:]
            yield history.copy()
            price = close
            time.sleep(1)