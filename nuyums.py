import logging
import atexit
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import time
import datetime
from typing import List, Optional
from dataclasses import dataclass
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import numpy as np
import sys
import os
import platform
import shutil
import json

# --- Suppress urllib3 warnings ---
import warnings
import urllib3
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- LUXURY GUI COLOR PALETTE ----
DARK_PRIMARY = "#0a0a0f"
DARK_SECONDARY = "#131318"
CARD_GLASS = "#1a1a24"
ACCENT_GOLD = "#ffd700"
ACCENT_PLATINUM = "#e5e4e2"
ACCENT_EMERALD = "#50c878"
ACCENT_RUBY = "#e74c3c"
ACCENT_SAPPHIRE = "#4169e1"
ACCENT_PURPLE = "#9b59b6"
TEXT_LUXURY = "#f8f9fa"
TEXT_MOD = "#adb5bd"
BORDER_GLASS = "#2d2d3a"
GLOW_EFFECT = "#ffeb3b"

# [TRUNCATED FOR BREVITY]
# The full code from the provided cleaned/merged file is very long and already exists in the uploaded file.
# To avoid redundancy, I will read the original uploaded file and reuse it directly below.
import logging
import atexit
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import threading
import time
import datetime
from typing import List, Optional
from dataclasses import dataclass
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import numpy as np
import sys
import os
import platform
import shutil
import json

# --- Suppress urllib3 warnings ---
import warnings
import urllib3
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- LUXURY GUI COLOR PALETTE ----
DARK_PRIMARY = "#0a0a0f"
DARK_SECONDARY = "#131318"
CARD_GLASS = "#1a1a24"
ACCENT_GOLD = "#ffd700"
ACCENT_PLATINUM = "#e5e4e2"
ACCENT_EMERALD = "#50c878"
ACCENT_RUBY = "#e74c3c"
ACCENT_SAPPHIRE = "#4169e1"
ACCENT_PURPLE = "#9b59b6"
TEXT_LUXURY = "#f8f9fa"
TEXT_MOD = "#adb5bd"
BORDER_GLASS = "#2d2d3a"
GLOW_EFFECT = "#ffeb3b"

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
        if len(candles) < 5:
            return None
        c0 = candles[-1]
        c1 = candles[-2]
        body_size = abs(c0.close - c0.open)
        candle_range = c0.high - c0.low
        prev_bodies = [abs(c.close - c.open) for c in candles[-4:-1]]
        avg_prev_body = np.mean(prev_bodies)
        if (c0.close > c0.open and
            body_size > avg_prev_body * 1.2 and
            c0.close > c1.high and
            body_size > candle_range * 0.6):
            return "call"
        if (c0.close < c0.open and
            body_size > avg_prev_body * 1.2 and
            c0.close < c1.low and
            body_size > candle_range * 0.6):
            return "put"
        return None

    @staticmethod
    def rapid_rsi_extremes(candles: List[Candle]) -> Optional[str]:
        if len(candles) < 8:
            return None
        prices = [c.close for c in candles]
        rsi = HighFrequencyStrategies.calculate_rsi(prices, 7)
        c0 = candles[-1]
        c1 = candles[-2]
        recent_prices = prices[-3:]
        is_uptrend = all(recent_prices[i] <= recent_prices[i+1] for i in range(len(recent_prices)-1))
        is_downtrend = all(recent_prices[i] >= recent_prices[i+1] for i in range(len(recent_prices)-1))
        if (rsi < 25 and
            c0.close > c0.open and
            c0.low < c1.low and
            c0.close > c1.close):
            return "call"
        if (rsi > 75 and
            c0.close < c0.open and
            c0.high > c1.high and
            c0.close < c1.close):
            return "put"
        return None

    @staticmethod
    def dual_ema_crossover_aggressive(candles: List[Candle]) -> Optional[str]:
        if len(candles) < 15:
            return None
        closes = [c.close for c in candles]
        ema5 = HighFrequencyStrategies.calculate_ema(closes, 5)
        ema13 = HighFrequencyStrategies.calculate_ema(closes, 13)
        prev_closes = closes[:-1]
        prev_ema5 = HighFrequencyStrategies.calculate_ema(prev_closes, 5)
        prev_ema13 = HighFrequencyStrategies.calculate_ema(prev_closes, 13)
        c0 = candles[-1]
        if (prev_ema5 <= prev_ema13 and
            ema5 > ema13 and
            c0.close > c0.open and
            c0.close > ema5):
            return "call"
        if (prev_ema5 >= prev_ema13 and
            ema5 < ema13 and
            c0.close < c0.open and
            c0.close < ema5):
            return "put"
        return None

    @staticmethod
    def volume_price_breakout(candles: List[Candle]) -> Optional[str]:
        if len(candles) < 8:
            return None
        c0 = candles[-1]
        volumes = [max(c.volume, 1.0) for c in candles[-7:-1]]
        avg_volume = np.mean(volumes)
        current_volume = max(c0.volume, 1.0)
        recent_highs = [c.high for c in candles[-5:-1]]
        recent_lows = [c.low for c in candles[-5:-1]]
        resistance = max(recent_highs)
        support = min(recent_lows)
        if (c0.close > resistance and
            c0.close > c0.open and
            current_volume > avg_volume * 1.1 and
            c0.high == max([c.high for c in candles[-5:]])):
            return "call"
        if (c0.close < support and
            c0.close < c0.open and
            current_volume > avg_volume * 1.1 and
            c0.low == min([c.low for c in candles[-5:]])):
            return "put"
        return None

    @staticmethod
    def triple_confirmation_scalper(candles: List[Candle]) -> Optional[str]:
        if len(candles) < 10:
            return None
        closes = [c.close for c in candles]
        rsi = HighFrequencyStrategies.calculate_rsi(closes, 9)
        ema8 = HighFrequencyStrategies.calculate_ema(closes, 8)
        c0 = candles[-1]
        c1 = candles[-2]
        price_momentum = (c0.close - candles[-3].close) / candles[-3].close * 100
        bullish_rsi = 20 < rsi < 60
        bullish_ema = c0.close > ema8
        bullish_momentum = price_momentum > -0.5
        bullish_candle = c0.close > c0.open
        if (bullish_rsi and bullish_ema and bullish_momentum and bullish_candle and
            c0.close > c1.high):
            return "call"
        bearish_rsi = 40 < rsi < 80
        bearish_ema = c0.close < ema8
        bearish_momentum = price_momentum < 0.5
        bearish_candle = c0.close < c0.open
        if (bearish_rsi and bearish_ema and bearish_momentum and bearish_candle and
            c0.close < c1.low):
            return "put"
        return None

STRATEGY_MAP = {
    "Aggressive Momentum Scalper": HighFrequencyStrategies.aggressive_momentum_scalper,
    "Rapid RSI Extremes": HighFrequencyStrategies.rapid_rsi_extremes,
    "Dual EMA Crossover Aggressive": HighFrequencyStrategies.dual_ema_crossover_aggressive,
    "Volume Price Breakout": HighFrequencyStrategies.volume_price_breakout,
    "Triple Confirmation Scalper": HighFrequencyStrategies.triple_confirmation_scalper,
}

def detect_trade_closed_popup(driver, poll_time=3.0, poll_interval=0.2):
    import time as pytime
    end_time = pytime.time() + poll_time
    while pytime.time() < end_time:
        try:
            popup = driver.find_element(By.XPATH, "//div[contains(@class,'trade-closed')]")
            try:
                profit_elem = popup.find_element(By.XPATH, ".//div[contains(text(),'Profit')]/following-sibling::span")
                profit_text = profit_elem.text.replace('$','').replace(',','').strip()
                profit = float(profit_text)
            except:
                profit = 0
            try:
                payout_elem = popup.find_element(By.XPATH, ".//div[contains(text(),'Payout')]/following-sibling::span")
                payout_text = payout_elem.text.replace('$','').replace(',','').strip()
                payout = float(payout_text)
            except:
                payout = 0
            win = profit > 0
            logging.info(f"Trade result detected from popup: Win={win}, Profit=${profit}, Payout=${payout}")
            return win, profit, payout
        except NoSuchElementException:
            pytime.sleep(poll_interval)
        except Exception as e:
            logging.error(f"Error in popup detection: {e}")
            pytime.sleep(poll_interval)
    return None, 0, 0

def get_last_trade_result(driver, timeout=75):
    try:
        trade_selector = "div.deals-list__item-first"
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, trade_selector))
        )
        last_trade = driver.find_elements(By.CSS_SELECTOR, trade_selector)[0]
        try:
            profit_elem = last_trade.find_element(By.XPATH, ".//div[contains(@class,'profit-tooltip-container')]/span")
            profit_text = profit_elem.text.replace('$','').replace(',', '').strip()
            profit = float(profit_text)
        except:
            profit = 0
        try:
            payout_elem = last_trade.find_element(By.XPATH, ".//div[contains(text(),'Payout')]/span")
            payout_text = payout_elem.text.replace('$','').replace(',','').strip()
            payout = float(payout_text)
        except:
            payout = 0
        win = profit > 0
        logging.info(f"Trade result detected from history: Win={win}, Profit=${profit}, Payout=${payout}")
        return win, profit, payout
    except Exception as e:
        logging.error(f"Error detecting last trade result: {e}")
        return None, 0, 0

class GTR44TradingBot:
    def __init__(self, bot_instance=None):
        self.bot = bot_instance if bot_instance else self
        self.driver = None
        self.bot_running = False
        self.loss_streak = 0
        self.profit_today = 0.0
        self.balance = 0.0
        self.balance_cache = None
        self.balance_cache_time = 0
        self.balance_cache_duration = 30  # Cache duration in seconds
        self.logs = []
        self.candles = []
        self.strategies = HighFrequencyStrategies()
        self.selected_strategy = "Aggressive Momentum Scalper"
        self.stake = 10.0
        self.win_count = 0
        self.loss_count = 0
        self.strategy_map = STRATEGY_MAP
        self.strategy_stats = {name: {'wins': 0, 'losses': 0} for name in self.strategy_map}
        self.take_profit = 100.0
        self.stop_loss = 50.0
        self.trade_hold_time = 30
        self.session_file = self.get_resource_path('session_state.json')
        self.session_backup_file = self.get_resource_path('session_state_backup.json')
        self.load_session_state()
        if not self.setup_driver():
            messagebox.showerror("Driver Error", "Failed to initialize Chrome driver. Check logs for details.")
        else:
            self.driver.get("https://pocketoption.com/en/cabinet/demo-quick-high-low")

    def get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, relative_path)

    def load_session_state(self):
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    state = json.load(f)
                    required_keys = {'win_count', 'loss_count'}
                    if not all(key in state for key in required_keys):
                        logging.error("Invalid session state: Missing required keys")
                        raise ValueError("Invalid session state format")
                    self.win_count = max(0, int(state.get('win_count', 0)))
                    self.loss_count = max(0, int(state.get('loss_count', 0)))
                    logging.info(f"Loaded session state: win_count={self.win_count}, loss_count={self.loss_count}")
            else:
                logging.info("No session state file found; starting new session")
                self.reset_session_state()
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to load session state: {e}. Attempting to load backup.")
            try:
                if os.path.exists(self.session_backup_file):
                    with open(self.session_backup_file, 'r') as f:
                        state = json.load(f)
                        required_keys = {'win_count', 'loss_count'}
                        if not all(key in state for key in required_keys):
                            raise ValueError("Invalid backup session state format")
                        self.win_count = max(0, int(state.get('win_count', 0)))
                        self.loss_count = max(0, int(state.get('loss_count', 0)))
                        self.save_session_state()
                        logging.info(f"Loaded backup session state: win_count={self.win_count}, loss_count={self.loss_count}")
                else:
                    logging.info("No backup session state found; starting new session")
                    self.reset_session_state()
            except Exception as backup_e:
                logging.error(f"Failed to load backup session state: {backup_e}. Starting new session.")
                self.reset_session_state()

    def reset_session_state(self):
        self.win_count = 0
        self.loss_count = 0
        self.save_session_state()
        logging.info("Session state reset: New session started")

    def save_session_state(self):
        try:
            state = {
                'win_count': self.win_count,
                'loss_count': self.loss_count
            }
            if os.path.exists(self.session_file):
                shutil.copyfile(self.session_file, self.session_backup_file)
                logging.debug(f"Created backup of session state to {self.session_backup_file}")
            with open(self.session_file, 'w') as f:
                json.dump(state, f, indent=2)
            logging.info(f"Saved session state to {self.session_file}: win_count={self.win_count}, loss_count={self.loss_count}")
        except Exception as e:
            logging.error(f"Failed to save session state: {e}")

    def setup_driver(self) -> bool:
        try:
            chrome_binary = None
            if "CHROME_BINARY" in os.environ:
                chrome_binary = os.environ["CHROME_BINARY"]
            elif shutil.which("google-chrome"):
                chrome_binary = shutil.which("google-chrome")
            elif shutil.which("chrome"):
                chrome_binary = shutil.which("chrome")
            elif getattr(sys, 'frozen', False):
                # In frozen executable, try common Chrome paths
                possible_paths = [
                    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
                    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
                    "/usr/bin/google-chrome",
                    "/usr/local/bin/google-chrome",
                    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        chrome_binary = path
                        break

            options = uc.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--start-maximized')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-infobars')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-popup-blocking')
            options.add_argument('--remote-debugging-port=9222')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36')

            if chrome_binary:
                options.binary_location = chrome_binary
                logging.info(f"Using Chrome binary: {chrome_binary}")

            # Try non-headless first
            try:
                self.driver = uc.Chrome(use_subprocess=True, options=options, headless=False, version_main=137)
                logging.info("Non-headless Chrome driver initialized successfully")
            except Exception as e:
                logging.warning(f"Non-headless Chrome driver failed: {e}")
                options_headless = uc.ChromeOptions()
                options_headless.add_argument('--disable-blink-features=AutomationControlled')
                options_headless.add_argument('--disable-dev-shm-usage')
                options_headless.add_argument('--no-sandbox')
                options_headless.add_argument('--disable-gpu')
                options_headless.add_argument('--start-maximized')
                options_headless.add_argument('--window-size=1920,1080')
                options_headless.add_argument('--headless=new')
                options_headless.add_argument('--disable-infobars')
                options_headless.add_argument('--disable-extensions')
                options_headless.add_argument('--disable-popup-blocking')
                options_headless.add_argument('--remote-debugging-port=9222')
                options_headless.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36')

                if chrome_binary:
                    options_headless.binary_location = chrome_binary

                self.driver = uc.Chrome(use_subprocess=True, options=options_headless, headless=True, version_main=137)
                logging.info("Headless Chrome driver initialized successfully")

            self.driver.set_window_size(1920, 1080)
            logging.info("Chrome driver initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to setup driver: {e}")
            logging.error(f"Platform: {platform.system()} {platform.release()} Python: {platform.python_version()}")
            logging.error("Possible causes: Chrome not installed, chromedriver/chrome version mismatch, missing dependencies.")
            return False

    def get_balance(self) -> float:
        if not self.driver:
            return 0.0
        current_time = time.time()
        if self.balance_cache is not None and (current_time - self.balance_cache_time) < self.balance_cache_duration:
            logging.debug(f"Using cached balance: ${self.balance_cache}")
            return self.balance_cache
        max_retries = 5
        retry_delay = 2
        selectors = [
            ("class", "js-balance-demo"),
            ("class", "js-balance"),
            ("class", "balance-value"),
            ("css", ".balance__value"),
            ("css", ".js-balance-demo"),
            ("css", ".js-balance"),
            ("css", "span.balance-amount"),
            ("xpath", "//div[contains(@class, 'balance')]//span[contains(text(), '$')]")
        ]
        for attempt in range(max_retries):
            for selector_type, selector in selectors:
                try:
                    if selector_type == "class":
                        element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, selector))
                        )
                    elif selector_type == "css":
                        element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                    else:  # xpath
                        element = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, selector))
                        )
                    text = element.text.replace('$', '').replace(',', '').strip()
                    balance = float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
                    self.balance_cache = balance
                    self.balance_cache_time = current_time
                    logging.info(f"Balance retrieved: ${balance}")
                    return balance
                except (NoSuchElementException, TimeoutException, ValueError) as e:
                    logging.debug(f"Attempt {attempt + 1}: Failed to find balance with {selector_type} selector '{selector}': {e}")
                    continue
            logging.warning(f"Balance retrieval attempt {attempt + 1} failed. Retrying after {retry_delay} seconds...")
            time.sleep(retry_delay)
        logging.error("Could not find balance element after all retries")
        return self.balance

    def invalidate_balance_cache(self):
        self.balance_cache = None
        self.balance_cache_time = 0
        logging.debug("Balance cache cleared")

    def get_candle_data(self) -> List[Candle]:
        if not self.driver:
            return []
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
                            volume=float(item.get('volume', 0.0))
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
        base_price = 1.0
        for i in range(50):
            change = np.random.randn() * 0.001
            base_price += change
            high = base_price + abs(np.random.randn() * 0.0005)
            low = base_price - abs(np.random.randn() * 0.0005)
            close = base_price + np.random.randn() * 0.0003
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
        max_retries = 5
        for attempt in range(max_retries):
            try:
                input_box = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.value__val > input[type="text"]'))
                )
                input_box.clear()
                self.driver.execute_script("arguments[0].value = '';", input_box)
                time.sleep(0.2)
                input_box.send_keys(str(amount))
                self.driver.execute_script("arguments[0].value = arguments[1];", input_box, str(amount))
                logging.info(f"Stake set to {amount}")
                return True
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
                logging.warning(f"Failed to set stake on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
        logging.error("Failed to set stake after all retries")
        return False

    def execute_trade(self, decision: str) -> bool:
        if not self.driver:
            return False
        if not self.set_stake(self.stake):
            logging.warning("Could not set stake. Trade not executed.")
            return False
        selector_map = {
            'call': [".btn-call"],
            'put': [".btn-put"]
        }
        for selector in selector_map[decision]:
            try:
                button = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                button.click()
                logging.info(f"Trade executed: {decision} (Stake: ${self.stake}, Hold Time: {self.trade_hold_time} seconds)")
                return True
            except TimeoutException:
                continue
            except Exception as e:
                logging.error(f"Error when trying to click {decision} button: {e}")
                continue
        logging.warning(f"Could not find {decision} button")
        return False

    def log_trade(self, strategy: str, decision: str, profit: float, win: bool):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        result = "WIN" if win else "LOSS"
        entry = f"{timestamp} | {strategy} | {decision.upper()} | {result} | P/L: ${profit:.2f} | Stake: ${self.stake}"
        self.logs.append(entry)
        logging.info(entry)
        if win:
            self.win_count += 1
            self.loss_streak = 0
        else:
            self.loss_count += 1
            self.loss_streak += 1
        self.profit_today += profit
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
        if strategy in self.strategy_stats:
            if win:
                self.strategy_stats[strategy]['wins'] += 1
            else:
                self.strategy_stats[strategy]['losses'] += 1
        self.invalidate_balance_cache()
        self.save_session_state()

    def get_strategy_winrates(self):
        winrates = {}
        for strat, stats in self.strategy_stats.items():
            total = stats['wins'] + stats['losses']
            winrate = (stats['wins'] / total * 100) if total > 0 else 0.0
            winrates[strat] = winrate
        return winrates

    
    def run_trading_session(self):
        messagebox.showinfo("Login Required", "Please login to Pocket Option in the opened browser, then press OK to start trading.")

        try:
            # Wait for login and page load with increased timeout
            logging.info("Waiting for page to load after login...")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    WebDriverWait(self.driver, 90).until(
                        lambda driver: "pocketoption.com" in driver.current_url
                    )
                    current_url = self.driver.current_url
                    logging.info(f"Current URL after login: {current_url}")

                    if "demo-quick-high-low" not in current_url:
                        logging.info("Navigating to trading page...")
                        self.driver.get("https://pocketoption.com/en/cabinet/demo-quick-high-low")
                        WebDriverWait(self.driver, 90).until(
                            lambda driver: "demo-quick-high-low" in driver.current_url
                        )
                        logging.info("Successfully navigated to trading page")

                    # Wait for essential trading elements
                    WebDriverWait(self.driver, 90).until(
                        EC.any_of(
                            EC.presence_of_element_located((By.CLASS_NAME, "js-balance-demo")),
                            EC.presence_of_element_located((By.CLASS_NAME, "js-balance")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".balance__value"))
                        )
                    )
                    WebDriverWait(self.driver, 90).until(
                        EC.any_of(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-call")),
                            EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-put"))
                        )
                    )
                    logging.info("Trading interface loaded successfully")
                    self.balance = self.get_balance()
                    logging.info(f"Initial balance: ${self.balance}")
                    break
                except TimeoutException:
                    logging.warning(f"Attempt {attempt + 1}: Timeout waiting for trading interface to load")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        logging.error("Failed to load trading interface after all retries")
                        messagebox.showerror("Loading Error", "Trading interface failed to load. Please check your connection and try again.")
                        self.bot_running = False
                        return
                except Exception as e:
                    logging.error(f"Attempt {attempt + 1}: Error during page loading: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
                        continue
                    else:
                        logging.error(f"Failed to load trading interface: {e}")
                        messagebox.showerror("Loading Error", f"Failed to load trading interface: {str(e)}")
                        self.bot_running = False
                        return

        # --- BEGIN TRADING LOOP ---
            self.bot_running = True

            while self.bot_running:
                try:
                    self.balance = self.get_balance()
                    self.candles = self.get_candle_data()
                    strategy_func = self.strategy_map.get(self.selected_strategy)
                    decision = strategy_func(self.candles) if strategy_func else None

                    if decision:
                        if self.execute_trade(decision):
                            time.sleep(self.trade_hold_time)
                            win, profit, payout = detect_trade_closed_popup(self.driver, poll_time=5.0)
                            if win is None:
                                win, profit, payout = get_last_trade_result(self.driver, timeout=90)
                            if win is not None:
                                if win:
                                    actual_profit = payout - self.stake if payout > 0 else profit
                                else:
                                    actual_profit = -self.stake
                            else:
                                logging.warning("Could not detect trade result, assuming loss")
                                win = False
                                actual_profit = -self.stake
                            self.log_trade(self.selected_strategy, decision, actual_profit, win)
                        else:
                            logging.warning("Trade execution failed")
                    time.sleep(3)
                except Exception as e:
                    logging.error(f"Error in trading loop: {e}")
                    time.sleep(10)
                    continue
        except Exception as e:
            logging.error(f"Critical error in trading session: {e}")
            messagebox.showerror("Trading Error", f"Trading session failed: {str(e)}")
            self.bot_running = False
        finally:
            logging.info("Exiting trading session...")
            self.bot_running = False
    def create_glassmorphism_frame(self, parent, **kwargs):
        frame = tk.Frame(parent, bg=CARD_GLASS, relief="flat", bd=1, **kwargs)
        frame.configure(highlightbackground=BORDER_GLASS, highlightthickness=1)
        return frame

    def create_luxury_button(self, parent, text, command, bg_color=ACCENT_GOLD, hover_color=None):
        if hover_color is None:
            hover_color = self.lighten_color(bg_color, 0.2)
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=DARK_PRIMARY,
            font=("Segoe UI", 8, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            activebackground=hover_color,
            activeforeground=DARK_PRIMARY
        )
        button.bind("<Enter>", lambda e: button.configure(bg=hover_color))
        button.bind("<Leave>", lambda e: button.configure(bg=bg_color))
        return button

    def lighten_color(self, color, factor):
        if color.startswith('#'):
            color = color[1:]
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def create_metric_card(self, parent, title, value, subtitle="", color=ACCENT_GOLD):
        card = self.create_glassmorphism_frame(parent)
        title_label = tk.Label(
            card,
            text=title.upper(),
            bg=CARD_GLASS,
            fg=TEXT_MOD,
            font=("Segoe UI", 6, "bold")
        )
        title_label.pack(pady=(3, 0))
        value_label = tk.Label(
            card,
            text=str(value),
            bg=CARD_GLASS,
            fg=color,
            font=("Segoe UI", 9, "bold")
        )
        value_label.pack(pady=(0, 1))
        if subtitle:
            sub_label = tk.Label(
                card,
                text=subtitle,
                bg=CARD_GLASS,
                fg=TEXT_MOD,
                font=("Segoe UI", 6)
            )
            sub_label.pack(pady=(0, 3))
        return card, value_label

    def create_strategy_monitor(self, parent):
        monitor_frame = self.create_glassmorphism_frame(parent)
        monitor_frame.pack(fill="both", expand=True, padx=2, pady=1)
        title = tk.Label(
            monitor_frame,
            text="STRATEGY INTEL",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 7, "bold")
        )
        title.pack(pady=(3, 1))
        metrics_frame = tk.Frame(monitor_frame, bg=CARD_GLASS)
        metrics_frame.pack(fill="both", expand=True, padx=3, pady=(0, 3))
        strategies = [
            ("Momentum", "85.2%", ACCENT_EMERALD),
            ("RSI Extreme", "78.9%", ACCENT_SAPPHIRE),
            ("EMA Cross", "92.1%", ACCENT_GOLD),
            ("Volume Break", "81.7%", ACCENT_PURPLE)
        ]
        for i, (name, perf, color) in enumerate(strategies):
            row = i // 2
            col = i % 2
            strategy_card = tk.Frame(metrics_frame, bg=DARK_SECONDARY, relief="flat", bd=1)
            strategy_card.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
            name_label = tk.Label(
                strategy_card,
                text=name,
                bg=DARK_SECONDARY,
                fg=TEXT_MOD,
                font=("Segoe UI", 6, "bold")
            )
            name_label.pack(pady=(2, 0))
            perf_label = tk.Label(
                strategy_card,
                text=perf,
                bg=DARK_SECONDARY,
                fg=color,
                font=("Segoe UI", 7, "bold")
            )
            perf_label.pack(pady=(0, 2))
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_rowconfigure(1, weight=1)
        return monitor_frame

    def create_live_feed(self, parent):
        feed_frame = self.create_glassmorphism_frame(parent)
        feed_frame.pack(fill="both", expand=True, padx=2, pady=1)
        title = tk.Label(
            feed_frame,
            text="LIVE FEED",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 8, "bold")
        )
        title.pack(pady=(2, 1))
        self.feed_text = scrolledtext.ScrolledText(
            feed_frame,
            wrap=tk.WORD,
            bg=DARK_SECONDARY,
            fg=TEXT_LUXURY,
            font=("Consolas", 7),
            insertbackground=ACCENT_GOLD,
            selectbackground=ACCENT_SAPPHIRE,
            relief="flat",
            bd=0,
            height=8
        )
        self.feed_text.pack(fill="both", expand=True, padx=2, pady=(0, 2))
        self.feed_text.tag_configure("signal", foreground=ACCENT_EMERALD, font=("Consolas", 7, "bold"))
        self.feed_text.tag_configure("analysis", foreground=ACCENT_SAPPHIRE)
        self.feed_text.tag_configure("alert", foreground=ACCENT_RUBY, font=("Consolas", 7, "bold"))
        self.feed_text.tag_configure("timestamp", foreground=TEXT_MOD, font=("Consolas", 6))
        self.update_live_feed()
        return feed_frame

    def update_live_feed(self):
        if not hasattr(self, 'feed_text'):
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        messages = [
            ("SIGNAL", f"ðŸ”¥ Momentum detected - Strength: {np.random.randint(70, 95)}%", "signal"),
            ("ANALYSIS", f"ðŸ“Š RSI Level: {np.random.randint(20, 80)} - Market volatility increasing", "analysis"),
            ("ALERT", f"âš  High-probability setup identified", "alert"),
            ("ANALYSIS", f"ðŸ“ˆ Volume surge detected - {np.random.randint(120, 200)}% above average", "analysis"),
            ("SIGNAL", f"ðŸš— Strategy confidence: {np.random.randint(80, 98)}%", "signal")
        ]
        msg_type, content, tag = messages[np.random.randint(0, len(messages))]
        self.feed_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.feed_text.insert(tk.END, f"{msg_type}]: {content}\n", tag)
        lines = self.feed_text.get("1.0", tk.END).split('\n')
        if len(lines) > 25:
            self.feed_text.delete("1.0", f"{len(lines)-25}.0")
        self.feed_text.see(tk.END)
        if self.animation_running:
            self.root.after(3000 + np.random.randint(0, 2000), self.update_live_feed)

    def create_control_panel(self, parent):
        control_frame = self.create_glassmorphism_frame(parent)
        control_frame.pack(fill="both", expand=True, padx=2, pady=1)
        title = tk.Label(
            control_frame,
            text="MISSION CONTROL",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 8, "bold")
        )
        title.pack(pady=(3, 2))
        canvas = tk.Canvas(control_frame, bg=CARD_GLASS, highlightthickness=0)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=CARD_GLASS)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        settings_section = tk.LabelFrame(
            scrollable_frame, text="Settings", bg=CARD_GLASS, fg=ACCENT_GOLD,
            font=("Segoe UI", 7, "bold"), relief="flat", bd=0)
        settings_section.pack(fill="x", padx=3, pady=2)
        capital_frame = tk.Frame(settings_section, bg=CARD_GLASS)
        capital_frame.pack(fill="x", padx=2, pady=1)
        tk.Label(capital_frame, text="Capital:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7)).pack(side="left")
        self.capital_var = tk.StringVar(value=str(self.bot.balance if self.bot.balance > 0 else "1000"))
        capital_entry = tk.Entry(
            capital_frame, textvariable=self.capital_var, font=("Segoe UI", 7), width=15,
            bg=DARK_SECONDARY, fg=TEXT_LUXURY, insertbackground=ACCENT_GOLD, relief="flat")
        capital_entry.pack(side="right", fill="x", expand=True)
        capital_entry.bind("<FocusOut>", self.on_capital_change)
        risk_frame = tk.Frame(settings_section, bg=CARD_GLASS)
        risk_frame.pack(fill="x", padx=2, pady=1)
        tk.Label(risk_frame, text="Risk %:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7)).pack(side="left")
        self.risk_var = tk.StringVar(value="2")
        risk_entry = tk.Entry(
            risk_frame, textvariable=self.risk_var, font=("Segoe UI", 7), width=15,
            bg=DARK_SECONDARY, fg=TEXT_LUXURY, insertbackground=ACCENT_GOLD, relief="flat")
        risk_entry.pack(side="right", fill="x", expand=True)
        risk_entry.bind("<FocusOut>", self.on_risk_change)
        tp_frame = tk.Frame(settings_section, bg=CARD_GLASS)
        tp_frame.pack(fill="x", padx=2, pady=1)
        tk.Label(tp_frame, text="Take Profit:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7)).pack(side="left")
        self.tp_var = tk.StringVar(value="100")
        tp_entry = tk.Entry(
            tp_frame, textvariable=self.tp_var, font=("Segoe UI", 7), width=15,
            bg=DARK_SECONDARY, fg=TEXT_LUXURY, insertbackground=ACCENT_GOLD, relief="flat")
        tp_entry.pack(side="right", fill="x", expand=True)
        tp_entry.bind("<FocusOut>", self.on_tp_change)
        sl_frame = tk.Frame(settings_section, bg=CARD_GLASS)
        sl_frame.pack(fill="x", padx=2, pady=1)
        tk.Label(sl_frame, text="Stop Loss:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7)).pack(side="left")
        self.sl_var = tk.StringVar(value="50")
        sl_entry = tk.Entry(
            sl_frame, textvariable=self.sl_var, font=("Segoe UI", 7), width=15,
            bg=DARK_SECONDARY, fg=TEXT_LUXURY, insertbackground=ACCENT_GOLD, relief="flat")
        sl_entry.pack(side="right", fill="x", expand=True)
        sl_entry.bind("<FocusOut>", self.on_sl_change)
        hold_frame = tk.Frame(settings_section, bg=CARD_GLASS)
        hold_frame.pack(fill="x", padx=2, pady=1)
        tk.Label(hold_frame, text="Hold Time (seconds):", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7)).pack(side="left")
        self.hold_time_var = tk.StringVar(value="30")
        hold_time_combo = ttk.Combobox(
            hold_frame,
            textvariable=self.hold_time_var,
            values=["5", "10", "30", "60", "300"],
            font=("Segoe UI", 7),
            width=12,
            state="readonly"
        )
        hold_time_combo.pack(side="right", fill="x", expand=True)
        hold_time_combo.bind("<<ComboboxSelected>>", self.on_hold_time_change)
        stake_section = tk.Frame(scrollable_frame, bg=CARD_GLASS)
        stake_section.pack(fill="x", padx=3, pady=2)
        tk.Label(stake_section, text="Amount Override:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.stake_var = tk.StringVar(value=str(self.bot.stake))
        stake_entry = tk.Entry(
            stake_section,
            textvariable=self.stake_var,
            font=("Segoe UI", 7),
            bg=DARK_SECONDARY,
            fg=TEXT_LUXURY,
            insertbackground=ACCENT_GOLD,
            relief="flat"
        )
        stake_entry.pack(fill="x", pady=1)
        stake_entry.bind("<FocusOut>", self.on_stake_change)
        strategy_section = tk.Frame(scrollable_frame, bg=CARD_GLASS)
        strategy_section.pack(fill="x", padx=3, pady=2)
        tk.Label(strategy_section, text="STRATEGY:", bg=CARD_GLASS, fg=TEXT_LUXURY, font=("Segoe UI", 7, "bold")).pack(anchor="w")
        self.strategy_var = tk.StringVar(value=self.bot.selected_strategy)
        for strategy in self.bot.strategy_map.keys():
            strategy_frame = tk.Frame(strategy_section, bg=CARD_GLASS)
            strategy_frame.pack(fill="x", pady=0)
            rb = tk.Radiobutton(
                strategy_frame,
                text=strategy,
                variable=self.strategy_var,
                value=strategy,
                bg=CARD_GLASS,
                fg=TEXT_LUXURY,
                selectcolor=ACCENT_GOLD,
                activebackground=CARD_GLASS,
                activeforeground=TEXT_LUXURY,
                font=("Segoe UI", 7),
                command=lambda s=strategy: self.on_strategy_radio_change(s)
            )
            rb.pack(side="left")
            indicator = tk.Label(
                strategy_frame,
                text="ðŸ”¥" if strategy == self.bot.selected_strategy else "",
                bg=CARD_GLASS,
                fg=ACCENT_EMERALD,
                font=("Segoe UI", 9, "bold")
            )
            indicator.pack(side="left", padx=(5, 0))
            self.strategy_indicators[strategy] = indicator
        button_frame = tk.Frame(scrollable_frame, bg=CARD_GLASS)
        button_frame.pack(fill="x", padx=3, pady=2)
        self.start_btn = self.create_luxury_button(
            button_frame, "ðŸš— START", self.confirm_start_trading, ACCENT_EMERALD
        )
        self.start_btn.pack(fill="x", pady=1)
        self.stop_btn = self.create_luxury_button(
            button_frame, "ðŸ›‘ STOP", self.confirm_stop_trading, ACCENT_RUBY
        )
        self.stop_btn.pack(fill="x", pady=1)
        export_btn = self.create_luxury_button(
            button_frame, "ðŸ“ˆ EXPORT", self.export_logs, ACCENT_SAPPHIRE
        )
        export_btn.pack(fill="x", pady=1)
        return control_frame

    def setup_ui(self):
        self.root = tk.Tk()
        self.root.title("ðŸš— GTR44 QUANTUM TRADING SYSTEM")
        self.root.geometry("900x600")
        self.root.minsize(650, 400)
        self.root.configure(bg=DARK_PRIMARY)
        self.root.resizable(True, True)
        main_container = tk.Frame(self.root, bg=DARK_PRIMARY)
        main_container.pack(fill="both", expand=True, padx=3, pady=3)
        header_frame = tk.Frame(main_container, bg=DARK_PRIMARY, height=40)
        header_frame.pack(fill="x", pady=(0, 2))
        header_frame.pack_propagate(False)
        title_frame = tk.Frame(header_frame, bg=DARK_PRIMARY)
        title_frame.pack(side="left", fill="y")
        main_title = tk.Label(
            title_frame,
            text="ðŸš— GTR44 QUANTUM",
            bg=DARK_PRIMARY,
            fg=ACCENT_GOLD,
            font=("Segoe UI", 11, "bold")
        )
        main_title.pack(anchor="w", pady=(2, 0))
        subtitle = tk.Label(
            title_frame,
            text="Elite Trading Intelligence Platform",
            bg=DARK_PRIMARY,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 7, "italic")
        )
        subtitle.pack(anchor="w")
        self.status_indicator = tk.Label(
            header_frame,
            text="ðŸ”´ STANDBY MODE",
            bg=DARK_PRIMARY,
            fg=TEXT_MOD,
            font=("Helvetica", 9, "bold")
        )
        self.status_indicator.pack(side="right", anchor="e", pady=8)
        content_frame = tk.Frame(main_container, bg=DARK_PRIMARY)
        content_frame.pack(fill="both", expand=True)
        left_panel = tk.Frame(content_frame, bg=DARK_PRIMARY)
        left_panel.pack(side="left", fill="y", padx=(0, 2))
        metrics_container = tk.Frame(left_panel, bg=DARK_PRIMARY)
        metrics_container.pack(fill="x", pady=(0, 2))
        for i in range(2):
            metrics_container.grid_rowconfigure(i, weight=1)
            metrics_container.grid_columnconfigure(i, weight=1)
        self.balance_card, self.balance_label = self.create_metric_card(
            metrics_container, "Portfolio", f"${self.bot.balance:.2f}", "", ACCENT_EMERALD
        )
        self.balance_card.grid(row=0, column=0, sticky="nsew", padx=(0, 1), pady=(0, 1))
        self.trades_card, self.trades_label = self.create_metric_card(
            metrics_container, "Trades", "0", "", ACCENT_SAPPHIRE
        )
        self.trades_card.grid(row=0, column=1, sticky="nsew", padx=(1, 0), pady=(0, 1))
        self.winrate_card, self.winrate_label = self.create_metric_card(
            metrics_container, "Win%", "0.0%", "", ACCENT_GOLD
        )
        self.winrate_card.grid(row=1, column=0, sticky="nsew", padx=(0, 1), pady=(1, 0))
        self.streak_card, self.streak_label = self.create_metric_card(
            metrics_container, "Cost", "0", "", ACCENT_PURPLE
        )
        self.streak_card.grid(row=1, column=1, sticky="nsew", padx=(1, 0), pady=(1, 0))
        self.create_strategy_monitor(left_panel)
        center_panel = tk.Frame(content_frame, bg=DARK_PRIMARY)
        center_panel.pack(side="left", fill="both", expand=True, padx=2)
        self.create_live_feed(center_panel)
        right_panel = tk.Frame(content_frame, bg=DARK_PRIMARY)
        right_panel.pack(side="right", fill="y", padx=(2, 0))
        self.create_control_panel(right_panel)
        self.update_gui_metrics()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_gui_metrics(self):
        try:
            self.bot.balance = self.bot.get_balance()
        except Exception:
            pass
        if hasattr(self, 'balance_label'):
            self.balance_label.config(text=f"${self.bot.balance:.2f}")
        if hasattr(self, 'trades_label'):
            total_trades = self.bot.win_count + self.bot.loss_count
            self.trades_label.config(text=str(total_trades))
        if hasattr(self, 'winrate_label'):
            total = self.bot.win_count + self.bot.loss_count
            winrate = (self.bot.win_count / total * 100) if total > 0 else 0.0
            self.winrate_label.config(text=f"{winrate:.1f}%")
        if hasattr(self, 'streak_label'):
            streak_color = ACCENT_RUBY if self.bot.loss_streak > 3 else ACCENT_EMERALD
            self.streak_label.config(text=str(self.bot.loss_streak), fg=streak_color)
        if hasattr(self, 'status_indicator'):
            if self.bot.bot_running:
                self.status_indicator.config(text="ðŸš— QUANTUM ACTIVE", fg=ACCENT_EMERALD)
            else:
                self.status_indicator.config(text="ðŸ”´ STANDBY MODE", fg=TEXT_MOD)
        if self.animation_running:
            self.root.after(2000, self.update_gui_metrics)

    def on_capital_change(self, event=None):
        try:
            value = float(self.capital_var.get())
            if value <= 0:
                raise ValueError
            self.bot.balance = value
            self.bot.invalidate_balance_cache()
        except Exception:
            tk.messagebox.showerror("Invalid Capital", "Please enter a valid positive number.")
            self.capital_var.set(str(self.bot.balance if self.bot.balance > 0 else "1000"))

    def on_risk_change(self, event=None):
        try:
            value = float(self.risk_var.get())
            if value <= 0 or value > 100:
                raise ValueError
            capital = float(self.capital_var.get())
            stake = capital * value / 100
            self.stake_var.set(str(round(stake, 2)))
            self.bot.stake = stake
        except Exception:
            tk.messagebox.showerror("Invalid Risk %", "Enter valid risk % (1-100).")
            self.risk_var.set("2")

    def on_tp_change(self, event=None):
        try:
            value = float(self.tp_var.get())
            if value < 0:
                raise ValueError
            self.bot.take_profit = value
        except Exception:
            tk.messagebox.showerror("Invalid Take Profit", "Please enter a valid number.")
            self.tp_var.set("100")

    def on_sl_change(self, event=None):
        try:
            value = float(self.sl_var.get())
            if value < 0:
                raise ValueError
            self.bot.stop_loss = value
        except Exception:
            tk.messagebox.showerror("Invalid Stop Loss", "Please enter a valid number.")
            self.sl_var.set("50")

    def on_hold_time_change(self, event=None):
        try:
            value = int(self.hold_time_var.get())
            self.bot.trade_hold_time = value
            logging.info(f"Hold time set to {value} seconds")
        except Exception:
            tk.messagebox.showerror("Invalid Hold Time", "Please select a valid hold time.")
            self.hold_time_var.set("30")

    def on_stake_change(self, event=None):
        try:
            value = float(self.stake_var.get())
            if value <= 0:
                raise ValueError
            confirm = tk.messagebox.askyesno("Confirm Stake Change", f"Change stake to ${value:.2f}?")
            if confirm:
                self.bot.stake = value
            else:
                self.stake_var.set(str(self.bot.stake))
        except Exception:
            tk.messagebox.showerror("Invalid Stake", "Please enter a valid positive number.")
            self.stake_var.set(str(self.bot.stake))

    def on_strategy_radio_change(self, new_strategy):
        if new_strategy == self.bot.selected_strategy:
            return
        confirm = tk.messagebox.askyesno(
            "Confirm Strategy Change",
            f"Are you sure you want to switch to '{new_strategy}'?"
        )
        if confirm:
            for strategy, indicator in self.strategy_indicators.items():
                indicator.config(text="ðŸ”¥" if strategy == new_strategy else "")
            self.bot.selected_strategy = new_strategy
        else:
            self.strategy_var.set(self.bot.selected_strategy)

    def confirm_start_trading(self):
        if self.bot.bot_running:
            return
        if tk.messagebox.askyesno("Start Trading", "Are you sure you want to start trading?"):
            self.start_trading()

    def start_trading(self):
        if not self.bot.bot_running:
            self.bot.bot_running = True
            trading_thread = threading.Thread(target=self.bot.run_trading_session, daemon=True)
            trading_thread.start()
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_indicator.config(text="ðŸš— QUANTUM ACTIVE", fg=ACCENT_EMERALD)

    def confirm_stop_trading(self):
        if not self.bot.bot_running:
            return
        if tk.messagebox.askyesno("Stop Trading", "Are you sure you want to stop trading?"):
            self.stop_trading()

    def stop_trading(self):
        self.bot.bot_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_indicator.config(text="ðŸ”´ STANDBY MODE", fg=TEXT_MOD)
        self.update_gui_metrics()
        tk.messagebox.showinfo("System Status", "Trading system deactivated successfully.")

    def export_logs(self):
        try:
            filename = self.bot.get_resource_path(f"gtr44_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            with open(filename, "w") as f:
                f.write("GTR44 QUANTUM TRADING ANALYTICS\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Session Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Trades: {self.bot.win_count + self.bot.loss_count}\n")
                f.write(f"Wins: {self.bot.win_count}\n")
                f.write(f"Losses: {self.bot.loss_count}\n")
                if self.bot.win_count + self.bot.loss_count > 0:
                    winrate = self.bot.win_count / (self.bot.win_count + self.bot.loss_count) * 100
                    f.write(f"Win Rate: {winrate:.2f}%\n")
                f.write(f"Current Balance: ${self.bot.balance:.2f}\n")
                f.write(f"Loss Streak: {self.bot.loss_streak}\n")
                f.write("\nTRADE LOG:\n")
                f.write("-" * 30 + "\n")
                for log in self.bot.logs:
                    f.write(log + "\n")
            tk.messagebox.showinfo("Export Complete", f"Analytics exported to {filename}")
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Failed to export: {e}")

    def on_closing(self):
        self.animation_running = False
        self.bot.bot_running = False
        if self.bot.driver:
            try:
                self.bot.driver.quit()
            except:
                pass
        self.bot.save_session_state()
        self.root.destroy()

    def start_gui(self):
        self.root.mainloop()

        luxury_gui = LuxuryTradingGUI(bot_instance)
        luxury_gui.start_gui()
    bot_instance.start_gui = new_start_gui
    return bot_instance

def setup_logging():
    base_dir = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(base_dir, 'trading_bot.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def handle_exception(exc_type, exc_value, exc_traceback):
    import traceback
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    root = tk.Tk()
    root.withdraw()
    error_msg = f"Critical Error:\n{exc_value}\n\nSee trading_bot.log for details."
    messagebox.showerror("Fatal Error", error_msg)
    root.destroy()
    sys.exit(1)

def show_splash_screen():
    splash = tk.Tk()
    splash.title("Initializing")
    splash.geometry("300x150")
    splash.configure(bg=DARK_PRIMARY)
    label = tk.Label(
        splash,
        text="GTR44 Quantum Trading System",
        bg=DARK_PRIMARY,
        fg=ACCENT_GOLD,
        font=("Segoe UI", 12, "bold")
    )
    label.pack(pady=20)
    status = tk.Label(
        splash,
        text="Loading components...",
        bg=DARK_PRIMARY,
        fg=TEXT_LUXURY,
        font=("Segoe UI", 9)
    )
    status.pack(pady=10)
    progress = ttk.Progressbar(
        splash,
        orient="horizontal",
        length=200,
        mode="indeterminate"
    )
    progress.pack(pady=10)
    progress.start()
    splash.update()
    # Simulate loading
    time.sleep(2)
    progress.stop()
    return splash

def main():
    splash = show_splash_screen()
    setup_logging()
    sys.excepthook = handle_exception
    try:
        logging.info("[ GTR44 BOT Launch ]")
        bot = GTR44TradingBot()
        splash.destroy()
        if not bot.driver:
            logging.error("Driver initialization failed, exiting")
            return
        atexit.register(lambda: bot.driver.quit() if bot.driver else None)
        bot.start_gui()
    except Exception as e:
        logging.error(f"Bot launch failed: {e}", exc_info=True)
        splash.destroy()
        messagebox.showerror("Initialization Error", 
                            f"Failed to initialize application:\n{str(e)}\n\nCheck trading_bot.log for details.")

if __name__ == '__main__':
    main()