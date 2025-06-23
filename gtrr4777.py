import logging
import atexit
import tkinter as tk
from tkinter import messagebox, scrolledtext
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

# --- Suppress urllib3 "connection pool is full" warnings ---
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
TEXT_MUTED = "#adb5bd"
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
    def __init__(self):
        self.driver = None
        self.bot_running = False
        self.loss_streak = 0
        self.profit_today = 0.0
        self.balance = 0.0
        self.logs = []
        self.candles = []
        self.strategies = HighFrequencyStrategies()
        self.selected_strategy = "Aggressive Momentum Scalper"
        self.stake = 10.0
        self.win_count = 0
        self.loss_count = 0
        self.strategy_map = STRATEGY_MAP
        # Track wins/losses per strategy
        self.strategy_stats = {name: {'wins': 0, 'losses': 0} for name in self.strategy_map}
        self.setup_driver()
        self.driver.get("https://pocketoption.com/en/cabinet/demo-quick-high-low")

    def setup_driver(self) -> bool:
        try:
            options = uc.ChromeOptions()
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            self.driver = uc.Chrome(use_subprocess=True, options=options)
            self.driver.set_window_size(1920, 1080)
            logging.info("Chrome driver initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to setup driver: {e}")
            return False

    def get_balance(self) -> float:
        if not self.driver:
            return 0.0
        selectors = [
            "js-balance-demo",   # Demo account balance
            "js-balance",        # Real account balance
            "balance-value",     # Fallback class for balance
        ]
        for selector in selectors:
            try:
                element = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CLASS_NAME, selector))
                )
                text = element.text.replace('$', '').replace(',', '').strip()
                return float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
            except (NoSuchElementException, TimeoutException, ValueError):
                continue
        css_selectors = [
            ".balance__value",
            ".js-balance-demo",
            ".js-balance",
        ]
        for css in css_selectors:
            try:
                element = WebDriverWait(self.driver, 2).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, css))
                )
                text = element.text.replace('$', '').replace(',', '').strip()
                return float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
            except Exception:
                continue
        logging.warning("Could not find balance element")
        return self.balance

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
        try:
            input_box = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.value__val > input[type="text"]'))
            )
            input_box.clear()
            self.driver.execute_script("arguments[0].value = '';", input_box)
            time.sleep(0.1)
            input_box.send_keys(str(amount))
            self.driver.execute_script("arguments[0].value = arguments[1];", input_box, str(amount))
            logging.info(f"Stake set to {amount}")
            return True
        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
            logging.error(f"Failed to set stake: {e}")
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
                button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                button.click()
                logging.info(f"Trade executed: {decision} (Stake: {self.stake})")
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
        # Track per-strategy stats
        if strategy in self.strategy_stats:
            if win:
                self.strategy_stats[strategy]['wins'] += 1
            else:
                self.strategy_stats[strategy]['losses'] += 1

    def get_strategy_winrates(self):
        winrates = {}
        for strat, stats in self.strategy_stats.items():
            total = stats['wins'] + stats['losses']
            winrate = (stats['wins'] / total * 100) if total > 0 else 0.0
            winrates[strat] = winrate
        return winrates

    def run_trading_session(self):
        messagebox.showinfo("Login Required", "Please login to Pocket Option in the opened browser, then press OK to start trading.")
        self.bot_running = True
        self.loss_streak = 0
        self.profit_today = 0.0
        self.win_count = 0
        self.loss_count = 0
        logging.info("Trading session started")

        session_start_time = time.time()
        session_time_limit = 2 * 60 * 60

        while self.bot_running:
            try:
                if time.time() - session_start_time >= session_time_limit:
                    self.bot_running = False
                    messagebox.showinfo("Session Complete", "2-hour trading session complete. Bot is stopping.")
                    logging.info("2-hour time limit reached - trading session stopped.")
                    break

                self.balance = self.get_balance()
                self.candles = self.get_candle_data()
                strategy_func = self.strategy_map.get(self.selected_strategy)
                decision = strategy_func(self.candles) if strategy_func else None

                if decision:
                    if self.execute_trade(decision):
                        time.sleep(58)
                        win, profit, payout = detect_trade_closed_popup(self.driver, poll_time=3.0)
                        if win is None:
                            win, profit, payout = get_last_trade_result(self.driver, timeout=75)
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
                    time.sleep(3)
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(5)
        logging.info("Exiting trading session...")

class LuxuryTradingGUI:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.root = None
        self.animation_running = True
        self.performance_metrics = {}
        self.chart_data = []
        self.strategy_winrate_labels = {}
        self.setup_luxury_gui()

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
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            activebackground=hover_color,
            activeforeground=DARK_PRIMARY
        )
        def on_enter(e):
            button.configure(bg=hover_color)
        def on_leave(e):
            button.configure(bg=bg_color)
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        return button

    def lighten_color(self, color, factor):
        if color.startswith('#'):
            color = color[1:]
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def create_metric_card(self, parent, title, value, subtitle="", color=ACCENT_GOLD):
        card = self.create_glassmorphism_frame(parent)
        card.pack(fill="both", expand=True, padx=4, pady=2)
        title_label = tk.Label(
            card, 
            text=title.upper(),
            bg=CARD_GLASS,
            fg=TEXT_MUTED,
            font=("Segoe UI", 7, "bold")
        )
        title_label.pack(pady=(7, 1))
        value_label = tk.Label(
            card,
            text=str(value),
            bg=CARD_GLASS,
            fg=color,
            font=("Segoe UI", 13, "bold")
        )
        value_label.pack(pady=(0, 1))
        if subtitle:
            sub_label = tk.Label(
                card,
                text=subtitle,
                bg=CARD_GLASS,
                fg=TEXT_MUTED,
                font=("Segoe UI", 7)
            )
            sub_label.pack(pady=(0, 5))
        return card, value_label

    def create_performance_chart(self, parent):
        # Overridden by setup_luxury_gui for compactness
        pass

    def create_strategy_winrate_panel(self, parent):
        frame = self.create_glassmorphism_frame(parent)
        frame.pack(fill="both", expand=False, padx=4, pady=(0, 5))
        title = tk.Label(
            frame,
            text="STRATEGY WINRATES",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 9, "bold")
        )
        title.pack(pady=(6, 1))
        self.strategy_winrate_labels = {}
        for strat in self.bot.strategy_map.keys():
            row = tk.Frame(frame, bg=CARD_GLASS)
            row.pack(fill="x", padx=7, pady=0)
            name = tk.Label(row, text=strat, bg=CARD_GLASS, fg=TEXT_MUTED, font=("Segoe UI", 8))
            name.pack(side="left")
            perc = tk.Label(row, text="0.0%", bg=CARD_GLASS, fg=ACCENT_GOLD, font=("Segoe UI", 8, "bold"))
            perc.pack(side="right")
            self.strategy_winrate_labels[strat] = perc
        return frame

    def update_strategy_winrates(self):
        winrates = self.bot.get_strategy_winrates()
        for strat, label in self.strategy_winrate_labels.items():
            label.config(text=f"{winrates.get(strat, 0.0):.1f}%")
        if self.animation_running:
            self.root.after(2000, self.update_strategy_winrates)

    def update_performance_chart(self):
        if not hasattr(self, 'chart_canvas'):
            return
        self.chart_canvas.delete("all")
        if len(self.chart_data) < 30:
            base_value = 1000 + (len(self.chart_data) * 7)
            noise = np.random.normal(0, 10)
            self.chart_data.append(base_value + noise)
        else:
            self.chart_data = self.chart_data[-29:] + [self.chart_data[-1] + np.random.normal(0, 10)]
        if len(self.chart_data) < 2:
            return
        width = 200
        height = 85
        for i in range(0, width, 40):
            self.chart_canvas.create_line(i, 0, i, height, fill=BORDER_GLASS, width=1)
        for i in range(0, height, 20):
            self.chart_canvas.create_line(0, i, width, i, fill=BORDER_GLASS, width=1)
        if len(self.chart_data) > 1:
            min_val = min(self.chart_data)
            max_val = max(self.chart_data)
            if max_val == min_val:
                max_val = min_val + 1
            points = []
            for i, value in enumerate(self.chart_data):
                x = (i / (len(self.chart_data) - 1)) * (width - 10) + 5
                y = height - 10 - ((value - min_val) / (max_val - min_val)) * (height - 25)
                points.extend([x, y])
            if len(points) >= 4:
                self.chart_canvas.create_line(points, fill=ACCENT_EMERALD, width=2, smooth=True)
                self.chart_canvas.create_line(points, fill=GLOW_EFFECT, width=1, smooth=True)
        if self.animation_running:
            self.root.after(2000, self.update_performance_chart)

    def create_strategy_monitor(self, parent):
        monitor_frame = self.create_glassmorphism_frame(parent)
        monitor_frame.pack(fill="both", expand=True, padx=4, pady=2)
        title = tk.Label(
            monitor_frame,
            text="STRATEGY INTEL",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 9, "bold")
        )
        title.pack(pady=(6, 2))
        metrics_frame = tk.Frame(monitor_frame, bg=CARD_GLASS)
        metrics_frame.pack(fill="both", expand=True, padx=6, pady=(0, 7))
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
            strategy_card.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            name_label = tk.Label(
                strategy_card,
                text=name,
                bg=DARK_SECONDARY,
                fg=TEXT_MUTED,
                font=("Segoe UI", 7, "bold")
            )
            name_label.pack(pady=(3, 0))
            perf_label = tk.Label(
                strategy_card,
                text=perf,
                bg=DARK_SECONDARY,
                fg=color,
                font=("Segoe UI", 9, "bold")
            )
            perf_label.pack(pady=(0, 2))
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        return monitor_frame

    def create_live_feed(self, parent):
        # Overridden by setup_luxury_gui for compactness
        pass

    def update_live_feed(self):
        if not hasattr(self, 'feed_text'):
            return
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        messages = [
            ("SIGNAL", f"📈 Momentum detected - Strength: {np.random.randint(70, 95)}%", "signal"),
            ("ANALYSIS", f"📊 RSI Level: {np.random.randint(20, 80)} - Market volatility increasing", "analysis"),
            ("ALERT", f"⚡ High-probability setup identified", "alert"),
            ("ANALYSIS", f"🔍 Volume surge detected - {np.random.randint(120, 200)}% above average", "analysis"),
            ("SIGNAL", f"🎯 Strategy confidence: {np.random.randint(80, 98)}%", "signal")
        ]
        msg_type, content, tag = messages[np.random.randint(0, len(messages))]
        self.feed_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.feed_text.insert(tk.END, f"{msg_type}: {content}\n", tag)
        lines = self.feed_text.get("1.0", tk.END).split('\n')
        if len(lines) > 25:
            self.feed_text.delete("1.0", f"{len(lines)-25}.0")
        self.feed_text.see(tk.END)
        if self.animation_running:
            self.root.after(3000 + np.random.randint(0, 2000), self.update_live_feed)

    def create_control_panel(self, parent):
        control_frame = self.create_glassmorphism_frame(parent)
        control_frame.pack(fill="x", padx=2, pady=2)

        title = tk.Label(
            control_frame,
            text="MISSION CONTROL",
            bg=CARD_GLASS,
            fg=TEXT_LUXURY,
            font=("Segoe UI", 9, "bold")
        )
        title.pack(pady=(6, 3))

        stake_section = tk.Frame(control_frame, bg=CARD_GLASS)
        stake_section.pack(fill="x", padx=6, pady=(0, 4))
        stake_label = tk.Label(
            stake_section,
            text="AMOUNT ($)",
            bg=CARD_GLASS,
            fg=TEXT_MUTED,
            font=("Segoe UI", 7, "bold")
        )
        stake_label.pack(anchor="w", pady=(0, 2))
        self.stake_var = tk.StringVar(value=str(self.bot.stake))
        stake_entry = tk.Entry(
            stake_section,
            textvariable=self.stake_var,
            font=("Segoe UI", 9),
            bg=DARK_SECONDARY,
            fg=TEXT_LUXURY,
            insertbackground=ACCENT_GOLD,
            relief="flat",
            bd=2,
            width=8
        )
        stake_entry.pack(fill="x", pady=(0, 3))
        stake_entry.bind("<FocusOut>", self.on_stake_change)
        stake_entry.bind("<Return>", self.on_stake_change)

        strategy_section = tk.Frame(control_frame, bg=CARD_GLASS)
        strategy_section.pack(fill="x", padx=6, pady=(0, 4))
        strategy_label = tk.Label(
            strategy_section,
            text="STRATEGY",
            bg=CARD_GLASS,
            fg=TEXT_MUTED,
            font=("Segoe UI", 7, "bold")
        )
        strategy_label.pack(anchor="w", pady=(0, 2))
        self.strategy_var = tk.StringVar(value=self.bot.selected_strategy)
        for strategy in self.bot.strategy_map.keys():
            rb_frame = tk.Frame(strategy_section, bg=CARD_GLASS)
            rb_frame.pack(fill="x", pady=0)
            rb = tk.Radiobutton(
                rb_frame,
                text=strategy,
                variable=self.strategy_var,
                value=strategy,
                bg=CARD_GLASS,
                fg=TEXT_LUXURY,
                selectcolor=ACCENT_GOLD,
                activebackground=CARD_GLASS,
                activeforeground=TEXT_LUXURY,
                font=("Segoe UI", 8),
                command=lambda s=strategy: self.on_strategy_radio_change(s)
            )
            rb.pack(anchor="w")

        button_frame = tk.Frame(control_frame, bg=CARD_GLASS)
        button_frame.pack(fill="x", padx=6, pady=(0, 6))
        self.start_btn = self.create_luxury_button(
            button_frame,
            "🚀 LAUNCH",
            self.confirm_start_trading,
            ACCENT_EMERALD
        )
        self.start_btn.pack(fill="x", pady=1)
        self.stop_btn = self.create_luxury_button(
            button_frame,
            "⏹ STOP",
            self.confirm_stop_trading,
            ACCENT_RUBY
        )
        self.stop_btn.pack(fill="x", pady=1)
        export_btn = self.create_luxury_button(
            button_frame,
            "📊 EXPORT",
            self.export_logs,
            ACCENT_SAPPHIRE
        )
        export_btn.pack(fill="x", pady=1)

        return control_frame

    def setup_luxury_gui(self):
        self.root = tk.Tk()
        self.root.title("💎 GTR44 QUANTUM TRADING SYSTEM")
        self.root.geometry("700x430")
        self.root.minsize(660, 400)
        self.root.configure(bg=DARK_PRIMARY)
        self.root.resizable(True, True)

        main_container = tk.Frame(self.root, bg=DARK_PRIMARY)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # --- HEADER FRAME (Banner, now much more compact) ---
        header_frame = tk.Frame(main_container, bg=DARK_PRIMARY)
        header_frame.pack(fill="x", pady=(0, 2))  # Less vertical space

        # Title at top left, now minimal fonts/padding
        title_frame = tk.Frame(header_frame, bg=DARK_PRIMARY)
        title_frame.pack(side="left", anchor="nw", padx=(0, 0))
        main_title = tk.Label(
            title_frame,
            text="💎 GTR44 QUANTUM",
            bg=DARK_PRIMARY,
            fg=ACCENT_GOLD,
            font=("Segoe UI", 11, "bold"),  # Smaller font
            padx=0,
            pady=0
        )
        main_title.pack(side="left", padx=(0, 2))
        subtitle = tk.Label(
            title_frame,
            text="Elite Trading Intelligence Platform",
            bg=DARK_PRIMARY,
            fg=TEXT_MUTED,
            font=("Segoe UI", 7, "italic"),  # Smaller font
            padx=0,
            pady=0
        )
        subtitle.pack(side="left", padx=(2, 0))

        # Top right: Status and Mission Control panel
        top_right_frame = tk.Frame(header_frame, bg=DARK_PRIMARY)
        top_right_frame.pack(side="right", anchor="ne", padx=(0, 0))

        self.status_indicator = tk.Label(
            top_right_frame,
            text="● STANDBY MODE",
            bg=DARK_PRIMARY,
            fg=TEXT_MUTED,
            font=("Segoe UI", 9, "bold")
        )
        self.status_indicator.pack(anchor="e")

        self.create_control_panel(top_right_frame)

        # --- MAIN CONTENT ---
        content_frame = tk.Frame(main_container, bg=DARK_PRIMARY)
        content_frame.pack(fill="both", expand=True)

        # Left panel (metrics, strategy monitor)
        left_panel = tk.Frame(content_frame, bg=DARK_PRIMARY, width=140)  # Less width
        left_panel.pack(side="left", fill="y", padx=(0, 2), ipadx=0)
        left_panel.pack_propagate(True)

        metrics_frame = tk.Frame(left_panel, bg=DARK_PRIMARY)
        metrics_frame.pack(fill="x", pady=(0, 2))
        row1_frame = tk.Frame(metrics_frame, bg=DARK_PRIMARY)
        row1_frame.pack(fill="x", pady=(0, 1))
        self.balance_card, self.balance_label = self.create_metric_card(
            row1_frame, "Portfolio", f"${self.bot.balance:.2f}", "", ACCENT_EMERALD
        )
        self.balance_card.pack(side="left", fill="both", expand=True, padx=(0, 1))
        self.trades_card, self.trades_label = self.create_metric_card(
            row1_frame, "Trades", "0", "", ACCENT_SAPPHIRE
        )
        self.trades_card.pack(side="right", fill="both", expand=True, padx=(1, 0))
        row2_frame = tk.Frame(metrics_frame, bg=DARK_PRIMARY)
        row2_frame.pack(fill="x")
        self.winrate_card, self.winrate_label = self.create_metric_card(
            row2_frame, "Win%", "0.0%", "", ACCENT_GOLD
        )
        self.winrate_card.pack(side="left", fill="both", expand=True, padx=(0, 1))
        self.streak_card, self.streak_label = self.create_metric_card(
            row2_frame, "Streak", "0", "", ACCENT_PURPLE
        )
        self.streak_card.pack(side="right", fill="both", expand=True, padx=(1, 0))

        self.create_strategy_monitor(left_panel)

        # Right panel (charts, winrate, live feed)
        right_panel = tk.Frame(content_frame, bg=DARK_PRIMARY)
        right_panel.pack(side="right", fill="both", expand=True)

        # Compact chart
        def _create_performance_chart_shrink(parent):
            chart_frame = self.create_glassmorphism_frame(parent)
            chart_frame.pack(fill="both", expand=True, padx=4, pady=2)
            title = tk.Label(
                chart_frame,
                text="PERFORMANCE",
                bg=CARD_GLASS,
                fg=TEXT_LUXURY,
                font=("Segoe UI", 10, "bold")
            )
            title.pack(pady=(2, 2))
            self.chart_canvas = tk.Canvas(
                chart_frame,
                width=200,
                height=85,
                bg=DARK_SECONDARY,
                highlightthickness=0
            )
            self.chart_canvas.pack(pady=(0, 2), padx=5)
            self.update_performance_chart()
            return chart_frame

        self.create_performance_chart = _create_performance_chart_shrink
        self.create_performance_chart(right_panel)
        self.create_strategy_winrate_panel(right_panel)
        self.update_strategy_winrates()

        # Compact live feed
        def _create_live_feed_shrink(parent):
            feed_frame = self.create_glassmorphism_frame(parent)
            feed_frame.pack(fill="both", expand=True, padx=4, pady=2)
            title = tk.Label(
                feed_frame,
                text="LIVE FEED",
                bg=CARD_GLASS,
                fg=TEXT_LUXURY,
                font=("Segoe UI", 10, "bold")
            )
            title.pack(pady=(2, 1))
            self.feed_text = scrolledtext.ScrolledText(
                feed_frame,
                wrap=tk.WORD,
                bg=DARK_SECONDARY,
                fg=TEXT_LUXURY,
                font=("Consolas", 8),
                insertbackground=ACCENT_GOLD,
                selectbackground=ACCENT_SAPPHIRE,
                relief="flat",
                bd=0,
                height=4
            )
            self.feed_text.pack(fill="both", expand=True, padx=4, pady=(0, 2))
            self.feed_text.tag_configure("signal", foreground=ACCENT_EMERALD, font=("Consolas", 8, "bold"))
            self.feed_text.tag_configure("analysis", foreground=ACCENT_SAPPHIRE)
            self.feed_text.tag_configure("alert", foreground=ACCENT_RUBY, font=("Consolas", 8, "bold"))
            self.feed_text.tag_configure("timestamp", foreground=TEXT_MUTED, font=("Consolas", 7))
            self.update_live_feed()
            return feed_frame

        self.create_live_feed = _create_live_feed_shrink
        self.create_live_feed(right_panel)
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
            winrate = (self.bot.win_count / total * 100) if total > 0 else 0
            self.winrate_label.config(text=f"{winrate:.1f}%")
        if hasattr(self, 'streak_label'):
            streak_color = ACCENT_RUBY if self.bot.loss_streak > 3 else ACCENT_EMERALD
            self.streak_label.config(text=str(self.bot.loss_streak), fg=streak_color)
        if self.animation_running:
            self.root.after(2000, self.update_gui_metrics)

    def on_stake_change(self, event=None):
        try:
            value = float(self.stake_var.get())
            if value <= 0:
                raise ValueError
            confirm = messagebox.askyesno("Confirm Stake Change", f"Change stake to ${value:.2f}?")
            if confirm:
                self.bot.stake = value
            else:
                self.stake_var.set(str(self.bot.stake))
        except Exception:
            messagebox.showerror("Invalid Stake", "Please enter a valid positive number.")
            self.stake_var.set(str(self.bot.stake))

    def on_strategy_radio_change(self, new_strategy):
        if new_strategy == self.bot.selected_strategy:
            return
        confirm = messagebox.askyesno(
            "Confirm Strategy Change",
            f"Are you sure you want to switch to '{new_strategy}'?"
        )
        if confirm:
            self.bot.selected_strategy = new_strategy
        else:
            self.strategy_var.set(self.bot.selected_strategy)

    def confirm_start_trading(self):
        if self.bot.bot_running:
            return
        if messagebox.askyesno("Start Trading", "Are you sure you want to start trading?"):
            self.start_trading()

    def start_trading(self):
        if not self.bot.bot_running:
            self.bot.bot_running = True
            trading_thread = threading.Thread(target=self.bot.run_trading_session, daemon=True)
            trading_thread.start()
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_indicator.config(text="● QUANTUM ACTIVE", fg=ACCENT_EMERALD)

    def confirm_stop_trading(self):
        if not self.bot.bot_running:
            return
        if messagebox.askyesno("Stop Trading", "Are you sure you want to stop trading?"):
            self.stop_trading()

    def stop_trading(self):
        self.bot.bot_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_indicator.config(text="● STANDBY MODE", fg=TEXT_MUTED)
        messagebox.showinfo("System Status", "Trading system deactivated successfully.")

    def export_logs(self):
        try:
            filename = f"gtr44_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
                f.write(f"Loss Streak: {self.bot.loss_streak}\n\n")
                f.write("TRADE LOG:\n")
                f.write("-" * 30 + "\n")
                for log in self.bot.logs:
                    f.write(log + "\n")
                # Export strategy winrates
                f.write("\nSTRATEGY WINRATES:\n")
                for strat, rate in self.bot.get_strategy_winrates().items():
                    f.write(f"{strat}: {rate:.1f}%\n")
            messagebox.showinfo("Export Complete", f"Analytics exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def on_closing(self):
        self.animation_running = False
        self.bot.bot_running = False
        if self.bot.driver:
            try:
                self.bot.driver.quit()
            except:
                pass
        self.root.destroy()

    def start_gui(self):
        self.root.mainloop()

def integrate_luxury_gui(bot_instance):
    def new_start_gui():
        luxury_gui = LuxuryTradingGUI(bot_instance)
        luxury_gui.start_gui()
    bot_instance.start_gui = new_start_gui
    return bot_instance

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logging.info("[ GTR48 BOT Launch ]")
    bot = GTR44TradingBot()
    integrate_luxury_gui(bot)
    atexit.register(lambda: bot.driver.quit() if bot.driver else None)
    try:
        bot.start_gui()
    except Exception as e:
        logging.error(f"Bot launch failed: {e}")
        if bot.driver:
            try:
                bot.driver.quit()
            except Exception:
                pass

if __name__ == '__main__':
    main()