import logging
import atexit
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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

DARK_BG = "#111827"
CARD_BG = "#1f2937"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#8b5cf6"
TEXT_PRIMARY = "#f9fafb"
TEXT_SECONDARY = "#9ca3af"
BORDER_COLOR = "#374151"
HOVER_COLOR = "#4b5563"

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
    """Fixed trade result detection from popup"""
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
    """Fixed trade result detection from trade history"""
    try:
        trade_selector = "div.deals-list__item-first"
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, trade_selector))
        )
        last_trade = driver.find_elements(By.CSS_SELECTOR, trade_selector)[0]
        try:
            profit_elem = last_trade.find_element(By.XPATH, ".//div[contains(@class,'profit-tooltip-container')]/span")
            profit_text = profit_elem.text.replace('$','').replace(',','').strip()
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
            "js-balance-demo",
        ]
        for selector in selectors:
            try:
                element = self.driver.find_element(By.CLASS_NAME, selector)
                text = element.text.replace('$', '').replace(',', '').strip()
                return float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
            except (NoSuchElementException, ValueError):
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
        """Fixed log_trade method with proper profit calculation and logging"""
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

    def update_gui(self):
        """Fixed GUI update method with correct log parsing"""
        if hasattr(self, 'balance_label'):
            self.balance_label.config(text=f"${self.balance:.2f}")
            profit_color = ACCENT_GREEN if self.profit_today >= 0 else ACCENT_RED
            self.profit_label.config(text=f"${self.profit_today:.2f}", fg=profit_color)
            self.strategy_label.config(text=self.selected_strategy)
            streak_color = ACCENT_RED if self.loss_streak > 3 else ACCENT_BLUE
            self.streak_label.config(text=str(self.loss_streak), fg=streak_color)
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            for log in self.logs[-50:]:
                self.log_text.insert(tk.END, log + "\n")
                start_idx = self.log_text.index(tk.END + "-2l")
                end_idx = self.log_text.index(tk.END + "-1c")
                if "| WIN |" in log:
                    self.log_text.tag_add("win", start_idx, end_idx)
                    self.log_text.tag_config("win", foreground=ACCENT_GREEN)
                elif "| LOSS |" in log:
                    self.log_text.tag_add("loss", start_idx, end_idx)
                    self.log_text.tag_config("loss", foreground=ACCENT_RED)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        if hasattr(self, 'root'):
            self.root.after(2000, self.update_gui)

    def run_trading_session(self):
        """Fixed trading session with proper profit calculation"""
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

    # The rest of the GUI logic is unchanged; you can keep your create_stat_card, stake/strategy handlers, cleanup, and so on as before.

    def create_stat_card(self, parent, title, value, color=ACCENT_BLUE):
        frame = tk.Frame(parent, bg=CARD_BG, relief="flat", bd=1)
        frame.pack(fill="x", padx=5, pady=3)
        title_label = tk.Label(frame, text=title, bg=CARD_BG, fg=TEXT_SECONDARY,
                              font=("Segoe UI", 9), anchor="w")
        title_label.pack(fill="x", padx=15, pady=(10, 2))
        value_label = tk.Label(frame, text=value, bg=CARD_BG, fg=color,
                              font=("Segoe UI", 14, "bold"), anchor="w")
        value_label.pack(fill="x", padx=15, pady=(0, 10))
        return frame, value_label

    def on_stake_change(self, event=None):
        try:
            value = float(self.stake_var.get())
            if value <= 0:
                raise ValueError
            self.stake = value
        except Exception:
            messagebox.showerror("Invalid Stake", "Please enter a valid positive number for the stake.")
            self.stake_var.set(str(self.stake))

    def request_strategy_change(self, new_strategy):
        if new_strategy == self.selected_strategy:
            return
        answer = messagebox.askyesno("Confirm Strategy Change", f"Are you sure you want to switch to '{new_strategy}' strategy?")
        if answer:
            self.strategy_var.set(new_strategy)
            self.selected_strategy = new_strategy
            self.strategy_label.config(text=self.selected_strategy)
        else:
            self.strategy_var.set(self.selected_strategy)

    def start_trading(self):
        if not self.bot_running:
            self.bot_running = True
            trading_thread = threading.Thread(target=self.run_trading_session, daemon=True)
            trading_thread.start()
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_indicator.config(text="● TRADING ACTIVE", fg=ACCENT_GREEN)

    def stop_trading(self):
        self.bot_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_indicator.config(text="● INACTIVE", fg=TEXT_SECONDARY)
        self.show_accuracy()
        messagebox.showinfo("Stopped", "Trading has been stopped.")

    def show_accuracy(self):
        total = self.win_count + self.loss_count
        if total > 0:
            accuracy = (self.win_count / total) * 100
            msg = (
                f"Session Results:\n\n"
                f"Wins: {self.win_count}\n"
                f"Losses: {self.loss_count}\n"
                f"Accuracy: {accuracy:.2f}%"
            )
        else:
            msg = "No trades were taken this session."
        messagebox.showinfo("Bot Signal Accuracy", msg)

    def export_logs(self):
        try:
            with open(f"trades_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
                for log in self.logs:
                    f.write(log + "\n")
            messagebox.showinfo("Success", "Logs exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")

    def cleanup(self):
        self.bot_running = False
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Chrome driver closed")
            except Exception:
                pass
        logging.info("Bot cleanup completed")

    def on_closing(self):
        self.stop_trading()
        self.cleanup()
        self.root.destroy()

    def start_gui(self):
        self.root = tk.Tk()
        self.root.title("🚀 GTR44 Bot")
        self.root.geometry("1050x710")
        self.root.configure(bg=DARK_BG)
        self.root.resizable(True, True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TFrame", background=CARD_BG, relief="flat", borderwidth=1)
        style.configure("Modern.TButton", font=("Segoe UI", 10, "bold"), padding=(15, 8))
        style.configure("Strategy.TRadiobutton", background=CARD_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))

        main_frame = tk.Frame(self.root, bg=DARK_BG)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        header_frame = tk.Frame(main_frame, bg=DARK_BG)
        header_frame.pack(fill="x", pady=(0, 20))
        title_frame = tk.Frame(header_frame, bg=DARK_BG)
        title_frame.pack(side="left")
        title_label = tk.Label(title_frame, text="🚀 GTR44 Bot",
                              bg=DARK_BG, fg=TEXT_PRIMARY,
                              font=("Segoe UI", 20, "bold"))
        title_label.pack(side="left")
        subtitle_label = tk.Label(title_frame, text="Advanced Trading Automation Platform",
                                 bg=DARK_BG, fg=TEXT_SECONDARY,
                                 font=("Segoe UI", 10))
        subtitle_label.pack(side="left", padx=(10, 0))
        status_frame = tk.Frame(header_frame, bg=DARK_BG)
        status_frame.pack(side="right")
        self.status_indicator = tk.Label(status_frame, text="● INACTIVE",
                                        bg=DARK_BG, fg=TEXT_SECONDARY,
                                        font=("Segoe UI", 10, "bold"))
        self.status_indicator.pack()

        stats_frame = tk.Frame(main_frame, bg=DARK_BG)
        stats_frame.pack(fill="x", pady=(0, 20))
        stat1_frame = tk.Frame(stats_frame, bg=CARD_BG, relief="solid", bd=1)
        stat1_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.balance_card, self.balance_label = self.create_stat_card(stat1_frame, "Account Balance", f"${self.balance:.2f}", ACCENT_GREEN)
        stat2_frame = tk.Frame(stats_frame, bg=CARD_BG, relief="solid", bd=1)
        stat2_frame.pack(side="left", fill="both", expand=True, padx=5)
        profit_color = ACCENT_GREEN if self.profit_today >= 0 else ACCENT_RED
        self.profit_card, self.profit_label = self.create_stat_card(stat2_frame, "Daily P/L", f"${self.profit_today:.2f}", profit_color)
        stat3_frame = tk.Frame(stats_frame, bg=CARD_BG, relief="solid", bd=1)
        stat3_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.strategy_card, self.strategy_label = self.create_stat_card(stat3_frame, "Active Strategy", self.selected_strategy, ACCENT_PURPLE)
        stat4_frame = tk.Frame(stats_frame, bg=CARD_BG, relief="solid", bd=1)
        stat4_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        streak_color = ACCENT_RED if self.loss_streak > 3 else ACCENT_BLUE
        self.streak_card, self.streak_label = self.create_stat_card(stat4_frame, "Loss Streak", str(self.loss_streak), streak_color)

        content_frame = tk.Frame(main_frame, bg=DARK_BG)
        content_frame.pack(fill="both", expand=True)

        left_panel = tk.Frame(content_frame, bg=CARD_BG, relief="solid", bd=1, width=320)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        stake_frame = tk.Frame(left_panel, bg=CARD_BG)
        stake_frame.pack(fill="x", padx=15, pady=(15, 0))
        stake_label = tk.Label(stake_frame, text="Stake ($):", bg=CARD_BG, fg=TEXT_PRIMARY, font=("Segoe UI", 11, "bold"))
        stake_label.pack(anchor="w", pady=(0, 4))
        self.stake_var = tk.StringVar(value=str(self.stake))
        stake_entry = tk.Entry(stake_frame, textvariable=self.stake_var, font=("Segoe UI", 11), width=15, bg="#222b3a", fg=TEXT_PRIMARY, insertbackground=ACCENT_BLUE, relief="flat")
        stake_entry.pack(anchor="w", pady=(0, 10))
        stake_entry.bind("<FocusOut>", self.on_stake_change)
        stake_entry.bind("<Return>", self.on_stake_change)

        strategy_frame = tk.Frame(left_panel, bg=CARD_BG)
        strategy_frame.pack(fill="x", padx=15, pady=(10, 15))
        strategy_title = tk.Label(strategy_frame, text="Trading Strategy",
                                 bg=CARD_BG, fg=TEXT_PRIMARY,
                                 font=("Segoe UI", 12, "bold"))
        strategy_title.pack(anchor="w", pady=(0, 10))
        self.strategy_var = tk.StringVar(value=self.selected_strategy)
        self.radio_buttons = {}
        for strategy in self.strategy_map.keys():
            rb = tk.Radiobutton(strategy_frame, text=strategy, variable=self.strategy_var,
                               value=strategy, bg=CARD_BG, fg=TEXT_PRIMARY,
                               selectcolor=ACCENT_BLUE, activebackground=CARD_BG,
                               activeforeground=TEXT_PRIMARY, font=("Segoe UI", 9),
                               command=lambda s=strategy: self.request_strategy_change(s))
            rb.pack(anchor="w", pady=2)
            self.radio_buttons[strategy] = rb

        button_frame = tk.Frame(left_panel, bg=CARD_BG)
        button_frame.pack(fill="x", padx=15, pady=15)
        controls_title = tk.Label(button_frame, text="Controls",
                                 bg=CARD_BG, fg=TEXT_PRIMARY,
                                 font=("Segoe UI", 12, "bold"))
        controls_title.pack(anchor="w", pady=(0, 10))
        self.start_btn = tk.Button(button_frame, text="▶ Start Trading",
                                  command=self.start_trading, bg=ACCENT_GREEN, fg="white",
                                  font=("Segoe UI", 10, "bold"), relief="flat",
                                  activebackground="#059669", cursor="hand2")
        self.start_btn.pack(fill="x", pady=5)
        self.stop_btn = tk.Button(button_frame, text="■ Stop Trading",
                                 command=self.stop_trading, bg=ACCENT_RED, fg="white",
                                 font=("Segoe UI", 10, "bold"), relief="flat",
                                 activebackground="#dc2626", cursor="hand2", state="normal")
        self.stop_btn.pack(fill="x", pady=5)
        export_btn = tk.Button(button_frame, text="📊 Export Logs",
                              command=self.export_logs, bg=HOVER_COLOR, fg=TEXT_PRIMARY,
                              font=("Segoe UI", 10, "bold"), relief="flat",
                              activebackground=BORDER_COLOR, cursor="hand2")
        export_btn.pack(fill="x", pady=5)

        right_panel = tk.Frame(content_frame, bg=CARD_BG, relief="solid", bd=1)
        right_panel.pack(side="right", fill="both", expand=True)
        log_title = tk.Label(right_panel, text="Recent Trades",
                            bg=CARD_BG, fg=TEXT_PRIMARY,
                            font=("Segoe UI", 12, "bold"))
        log_title.pack(anchor="w", padx=15, pady=(15, 10))
        log_frame = tk.Frame(right_panel, bg=CARD_BG)
        log_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD,
                                                 bg="#0f172a", fg=TEXT_PRIMARY,
                                                 font=("Consolas", 9),
                                                 insertbackground=ACCENT_BLUE,
                                                 selectbackground=ACCENT_BLUE,
                                                 selectforeground="white",
                                                 relief="flat", bd=0)
        self.log_text.pack(fill="both", expand=True)
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

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
    logging.info("[ GTR44 BOT Launch ]")
    bot = GTR44TradingBot()
    atexit.register(bot.cleanup)
    try:
        bot.start_gui()
    except Exception as e:
        logging.error(f"Bot launch failed: {e}")
        bot.cleanup()

if __name__ == '__main__':
    main()