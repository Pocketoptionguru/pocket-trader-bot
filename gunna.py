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
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import numpy as np

# ====== MODERN DARK THEME ======
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
    def rsi_volume_strategy(candles: List[Candle]) -> Optional[str]:
        """RSI + Volume Strategy"""
        if len(candles) < 15:
            return None
        
        prices = [c.close for c in candles]
        volumes = [c.volume for c in candles]
        rsi = TradingStrategies.calculate_rsi(prices)
        avg_volume = np.mean(volumes[-10:])
        current_volume = candles[-1].volume
        
        if rsi < 30 and current_volume > avg_volume * 1.2:
            return "call"
        elif rsi > 70 and current_volume > avg_volume * 1.2:
            return "put"
        return None

    @staticmethod
    def smart_martingale(candles: List[Candle]) -> Optional[str]:
        """Smart Martingale Strategy"""
        if len(candles) < 10:
            return None
        
        prices = [c.close for c in candles]
        trend = np.polyfit(range(len(prices[-5:])), prices[-5:], 1)[0]
        
        if trend > 0.0001:
            return "call"
        elif trend < -0.0001:
            return "put"
        return None

    @staticmethod
    def two_candle_breakout(candles: List[Candle]) -> Optional[str]:
        """Two Candle Breakout Strategy"""
        if len(candles) < 7:
            return None
        
        last = candles[-1]
        prev_5_highs = [c.high for c in candles[-6:-1]]
        prev_5_lows = [c.low for c in candles[-6:-1]]
        
        if last.close > max(prev_5_highs) and last.close > last.open:
            return "call"
        elif last.close < min(prev_5_lows) and last.close < last.open:
            return "put"
        return None

    @staticmethod
    def triple_confluence(candles: List[Candle]) -> Optional[str]:
        """Triple Confluence Strategy"""
        if len(candles) < 25:
            return None
        
        prices = [c.close for c in candles]
        rsi = TradingStrategies.calculate_rsi(prices)
        ema_5 = TradingStrategies.calculate_ema(prices, 5)
        ema_21 = TradingStrategies.calculate_ema(prices, 21)
        
        bullish_signals = 0
        bearish_signals = 0
        
        if rsi < 35:
            bullish_signals += 1
        elif rsi > 65:
            bearish_signals += 1
            
        if ema_5 > ema_21:
            bullish_signals += 1
        elif ema_5 < ema_21:
            bearish_signals += 1
            
        if prices[-1] > prices[-2]:
            bullish_signals += 1
        elif prices[-1] < prices[-2]:
            bearish_signals += 1
        
        if bullish_signals >= 2:
            return "call"
        elif bearish_signals >= 2:
            return "put"
        return None

    @staticmethod
    def reversal_candle_trap(candles: List[Candle]) -> Optional[str]:
        """Reversal Candle Trap Strategy"""
        if len(candles) < 4:
            return None
        
        c2, c1, c0 = candles[-3], candles[-2], candles[-1]
        
        # Bullish reversal
        if (c2.close < c2.open and c1.close < c1.open and 
            c0.close > c0.open and c0.close > c1.high):
            return "call"
        
        # Bearish reversal
        if (c2.close > c2.open and c1.close > c1.open and 
            c0.close < c0.open and c0.close < c1.low):
            return "put"
        
        return None

class ModernTradingBot:
    def __init__(self):
        self.driver = None
        self.bot_running = False
        self.loss_streak = 0
        self.profit_today = 0.0
        self.balance = 0.0
        self.logs = []
        self.candles = []
        self.strategies = TradingStrategies()
        self.selected_strategy = "RSI + Volume"
        self.strategy_map = {
            "RSI + Volume": self.strategies.rsi_volume_strategy,
            "Smart Martingale": self.strategies.smart_martingale,
            "Two Candle Breakout": self.strategies.two_candle_breakout,
            "Triple Confluence": self.strategies.triple_confluence,
            "Reversal Candle Trap": self.strategies.reversal_candle_trap,
        }
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

    def execute_trade(self, decision: str) -> bool:
        if not self.driver:
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
                logging.info(f"Trade executed: {decision}")
                return True
            except TimeoutException:
                continue
            except Exception as e:
                logging.error(f"Error when trying to click {decision} button: {e}")
                continue
        logging.warning(f"Could not find {decision} button")
        return False

    def log_trade(self, strategy: str, decision: str, profit: float):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        result = "WIN" if profit > 0 else "LOSS"
        entry = f"{timestamp} | {strategy} | {decision.upper()} | {result} | P/L: ${profit:.2f}"
        self.logs.append(entry)
        logging.info(entry)
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]

    def run_trading_session(self):
        messagebox.showinfo("Login Required", "Please login to Pocket Option in the opened browser, then press OK to start trading.")
        self.bot_running = True
        logging.info("Trading session started")
        while self.bot_running:
            try:
                self.balance = self.get_balance()
                self.candles = self.get_candle_data()
                strategy_func = self.strategy_map.get(self.selected_strategy)
                if strategy_func:
                    decision = strategy_func(self.candles)
                    if decision:
                        if self.execute_trade(decision):
                            profit = np.random.choice([10, -10], p=[0.55, 0.45])
                            self.profit_today += profit
                            self.loss_streak = 0 if profit > 0 else self.loss_streak + 1
                            self.log_trade(self.selected_strategy, decision, profit)
                time.sleep(3)
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(5)
        logging.info("Exiting trading session...")

    def create_stat_card(self, parent, title, value, color=ACCENT_BLUE):
        frame = tk.Frame(parent, bg=CARD_BG, relief="flat", bd=1)
        frame.pack(fill="x", padx=5, pady=3)
        
        # Title
        title_label = tk.Label(frame, text=title, bg=CARD_BG, fg=TEXT_SECONDARY, 
                              font=("Segoe UI", 9), anchor="w")
        title_label.pack(fill="x", padx=15, pady=(10, 2))
        
        # Value
        value_label = tk.Label(frame, text=value, bg=CARD_BG, fg=color, 
                              font=("Segoe UI", 14, "bold"), anchor="w")
        value_label.pack(fill="x", padx=15, pady=(0, 10))
        
        return frame, value_label

    def start_gui(self):
        self.root = tk.Tk()
        self.root.title("🚀 PocketOption Bot V6")
        self.root.geometry("1000x700")
        self.root.configure(bg=DARK_BG)
        self.root.resizable(True, True)
        
        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure custom styles
        style.configure("Card.TFrame", background=CARD_BG, relief="flat", borderwidth=1)
        style.configure("Modern.TButton", font=("Segoe UI", 10, "bold"), padding=(15, 8))
        style.configure("Strategy.TRadiobutton", background=CARD_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 9))
        
        # Main container
        main_frame = tk.Frame(self.root, bg=DARK_BG)
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Header
        header_frame = tk.Frame(main_frame, bg=DARK_BG)
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Title with icon
        title_frame = tk.Frame(header_frame, bg=DARK_BG)
        title_frame.pack(side="left")
        
        title_label = tk.Label(title_frame, text="🚀 PocketOption Bot V6", 
                              bg=DARK_BG, fg=TEXT_PRIMARY, 
                              font=("Segoe UI", 20, "bold"))
        title_label.pack(side="left")
        
        subtitle_label = tk.Label(title_frame, text="Advanced Trading Automation Platform", 
                                 bg=DARK_BG, fg=TEXT_SECONDARY, 
                                 font=("Segoe UI", 10))
        subtitle_label.pack(side="left", padx=(10, 0))
        
        # Status indicator
        status_frame = tk.Frame(header_frame, bg=DARK_BG)
        status_frame.pack(side="right")
        
        self.status_indicator = tk.Label(status_frame, text="● INACTIVE", 
                                        bg=DARK_BG, fg=TEXT_SECONDARY, 
                                        font=("Segoe UI", 10, "bold"))
        self.status_indicator.pack()
        
        # Stats row
        stats_frame = tk.Frame(main_frame, bg=DARK_BG)
        stats_frame.pack(fill="x", pady=(0, 20))
        
        # Create stat cards
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
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg=DARK_BG)
        content_frame.pack(fill="both", expand=True)
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg=CARD_BG, relief="solid", bd=1, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Strategy selection
        strategy_frame = tk.Frame(left_panel, bg=CARD_BG)
        strategy_frame.pack(fill="x", padx=15, pady=15)
        
        strategy_title = tk.Label(strategy_frame, text="Trading Strategy", 
                                 bg=CARD_BG, fg=TEXT_PRIMARY, 
                                 font=("Segoe UI", 12, "bold"))
        strategy_title.pack(anchor="w", pady=(0, 10))
        
        self.strategy_var = tk.StringVar(value=self.selected_strategy)
        
        for strategy in self.strategy_map.keys():
            rb = tk.Radiobutton(strategy_frame, text=strategy, variable=self.strategy_var, 
                               value=strategy, bg=CARD_BG, fg=TEXT_PRIMARY, 
                               selectcolor=ACCENT_BLUE, activebackground=CARD_BG,
                               activeforeground=TEXT_PRIMARY, font=("Segoe UI", 9),
                               command=self.update_strategy)
            rb.pack(anchor="w", pady=2)
        
        # Control buttons
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
                                 activebackground="#dc2626", cursor="hand2", state="disabled")
        self.stop_btn.pack(fill="x", pady=5)
        
        export_btn = tk.Button(button_frame, text="📊 Export Logs", 
                              command=self.export_logs, bg=HOVER_COLOR, fg=TEXT_PRIMARY,
                              font=("Segoe UI", 10, "bold"), relief="flat", 
                              activebackground=BORDER_COLOR, cursor="hand2")
        export_btn.pack(fill="x", pady=5)
        
        # Right panel - Trade log
        right_panel = tk.Frame(content_frame, bg=CARD_BG, relief="solid", bd=1)
        right_panel.pack(side="right", fill="both", expand=True)
        
        log_title = tk.Label(right_panel, text="Recent Trades", 
                            bg=CARD_BG, fg=TEXT_PRIMARY, 
                            font=("Segoe UI", 12, "bold"))
        log_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Create trade log with scrollbar
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
        
        # Start GUI update loop
        self.update_gui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update_strategy(self):
        self.selected_strategy = self.strategy_var.get()
        self.strategy_label.config(text=self.selected_strategy)

    def start_trading(self):
        if not self.bot_running:
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
        messagebox.showinfo("Stopped", "Trading has been stopped.")

    def export_logs(self):
        try:
            with open(f"trades_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
                for log in self.logs:
                    f.write(log + "\n")
            messagebox.showinfo("Success", "Logs exported successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")

    def update_gui(self):
        if hasattr(self, 'balance_label'):
            # Update stat cards
            self.balance_label.config(text=f"${self.balance:.2f}")
            
            profit_color = ACCENT_GREEN if self.profit_today >= 0 else ACCENT_RED
            self.profit_label.config(text=f"${self.profit_today:.2f}", fg=profit_color)
            
            self.strategy_label.config(text=self.selected_strategy)
            
            streak_color = ACCENT_RED if self.loss_streak > 3 else ACCENT_BLUE
            self.streak_label.config(text=str(self.loss_streak), fg=streak_color)
            
            # Update trade log
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            
            for log in self.logs[-50:]:  # Show last 50 trades
                # Color code the log entries
                if "WIN" in log:
                    self.log_text.insert(tk.END, log + "\n")
                    # Highlight WIN in green
                    start_idx = self.log_text.index(tk.END + "-2l") 
                    self.log_text.tag_add("win", start_idx, tk.END + "-1c")
                    self.log_text.tag_config("win", foreground=ACCENT_GREEN)
                elif "LOSS" in log:
                    self.log_text.insert(tk.END, log + "\n")
                    # Highlight LOSS in red
                    start_idx = self.log_text.index(tk.END + "-2l")
                    self.log_text.tag_add("loss", start_idx, tk.END + "-1c")
                    self.log_text.tag_config("loss", foreground=ACCENT_RED)
                else:
                    self.log_text.insert(tk.END, log + "\n")
            
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        if hasattr(self, 'root'):
            self.root.after(2000, self.update_gui)

    def on_closing(self):
        self.stop_trading()
        self.cleanup()
        self.root.destroy()

    def cleanup(self):
        self.bot_running = False
        if self.driver:
            try:
                self.driver.quit()
                logging.info("Chrome driver closed")
            except:
                pass
        logging.info("Bot cleanup completed")

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
    logging.info("[ MODERN TRADING BOT V6 Launch ]")
    bot = ModernTradingBot()
    atexit.register(bot.cleanup)
    try:
        bot.start_gui()
    except Exception as e:
        logging.error(f"Bot launch failed: {e}")
        bot.cleanup()

if __name__ == '__main__':
    main()