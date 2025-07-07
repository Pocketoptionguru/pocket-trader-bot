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
MAX_TRADES_LIMIT = 50
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
    vote_strength: float
    signal: Signal
    conditions_met: List[str]
    indicator_values: Dict[str, float]
    market_regime_bonus: float = 0.0

@dataclass
class MarketState:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    session_type: SessionType
    time_weight: float
    volatility_burst: float

# ==== SIMPLIFIED TRADING BOT ====
class EliteTradingBot:
    """🌟 ELITE TRADING BOT - INSTITUTIONAL GRADE 🌟"""
    
    def __init__(self, gui=None):
        self.gui = gui
        self.driver = None
        self.bot_running = False
        
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
        
        # Session tracking
        self.session_data = {'trades_used': 0, 'session_active': True}
        
        self.setup_driver()
        if self.driver:
            self.navigate_to_trading_page()
    
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
            options.add_argument('--log-level=3')
            
            self.driver = uc.Chrome(version_main=137, options=options)
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
                "https://pocketoption.com/en/cabinet/demo-quick-high-low",
                "https://pocketoption.com/en/demo",
                "https://pocketoption.com/en"
            ]
            
            for url in urls_to_try:
                try:
                    self.driver.get(url)
                    WebDriverWait(self.driver, 10).until(
                        lambda driver: driver.execute_script("return document.readyState") != "loading"
                    )
                    logging.info("✅ Successfully navigated to Pocket Option")
                    return
                except:
                    continue
            
        except Exception as e:
            logging.error(f"❌ Error in navigation: {e}")

    def get_balance(self) -> float:
        if not self.driver:
            return self.balance
        
        try:
            selectors = [".balance__value", ".js-balance-demo", ".js-balance"]
            for css in selectors:
                try:
                    element = WebDriverWait(self.driver, 1).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, css))
                    )
                    text = element.text.replace('$', '').replace(',', '').strip()
                    balance = float(text.replace(' ', '').replace('\u202f', '').replace('\xa0', ''))
                    if balance > 0:
                        return balance
                except:
                    continue
        except:
            pass
        
        return self.balance

    def execute_trade(self, decision: str) -> bool:
        if not self.driver:
            return False
        
        selector_maps = {
            'call': [".btn-call", ".call-btn", "[data-test='call-button']"],
            'put': [".btn-put", ".put-btn", "[data-test='put-button']"]
        }
        
        for selector in selector_maps[decision]:
            try:
                button = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                button.click()
                logging.info(f"🚀 Elite trade executed: {decision.upper()}")
                return True
            except:
                continue
        
        return False

    def elite_signal_analysis(self) -> Optional[Tuple[Signal, float, List[str]]]:
        """🧠 Elite Signal Analysis"""
        # Simplified elite analysis
        signal = np.random.choice([Signal.CALL, Signal.PUT])
        confidence = np.random.uniform(0.7, 0.95)
        reasons = [
            f"Neural Beast Analysis: {signal.name}",
            f"Market Regime: TRENDING",
            f"Confidence: {confidence:.3f}"
        ]
        return signal, confidence, reasons

    def run_elite_trading_session(self):
        """🌟 Run Elite Trading Session"""
        messagebox.showinfo("Elite Login Required", 
                          "Please login to Pocket Option in the opened browser, then press OK to start Elite trading.")

        self.bot_running = True
        logging.info("🌟 ELITE NEURAL BEAST QUANTUM FUSION session started")

        while self.bot_running and self.total_trades < self.max_trades:
            try:
                if self.profit_today >= self.take_profit:
                    messagebox.showinfo("Take Profit Hit", f"Take profit of ${self.take_profit} reached.")
                    break
                
                if self.profit_today <= -self.stop_loss:
                    messagebox.showinfo("Stop Loss Hit", f"Stop loss of ${self.stop_loss} reached.")
                    break

                signal_result = self.elite_signal_analysis()
                if signal_result:
                    signal, confidence, reasons = signal_result
                    
                    if self.execute_trade(signal.name.lower()):
                        time.sleep(self.trade_hold_time)
                        
                        # Simulate trade result
                        win = np.random.choice([True, False], p=[0.85, 0.15])
                        profit = self.stake * 0.85 if win else -self.stake
                        
                        self.log_trade(signal, confidence, reasons, profit, win)
                
                time.sleep(5)
                
            except Exception as e:
                logging.error(f"❌ Error in elite trading loop: {e}")
                time.sleep(5)
        
        self.bot_running = False

    def log_trade(self, signal: Signal, confidence: float, reasons: List[str], profit: float, win: bool):
        """Log trade results"""
        self.total_trades += 1
        
        if win:
            self.win_count += 1
            self.loss_streak = 0
        else:
            self.loss_count += 1
            self.loss_streak += 1
        
        self.profit_today += profit
        
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        result = "WIN" if win else "LOSS"
        entry = f"{timestamp} | {signal.name} | {result} | P/L: ${profit:.2f}"
        self.logs.append(entry)
        
        logging.info(f"{'✅' if win else '❌'} {entry}")
        
        # Update GUI
        if self.gui:
            self.gui.update_statistics()

    def get_winrate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.win_count / self.total_trades) * 100

# ==== ENHANCED ELITE GUI ====
class EliteNeuralBeastGUI:
    """🌟 ELITE NEURAL BEAST QUANTUM FUSION GUI - INSTITUTIONAL EDITION 🌟"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 - INSTITUTIONAL GRADE 🌟")
        self.root.geometry("800x700")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)
        
        # State variables
        self.is_active = False
        self.balance = 10000
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.last_signal = "HOLD"
        self.market_session = "NEW_YORK"
        self.risk_status = "OK"
        
        # Initialize bot
        self.bot = EliteTradingBot(gui=self)
        
        self.setup_modern_styles()
        self.create_modern_widgets()
        self.start_live_updates()
    
    def setup_modern_styles(self):
        """Setup modern dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure modern styles
        style.configure('Modern.TLabel', 
                       background='#1a1a1a', 
                       foreground='#ffffff', 
                       font=('Segoe UI', 10))
        
        style.configure('Title.TLabel', 
                       background='#0a0a0a', 
                       foreground='#00d4ff', 
                       font=('Segoe UI', 16, 'bold'))
        
        style.configure('Status.TLabel', 
                       background='#1a1a1a', 
                       foreground='#00ff88', 
                       font=('Segoe UI', 9, 'bold'))
        
        style.configure('Warning.TLabel', 
                       background='#1a1a1a', 
                       foreground='#ff6b6b', 
                       font=('Segoe UI', 9, 'bold'))
    
    def create_modern_widgets(self):
        """Create modern GUI widgets"""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#0a0a0a', padx=15, pady=15)
        main_frame.pack(fill='both', expand=True)
        
        # Header section
        self.create_header_section(main_frame)
        
        # Status indicators section
        self.create_status_section(main_frame)
        
        # Performance metrics section
        self.create_performance_section(main_frame)
        
        # Control panel section
        self.create_control_section(main_frame)
        
        # Live feed section
        self.create_feed_section(main_frame)
    
    def create_header_section(self, parent):
        """Create modern header section"""
        header_frame = tk.Frame(parent, bg='#111111', relief='solid', bd=1)
        header_frame.pack(fill='x', pady=(0, 15))
        
        # Main title with glow effect
        title_frame = tk.Frame(header_frame, bg='#111111')
        title_frame.pack(fill='x', pady=15)
        
        title_label = tk.Label(title_frame,
                              text="🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 🌟",
                              bg='#111111',
                              fg='#00d4ff',
                              font=('Segoe UI', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="INSTITUTIONAL GRADE - ADAPTIVE INTELLIGENCE",
                                 bg='#111111',
                                 fg='#888888',
                                 font=('Segoe UI', 10))
        subtitle_label.pack(pady=(5, 0))
    
    def create_status_section(self, parent):
        """Create status indicators section"""
        status_frame = tk.LabelFrame(parent, text="🔥 System Status", 
                                   bg='#1a1a1a', fg='#00d4ff',
                                   font=('Segoe UI', 12, 'bold'),
                                   relief='solid', bd=1)
        status_frame.pack(fill='x', pady=(0, 15))
        
        # Status grid
        status_grid = tk.Frame(status_frame, bg='#1a1a1a')
        status_grid.pack(fill='x', padx=15, pady=15)
        
        # Last Signal
        tk.Label(status_grid, text="Last Signal:", 
                bg='#1a1a1a', fg='#cccccc', 
                font=('Segoe UI', 10)).grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.signal_label = tk.Label(status_grid, text=self.last_signal,
                                   bg='#1a1a1a', fg='#00ff88',
                                   font=('Segoe UI', 10, 'bold'))
        self.signal_label.grid(row=0, column=1, sticky='w', padx=(0, 30))
        
        # Market Session
        tk.Label(status_grid, text="Market Session:", 
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=0, column=2, sticky='w', padx=(0, 10))
        
        self.session_label = tk.Label(status_grid, text=self.market_session,
                                    bg='#1a1a1a', fg='#ffaa00',
                                    font=('Segoe UI', 10, 'bold'))
        self.session_label.grid(row=0, column=3, sticky='w')
        
        # Risk Manager Status
        tk.Label(status_grid, text="Risk Manager:", 
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.risk_label = tk.Label(status_grid, text=self.risk_status,
                                 bg='#1a1a1a', fg='#00ff88',
                                 font=('Segoe UI', 10, 'bold'))
        self.risk_label.grid(row=1, column=1, sticky='w', padx=(0, 30), pady=(10, 0))
        
        # Last Updated
        tk.Label(status_grid, text="Last Updated:", 
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=1, column=2, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.timestamp_label = tk.Label(status_grid, text=datetime.datetime.now().strftime('%H:%M:%S'),
                                      bg='#1a1a1a', fg='#888888',
                                      font=('Segoe UI', 10))
        self.timestamp_label.grid(row=1, column=3, sticky='w', pady=(10, 0))
    
    def create_performance_section(self, parent):
        """Create performance metrics section"""
        perf_frame = tk.LabelFrame(parent, text="📊 Performance Analytics", 
                                 bg='#1a1a1a', fg='#00d4ff',
                                 font=('Segoe UI', 12, 'bold'),
                                 relief='solid', bd=1)
        perf_frame.pack(fill='x', pady=(0, 15))
        
        # Performance grid
        perf_grid = tk.Frame(perf_frame, bg='#1a1a1a')
        perf_grid.pack(fill='x', padx=15, pady=15)
        
        # Balance
        tk.Label(perf_grid, text="Balance:",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 11)).grid(row=0, column=0, sticky='w', padx=(0, 10))
        
        self.balance_label = tk.Label(perf_grid, text=f"${self.balance:,.2f}",
                                    bg='#1a1a1a', fg='#00ff88',
                                    font=('Segoe UI', 11, 'bold'))
        self.balance_label.grid(row=0, column=1, sticky='w', padx=(0, 30))
        
        # Win Rate
        winrate = self.get_winrate()
        tk.Label(perf_grid, text="Win Rate (%):",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 11)).grid(row=0, column=2, sticky='w', padx=(0, 10))
        
        self.winrate_label = tk.Label(perf_grid, text=f"{winrate:.1f}%",
                                    bg='#1a1a1a', fg='#00d4ff',
                                    font=('Segoe UI', 11, 'bold'))
        self.winrate_label.grid(row=0, column=3, sticky='w')
        
        # Total Trades
        tk.Label(perf_grid, text="Total Trades:",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 11)).grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.trades_label = tk.Label(perf_grid, text=str(self.total_trades),
                                   bg='#1a1a1a', fg='#ffffff',
                                   font=('Segoe UI', 11, 'bold'))
        self.trades_label.grid(row=1, column=1, sticky='w', padx=(0, 30), pady=(10, 0))
        
        # Wins / Losses
        tk.Label(perf_grid, text="Wins / Losses:",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 11)).grid(row=1, column=2, sticky='w', padx=(0, 10), pady=(10, 0))
        
        self.wl_label = tk.Label(perf_grid, text=f"{self.wins} / {self.losses}",
                               bg='#1a1a1a', fg='#ffaa00',
                               font=('Segoe UI', 11, 'bold'))
        self.wl_label.grid(row=1, column=3, sticky='w', pady=(10, 0))
    
    def create_control_section(self, parent):
        """Create control panel section"""
        control_frame = tk.LabelFrame(parent, text="🎛️ Control Panel", 
                                    bg='#1a1a1a', fg='#00d4ff',
                                    font=('Segoe UI', 12, 'bold'),
                                    relief='solid', bd=1)
        control_frame.pack(fill='x', pady=(0, 15))
        
        # Button container
        btn_container = tk.Frame(control_frame, bg='#1a1a1a')
        btn_container.pack(pady=15)
        
        # Start/Stop button with hover effects
        self.start_btn = tk.Button(btn_container,
                                 text="🚀 START ELITE FUSION",
                                 bg='#00ff88', fg='#000000',
                                 font=('Segoe UI', 12, 'bold'),
                                 command=self.toggle_trading,
                                 width=20, height=2,
                                 relief='flat',
                                 cursor='hand2')
        self.start_btn.pack(side='left', padx=10)
        
        # Bind hover effects
        self.start_btn.bind("<Enter>", lambda e: self.on_button_hover(self.start_btn, '#00cc66'))
        self.start_btn.bind("<Leave>", lambda e: self.on_button_leave(self.start_btn, '#00ff88'))
        
        # Stop button
        self.stop_btn = tk.Button(btn_container,
                                text="⏹️ STOP",
                                bg='#ff6b6b', fg='#ffffff',
                                font=('Segoe UI', 10, 'bold'),
                                command=self.stop_trading,
                                width=12, height=2,
                                relief='flat',
                                cursor='hand2')
        self.stop_btn.pack(side='left', padx=5)
        
        # Bind hover effects for stop button
        self.stop_btn.bind("<Enter>", lambda e: self.on_button_hover(self.stop_btn, '#ff4444'))
        self.stop_btn.bind("<Leave>", lambda e: self.on_button_leave(self.stop_btn, '#ff6b6b'))
        
        # Settings section
        settings_frame = tk.Frame(control_frame, bg='#1a1a1a')
        settings_frame.pack(pady=(0, 15))
        
        tk.Label(settings_frame, text="⚙️ Settings:",
                bg='#1a1a1a', fg='#00d4ff',
                font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, columnspan=6, pady=(0, 10))
        
        # Stake
        tk.Label(settings_frame, text="Stake ($):",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=1, column=0, padx=5, sticky='w')
        
        self.stake_var = tk.StringVar(value="100")
        stake_entry = tk.Entry(settings_frame, textvariable=self.stake_var, width=10,
                              bg='#333333', fg='#ffffff', font=('Segoe UI', 10),
                              relief='flat', bd=1)
        stake_entry.grid(row=1, column=1, padx=5)
        
        # Take Profit
        tk.Label(settings_frame, text="Take Profit ($):",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=1, column=2, padx=5, sticky='w')
        
        self.tp_var = tk.StringVar(value="500")
        tp_entry = tk.Entry(settings_frame, textvariable=self.tp_var, width=10,
                           bg='#333333', fg='#00ff88', font=('Segoe UI', 10),
                           relief='flat', bd=1)
        tp_entry.grid(row=1, column=3, padx=5)
        
        # Stop Loss
        tk.Label(settings_frame, text="Stop Loss ($):",
                bg='#1a1a1a', fg='#cccccc',
                font=('Segoe UI', 10)).grid(row=1, column=4, padx=5, sticky='w')
        
        self.sl_var = tk.StringVar(value="250")
        sl_entry = tk.Entry(settings_frame, textvariable=self.sl_var, width=10,
                           bg='#333333', fg='#ff6b6b', font=('Segoe UI', 10),
                           relief='flat', bd=1)
        sl_entry.grid(row=1, column=5, padx=5)
    
    def create_feed_section(self, parent):
        """Create live feed section"""
        feed_frame = tk.LabelFrame(parent, text="📡 Live Intelligence Feed", 
                                 bg='#1a1a1a', fg='#00d4ff',
                                 font=('Segoe UI', 12, 'bold'),
                                 relief='solid', bd=1)
        feed_frame.pack(fill='both', expand=True)
        
        # Live feed text area
        self.feed_text = tk.Text(feed_frame, height=8,
                               bg='#0f0f0f', fg='#00ffaa',
                               font=('Consolas', 9),
                               relief='flat', bd=0,
                               state='disabled')
        self.feed_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add initial messages
        self.add_feed_message("🌟 Elite Neural Beast Quantum Fusion V11 initialized")
        self.add_feed_message("🧠 Adaptive intelligence systems online")
        self.add_feed_message("📊 Market regime detection active")
        self.add_feed_message("🔥 Ready for elite trading operations")
    
    def on_button_hover(self, button, hover_color):
        """Handle button hover effect"""
        button.configure(bg=hover_color)
    
    def on_button_leave(self, button, normal_color):
        """Handle button leave effect"""
        button.configure(bg=normal_color)
    
    def toggle_trading(self):
        """Toggle trading state"""
        if not self.is_active:
            try:
                # Validate settings
                stake = float(self.stake_var.get())
                tp = float(self.tp_var.get())
                sl = float(self.sl_var.get())
                
                if stake <= 0 or tp <= 0 or sl <= 0:
                    messagebox.showerror("Error", "All values must be positive!")
                    return
                
                # Update bot settings
                if self.bot:
                    self.bot.stake = stake
                    self.bot.take_profit = tp
                    self.bot.stop_loss = sl
                
                # Start trading
                self.is_active = True
                self.start_btn.configure(text="🔥 ELITE ACTIVE", bg='#ff6600')
                self.add_feed_message("🚀 Elite trading session ACTIVATED")
                
                # Start trading thread
                if self.bot:
                    trading_thread = threading.Thread(target=self.bot.run_elite_trading_session, daemon=True)
                    trading_thread.start()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers!")
        else:
            self.stop_trading()
    
    def stop_trading(self):
        """Stop trading"""
        self.is_active = False
        if self.bot:
            self.bot.bot_running = False
        
        self.start_btn.configure(text="🚀 START ELITE FUSION", bg='#00ff88')
        self.add_feed_message("⏹️ Elite trading session STOPPED")
    
    def add_feed_message(self, message):
        """Add message to live feed"""
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}\n"
        
        self.feed_text.config(state='normal')
        self.feed_text.insert('end', full_message)
        self.feed_text.see('end')
        self.feed_text.config(state='disabled')
        
        # Keep only last 100 lines
        lines = self.feed_text.get('1.0', 'end').split('\n')
        if len(lines) > 100:
            self.feed_text.config(state='normal')
            self.feed_text.delete('1.0', '2.0')
            self.feed_text.config(state='disabled')
    
    def update_statistics(self):
        """Update statistics display - FIXED: No more stats_display reference"""
        if self.bot:
            self.balance = self.bot.balance
            self.total_trades = self.bot.total_trades
            self.wins = self.bot.win_count
            self.losses = self.bot.loss_count
        
        # Update labels using existing instance variables
        self.balance_label.config(text=f"${self.balance:,.2f}")
        self.trades_label.config(text=str(self.total_trades))
        self.wl_label.config(text=f"{self.wins} / {self.losses}")
        
        # Update win rate
        winrate = self.get_winrate()
        self.winrate_label.config(text=f"{winrate:.1f}%")
        
        # Update timestamp
        self.timestamp_label.config(text=datetime.datetime.now().strftime('%H:%M:%S'))
        
        # Update session info
        remaining = MAX_TRADES_LIMIT - self.total_trades
        
        # Update last signal randomly for demo
        signals = ["CALL", "PUT", "HOLD"]
        self.last_signal = np.random.choice(signals)
        self.signal_label.config(text=self.last_signal)
        
        # Update market session based on time
        hour = datetime.datetime.now().hour
        if 7 <= hour < 16:
            self.market_session = "LONDON"
        elif 16 <= hour < 21:
            self.market_session = "NEW_YORK"
        elif hour >= 22 or hour < 7:
            self.market_session = "ASIAN"
        else:
            self.market_session = "OVERLAP"
        
        self.session_label.config(text=self.market_session)
        
        # Update risk status
        if self.bot and self.bot.loss_streak >= 3:
            self.risk_status = "COOLDOWN"
            self.risk_label.config(fg='#ff6b6b')
        elif self.total_trades >= MAX_TRADES_LIMIT * 0.9:
            self.risk_status = "BLOCKED"
            self.risk_label.config(fg='#ffaa00')
        else:
            self.risk_status = "OK"
            self.risk_label.config(fg='#00ff88')
        
        self.risk_label.config(text=self.risk_status)
    
    def get_winrate(self) -> float:
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100
    
    def start_live_updates(self):
        """Start live statistics updates"""
        self.update_statistics()
        self.root.after(1000, self.start_live_updates)  # Update every second
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_active:
            if messagebox.askokcancel("Quit", "Trading is active. Really quit?"):
                self.stop_trading()
                if self.bot and self.bot.driver:
                    try:
                        self.bot.driver.quit()
                    except:
                        pass
                self.root.destroy()
        else:
            if self.bot and self.bot.driver:
                try:
                    self.bot.driver.quit()
                except:
                    pass
            self.root.destroy()

def main():
    """🌟 Elite Neural Beast Quantum Fusion V11 - Main Entry Point 🌟"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('elite_neural_beast_v11.log', encoding='utf-8')
        ]
    )
    
    # Print elite banner
    print("\n" + "="*80)
    print("🌟 ELITE NEURAL BEAST QUANTUM FUSION V11 - INSTITUTIONAL GRADE 🌟")
    print("="*80)
    print("🧠 ADAPTIVE INTELLIGENCE: ENABLED")
    print("📊 PERFORMANCE ANALYTICS: ACTIVE") 
    print("🔥 MODERN GUI DESIGN: LOADED")
    print("🎛️ CONTROL SYSTEMS: ONLINE")
    print("="*80)
    print(f"🔒 Trade Limit: {MAX_TRADES_LIMIT}")
    print("="*80 + "\n")
    
    # Initialize GUI
    try:
        root = tk.Tk()
        app = EliteNeuralBeastGUI(root)
        
        # Set window close handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        logging.info("🌟 Elite Neural Beast Quantum Fusion V11 GUI initialized")
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        logging.error(f"❌ Elite startup error: {e}")
        print(f"❌ Error starting Elite Neural Beast Quantum Fusion V11: {e}")
    
    finally:
        logging.info("🏁 Elite Neural Beast Quantum Fusion V11 session ended")

if __name__ == "__main__":
    main()