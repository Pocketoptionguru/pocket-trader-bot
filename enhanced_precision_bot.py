# ==== ENHANCED PRECISION TRADING BOT WITH DETAILED TERMINAL OUTPUT ====
# Enhanced version with comprehensive strategy voting transparency

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import random
import time
import math
# import numpy as np  # Not needed for demo
import logging
# Selenium imports commented out for standalone demo
# import undetected_chromedriver as uc
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.common.exceptions import TimeoutException, NoSuchElementException
# import tkinter as tk
# from tkinter import ttk, messagebox, simpledialog
# import threading
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== DATA STRUCTURES ====
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
        """Size of upper wick"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        """Size of lower wick"""
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        """Total range from high to low"""
        return self.high - self.low
    
    @property
    def body_to_wick_ratio(self) -> float:
        """Ratio of body size to total wick size"""
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
    """Complete analysis of all strategy votes"""
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

    @property
    def total_confirmations(self) -> int:
        """Number of strategies that voted for the signal"""
        return len([vote for vote in self.strategy_votes if vote.signal == self.signal])
    
    @property
    def contributing_strategies(self) -> List[str]:
        """List of strategy names that contributed to the decision"""
        return [vote.strategy_name for vote in self.strategy_votes if vote.signal == self.signal]

@dataclass
class TradeRecord:
    """Record of a completed trade"""
    timestamp: datetime
    signal: Signal
    entry_price: float
    exit_price: Optional[float] = None
    result: Optional[str] = None  # 'win', 'loss', 'pending'
    confidence: float = 0.0

# ==== STRATEGY BASE CLASS ====
class BaseStrategy:
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled

    def get_required_history_length(self) -> int:
        """Return minimum number of candles needed for analysis"""
        return self.config.get('required_history', 20)
    
    def has_sufficient_data(self, candles: List[Candle], current_index: int) -> bool:
        """Check if there's enough historical data for analysis"""
        return current_index >= self.get_required_history_length()
    
    def create_vote(self, signal: Signal, strength: float, conditions: List[str], 
                indicators: Dict[str, float]) -> StrategyVote:
        """Helper method to create a standardized vote"""
        return StrategyVote(
            strategy_name=self.name,
            vote_strength=max(0.0, min(1.0, strength)),  # Clamp between 0 and 1
            signal=signal,
            conditions_met=conditions,
            indicator_values=indicators
        )

# ==== RISK MANAGER ====
class RiskManager:
    def __init__(self, config: Dict):
        self.config = config
        
        # Risk limits
        self.max_trades_per_day = config.get('max_trades_per_day', 50)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 2)
        self.min_win_rate_threshold = config.get('min_win_rate_threshold', 0.6)  # 60%
        self.cooldown_seconds = config.get('cooldown_seconds', 60)
        self.min_trades_for_winrate = config.get('min_trades_for_winrate', 10)
        
        # Session management
        self.session_start_time = datetime.now()
        self.last_trade_time: Optional[datetime] = None
        
        # Trade tracking
        self.trade_history: List[TradeRecord] = []
        self.consecutive_losses = 0
        self.total_wins = 0
        self.total_losses = 0
        self.trades_today = 0
        
        # State flags
        self.trading_enabled = True
        self.shutdown_reason: Optional[str] = None
    
    def can_trade(self, current_time: datetime) -> bool:
        """
        Check if trading is allowed based on all risk criteria
        
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        if not self.trading_enabled:
            return False
        
        # Check daily trade limit
        if self.trades_today >= self.max_trades_per_day:
            self.shutdown_reason = f"Daily trade limit reached ({self.max_trades_per_day})"
            return False
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.shutdown_reason = f"Max consecutive losses reached ({self.max_consecutive_losses})"
            return False
        
        # Check cooldown period
        if self.last_trade_time and current_time:
            time_since_last = (current_time - self.last_trade_time).total_seconds()
            if time_since_last < self.cooldown_seconds:
                remaining = self.cooldown_seconds - time_since_last
                self.shutdown_reason = f"Cooldown active ({remaining:.0f}s remaining)"
                return False
        
        # Check win rate threshold
        total_completed_trades = self.total_wins + self.total_losses
        if total_completed_trades >= self.min_trades_for_winrate:
            win_rate = self.total_wins / total_completed_trades
            if win_rate < self.min_win_rate_threshold:
                self.shutdown_reason = f"Win rate below threshold ({win_rate:.1%} < {self.min_win_rate_threshold:.1%})"
                return False
        
        # Clear any previous shutdown reason
        self.shutdown_reason = None
        return True
    
    def record_trade_signal(self, signal: Signal, confidence: float, 
                        entry_price: float, timestamp: datetime) -> bool:
        """
        Record a new trade signal
        
        Returns:
            bool: True if trade was recorded, False if rejected
        """
        if not self.can_trade(timestamp):
            return False
        
        # Create trade record
        trade = TradeRecord(
            timestamp=timestamp,
            signal=signal,
            entry_price=entry_price,
            confidence=confidence,
            result='pending'
        )
        
        self.trade_history.append(trade)
        self.trades_today += 1
        self.last_trade_time = timestamp
        
        return True
    
    def update_trade_result(self, trade_index: int, exit_price: float, 
                        result: str) -> None:
        """
        Update the result of a trade
        
        Args:
            trade_index: Index of trade in history
            exit_price: Exit price of the trade
            result: 'win' or 'loss'
        """
        if 0 <= trade_index < len(self.trade_history):
            trade = self.trade_history[trade_index]
            trade.exit_price = exit_price
            trade.result = result
            
            if result == 'win':
                self.total_wins += 1
                self.consecutive_losses = 0  # Reset consecutive losses
            elif result == 'loss':
                self.total_losses += 1
                self.consecutive_losses += 1
    
    def get_risk_status(self) -> Dict:
        """Get current risk management status"""
        total_completed = self.total_wins + self.total_losses
        win_rate = self.total_wins / total_completed if total_completed > 0 else 0.0
        
        cooldown_remaining = 0
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).total_seconds()
            cooldown_remaining = max(0, self.cooldown_seconds - elapsed)
        
        return {
            'trading_enabled': self.trading_enabled,
            'can_trade_now': self.can_trade(datetime.now()),
            'shutdown_reason': self.shutdown_reason,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': win_rate,
            'min_win_rate_threshold': self.min_win_rate_threshold,
            'cooldown_remaining': cooldown_remaining,
            'session_duration': (datetime.now() - self.session_start_time).total_seconds()
        }

# ==== ENHANCED SIGNAL ENGINE WITH DETAILED ANALYSIS ====
class SignalEngine:
    def __init__(self, strategies: List[BaseStrategy], risk_manager: RiskManager, config: Dict):
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.config = config
        
        # Confirmation requirements
        self.min_confirmations = config.get('min_confirmations', 2)
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.5)
        self.confirmation_weight_threshold = config.get('confirmation_weight_threshold', 0.4)
        
        # Session context
        self.session_contexts = {
            'asian': (0, 9),      # 00:00 - 09:00 UTC
            'london': (8, 17),    # 08:00 - 17:00 UTC
            'ny': (13, 22),       # 13:00 - 22:00 UTC
            'overlap_london_ny': (13, 17)  # 13:00 - 17:00 UTC
        }
    
    def determine_session_context(self, timestamp: datetime) -> str:
        """Determine the current trading session"""
        hour = timestamp.hour
        
        # Check for overlaps first
        if 13 <= hour < 17:
            return "London-NY Overlap"
        elif 8 <= hour < 13:
            return "London Session"
        elif 17 <= hour < 22:
            return "NY Session"
        elif 0 <= hour < 8 or hour >= 22:
            return "Asian Session"
        else:
            return "Off-Hours"
    
    def analyze_candle(self, candles: List[Candle], current_index: int) -> VoteAnalysis:
        """
        Enhanced analysis that ALWAYS returns detailed vote information
        """
        if current_index >= len(candles):
            return VoteAnalysis(
                all_votes=[],
                call_votes=[],
                put_votes=[],
                hold_votes=[],
                call_confidence=0.0,
                put_confidence=0.0,
                call_confirmations=0,
                put_confirmations=0,
                rejection_reasons=["Index exceeds candle list"],
                final_decision=None
            )
        
        current_candle = candles[current_index]
        timestamp = current_candle.timestamp
        
        # Collect votes from all enabled strategies
        all_strategy_votes = []
        
        for strategy in self.strategies:
            if strategy.is_enabled():
                try:
                    if strategy.has_sufficient_data(candles, current_index):
                        vote = strategy.analyze(candles, current_index)
                        if vote:
                            all_strategy_votes.append(vote)
                        else:
                            # Create a HOLD vote for strategies that didn't vote
                            hold_vote = StrategyVote(
                                strategy_name=strategy.name,
                                vote_strength=0.0,
                                signal=Signal.HOLD,
                                conditions_met=["No conditions met"],
                                indicator_values={}
                            )
                            all_strategy_votes.append(hold_vote)
                    else:
                        # Strategy doesn't have enough data
                        insufficient_vote = StrategyVote(
                            strategy_name=strategy.name,
                            vote_strength=0.0,
                            signal=Signal.HOLD,
                            conditions_met=["Insufficient historical data"],
                            indicator_values={}
                        )
                        all_strategy_votes.append(insufficient_vote)
                except Exception as e:
                    error_vote = StrategyVote(
                        strategy_name=strategy.name,
                        vote_strength=0.0,
                        signal=Signal.HOLD,
                        conditions_met=[f"Error: {str(e)}"],
                        indicator_values={}
                    )
                    all_strategy_votes.append(error_vote)
        
        # Group votes by signal type
        call_votes = [v for v in all_strategy_votes if v.signal == Signal.CALL and v.vote_strength > 0]
        put_votes = [v for v in all_strategy_votes if v.signal == Signal.PUT and v.vote_strength > 0]
        hold_votes = [v for v in all_strategy_votes if v.signal == Signal.HOLD or v.vote_strength == 0]
        
        # Calculate weighted confidence for each signal
        call_confidence = sum(v.vote_strength for v in call_votes)
        put_confidence = sum(v.vote_strength for v in put_votes)
        
        # Determine rejection reasons
        rejection_reasons = []
        final_decision = None
        
        # Check if risk manager allows trading
        if not self.risk_manager.can_trade(timestamp):
            rejection_reasons.append(f"Risk Manager Block: {self.risk_manager.shutdown_reason}")
        
        # Check confirmation requirements
        if len(call_votes) < self.min_confirmations and len(put_votes) < self.min_confirmations:
            rejection_reasons.append(f"Insufficient confirmations (need {self.min_confirmations}, got CALL:{len(call_votes)}, PUT:{len(put_votes)})")
        
        # Check confidence thresholds
        if call_confidence < self.confirmation_weight_threshold and put_confidence < self.confirmation_weight_threshold:
            rejection_reasons.append(f"Low confidence (need {self.confirmation_weight_threshold}, got CALL:{call_confidence:.2f}, PUT:{put_confidence:.2f})")
        
        # If risk manager allows and we have sufficient confidence/confirmations, try to make decision
        if self.risk_manager.can_trade(timestamp):
            final_signal = Signal.HOLD
            contributing_votes = []
            final_confidence = 0.0
            
            if call_confidence > put_confidence and len(call_votes) >= self.min_confirmations:
                if call_confidence >= self.confirmation_weight_threshold:
                    final_signal = Signal.CALL
                    contributing_votes = call_votes
                    final_confidence = min(1.0, call_confidence)
            
            elif put_confidence > call_confidence and len(put_votes) >= self.min_confirmations:
                if put_confidence >= self.confirmation_weight_threshold:
                    final_signal = Signal.PUT
                    contributing_votes = put_votes
                    final_confidence = min(1.0, put_confidence)
            
            # Check if final confidence meets minimum threshold
            if final_signal != Signal.HOLD and final_confidence >= self.min_confidence_threshold:
                # Get session context
                session_context = self.determine_session_context(timestamp)
                
                # Get current risk status
                risk_status = self.risk_manager.get_risk_status()
                
                # Create final trade decision
                final_decision = TradeDecision(
                    signal=final_signal,
                    confidence=final_confidence,
                    strategy_votes=contributing_votes,
                    current_candle=current_candle,
                    session_context=session_context,
                    risk_status=risk_status
                )
                
                # Clear rejection reasons since we have a valid decision
                rejection_reasons = []
            else:
                if final_signal == Signal.HOLD:
                    rejection_reasons.append("No clear signal direction determined")
                if final_confidence < self.min_confidence_threshold:
                    rejection_reasons.append(f"Final confidence too low ({final_confidence:.2f} < {self.min_confidence_threshold})")
        
        return VoteAnalysis(
            all_votes=all_strategy_votes,
            call_votes=call_votes,
            put_votes=put_votes,
            hold_votes=hold_votes,
            call_confidence=call_confidence,
            put_confidence=put_confidence,
            call_confirmations=len(call_votes),
            put_confirmations=len(put_votes),
            rejection_reasons=rejection_reasons,
            final_decision=final_decision
        )
    
    def format_vote_analysis(self, analysis: VoteAnalysis, timestamp: datetime) -> str:
        """
        Format comprehensive vote analysis for terminal output
        """
        output = []
        
        # Header
        output.append("🎯 PRECISION TRADING ANALYSIS")
        output.append("=" * 60)
        output.append(f"⏰ Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"🌐 Session: {self.determine_session_context(timestamp)}")
        output.append("")
        
        # Strategy Vote Breakdown
        output.append("📊 STRATEGY VOTE BREAKDOWN:")
        output.append("-" * 40)
        
        for vote in analysis.all_votes:
            if vote.vote_strength > 0:
                strength_bar = "█" * int(vote.vote_strength * 10) + "░" * (10 - int(vote.vote_strength * 10))
                output.append(f"   {vote.strategy_name}: {vote.signal.name} ({vote.vote_strength:.2f}) [{strength_bar}]")
                if vote.conditions_met:
                    for condition in vote.conditions_met[:2]:  # Show top 2 conditions
                        output.append(f"     └─ {condition}")
            else:
                output.append(f"   {vote.strategy_name}: HOLD/NO VOTE")
                if vote.conditions_met and vote.conditions_met[0] != "No conditions met":
                    output.append(f"     └─ {vote.conditions_met[0]}")
        
        output.append("")
        
        # Vote Summary
        output.append("📈 VOTE SUMMARY:")
        output.append("-" * 40)
        output.append(f"   CALL Votes: {analysis.call_confirmations} strategies | Confidence: {analysis.call_confidence:.2f}")
        output.append(f"   PUT Votes:  {analysis.put_confirmations} strategies | Confidence: {analysis.put_confidence:.2f}")
        output.append(f"   HOLD Votes: {len(analysis.hold_votes)} strategies")
        output.append("")
        
        # Requirements Check
        output.append("✅ REQUIREMENTS CHECK:")
        output.append("-" * 40)
        output.append(f"   Min Confirmations: {self.min_confirmations} (CALL:{analysis.call_confirmations}, PUT:{analysis.put_confirmations})")
        output.append(f"   Min Confidence: {self.min_confidence_threshold} (CALL:{analysis.call_confidence:.2f}, PUT:{analysis.put_confidence:.2f})")
        output.append(f"   Weight Threshold: {self.confirmation_weight_threshold}")
        output.append("")
        
        # Risk Manager Status
        risk_status = self.risk_manager.get_risk_status()
        output.append("🛡️ RISK MANAGER STATUS:")
        output.append("-" * 40)
        output.append(f"   Trading Enabled: {'✅ YES' if risk_status['can_trade_now'] else '🚫 NO'}")
        if not risk_status['can_trade_now']:
            output.append(f"   Block Reason: {risk_status.get('shutdown_reason', 'Unknown')}")
        output.append(f"   Daily Trades: {risk_status['trades_today']}/{risk_status['max_trades_per_day']}")
        output.append(f"   Consecutive Losses: {risk_status['consecutive_losses']}/{risk_status['max_consecutive_losses']}")
        output.append(f"   Win Rate: {risk_status['win_rate']:.1%} (threshold: {risk_status['min_win_rate_threshold']:.1%})")
        if risk_status['cooldown_remaining'] > 0:
            output.append(f"   Cooldown: {risk_status['cooldown_remaining']:.0f}s remaining")
        output.append("")
        
        # Final Decision
        if analysis.final_decision:
            output.append("🎯 SIGNAL GENERATED!")
            output.append("-" * 40)
            output.append(f"   Signal: {analysis.final_decision.signal.name}")
            output.append(f"   Confidence: {analysis.final_decision.confidence:.2f}")
            output.append(f"   Contributing Strategies: {', '.join(analysis.final_decision.contributing_strategies)}")
        else:
            output.append("🚫 NO SIGNAL GENERATED")
            output.append("-" * 40)
            if analysis.rejection_reasons:
                output.append("   Rejection Reasons:")
                for reason in analysis.rejection_reasons:
                    output.append(f"     • {reason}")
            else:
                output.append("   • No specific rejection reasons identified")
        
        output.append("=" * 60)
        output.append("")
        
        return "\n".join(output)
    
    def format_signal_output(self, decision: TradeDecision) -> str:
        """
        Format trade decision into readable terminal output (existing method)
        """
        if not decision:
            return ""
        
        # Header with signal and confidence
        output = [
            f"🎯 SIGNAL: {decision.signal.name} | Confidence: {decision.confidence:.2f}",
            "✅ CONDITIONS MET:"
        ]
        
        # List all conditions from contributing strategies
        for vote in decision.strategy_votes:
            for condition in vote.conditions_met:
                output.append(f"   - {condition}")
        
        # Strategy votes section
        output.append("📊 STRATEGY VOTES:")
        for vote in decision.strategy_votes:
            output.append(f"   - {vote.strategy_name}: {vote.vote_strength:.2f}")
        
        # Add key indicator values
        output.append("📈 KEY INDICATORS:")
        all_indicators = {}
        for vote in decision.strategy_votes:
            all_indicators.update(vote.indicator_values)
        
        # Show most relevant indicators
        for key, value in all_indicators.items():
            if isinstance(value, (int, float)):
                output.append(f"   - {key}: {value:.3f}")
        
        # Session and timing info
        output.append(f"🌐 SESSION: {decision.session_context}")
        output.append(f"⏰ TIME: {decision.current_candle.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Risk status
        risk = decision.risk_status
        risk_line = f"🛡️ Risk OK: {risk['consecutive_losses']} consecutive losses, Win rate {risk['win_rate']:.0%}"
        
        if risk['trades_today'] > 0:
            risk_line += f", {risk['trades_today']}/{risk['max_trades_per_day']} trades today"
        
        output.append(risk_line)
        
        # Add separation line
        output.append("=" * 60)
        
        return "\n".join(output)

# ==== ENHANCED PRECISION TRADING BOT ====
class PrecisionTradingBot:
    """
    Enhanced Precision Trading Bot with Complete Transparency
    
    Key Features:
    - Detailed strategy vote analysis
    - Complete rejection reason reporting
    - Risk manager status visibility
    - Full decision transparency
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the enhanced trading bot with configuration
        """
        self.config = config
        self.is_running = False
        
        # Initialize risk manager
        risk_config = config.get('risk_management', {})
        self.risk_manager = RiskManager(risk_config)
        
        # Initialize strategies
        self.strategies = self._initialize_strategies(config.get('strategies', {}))
        
        # Initialize enhanced signal engine
        engine_config = config.get('signal_engine', {})
        self.signal_engine = SignalEngine(self.strategies, self.risk_manager, engine_config)
        
        # Bot state
        self.current_candles: List[Candle] = []
        self.last_decision: Optional[TradeDecision] = None
        
        print("🤖 Enhanced Precision Trading Bot Initialized")
        print(f"✅ {len(self.strategies)} strategies loaded")
        print(f"✅ Risk management active")
        print(f"✅ Enhanced signal engine ready")
        print("=" * 60)
    
    def _initialize_strategies(self, strategy_configs: Dict) -> List[BaseStrategy]:
        """Initialize all trading strategies with their configurations"""
        strategies = []
        
        # Mock strategies for demonstration (replace with actual strategy implementations)
        if strategy_configs.get('momentum', {}).get('enabled', True):
            strategies.append(MockStrategy("Momentum", strategy_configs.get('momentum', {})))
        
        if strategy_configs.get('rsi', {}).get('enabled', True):
            strategies.append(MockStrategy("RSI", strategy_configs.get('rsi', {})))
        
        if strategy_configs.get('ma', {}).get('enabled', True):
            strategies.append(MockStrategy("MovingAverage", strategy_configs.get('ma', {})))
        
        if strategy_configs.get('bollinger', {}).get('enabled', True):
            strategies.append(MockStrategy("BollingerBands", strategy_configs.get('bollinger', {})))
        
        if strategy_configs.get('volume', {}).get('enabled', True):
            strategies.append(MockStrategy("Volume", strategy_configs.get('volume', {})))
        
        if strategy_configs.get('support_resistance', {}).get('enabled', True):
            strategies.append(MockStrategy("SupportResistance", strategy_configs.get('support_resistance', {})))
        
        return strategies
    
    def process_candle(self, candle: Candle) -> Optional[TradeDecision]:
        """
        ENHANCED process_candle with complete transparency
        
        This is the main method that was enhanced to provide detailed output
        """
        self.current_candles.append(candle)

        # Keep only necessary history (prevent memory bloat)
        max_history = max(strategy.get_required_history_length() for strategy in self.strategies) + 50
        if len(self.current_candles) > max_history:
            self.current_candles = self.current_candles[-max_history:]

        current_index = len(self.current_candles) - 1
        
        # Get COMPLETE analysis - this now always returns detailed information
        analysis = self.signal_engine.analyze_candle(self.current_candles, current_index)

        # ALWAYS show detailed analysis regardless of outcome
        detailed_output = self.signal_engine.format_vote_analysis(analysis, candle.timestamp)
        print(detailed_output)

        if analysis.final_decision:
            self.last_decision = analysis.final_decision

            # Try to record the trade
            trade_recorded = self.risk_manager.record_trade_signal(
                analysis.final_decision.signal,
                analysis.final_decision.confidence,
                analysis.final_decision.current_candle.close,
                analysis.final_decision.current_candle.timestamp
            )

            if trade_recorded:
                # Show additional signal details when trade is recorded
                signal_output = self.signal_engine.format_signal_output(analysis.final_decision)
                print("🚀 TRADE SIGNAL DETAILS:")
                print(signal_output)
                return analysis.final_decision
            else:
                # This shouldn't happen since we already checked risk manager in analysis
                print("⚠️ WARNING: Trade was not recorded despite passing analysis!")

        # No additional output needed here since we already showed the complete analysis above
        return None
    
    def run_live_simulation(self, candle_generator, sleep_seconds: float = 1.0):
        """
        Run the bot in live simulation mode with enhanced output
        """
        self.is_running = True
        print("🚀 Starting Enhanced Precision Trading Simulation...")
        print("📊 Complete strategy analysis will be shown for every candle")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            for candle in candle_generator:
                if not self.is_running:
                    break
                
                # Process the candle with enhanced output
                self.process_candle(candle)
                
                # Sleep between candles
                time.sleep(sleep_seconds)
                
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Error in live simulation: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the trading bot and print session summary"""
        self.is_running = False
        
        print("\n" + "=" * 60)
        print("📊 ENHANCED PRECISION TRADING SESSION SUMMARY")
        print("=" * 60)
        
        # Get session summary from risk manager
        summary = self.risk_manager.get_session_summary() if hasattr(self.risk_manager, 'get_session_summary') else {}
        
        print(f"⏱️  Session Duration: {summary.get('session_duration_hours', 0):.1f} hours")
        print(f"📡 Total Signals: {summary.get('total_signals', 0)}")
        print(f"✅ Completed Trades: {summary.get('completed_trades', 0)}")
        print(f"🎯 Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"🏆 Wins: {summary.get('wins', 0)} | 💔 Losses: {summary.get('losses', 0)}")
        print(f"📈 Average Confidence: {summary.get('average_confidence', 0):.2f}")
        
        if summary.get('consecutive_losses', 0) > 0:
            print(f"⚠️  Consecutive Losses: {summary['consecutive_losses']}")
        
        print("\n🤖 Enhanced Precision Trading Bot Stopped")


# ==== MOCK STRATEGY FOR DEMONSTRATION ====
class MockStrategy(BaseStrategy):
    """Mock strategy that demonstrates the enhanced output system"""
    
    def analyze(self, candles: List[Candle], current_index: int) -> Optional[StrategyVote]:
        if not self.has_sufficient_data(candles, current_index):
            return None
        
        current = candles[current_index]
        
        # Mock analysis with random but realistic behavior
        import random
        
        # Simulate different strategy behaviors
        vote_strength = random.uniform(0.0, 0.8)
        
        if vote_strength < 0.2:
            return None  # No vote
        
        # Randomly choose signal direction
        signals = [Signal.CALL, Signal.PUT, Signal.HOLD]
        weights = [0.35, 0.35, 0.3]  # Slightly favor CALL/PUT over HOLD
        signal = random.choices(signals, weights=weights)[0]
        
        # Create mock conditions based on strategy name
        conditions = []
        indicators = {}
        
        if self.name == "Momentum":
            conditions.append(f"Price momentum detected: {vote_strength:.2f}")
            indicators['momentum_strength'] = vote_strength
            indicators['price_change'] = (current.close - current.open) / current.open
            
        elif self.name == "RSI":
            mock_rsi = random.uniform(20, 80)
            indicators['rsi'] = mock_rsi
            if mock_rsi < 30:
                conditions.append(f"RSI oversold: {mock_rsi:.1f}")
            elif mock_rsi > 70:
                conditions.append(f"RSI overbought: {mock_rsi:.1f}")
            else:
                conditions.append(f"RSI neutral: {mock_rsi:.1f}")
                
        elif self.name == "MovingAverage":
            conditions.append("MA crossover pattern detected")
            indicators['ma_short'] = current.close * random.uniform(0.998, 1.002)
            indicators['ma_long'] = current.close * random.uniform(0.995, 1.005)
            
        elif self.name == "BollingerBands":
            bb_position = random.uniform(0, 1)
            indicators['bb_position'] = bb_position
            if bb_position > 0.8:
                conditions.append("Price near upper Bollinger Band")
            elif bb_position < 0.2:
                conditions.append("Price near lower Bollinger Band")
            else:
                conditions.append("Price within Bollinger Bands")
                
        elif self.name == "Volume":
            volume_ratio = random.uniform(0.5, 3.0)
            indicators['volume_ratio'] = volume_ratio
            if volume_ratio > 1.5:
                conditions.append(f"High volume detected: {volume_ratio:.1f}x average")
            else:
                conditions.append(f"Normal volume: {volume_ratio:.1f}x average")
                
        elif self.name == "SupportResistance":
            distance_to_level = random.uniform(0.001, 0.01)
            indicators['distance_to_level'] = distance_to_level
            if distance_to_level < 0.005:
                conditions.append(f"Near key level: {distance_to_level:.3f} distance")
            else:
                conditions.append("No key levels nearby")
        
        # Adjust vote strength based on signal type
        if signal == Signal.HOLD:
            vote_strength *= 0.3  # Lower strength for HOLD votes
        
        return self.create_vote(signal, vote_strength, conditions, indicators)


# ==== CONFIGURATION ====
ENHANCED_CONFIG = {
    'signal_engine': {
        'min_confirmations': 2,
        'min_confidence_threshold': 0.5,
        'confirmation_weight_threshold': 0.4
    },
    'risk_management': {
        'max_trades_per_day': 50,
        'max_consecutive_losses': 3,
        'min_win_rate_threshold': 0.6,
        'cooldown_seconds': 60,
        'min_trades_for_winrate': 5
    },
    'strategies': {
        'momentum': {'enabled': True, 'required_history': 10},
        'rsi': {'enabled': True, 'required_history': 15},
        'ma': {'enabled': True, 'required_history': 25},
        'bollinger': {'enabled': True, 'required_history': 20},
        'volume': {'enabled': True, 'required_history': 15},
        'support_resistance': {'enabled': True, 'required_history': 30}
    }
}


# ==== MOCK CANDLE GENERATOR FOR DEMONSTRATION ====
def generate_mock_candles():
    """Generate realistic mock candles for demonstration"""
    base_price = 1.0
    
    while True:
        # Random price movement
        change = random.gauss(0, 0.002)
        base_price += change
        
        # Create OHLC data
        open_price = base_price
        high_price = base_price + abs(random.gauss(0, 0.001))
        low_price = base_price - abs(random.gauss(0, 0.001))
        close_price = base_price + random.gauss(0, 0.0005)
        volume = random.uniform(1.0, 3.0)
        
        # Create candle
        candle = Candle(
            timestamp=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )
        
        base_price = close_price
        yield candle


# ==== DEMO SECTION ====
def run_demo():
    """Run a demonstration of the enhanced precision trading bot"""
    print("🎯 ENHANCED PRECISION TRADING BOT DEMO")
    print("=" * 60)
    print("This demo shows the complete transparency features:")
    print("✅ Detailed strategy voting")
    print("✅ Vote strength visualization")
    print("✅ Confidence calculations")
    print("✅ Risk manager status")
    print("✅ Rejection reason analysis")
    print("=" * 60)
    print()
    
    # Initialize the enhanced bot
    bot = PrecisionTradingBot(ENHANCED_CONFIG)
    
    # Generate some mock candles and process them
    candle_generator = generate_mock_candles()
    
    print("Processing sample candles to demonstrate enhanced output:")
    print("(Each candle will show complete strategy analysis)")
    print()
    
    # Process 5 sample candles to show the enhanced output
    for i in range(5):
        print(f"🕐 PROCESSING CANDLE #{i+1}")
        print("-" * 30)
        
        candle = next(candle_generator)
        result = bot.process_candle(candle)
        
        if result:
            print(f"✅ Signal generated: {result.signal.name}")
        else:
            print("📭 No signal generated (see detailed analysis above)")
        
        print("\n" + "⏳ Waiting 2 seconds before next candle...\n")
        time.sleep(2)
    
    print("🏁 Demo completed!")
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("• Complete strategy vote transparency")
    print("• Detailed rejection reason reporting")
    print("• Risk manager status visibility")
    print("• Vote strength visualization with progress bars")
    print("• Confidence calculation breakdowns")
    print("• Session context awareness")
    print("=" * 60)


if __name__ == "__main__":
    """
    Main execution - run the enhanced precision trading bot demo
    
    This demonstrates the complete fix for the 'No trading signal generated' issue
    by providing full transparency into:
    1. What each strategy voted
    2. Vote strength of each strategy
    3. Total confidence for CALL and PUT
    4. Number of strategy confirmations
    5. Risk manager blocking reasons
    6. Confidence/confirmation rejection details
    """
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()