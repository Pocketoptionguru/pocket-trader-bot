import logging
import time


class StopLossDrawdownManager:
    """Tracks equity curve and enforces various draw-down / stop-loss rules."""

    def __init__(self):
        self.drawdown_cfg = {
            'max_daily_drawdown': 0.25,
            'max_session_drawdown': 0.20,
            'max_peak_drawdown': 0.30,
            'recovery_threshold': 0.03,
        }
        self.stop_loss_levels = {
            'conservative': 0.05,
            'moderate': 0.10,
            'aggressive': 0.15,
        }
        self.peak_balance = 0.0
        self.session_start_balance = 0.0
        self.daily_start_balance = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5

    # ------------------------------------------------------------------
    #                       INITIALISATION / RECORDING
    # ------------------------------------------------------------------
    def initialize_balances(self, current_balance: float):
        self.peak_balance = current_balance
        self.session_start_balance = current_balance
        self.daily_start_balance = current_balance
        self.trades_today = 0
        self.consecutive_losses = 0
        logging.info(f"StopLossDrawdownManager: balances initialised at ${current_balance:.2f}")

    def record_trade_result(self, profit_loss: float):
        self.trades_today += 1
        if profit_loss < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # ------------------------------------------------------------------
    #                             DRAWDOWN
    # ------------------------------------------------------------------
    def check_drawdown_limits(self, current_balance: float, session_data: dict):
        # peak update
        self.peak_balance = max(self.peak_balance, current_balance)
        # consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"consecutive loss limit reached ({self.consecutive_losses})"
        daily_dd = (self.daily_start_balance - current_balance) / self.daily_start_balance if self.daily_start_balance else 0
        if daily_dd >= self.drawdown_cfg['max_daily_drawdown']:
            return False, f"daily draw-down {daily_dd:.1%} exceeds {self.drawdown_cfg['max_daily_drawdown']:.1%}"
        session_dd = (self.session_start_balance - current_balance) / self.session_start_balance if self.session_start_balance else 0
        if session_dd >= self.drawdown_cfg['max_session_drawdown']:
            return False, f"session draw-down {session_dd:.1%} exceeds {self.drawdown_cfg['max_session_drawdown']:.1%}"
        peak_dd = (self.peak_balance - current_balance) / self.peak_balance if self.peak_balance else 0
        if peak_dd >= self.drawdown_cfg['max_peak_drawdown']:
            return False, f"peak draw-down {peak_dd:.1%} exceeds {self.drawdown_cfg['max_peak_drawdown']:.1%}"
        # early session grace – allow first 3 trades no matter what
        if self.trades_today < 3:
            return True, "early-session grace period"
        return True, "all clear"

    # ------------------------------------------------------------------
    #                     GENERIC STOP-LOSS CHECKER
    # ------------------------------------------------------------------
    def check_stop_loss(self, method: str, current_balance: float, session_data: dict, **kwargs):
        if method == 'adaptive':
            return self._adaptive_stop_loss(current_balance, **kwargs)
        if method == 'percentage':
            return self._percentage_stop_loss(current_balance, kwargs.get('stop_loss_percentage', 0.10))
        return False, 'unknown stop-loss method'

    def _percentage_stop_loss(self, current_balance: float, perc: float):
        dd = (self.session_start_balance - current_balance) / self.session_start_balance
        return (dd >= perc, f"percentage stop-loss {'hit' if dd >= perc else 'ok'}: {dd:.1%} / {perc:.1%}")

    def _adaptive_stop_loss(self, current_balance: float, recent_performance: float, volatility: float):
        base = 0.15
        if recent_performance < -0.05:
            base *= 0.8
        elif recent_performance > 0.05:
            base *= 1.5
        base *= min(1.8, max(0.6, 1.0 + volatility))
        base = max(0.08, min(base, 0.30))
        dd = (self.session_start_balance - current_balance) / self.session_start_balance
        return (dd >= base, f"adaptive stop-loss {'hit' if dd >= base else 'ok'}: {dd:.1%} / {base:.1%}")

    # ------------------------------------------------------------------
    #                           REPORTING HELPERS
    # ------------------------------------------------------------------
    def get_drawdown_metrics(self, current_balance: float):
        daily_dd = (self.daily_start_balance - current_balance) / self.daily_start_balance * 100 if self.daily_start_balance else 0
        session_dd = (self.session_start_balance - current_balance) / self.session_start_balance * 100 if self.session_start_balance else 0
        peak_dd = (self.peak_balance - current_balance) / self.peak_balance * 100 if self.peak_balance else 0
        return {
            'daily_drawdown': daily_dd,
            'session_drawdown': session_dd,
            'peak_drawdown': peak_dd,
            'peak_balance': self.peak_balance,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
        }

    def reset_daily_limits(self):
        self.daily_start_balance = self.peak_balance
        self.trades_today = 0
        self.consecutive_losses = 0