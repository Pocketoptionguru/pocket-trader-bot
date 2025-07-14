import csv
import json
import logging
import datetime
from typing import Dict


class LiveLogger:
    """Persist signals & trade outcomes to CSV so performance can be analysed."""

    def __init__(self, signal_log_file: str = 'signal_log.csv', trade_log_file: str = 'trade_log.csv'):
        self.signal_log_file = signal_log_file
        self.trade_log_file = trade_log_file
        self.signal_id_counter = 0
        # Create files with header if they do not yet exist
        for file, headers in [
            (self.signal_log_file, ['timestamp', 'signal_id', 'direction', 'confidence', 'reason', 'trend', 'indicator_data']),
            (self.trade_log_file, ['timestamp', 'trade_id', 'signal_id', 'result', 'profit_loss', 'stake', 'balance_after'])
        ]:
            try:
                with open(file, 'x', newline='') as f:
                    csv.writer(f).writerow(headers)
            except FileExistsError:
                pass

    # ------------------------------------------------------------------
    #                             SIGNALS
    # ------------------------------------------------------------------
    def log_signal(self, direction: str, confidence: float, reason: str, trend: str, indicator_data: Dict) -> int:
        self.signal_id_counter += 1
        signal_id = self.signal_id_counter
        ts = datetime.datetime.utcnow().isoformat()
        with open(self.signal_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                ts, signal_id, direction, confidence, reason, trend, json.dumps(indicator_data)
            ])
        logging.info(f"LiveLogger: Signal {signal_id} logged")
        return signal_id

    # ------------------------------------------------------------------
    #                               TRADES
    # ------------------------------------------------------------------
    def log_trade(self, trade_id: int, signal_id: int, result: bool, profit_loss: float, stake: float, balance_after: float):
        ts = datetime.datetime.utcnow().isoformat()
        with open(self.trade_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                ts, trade_id, signal_id, 'WIN' if result else 'LOSS', profit_loss, stake, balance_after
            ])
        logging.info(f"LiveLogger: Trade {trade_id} ({'WIN' if result else 'LOSS'}) logged")