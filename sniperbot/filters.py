import logging
import datetime
from collections import deque
from typing import Dict

import numpy as np


class VolatilityFilter:
    """Detect low-volatility (dead) market zones so the bot can stay out."""

    def __init__(self, window: int = 20, min_range: float = 0.0007, min_body: float = 0.0003):
        self.window = window
        self.min_range = min_range
        self.min_body = min_body
        self.candles = deque(maxlen=window)

    def update(self, candle: Dict):
        self.candles.append(candle)

    def is_dead_zone(self) -> bool:
        if len(self.candles) < self.window:
            return False
        ranges = [c['high'] - c['low'] for c in self.candles]
        bodies = [abs(c['close'] - c['open']) for c in self.candles]
        avg_range = sum(ranges) / len(ranges)
        avg_body = sum(bodies) / len(bodies)
        dead = avg_range < self.min_range and avg_body < self.min_body
        if dead:
            logging.info(f"VolatilityFilter: Dead zone detected (Avg Range={avg_range:.5f}, Avg Body={avg_body:.5f})")
        return dead


class AdaptiveConfidence:
    """Dynamically raises required confidence after consecutive losses."""

    def __init__(self, base: float = 0.65, step: float = 0.05, max_conf: float = 0.9):
        self.base = base
        self.step = step
        self.max_conf = max_conf
        self.loss_streak = 0

    def get_threshold(self) -> float:
        return min(self.base + self.loss_streak * self.step, self.max_conf)

    def register_result(self, win: bool):
        self.loss_streak = 0 if win else self.loss_streak + 1


class SmartPause:
    """Pauses trading automatically after back-to-back losses for a time / candle count."""

    def __init__(self, pause_candles: int = 3, pause_minutes: int = 5):
        self.pause_until_time = None
        self.pause_until_candle_idx = None
        self.loss_streak = 0
        self.pause_candles = pause_candles
        self.pause_minutes = pause_minutes

    def register_result(self, win: bool, curr_time: float, curr_candle_idx: int):
        if win:
            self.loss_streak = 0
            return
        self.loss_streak += 1
        if self.loss_streak >= 2:
            self.pause_until_time = datetime.datetime.fromtimestamp(curr_time) + datetime.timedelta(minutes=self.pause_minutes)
            self.pause_until_candle_idx = curr_candle_idx + self.pause_candles
            logging.warning(f"SmartPause triggered – pausing until {self.pause_until_time} or candle {self.pause_until_candle_idx}")

    def is_paused(self, curr_time: float, curr_candle_idx: int) -> bool:
        now_dt = datetime.datetime.fromtimestamp(curr_time)
        if self.pause_until_time and now_dt < self.pause_until_time:
            return True
        if self.pause_until_candle_idx and curr_candle_idx < self.pause_until_candle_idx:
            return True
        # Reset if pause expired
        if self.pause_until_time and now_dt >= self.pause_until_time:
            self.pause_until_time = None
            self.loss_streak = 0
        if self.pause_until_candle_idx and curr_candle_idx >= self.pause_until_candle_idx:
            self.pause_until_candle_idx = None
            self.loss_streak = 0
        return False

    def clear_pause(self):
        self.pause_until_time = None
        self.pause_until_candle_idx = None
        self.loss_streak = 0