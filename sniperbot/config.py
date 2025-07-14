AGGRESSIVE_1MIN_CONFIG = {
    'momentum_candles': 3,
    'body_strength_threshold': 0.55,
    'trend_candles': 5,
    'cooldown_seconds': 18,
    'min_confidence': 0.65,
    'max_trades_per_session': 100,
    'trend_alignment_candles': 12,
    'spike_rejection_ratio': 1.8,
    'min_swing_candles': 2
}

CONSERVATIVE_1MIN_CONFIG = {
    'momentum_candles': 3,
    'body_strength_threshold': 0.7,
    'trend_candles': 7,
    'cooldown_seconds': 25,
    'min_confidence': 0.75,
    'max_trades_per_session': 60,
    'trend_alignment_candles': 18,
    'spike_rejection_ratio': 2.2,
    'min_swing_candles': 3
}

BALANCED_1MIN_CONFIG = {
    'momentum_candles': 3,
    'body_strength_threshold': 0.6,
    'trend_candles': 6,
    'cooldown_seconds': 20,
    'min_confidence': 0.7,
    'max_trades_per_session': 80,
    'trend_alignment_candles': 15,
    'spike_rejection_ratio': 2.0,
    'min_swing_candles': 2,
    'swing_continuation_threshold': 0.65,
    'trend_strength_weight': 0.3,
    'pullback_tolerance': 0.4
}