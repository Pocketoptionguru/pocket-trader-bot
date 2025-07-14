import logging
from typing import Dict
import numpy as np


class MLSignalFilter:
    """Optional ML gate that can veto low-quality signals when a trained model is provided."""

    def __init__(self, model=None):
        self.model = model
        if model:
            logging.info("MLSignalFilter initialised with model – quality gating active")
        else:
            logging.info("MLSignalFilter initialised without model – pass-through mode")

    def is_high_quality(self, signal_features: Dict) -> bool:
        if not self.model:
            return True  # No model means no filtering
        try:
            features_array = np.array(list(signal_features.values())).reshape(1, -1)
            pred = self.model.predict(features_array)
            return bool(pred[0])
        except Exception as e:
            logging.error(f"MLSignalFilter: prediction error – {e}")
            return True  # Fail-open if the model errors