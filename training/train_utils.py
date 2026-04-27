"""
Training utilities: learning rate schedulers, early stopping, logging.
"""
import numpy as np
import torch


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_metric: float) -> bool:
        if self.best_score is None:
            self.best_score = val_metric
            return False
        if val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        else:
            self.best_score = val_metric
            self.counter = 0
        return False


class MetricTracker:
    """Track and average metrics over epochs."""
    def __init__(self):
        self.history = {}

    def update(self, metrics_dict: dict):
        for k, v in metrics_dict.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def get_latest(self, key: str):
        return self.history.get(key, [0])[-1]

    def get_best(self, key: str, mode: str = 'max'):
        values = self.history.get(key, [])
        if not values:
            return None
        return max(values) if mode == 'max' else min(values)