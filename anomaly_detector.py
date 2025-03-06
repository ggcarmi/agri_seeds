from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any

class AnomalyDetectionStrategy(ABC):
    """Abstract base class for anomaly detection strategies"""
    @abstractmethod
    def detect(self, data: pd.Series, **kwargs) -> Tuple[pd.Series, pd.Series]:
        """Return anomaly labels and scores"""
        pass

class IsolationForestStrategy(AnomalyDetectionStrategy):
    """Isolation Forest implementation for anomaly detection"""
    def detect(self, data: pd.Series, **kwargs) -> Tuple[pd.Series, pd.Series]:
        contamination = kwargs.get('contamination', 0.1)
        n_estimators = kwargs.get('n_estimators', 100)
        max_samples = kwargs.get('max_samples', 'auto')
        
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        
        X = data.values.reshape(-1, 1)
        labels = pd.Series(model.fit_predict(X), index=data.index)
        scores = pd.Series(model.score_samples(X), index=data.index)
        
        return labels, scores

class AnomalyDetector:
    """Main class for detecting anomalies in time series data"""
    def __init__(self, strategy: AnomalyDetectionStrategy = None):
        self.strategy = strategy or IsolationForestStrategy()
        self.anomaly_labels = None
        self.anomaly_scores = None
        self.stats = None

    def detect(self, data: pd.Series, **kwargs) -> Dict[str, Any]:
        """Detect anomalies in the given data series"""
        self.anomaly_labels, self.anomaly_scores = self.strategy.detect(data, **kwargs)
        
        # Calculate statistics for anomalies
        anomaly_data = data[self.anomaly_labels == -1]
        self.stats = {
            'count': len(anomaly_data),
            'mean': anomaly_data.mean(),
            'std': anomaly_data.std(),
            'min': anomaly_data.min(),
            'max': anomaly_data.max()
        }
        
        return {
            'labels': self.anomaly_labels,
            'scores': self.anomaly_scores,
            'stats': self.stats
        }

    def get_anomaly_labels(self) -> pd.Series:
        """Get the anomaly labels (-1 for anomalies, 1 for normal points)"""
        return self.anomaly_labels if self.anomaly_labels is not None else pd.Series()

    def get_anomaly_scores(self) -> pd.Series:
        """Get the anomaly scores"""
        return self.anomaly_scores if self.anomaly_scores is not None else pd.Series()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies"""
        return self.stats if self.stats is not None else {}