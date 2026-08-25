from __future__ import annotations
import math


def adaptive_uncertainty_gate(uncertainty: float, threshold: float=0.03, temperature: float=0.02) -> float:
    """Higher model uncertainty -> larger RL correction weight."""
    t=max(float(temperature),1e-6)
    x=max(min((float(uncertainty)-float(threshold))/t, 30.0), -30.0)
    return float(1.0/(1.0+math.exp(-x)))
