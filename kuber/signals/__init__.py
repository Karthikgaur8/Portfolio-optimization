"""Signal generators for the KUBER portfolio optimization framework."""

from kuber.signals.base import Signal
from kuber.signals.momentum import (
    DualMomentumSignal,
    TSMOMSignal,
    XSMOMSignal,
)
from kuber.signals.mean_reversion import (
    BollingerBandSignal,
    RSISignal,
    ZScoreReversionSignal,
)
from kuber.signals.volatility import (
    GARCHVolSignal,
    RealizedVolSignal,
    VolRatioSignal,
)
from kuber.signals.sentiment import (
    SentimentMomentumSignal,
    SentimentSignal,
)
from kuber.signals.macro import (
    FedStanceSignal,
    VIXRegimeSignal,
    YieldCurveSignal,
)
from kuber.signals.composite import CompositeSignal

__all__ = [
    "Signal",
    # Momentum
    "TSMOMSignal",
    "XSMOMSignal",
    "DualMomentumSignal",
    # Mean reversion
    "RSISignal",
    "BollingerBandSignal",
    "ZScoreReversionSignal",
    # Volatility
    "RealizedVolSignal",
    "VolRatioSignal",
    "GARCHVolSignal",
    # Sentiment
    "SentimentSignal",
    "SentimentMomentumSignal",
    # Macro
    "YieldCurveSignal",
    "VIXRegimeSignal",
    "FedStanceSignal",
    # Composite
    "CompositeSignal",
]
