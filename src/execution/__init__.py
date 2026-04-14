"""Execution engine: broker integration, order management, risk, alerts, and scheduling."""

from .broker import BrokerInterface, AlpacaBroker, TradierBroker, PaperBroker
from .oms import OrderManagementSystem
from .risk_engine import RiskEngine
from .alerts import AlertManager
from .scheduler import TradingScheduler

try:
    from .binance_broker import BinanceBroker
except ImportError:
    BinanceBroker = None  # python-binance not installed

__all__ = [
    "BrokerInterface",
    "AlpacaBroker",
    "TradierBroker",
    "PaperBroker",
    "BinanceBroker",
    "OrderManagementSystem",
    "RiskEngine",
    "AlertManager",
    "TradingScheduler",
]
