from dataclasses import dataclass, field
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExchangeConfig:
    api_key: str
    secret_key: str
    use_futures: bool = True
    leverage: int = 3
    sandbox: bool = False
    rate_limit_ms: int = 200
    max_retries: int = 3


@dataclass
class TradingConfig:
    symbols: List[str]
    trading_mode: str = "paper"  # "paper" or "live"
    entry_timeframe: str = "5m"
    timeframes: List[str] = field(default_factory=lambda: ["5m", "15m", "30m", "1h", "4h", "1d"])
    min_conviction_score: int = 5
    max_concurrent_trades: int = 3
    target_trades_per_day: int = 40
    min_adx_threshold: float = 25.0
    volume_surge_multiplier: float = 1.5
    ai_confidence_threshold: float = 0.60
    min_rr_ratio: float = 2.0
    scan_interval_seconds: int = 60


@dataclass
class RiskConfig:
    capital_usdt: float
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 8.0
    min_rr_ratio: float = 2.0
    trailing_stop_activation_atr: float = 1.5
    max_sl_atr_multiplier: float = 1.5
    min_sl_atr_multiplier: float = 0.8


@dataclass
class AIConfig:
    model_path: str = "models/signal_filter.pkl"
    min_training_samples: int = 50
    retrain_interval_hours: int = 24
    cold_start_score: float = 0.70


@dataclass
class NotificationConfig:
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    notify_on_trade: bool = True
    notify_on_daily_stop: bool = True
    notify_on_error: bool = True


@dataclass
class AppConfig:
    exchange: ExchangeConfig
    trading: TradingConfig
    risk: RiskConfig
    ai: AIConfig
    notification: NotificationConfig
    log_level: str = "INFO"
    data_dir: str = "data/cache"
    models_dir: str = "models"
    logs_dir: str = "logs"


def load_config() -> AppConfig:
    symbols_raw = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,AVAX/USDT,LINK/USDT,MATIC/USDT")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    exchange_cfg = ExchangeConfig(
        api_key=os.getenv("MEXC_API_KEY", ""),
        secret_key=os.getenv("MEXC_SECRET_KEY", ""),
        use_futures=os.getenv("USE_FUTURES", "true").lower() == "true",
        leverage=int(os.getenv("LEVERAGE", "3")),
        sandbox=os.getenv("MEXC_SANDBOX", "false").lower() == "true",
    )

    trading_cfg = TradingConfig(
        symbols=symbols,
        trading_mode=os.getenv("TRADING_MODE", "paper"),
        min_conviction_score=int(os.getenv("MIN_CONVICTION_SCORE", "5")),
        max_concurrent_trades=int(os.getenv("MAX_CONCURRENT_TRADES", "3")),
        ai_confidence_threshold=float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.60")),
        min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "2.0")),
    )

    risk_cfg = RiskConfig(
        capital_usdt=float(os.getenv("CAPITAL_USDT", "1000.0")),
        max_risk_per_trade_pct=float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0")),
        max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "3.0")),
        min_rr_ratio=float(os.getenv("MIN_RR_RATIO", "2.0")),
    )

    ai_cfg = AIConfig()

    notif_cfg = NotificationConfig(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)

    return AppConfig(
        exchange=exchange_cfg,
        trading=trading_cfg,
        risk=risk_cfg,
        ai=ai_cfg,
        notification=notif_cfg,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
