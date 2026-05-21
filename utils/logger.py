import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
import json


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def get_logger(name: str, log_level: str = "INFO", logs_dir: str = "logs") -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    console_handler.setLevel(level)

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, "trading.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(level)

    trades_handler = RotatingFileHandler(
        os.path.join(logs_dir, "trades.log"),
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
    )
    trades_handler.setFormatter(JSONFormatter())
    trades_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    trades_logger = logging.getLogger(f"{name}.trades")
    trades_logger.addHandler(trades_handler)
    trades_logger.setLevel(logging.INFO)

    return logger


class BotLogger:
    def __init__(self, log_level: str = "INFO", logs_dir: str = "logs"):
        self.logger = get_logger("bot", log_level, logs_dir)
        self.trades_logger = get_logger("bot.trades", log_level, logs_dir)

    def _log(self, level: str, message: str, **kwargs):
        extra = {"extra": kwargs} if kwargs else {}
        getattr(self.logger, level)(message, extra=extra if kwargs else None)

    def info(self, message: str, **kwargs):
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("warning", message, **kwargs)

    def error(self, message: str, exc: Exception = None, **kwargs):
        if exc:
            kwargs["exception"] = str(exc)
        self._log("error", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("debug", message, **kwargs)

    def trade_signal(self, symbol: str, direction: str, entry: float, sl: float,
                     tp1: float, conviction: int, ai_score: float, checklist: dict):
        self.trades_logger.info("SIGNAL_GENERATED", extra={
            "extra": {
                "event": "signal",
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "conviction_score": conviction,
                "ai_score": round(ai_score, 4),
                "checklist": checklist,
            }
        })

    def trade_open(self, trade_id: str, symbol: str, side: str, entry: float,
                   sl: float, tp1: float, size: float, notional: float):
        self.trades_logger.info("TRADE_OPEN", extra={
            "extra": {
                "event": "trade_open",
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit_1": tp1,
                "size": size,
                "notional_usdt": notional,
            }
        })

    def trade_close(self, trade_id: str, symbol: str, side: str, entry: float,
                    exit_price: float, pnl_usdt: float, pnl_pct: float, reason: str):
        self.trades_logger.info("TRADE_CLOSE", extra={
            "extra": {
                "event": "trade_close",
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl_usdt": round(pnl_usdt, 4),
                "pnl_pct": round(pnl_pct, 4),
                "reason": reason,
                "result": "WIN" if pnl_usdt > 0 else "LOSS",
            }
        })

    def daily_summary(self, date: str, trades: int, wins: int, losses: int,
                      pnl: float, win_rate: float, halted: bool):
        self.trades_logger.info("DAILY_SUMMARY", extra={
            "extra": {
                "event": "daily_summary",
                "date": date,
                "total_trades": trades,
                "wins": wins,
                "losses": losses,
                "realized_pnl_usdt": round(pnl, 4),
                "win_rate_pct": round(win_rate * 100, 2),
                "trading_halted": halted,
            }
        })

    def risk_halt(self, reason: str, daily_pnl: float, daily_pnl_pct: float):
        self.logger.warning("TRADING_HALTED", extra={
            "extra": {
                "event": "risk_halt",
                "reason": reason,
                "daily_pnl": round(daily_pnl, 4),
                "daily_pnl_pct": round(daily_pnl_pct * 100, 2),
            }
        })
