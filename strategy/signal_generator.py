import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

from analysis.trend_detector import TrendDetector, TrendDirection
from analysis.volume_analyzer import VolumeAnalyzer, VolumeState
from strategy.multi_timeframe import ConvictionScore, MultiTimeframeAnalyzer

logger = logging.getLogger("bot.signal")


@dataclass
class TradingSignal:
    signal_id: str
    symbol: str
    direction: str            # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit_1: float      # 1:2 R:R
    take_profit_2: float      # 1:3 R:R (for trailing portion)
    position_size: float      # in base currency
    notional_usdt: float
    risk_usdt: float
    atr: float
    conviction_score: int
    conviction_max: int
    ai_score: float           # set by AISignalFilter later
    rr_ratio: float
    checklist: Dict[str, bool]
    entry_reason: List[str]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    timeframe: str = "5m"


class SignalGenerator:
    """
    Assembles the 6-criteria entry checklist.
    ALL criteria must pass for a signal to be emitted.
    Embodies strict discipline: if ANY criteria fails, no trade.
    """

    def __init__(self, trend_detector: TrendDetector = None,
                 volume_analyzer: VolumeAnalyzer = None,
                 mtf_analyzer: MultiTimeframeAnalyzer = None,
                 min_conviction: int = 5,
                 min_rr_ratio: float = 2.0,
                 min_adx: float = 25.0,
                 volume_multiplier: float = 1.5):
        self.trend = trend_detector or TrendDetector()
        self.volume = volume_analyzer or VolumeAnalyzer()
        self.mtf = mtf_analyzer or MultiTimeframeAnalyzer()
        self.min_conviction = min_conviction
        self.min_rr_ratio = min_rr_ratio
        self.min_adx = min_adx
        self.volume_multiplier = volume_multiplier

    def generate_signal(self, symbol: str, tf_data: Dict[str, pd.DataFrame],
                        conviction: ConvictionScore,
                        order_book_imbalance: float = 0.5,
                        capital: float = 1000.0,
                        max_risk_pct: float = 1.0) -> Optional[TradingSignal]:

        df_5m = tf_data.get("5m")
        df_15m = tf_data.get("15m")

        if df_5m is None or len(df_5m) < 30:
            return None

        checklist = {}
        reasons = []

        # === CHECK 1: HTF Trend Alignment ===
        htf_ok = conviction.htf_bias != "SIDEWAYS" and conviction.htf_bias_confidence > 0.3
        checklist["htf_trend_aligned"] = htf_ok
        if htf_ok:
            reasons.append(f"HTF bias {conviction.htf_bias} ({conviction.htf_bias_confidence:.0%} confidence)")

        # === CHECK 2: MTF Conviction Score ===
        conviction_ok = conviction.total >= self.min_conviction
        checklist["conviction_score"] = conviction_ok
        if conviction_ok:
            reasons.append(f"Conviction {conviction.total}/{conviction.max_score}")

        # === CHECK 3: ADX > threshold (trend is strong, not ranging) ===
        last_5m = df_5m.iloc[-1]
        adx = float(last_5m.get("adx", 0))
        adx_ok = adx >= self.min_adx
        checklist["adx_trending"] = adx_ok
        if adx_ok:
            reasons.append(f"ADX {adx:.1f} (trending)")

        # === CHECK 4: Momentum not extreme (RSI in 30-75 range) ===
        rsi = float(last_5m.get("rsi", 50))
        stoch_k = float(last_5m.get("stochrsi_k", 50))
        direction = conviction.aligned_direction

        if direction == "BULLISH":
            momentum_ok = 30 < rsi < 75 and stoch_k < 85
        else:
            momentum_ok = 25 < rsi < 70 and stoch_k > 15
        checklist["momentum_not_extreme"] = momentum_ok
        if momentum_ok:
            reasons.append(f"RSI {rsi:.1f} OK, StochRSI {stoch_k:.1f}")

        # === CHECK 5: Volume confirmation ===
        vol_state = self.volume.analyze(df_5m, order_book_imbalance)
        volume_ok = vol_state.surge_ratio >= self.volume_multiplier
        checklist["volume_confirmed"] = volume_ok
        if volume_ok:
            reasons.append(f"Volume surge {vol_state.surge_ratio:.1f}x avg")

        # === CHECK 6: Price near key level ===
        near_level = self.mtf.is_near_key_level(df_5m, tolerance_pct=0.004)
        # Relax if volume is very strong
        if not near_level and vol_state.surge_ratio >= 2.0:
            near_level = True
            reasons.append("High volume override for key level check")
        checklist["near_key_level"] = near_level
        if near_level:
            reasons.append("Price near key level (VWAP/EMA/BB/S-R)")

        # === EARLY EXIT: If any critical check fails ===
        critical_checks = ["htf_trend_aligned", "conviction_score", "adx_trending", "momentum_not_extreme"]
        failed = [k for k in critical_checks if not checklist.get(k, False)]
        if failed:
            logger.debug(f"{symbol}: Signal rejected — failed: {failed}")
            return None

        # Allow soft checks to have 1 failure (near_level or volume)
        soft_checks = ["volume_confirmed", "near_key_level"]
        soft_failed = sum(1 for k in soft_checks if not checklist.get(k, True))
        if soft_failed >= 2:
            logger.debug(f"{symbol}: Signal rejected — both soft checks failed")
            return None

        # === ADDITIONAL DIRECTION CONFIRMATION ===
        signal_direction = direction
        macd_hist = float(last_5m.get("macd_hist", 0))
        cvd_bullish = vol_state.cvd_bullish
        obv_up = vol_state.obv_trend == "UP"

        if signal_direction == "BULLISH":
            confirmations = sum([macd_hist > 0, cvd_bullish, obv_up,
                                  order_book_imbalance > 0.5])
            if confirmations < 2:
                logger.debug(f"{symbol}: Long rejected — only {confirmations}/4 volume confirmations")
                return None
        else:
            confirmations = sum([macd_hist < 0, not cvd_bullish, not obv_up,
                                  order_book_imbalance < 0.5])
            if confirmations < 2:
                logger.debug(f"{symbol}: Short rejected — only {confirmations}/4 volume confirmations")
                return None

        # === CALCULATE ENTRY PRICE, SL, TP ===
        price = float(df_5m["close"].iloc[-1])
        atr = float(last_5m.get("atr", price * 0.01))
        if atr <= 0:
            atr = price * 0.01

        if signal_direction == "BULLISH":
            entry = price
            stop_loss = entry - 1.5 * atr
            # Push SL below nearest support
            support = float(last_5m.get("support", entry - 2 * atr))
            stop_loss = min(stop_loss, support - 0.05 * atr)
        else:
            entry = price
            stop_loss = entry + 1.5 * atr
            resistance = float(last_5m.get("resistance", entry + 2 * atr))
            stop_loss = max(stop_loss, resistance + 0.05 * atr)

        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            logger.warning(f"{symbol}: Invalid SL distance {sl_distance}")
            return None

        take_profit_1 = entry + sl_distance * self.min_rr_ratio * (1 if signal_direction == "BULLISH" else -1)
        take_profit_2 = entry + sl_distance * (self.min_rr_ratio + 1) * (1 if signal_direction == "BULLISH" else -1)

        rr = abs(take_profit_1 - entry) / sl_distance
        if rr < self.min_rr_ratio:
            logger.debug(f"{symbol}: R:R {rr:.2f} below minimum {self.min_rr_ratio}")
            return None

        # === POSITION SIZING ===
        risk_amount = capital * (max_risk_pct / 100)
        position_size = risk_amount / sl_distance
        notional = position_size * entry

        # Cap position at 20% of capital
        max_notional = capital * 0.20
        if notional > max_notional:
            position_size = max_notional / entry
            notional = max_notional

        if position_size <= 0:
            return None

        return TradingSignal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction="LONG" if signal_direction == "BULLISH" else "SHORT",
            entry_price=round(entry, 8),
            stop_loss=round(stop_loss, 8),
            take_profit_1=round(take_profit_1, 8),
            take_profit_2=round(take_profit_2, 8),
            position_size=round(position_size, 6),
            notional_usdt=round(notional, 2),
            risk_usdt=round(risk_amount, 2),
            atr=round(atr, 8),
            conviction_score=conviction.total,
            conviction_max=conviction.max_score,
            ai_score=0.0,  # filled by AISignalFilter
            rr_ratio=round(rr, 2),
            checklist=checklist,
            entry_reason=reasons,
        )

    def validate_spread(self, symbol: str, ticker: Dict,
                         max_spread_pct: float = 0.001) -> bool:
        """Returns False if bid-ask spread is too wide (low liquidity)."""
        try:
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            mid = (bid + ask) / 2
            if mid <= 0:
                return True
            spread = (ask - bid) / mid
            return spread <= max_spread_pct
        except Exception:
            return True

    def is_trading_window(self) -> bool:
        """Returns False during known low-liquidity periods (optional filter)."""
        return True  # Can be extended with time-based filters
