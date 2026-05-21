import asyncio
import time
from typing import Dict, List, Optional, Set
import pandas as pd
import logging

from analysis.indicators import TechnicalIndicators
from analysis.trend_detector import TrendDetector
from analysis.volume_analyzer import VolumeAnalyzer
from strategy.multi_timeframe import MultiTimeframeAnalyzer, ConvictionScore
from strategy.signal_generator import SignalGenerator, TradingSignal
from ai.feature_engineering import FeatureEngineer
from ai.signal_filter import AISignalFilter

logger = logging.getLogger("bot.strategy")


class ScalpingStrategy:
    """
    Top-level strategy orchestrator. Embodies the 15-year experienced trader philosophy:
    - Strict multi-filter entry: ALL criteria must pass
    - No chasing entries (use limit orders)
    - Respects daily loss limits absolutely
    - Knows when NOT to trade (sideways, low conviction, bad risk/reward)
    - Capital protection above all else
    """

    def __init__(self, market_data_service, config,
                 signal_filter: AISignalFilter = None,
                 risk_manager=None, trade_manager=None,
                 bot_logger=None):
        self.market_data = market_data_service
        self.config = config

        self.trend_detector = TrendDetector()
        self.volume_analyzer = VolumeAnalyzer(
            surge_multiplier=config.trading.volume_surge_multiplier
        )
        self.mtf_analyzer = MultiTimeframeAnalyzer(
            trend_detector=self.trend_detector,
            volume_analyzer=self.volume_analyzer,
            min_conviction=config.trading.min_conviction_score,
        )
        self.signal_gen = SignalGenerator(
            trend_detector=self.trend_detector,
            volume_analyzer=self.volume_analyzer,
            mtf_analyzer=self.mtf_analyzer,
            min_conviction=config.trading.min_conviction_score,
            min_rr_ratio=config.trading.min_rr_ratio,
            min_adx=config.trading.min_adx_threshold,
            volume_multiplier=config.trading.volume_surge_multiplier,
        )
        self.feature_eng = FeatureEngineer()
        self.ai_filter = signal_filter or AISignalFilter(
            model_path=config.ai.model_path,
            min_training_samples=config.ai.min_training_samples,
            cold_start_score=config.ai.cold_start_score,
        )
        self.risk = risk_manager
        self.trade_mgr = trade_manager
        self.bot_logger = bot_logger

        # Cooldown tracking: after SL hit, don't trade symbol for N minutes
        self._symbol_cooldowns: Dict[str, float] = {}
        self.cooldown_minutes = 45

        # Track last signal time to avoid over-trading single symbols
        self._last_signal_time: Dict[str, float] = {}
        self.min_signal_interval_seconds = 600  # 10 min between signals per symbol

        # Consecutive loss circuit breaker per symbol
        self._consecutive_losses: Dict[str, int] = {}
        self.max_consecutive_losses = 3          # 2-hour cooldown after 3 losses in a row
        self.hard_cooldown_minutes = 120

        # Stats
        self.signals_generated = 0
        self.signals_filtered_ai = 0
        self.signals_traded = 0
        self.cycles_run = 0

    async def evaluate_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Full signal pipeline for one symbol. Returns TradingSignal or None."""

        # Pre-flight checks
        if self.should_skip_symbol(symbol):
            return None

        # Fetch all timeframes concurrently
        tf_data = await self.market_data.get_all_timeframes(symbol)
        if len(tf_data) < 3:
            logger.debug(f"{symbol}: Insufficient timeframe data ({len(tf_data)} TFs)")
            return None

        # Compute indicators on each timeframe
        processed_tf = {}
        for tf, df in tf_data.items():
            if df is not None and len(df) >= 30:
                processed_tf[tf] = TechnicalIndicators.compute_all(df)

        if "5m" not in processed_tf:
            return None

        # MTF conviction scoring (fast filter before AI)
        conviction = self.mtf_analyzer.analyze(processed_tf, symbol)

        if not conviction.is_tradeable:
            logger.debug(f"{symbol}: Low conviction {conviction.total}/{conviction.max_score} "
                         f"bias={conviction.htf_bias}")
            return None

        # Order book imbalance for volume confirmation
        ob_imbalance = await self.market_data.get_order_book_imbalance(symbol)

        # Generate signal (6-criteria checklist)
        capital = self.risk.capital if self.risk else self.config.risk.capital_usdt
        signal = self.signal_gen.generate_signal(
            symbol=symbol,
            tf_data=processed_tf,
            conviction=conviction,
            order_book_imbalance=ob_imbalance,
            capital=capital,
            max_risk_pct=self.config.risk.max_risk_per_trade_pct,
        )

        if signal is None:
            return None

        self.signals_generated += 1

        # AI signal scoring
        vol_state = self.volume_analyzer.analyze(processed_tf["5m"], ob_imbalance)
        features = self.feature_eng.extract_features(
            df_5m=processed_tf["5m"],
            conviction=conviction,
            volume_state=vol_state,
            order_book_imbalance=ob_imbalance,
        )
        ai_score = self.ai_filter.score_signal(features)
        signal.ai_score = ai_score

        if ai_score < self.config.trading.ai_confidence_threshold:
            self.signals_filtered_ai += 1
            logger.debug(f"{symbol}: Signal filtered by AI score {ai_score:.3f} < "
                         f"{self.config.trading.ai_confidence_threshold}")
            return None

        # Log signal
        if self.bot_logger:
            self.bot_logger.trade_signal(
                symbol=symbol,
                direction=signal.direction,
                entry=signal.entry_price,
                sl=signal.stop_loss,
                tp1=signal.take_profit_1,
                conviction=signal.conviction_score,
                ai_score=ai_score,
                checklist=signal.checklist,
            )

        logger.info(f"SIGNAL: {symbol} {signal.direction} @ {signal.entry_price:.6f} "
                    f"SL={signal.stop_loss:.6f} TP={signal.take_profit_1:.6f} "
                    f"R:R={signal.rr_ratio} Conviction={signal.conviction_score}/{signal.conviction_max} "
                    f"AI={ai_score:.2f}")

        self._last_signal_time[symbol] = time.time()
        return signal

    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a validated signal. Returns True if trade opened."""
        if not self.trade_mgr:
            logger.warning("No trade manager set — cannot execute signal")
            return False

        # Final risk check
        if self.risk:
            can_trade, reason = self.risk.can_open_trade(signal.symbol)
            if not can_trade:
                logger.info(f"Trade blocked by risk manager: {reason}")
                return False

        trade = await self.trade_mgr.open_trade(signal)
        if trade:
            self.signals_traded += 1
            return True
        return False

    async def run_cycle(self) -> Dict[str, int]:
        """Run one full scan cycle across all configured symbols."""
        self.cycles_run += 1
        results = {"scanned": 0, "signals": 0, "trades": 0, "skipped": 0}

        symbols = self.config.trading.symbols
        for symbol in symbols:
            results["scanned"] += 1

            # Monitor existing trades (non-blocking)
            if self.trade_mgr:
                await self.trade_mgr.monitor_trades(self.market_data)

            # Check risk limits
            if self.risk and not self.risk.can_open_trade()[0]:
                results["skipped"] += 1
                continue

            # Check max concurrent positions
            if self.trade_mgr and self.trade_mgr.get_open_count() >= self.config.trading.max_concurrent_trades:
                results["skipped"] += 1
                continue

            # Evaluate symbol
            try:
                signal = await self.evaluate_symbol(symbol)
                if signal:
                    results["signals"] += 1
                    success = await self.execute_signal(signal)
                    if success:
                        results["trades"] += 1
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {e}", exc_info=True)

        return results

    def should_skip_symbol(self, symbol: str) -> bool:
        # Already have position in this symbol
        if self.trade_mgr and self.trade_mgr.has_position_in(symbol):
            return True

        # Symbol in cooldown after SL hit
        if symbol in self._symbol_cooldowns:
            cooldown_end = self._symbol_cooldowns[symbol]
            if time.time() < cooldown_end:
                return True
            else:
                del self._symbol_cooldowns[symbol]

        # Too soon since last signal for this symbol (avoid overtrading single pair)
        if symbol in self._last_signal_time:
            elapsed = time.time() - self._last_signal_time[symbol]
            if elapsed < self.min_signal_interval_seconds:
                return True

        return False

    def on_trade_closed_with_loss(self, symbol: str):
        """Put symbol in cooldown after a losing trade. Escalates after consecutive losses."""
        self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
        consecutive = self._consecutive_losses[symbol]

        if consecutive >= self.max_consecutive_losses:
            cooldown_until = time.time() + self.hard_cooldown_minutes * 60
            self._symbol_cooldowns[symbol] = cooldown_until
            self._consecutive_losses[symbol] = 0  # reset after hard cooldown
            logger.warning(f"{symbol}: {consecutive} consecutive losses — hard cooldown {self.hard_cooldown_minutes} min")
        else:
            cooldown_until = time.time() + self.cooldown_minutes * 60
            self._symbol_cooldowns[symbol] = cooldown_until
            logger.info(f"{symbol} in cooldown {self.cooldown_minutes} min (loss #{consecutive})")

    def on_trade_closed_with_win(self, symbol: str):
        """Reset consecutive loss counter on a win."""
        self._consecutive_losses[symbol] = 0

    def record_trade_outcome(self, features: Dict, outcome: int):
        """Feed trade result back to AI model for learning."""
        self.ai_filter.record_trade_outcome(features, outcome)

    def try_retrain_model(self) -> bool:
        """Attempt AI model retraining. Called daily."""
        return self.ai_filter.retrain_if_due()

    def get_stats(self) -> Dict:
        return {
            "cycles_run": self.cycles_run,
            "signals_generated": self.signals_generated,
            "signals_filtered_ai": self.signals_filtered_ai,
            "signals_traded": self.signals_traded,
            "ai_trained": self.ai_filter.is_trained,
            "ai_samples": self.ai_filter.training_samples,
            "filter_rate": round(
                self.signals_filtered_ai / max(self.signals_generated, 1) * 100, 1
            ),
        }
