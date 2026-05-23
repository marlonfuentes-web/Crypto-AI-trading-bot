"""
MEXC Crypto AI Scalping Bot
===========================
A production-grade scalping bot for MEXC exchange.

Timeframes: 5m, 15m, 30m, 1h, 4h, 1d (multi-timeframe conviction)
Strategy: Trend-following scalping with AI signal filtering
Risk: 1% per trade, 3% daily limit, full SL/TP on every trade

Usage:
  python main.py               # Start bot (paper mode by default)
  python main.py --live        # Live trading mode
  python main.py --status      # Show current status
"""

import asyncio
import signal
import sys
import argparse
import time
from datetime import datetime, timezone
import logging

from config import load_config, AppConfig
from utils.logger import BotLogger
from utils.notifier import create_notifier
from exchange.mexc_client import MEXCClient
from data.market_data import MarketDataService
from ai.signal_filter import AISignalFilter
from risk.risk_manager import RiskManager
from risk.trade_manager import TradeManager
from strategy.scalping_strategy import ScalpingStrategy

logger = logging.getLogger("bot.main")


class TradingBot:
    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self._startup_time = time.time()
        self._last_daily_reset = datetime.now(timezone.utc).date()
        self._last_retrain_check = time.time()

        # Initialize logger
        self.bot_logger = BotLogger(config.log_level, config.logs_dir)

        # Initialize components
        self.exchange = MEXCClient(
            api_key=config.exchange.api_key,
            secret_key=config.exchange.secret_key,
            use_futures=config.exchange.use_futures,
            leverage=config.exchange.leverage,
            sandbox=config.exchange.sandbox,
        )

        self.market_data = MarketDataService(self.exchange, config)

        self.notifier = create_notifier(
            config.notification.telegram_bot_token,
            config.notification.telegram_chat_id,
        )

        self.risk = RiskManager(
            capital=config.risk.capital_usdt,
            max_risk_pct=config.risk.max_risk_per_trade_pct,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
            max_concurrent=config.trading.max_concurrent_trades,
            min_rr=config.trading.min_rr_ratio,
        )

        self.trade_manager = TradeManager(
            exchange_client=self.exchange,
            risk_manager=self.risk,
            mode=config.trading.trading_mode,
            notifier=self.notifier,
        )

        self.ai_filter = AISignalFilter(
            model_path=config.ai.model_path,
            min_training_samples=config.ai.min_training_samples,
            cold_start_score=config.ai.cold_start_score,
            retrain_interval_hours=config.ai.retrain_interval_hours,
        )

        self.strategy = ScalpingStrategy(
            market_data_service=self.market_data,
            config=config,
            signal_filter=self.ai_filter,
            risk_manager=self.risk,
            trade_manager=self.trade_manager,
            bot_logger=self.bot_logger,
        )

        # Quantum components (activated only when QUANTUM_ENABLED=true)
        if config.quantum is not None and config.quantum.enabled:
            self._init_quantum_components()

    def _init_quantum_components(self):
        """Wire quantum scorer, neural brain, Kelly sizer, and adaptive aggression."""
        from strategy.quantum_scorer import QuantumConvictionScorer
        from ai.lstm_model import LSTMSignalModel
        from ai.neural_brain import QuantumNeuralBrain
        from risk.kelly_sizer import KellyPositionSizer
        from strategy.adaptive_aggression import AdaptiveAggressionController

        cfg = self.config.quantum
        logger.info("Quantum mode ENABLED — initialising quantum components")

        # 1. Quantum conviction scorer
        quantum_scorer = QuantumConvictionScorer()
        self.strategy.quantum_scorer = quantum_scorer
        # Also wire into mtf_analyzer for inline scoring
        self.strategy.mtf_analyzer.quantum_scorer = quantum_scorer

        # 2. Neural brain (LSTM + XGBoost + MLP ensemble)
        lstm_model = LSTMSignalModel(
            model_path=cfg.lstm_model_path,
            mlp_path=cfg.mlp_model_path,
        )
        neural_brain = QuantumNeuralBrain(
            xgb_filter=self.ai_filter,
            lstm_model=lstm_model,
            retrain_interval_hours=cfg.retrain_interval_hours,
            min_training_samples=cfg.min_training_samples,
        )
        self.strategy.neural_brain = neural_brain

        # 3. Kelly position sizer
        if cfg.use_kelly:
            kelly = KellyPositionSizer(
                min_risk_pct=cfg.kelly_min_risk_pct,
                max_risk_pct=cfg.kelly_max_risk_pct,
                fallback_risk_pct=cfg.kelly_fallback_risk_pct,
            )
            self.strategy.kelly_sizer = kelly

        # 4. Adaptive aggression controller
        if cfg.adaptive_aggression:
            ctrl = AdaptiveAggressionController(
                high_clarity_threshold=cfg.high_clarity_threshold,
                mid_clarity_threshold=cfg.mid_clarity_threshold,
            )
            self.strategy.aggression_ctrl = ctrl

        logger.info(
            f"Quantum components ready: scorer=OK neural_brain=OK "
            f"kelly={'OK' if cfg.use_kelly else 'disabled'} "
            f"aggression={'OK' if cfg.adaptive_aggression else 'disabled'}"
        )

    async def initialize(self):
        self.bot_logger.info("=" * 60)
        self.bot_logger.info("MEXC Crypto AI Scalping Bot Starting")
        self.bot_logger.info(f"Mode: {self.config.trading.trading_mode.upper()}")
        self.bot_logger.info(f"Symbols: {', '.join(self.config.trading.symbols)}")
        self.bot_logger.info(f"Timeframes: {', '.join(self.config.trading.timeframes)}")
        self.bot_logger.info(f"Capital: ${self.config.risk.capital_usdt:.2f} USDT")
        self.bot_logger.info(f"Max risk/trade: {self.config.risk.max_risk_per_trade_pct}%")
        self.bot_logger.info(f"Daily loss limit: {self.config.risk.max_daily_loss_pct}%")
        self.bot_logger.info(f"AI threshold: {self.config.trading.ai_confidence_threshold}")
        if self.config.quantum and self.config.quantum.enabled:
            self.bot_logger.info("Quantum mode: ENABLED (scorer + neural brain + Kelly + aggression)")
        self.bot_logger.info("=" * 60)

        # Connect to exchange
        if self.config.trading.trading_mode == "live" or self.config.exchange.api_key:
            try:
                await self.exchange.initialize()
                self.bot_logger.info("Exchange connected successfully")
            except Exception as e:
                self.bot_logger.error("Exchange connection failed", exc=e)
                if self.config.trading.trading_mode == "live":
                    raise

        # Warm cache
        self.bot_logger.info("Warming market data cache...")
        await self.market_data.warm_cache(self.config.trading.symbols)
        self.bot_logger.info("Cache warmed")

        # Try to load AI model
        ai_status = "TRAINED" if self.ai_filter.is_trained else f"COLD START ({self.ai_filter.training_samples} samples)"
        self.bot_logger.info(f"AI model status: {ai_status}")

        # Send startup notification
        await self.notifier.send_startup(
            symbols=len(self.config.trading.symbols),
            mode=self.config.trading.trading_mode,
            capital=self.config.risk.capital_usdt,
        )

    async def run(self):
        self.running = True
        self._register_signal_handlers()

        self.bot_logger.info("Bot is running. Press Ctrl+C to stop.")

        scan_interval = self.config.trading.scan_interval_seconds

        while self.running:
            cycle_start = time.time()

            try:
                await self._tick()
            except Exception as e:
                self.bot_logger.error(f"Error in main tick: {e}", exc=e)
                await self.notifier.send_error_alert(f"Main tick error: {str(e)[:200]}")

            # Sleep for remaining interval
            elapsed = time.time() - cycle_start
            sleep_time = max(1, scan_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _tick(self):
        # Daily reset check (UTC midnight)
        await self._check_daily_reset()

        # Check if trading is halted
        if self.risk.daily_stats.is_trading_halted:
            if self.strategy.cycles_run % 10 == 0:
                summary = self.risk.get_summary()
                self.bot_logger.info(
                    f"Trading HALTED: {self.risk.daily_stats.halt_reason} | "
                    f"Daily P&L: {summary['realized_pnl']:+.2f} USDT"
                )
            return

        # Run strategy cycle
        results = await self.strategy.run_cycle()

        # Log cycle stats
        summary = self.risk.get_summary()
        stats = self.strategy.get_stats()

        self.bot_logger.debug(
            "Cycle complete",
            cycle=self.strategy.cycles_run,
            scanned=results["scanned"],
            signals=results["signals"],
            trades=results["trades"],
            open_positions=summary["open_positions"],
            daily_pnl=f"{summary['realized_pnl']:+.2f}",
            win_rate=f"{summary['win_rate']}%",
            ai_trained=stats["ai_trained"],
            ai_samples=stats["ai_samples"],
        )

        # Periodic detailed status log (every 10 cycles)
        if self.strategy.cycles_run % 10 == 0:
            self.bot_logger.info(
                f"STATUS | Trades today: {summary['trades']} | "
                f"Win rate: {summary['win_rate']}% | "
                f"P&L: {summary['realized_pnl']:+.2f} USDT | "
                f"Open: {summary['open_positions']}/{self.config.trading.max_concurrent_trades} | "
                f"AI: {stats['ai_samples']} samples"
            )

        # Periodic AI retraining check (every 6 hours)
        if time.time() - self._last_retrain_check > 6 * 3600:
            self._last_retrain_check = time.time()
            if self.strategy.try_retrain_model():
                self.bot_logger.info("AI model retrained successfully")

    async def _check_daily_reset(self):
        today = datetime.now(timezone.utc).date()
        if today > self._last_daily_reset:
            self._last_daily_reset = today

            # Log and send daily summary before reset
            summary = self.risk.get_summary()
            self.bot_logger.daily_summary(
                date=summary["date"],
                trades=summary["trades"],
                wins=summary["wins"],
                losses=summary["losses"],
                pnl=summary["realized_pnl"],
                win_rate=summary["win_rate"] / 100,
                halted=summary["halted"],
            )
            await self.notifier.send_daily_summary(
                date=summary["date"],
                trades=summary["trades"],
                wins=summary["wins"],
                losses=summary["losses"],
                pnl=summary["realized_pnl"],
                win_rate=summary["win_rate"] / 100,
            )

            # Reset daily stats
            self.risk.reset_daily_stats()
            self.bot_logger.info("Daily stats reset for new trading day")

            # Try AI retraining at midnight
            self.strategy.try_retrain_model()

    def _register_signal_handlers(self):
        def handler(sig, frame):
            self.bot_logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    async def shutdown(self):
        self.bot_logger.info("Shutting down bot...")
        self.running = False

        # Log final summary
        if self.trade_manager.get_open_count() > 0:
            self.bot_logger.warning(
                f"Shutting down with {self.trade_manager.get_open_count()} open trades. "
                f"Positions remain open on exchange."
            )

        summary = self.risk.get_summary()
        self.bot_logger.info(
            f"Final: Trades={summary['trades']} "
            f"Wins={summary['wins']} Losses={summary['losses']} "
            f"P&L={summary['realized_pnl']:+.2f} USDT"
        )

        await self.exchange.close()
        self.bot_logger.info("Bot stopped cleanly")

    def print_status(self):
        summary = self.risk.get_summary()
        stats = self.strategy.get_stats()
        print("\n" + "=" * 50)
        print("MEXC Scalping Bot Status")
        print("=" * 50)
        print(f"Mode:          {self.config.trading.trading_mode.upper()}")
        print(f"Capital:       ${self.config.risk.capital_usdt:.2f} USDT")
        print(f"Daily P&L:     {summary['realized_pnl']:+.2f} USDT ({summary['pnl_pct']:+.2f}%)")
        print(f"Trades:        {summary['trades']} (Wins: {summary['wins']}, Losses: {summary['losses']})")
        print(f"Win Rate:      {summary['win_rate']}%")
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Portfolio Heat: {summary['portfolio_heat_pct']}%")
        print(f"AI Model:      {'Trained' if stats['ai_trained'] else 'Cold Start'} ({stats['ai_samples']} samples)")
        print(f"Halted:        {summary['halted']}")
        if summary["halt_reason"]:
            print(f"Halt Reason:   {summary['halt_reason']}")
        print("=" * 50 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="MEXC Crypto AI Scalping Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--status", action="store_true", help="Show current status and exit")
    args = parser.parse_args()

    config = load_config()

    if args.live:
        config.trading.trading_mode = "live"
        print("⚠️  LIVE TRADING MODE ENABLED — real money at risk")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    bot = TradingBot(config)

    if args.status:
        bot.print_status()
        return

    await bot.initialize()

    try:
        await bot.run()
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
