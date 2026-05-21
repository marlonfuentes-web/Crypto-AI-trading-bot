import asyncio
import aiohttp
from typing import Optional
import logging

logger = logging.getLogger("bot.notifier")


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and chat_id)

    async def _send(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                }
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    async def send_trade_open(self, symbol: str, side: str, entry: float,
                               sl: float, tp1: float, size: float,
                               conviction: int, ai_score: float):
        emoji = "🟢" if side.upper() == "BUY" else "🔴"
        sl_pct = abs(entry - sl) / entry * 100
        tp_pct = abs(tp1 - entry) / entry * 100
        rr = tp_pct / sl_pct if sl_pct > 0 else 0
        msg = (
            f"{emoji} <b>TRADE OPEN</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: <b>{side.upper()}</b>\n"
            f"Entry: <b>${entry:.4f}</b>\n"
            f"Stop Loss: ${sl:.4f} (-{sl_pct:.2f}%)\n"
            f"Take Profit: ${tp1:.4f} (+{tp_pct:.2f}%)\n"
            f"R:R Ratio: 1:{rr:.1f}\n"
            f"Size: {size:.6f}\n"
            f"Conviction: {conviction}/8 | AI: {ai_score:.0%}"
        )
        await self._send(msg)

    async def send_trade_close(self, symbol: str, side: str, entry: float,
                                exit_price: float, pnl_usdt: float, pnl_pct: float, reason: str):
        result_emoji = "✅" if pnl_usdt >= 0 else "❌"
        pnl_sign = "+" if pnl_usdt >= 0 else ""
        msg = (
            f"{result_emoji} <b>TRADE CLOSED</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Side: {side.upper()}\n"
            f"Entry: ${entry:.4f} → Exit: ${exit_price:.4f}\n"
            f"P&L: <b>{pnl_sign}{pnl_usdt:.2f} USDT ({pnl_sign}{pnl_pct:.2f}%)</b>\n"
            f"Reason: {reason}"
        )
        await self._send(msg)

    async def send_daily_summary(self, date: str, trades: int, wins: int,
                                  losses: int, pnl: float, win_rate: float):
        pnl_emoji = "📈" if pnl >= 0 else "📉"
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"{pnl_emoji} <b>DAILY SUMMARY — {date}</b>\n"
            f"Total Trades: {trades}\n"
            f"Wins: {wins} | Losses: {losses}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Net P&L: <b>{pnl_sign}{pnl:.2f} USDT</b>"
        )
        await self._send(msg)

    async def send_halt_alert(self, reason: str, daily_pnl: float):
        pnl_sign = "+" if daily_pnl >= 0 else ""
        msg = (
            f"🛑 <b>TRADING HALTED</b>\n"
            f"Reason: {reason}\n"
            f"Daily P&L: {pnl_sign}{daily_pnl:.2f} USDT\n"
            f"Bot will resume at next UTC midnight reset."
        )
        await self._send(msg)

    async def send_error_alert(self, error: str):
        msg = f"⚠️ <b>BOT ERROR</b>\n{error}"
        await self._send(msg)

    async def send_startup(self, symbols: int, mode: str, capital: float):
        msg = (
            f"🤖 <b>BOT STARTED</b>\n"
            f"Mode: <b>{mode.upper()}</b>\n"
            f"Watching: {symbols} symbols\n"
            f"Capital: ${capital:.2f} USDT"
        )
        await self._send(msg)


class NullNotifier:
    """No-op notifier when Telegram is not configured."""

    async def send_trade_open(self, *args, **kwargs): pass
    async def send_trade_close(self, *args, **kwargs): pass
    async def send_daily_summary(self, *args, **kwargs): pass
    async def send_halt_alert(self, *args, **kwargs): pass
    async def send_error_alert(self, *args, **kwargs): pass
    async def send_startup(self, *args, **kwargs): pass


def create_notifier(token: Optional[str], chat_id: Optional[str]):
    if token and chat_id:
        return TelegramNotifier(token, chat_id)
    return NullNotifier()
