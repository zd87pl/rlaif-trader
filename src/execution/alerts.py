"""
Alert Manager - Multi-channel notifications for RLAIF options trading.

Supports Telegram, Discord webhooks, and console/log output.
Includes rate-limiting and retry queue for network failures.
"""

import json
import os
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ── Emoji indicators ─────────────────────────────────────────────────────
EMOJI = {
    "trade_executed": "🟢",
    "trade_rejected": "🟡",
    "risk_warning": "🟡",
    "kill_switch": "🔴",
    "critical": "🔴",
    "daily_summary": "📊",
    "signal": "💡",
    "info": "ℹ️",
}


class AlertManager:
    """Multi-channel alert dispatcher with rate limiting and retry."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self._config = config or {}

        # Channel credentials (env vars take precedence)
        self._telegram_token: str = os.getenv(
            "TELEGRAM_BOT_TOKEN",
            self._config.get("telegram_bot_token", ""),
        )
        self._telegram_chat_id: str = os.getenv(
            "TELEGRAM_CHAT_ID",
            self._config.get("telegram_chat_id", ""),
        )
        self._discord_webhook_url: str = os.getenv(
            "DISCORD_WEBHOOK_URL",
            self._config.get("discord_webhook_url", ""),
        )

        # Rate limiting: key = alert_type, value = last-sent unix timestamp
        self._rate_limit_interval: int = self._config.get("rate_limit_seconds", 60)
        self._last_sent: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Retry queue for failed network sends
        self._retry_queue: queue.Queue = queue.Queue(maxsize=500)
        self._retry_thread: Optional[threading.Thread] = None
        self._running = True
        self._start_retry_worker()

        logger.info(
            "AlertManager initialised",
            extra={
                "telegram_configured": bool(self._telegram_token and self._telegram_chat_id),
                "discord_configured": bool(self._discord_webhook_url),
            },
        )

    # ── public API ───────────────────────────────────────────────────────

    def send_alert(
        self,
        message: str,
        level: str = "info",
        channel: str = "all",
        alert_type: str = "info",
    ) -> None:
        """
        Dispatch an alert to the requested channel(s).

        Parameters
        ----------
        message   : human-readable alert body
        level     : 'info' | 'warning' | 'critical'
        channel   : 'all' | 'telegram' | 'discord' | 'console'
        alert_type: used for rate-limiting key
        """
        if self._is_rate_limited(alert_type):
            logger.debug("Alert rate-limited", extra={"alert_type": alert_type})
            return

        emoji = EMOJI.get(alert_type, EMOJI.get(level, "ℹ️"))
        formatted = f"{emoji} [{level.upper()}] {message}"
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        full_message = f"{formatted}\n⏰ {timestamp}"

        if channel in ("all", "console"):
            self._send_console(full_message, level)

        if channel in ("all", "telegram"):
            self._send_telegram(full_message)

        if channel in ("all", "discord"):
            self._send_discord(full_message)

        with self._lock:
            self._last_sent[alert_type] = time.time()

    # ── predefined alert types ───────────────────────────────────────────

    def trade_executed(self, order_details: Dict) -> None:
        symbol = order_details.get("symbol", "?")
        strategy = order_details.get("strategy", "?")
        cost = order_details.get("cost", 0)
        qty = order_details.get("quantity", 0)
        msg = (
            f"Trade Executed: {symbol} {strategy}\n"
            f"  Qty: {qty} | Cost: ${cost:,.2f}"
        )
        self.send_alert(msg, level="info", alert_type="trade_executed")

    def trade_rejected(self, reason: str) -> None:
        msg = f"Trade Rejected: {reason}"
        self.send_alert(msg, level="warning", alert_type="trade_rejected")

    def risk_warning(self, metric: str, value: Any, threshold: Any) -> None:
        msg = (
            f"Risk Warning: {metric}\n"
            f"  Current: {value} | Threshold: {threshold}"
        )
        self.send_alert(msg, level="warning", alert_type="risk_warning")

    def kill_switch_triggered(self, reason: str) -> None:
        msg = f"KILL SWITCH TRIGGERED\n  Reason: {reason}"
        self.send_alert(msg, level="critical", alert_type="kill_switch")

    def daily_summary(self, portfolio_state: Dict) -> None:
        equity = portfolio_state.get("equity", 0)
        daily_pnl = portfolio_state.get("daily_pnl", 0)
        weekly_pnl = portfolio_state.get("weekly_pnl", 0)
        positions = portfolio_state.get("positions_open", 0)
        trades = portfolio_state.get("daily_trade_count", 0)
        exposure = portfolio_state.get("total_exposure", 0)
        msg = (
            f"Daily Summary\n"
            f"  Equity:    ${equity:,.2f}\n"
            f"  Daily P/L: ${daily_pnl:+,.2f}\n"
            f"  Weekly P/L:${weekly_pnl:+,.2f}\n"
            f"  Positions: {positions}\n"
            f"  Trades:    {trades}\n"
            f"  Exposure:  ${exposure:,.2f}"
        )
        self.send_alert(msg, level="info", alert_type="daily_summary")

    def signal_generated(self, signal_details: Dict) -> None:
        symbol = signal_details.get("symbol", "?")
        strategy = signal_details.get("strategy", "?")
        confidence = signal_details.get("confidence", 0)
        msg = (
            f"Signal: {symbol} → {strategy}\n"
            f"  Confidence: {confidence:.1%}"
        )
        self.send_alert(msg, level="info", alert_type="signal")

    # ── channel implementations ──────────────────────────────────────────

    def _send_console(self, message: str, level: str = "info") -> None:
        log_fn = {
            "info": logger.info,
            "warning": logger.warning,
            "critical": logger.critical,
        }.get(level, logger.info)
        log_fn(message)

    def _send_telegram(self, message: str) -> None:
        if not (self._telegram_token and self._telegram_chat_id):
            return
        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": message,
            "parse_mode": "HTML",
        }
        self._post_with_retry(url, payload, channel="telegram")

    def _send_discord(self, message: str) -> None:
        if not self._discord_webhook_url:
            return
        payload = {"content": message}
        self._post_with_retry(self._discord_webhook_url, payload, channel="discord")

    # ── network helpers ──────────────────────────────────────────────────

    def _post_with_retry(self, url: str, payload: Dict, channel: str) -> None:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Alert delivery failed ({channel}), queuing for retry",
                extra={"error": str(exc)},
            )
            try:
                self._retry_queue.put_nowait({
                    "url": url,
                    "payload": payload,
                    "channel": channel,
                    "attempts": 1,
                })
            except queue.Full:
                logger.error("Retry queue full – alert dropped")

    def _start_retry_worker(self) -> None:
        def _worker() -> None:
            while self._running:
                try:
                    item = self._retry_queue.get(timeout=5)
                except queue.Empty:
                    continue
                # Exponential back-off: 2^attempts seconds, max 60
                delay = min(2 ** item["attempts"], 60)
                time.sleep(delay)
                try:
                    resp = requests.post(
                        item["url"], json=item["payload"], timeout=10
                    )
                    resp.raise_for_status()
                    logger.info(
                        f"Retry succeeded ({item['channel']})"
                    )
                except Exception:  # noqa: BLE001
                    item["attempts"] += 1
                    if item["attempts"] <= 5:
                        try:
                            self._retry_queue.put_nowait(item)
                        except queue.Full:
                            pass
                    else:
                        logger.error(
                            f"Alert permanently failed after 5 retries ({item['channel']})"
                        )

        self._retry_thread = threading.Thread(
            target=_worker, daemon=True, name="alert-retry"
        )
        self._retry_thread.start()

    # ── rate limiting ────────────────────────────────────────────────────

    def _is_rate_limited(self, alert_type: str) -> bool:
        with self._lock:
            last = self._last_sent.get(alert_type, 0)
            return (time.time() - last) < self._rate_limit_interval

    # ── cleanup ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Gracefully stop the retry worker."""
        self._running = False
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=10)
        logger.info("AlertManager shut down")
