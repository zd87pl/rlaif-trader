"""Portfolio Strategist: AI-driven capital allocation, timing, and strategy selection.

The "CIO agent" that sits above the autotrader loop. Given your wallet balance,
market regime, and risk appetite, it decides:

- What strategy style to use (scalping, day trading, swing, market making)
- How much capital to deploy (risk budget)
- How often to check the market and run experiments
- What composite metric weights to use
- What guidance to give the thesis generator

Produces a PortfolioDirective that configures the entire autotrader system.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── Capital tier definitions ────────────────────────────────────────────

CAPITAL_TIERS = {
    "micro":  {"min": 0,      "max": 1_000,   "label": "Micro (<$1K)"},
    "small":  {"min": 1_000,  "max": 10_000,  "label": "Small ($1K-$10K)"},
    "medium": {"min": 10_000, "max": 50_000,  "label": "Medium ($10K-$50K)"},
    "large":  {"min": 50_000, "max": 250_000, "label": "Large ($50K-$250K)"},
    "whale":  {"min": 250_000, "max": float("inf"), "label": "Whale ($250K+)"},
}


def classify_capital(equity: float) -> str:
    for tier, bounds in CAPITAL_TIERS.items():
        if bounds["min"] <= equity < bounds["max"]:
            return tier
    return "whale"


# ── Directive dataclass ─────────────────────────────────────────────────

@dataclass
class PortfolioDirective:
    """Output of the strategist: configures the entire autotrader system."""

    directive_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Core allocation
    strategy_style: str = "swing"  # scalping, day_trading, swing, market_making
    risk_budget_pct: float = 0.20  # % of equity to deploy
    max_position_pct: float = 0.02  # % per trade

    # Timing
    check_interval_seconds: int = 300  # sentinel scan frequency
    experiment_frequency: str = "hourly"  # continuous, hourly, 4h, daily
    reassess_after_minutes: int = 60  # when to re-evaluate this directive

    # Metric tuning
    composite_weights: Dict[str, float] = field(default_factory=lambda: {
        "sharpe": 0.35, "return": 0.30, "drawdown": 0.20, "hit_rate": 0.15,
    })
    improvement_threshold: float = 0.01

    # Expectations
    expected_daily_return: float = 0.003  # 0.3%
    expected_weekly_return: float = 0.015
    expected_sharpe: float = 1.5
    max_acceptable_drawdown: float = 0.10  # 10%

    # Compound projections (computed from expected_daily_return + wallet_balance)
    projection_daily_pnl: float = 0.0
    projection_weekly_pnl: float = 0.0
    projection_monthly_pnl: float = 0.0
    projection_yearly_pnl: float = 0.0
    projection_monthly_balance: float = 0.0
    projection_yearly_balance: float = 0.0
    projection_drawdown_dollar: float = 0.0

    # Guidance
    symbols_focus: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    thesis_guidance: str = ""  # injected into ThesisGenerator prompt
    reasoning: str = ""  # full explanation for dashboard
    confidence: float = 0.5

    # Context
    capital_tier: str = "small"
    wallet_balance: float = 0.0
    market_regime: str = "unknown"
    risk_preference: str = "moderate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PortfolioDirective:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Deterministic fallback tables ───────────────────────────────────────

_TIER_DEFAULTS = {
    # ── Crypto-calibrated targets ──────────────────────────────────────
    # Crypto: 24/7 = 365 trading days, higher vol = more edge to capture.
    # AI improves over time so targets are "after warm-up" steady state.
    #
    # Reality check (conservative = achievable, aggressive = top-decile):
    #   0.3%/day = ~3x/year compound   (good systematic crypto fund)
    #   0.5%/day = ~6x/year compound   (exceptional)
    #   1.0%/day = ~37x/year compound  (only during strong trends)
    #
    # Equities are ~40-60% of crypto targets (fewer hours, lower vol).
    "micro": {
        "conservative": dict(strategy_style="swing",         risk_budget_pct=0.15, max_position_pct=0.02, check_interval_seconds=900,  experiment_frequency="4h",         composite_weights={"sharpe":0.35,"return":0.25,"drawdown":0.25,"hit_rate":0.15}, improvement_threshold=0.015, expected_daily_return=0.002,  expected_sharpe=1.0, max_acceptable_drawdown=0.08),
        "moderate":     dict(strategy_style="day_trading",   risk_budget_pct=0.25, max_position_pct=0.02, check_interval_seconds=300,  experiment_frequency="hourly",     composite_weights={"sharpe":0.30,"return":0.30,"drawdown":0.20,"hit_rate":0.20}, improvement_threshold=0.01,  expected_daily_return=0.004,  expected_sharpe=1.3, max_acceptable_drawdown=0.12),
        "aggressive":   dict(strategy_style="day_trading",   risk_budget_pct=0.40, max_position_pct=0.02, check_interval_seconds=120,  experiment_frequency="continuous", composite_weights={"sharpe":0.25,"return":0.40,"drawdown":0.15,"hit_rate":0.20}, improvement_threshold=0.008, expected_daily_return=0.008,  expected_sharpe=1.5, max_acceptable_drawdown=0.18),
    },
    "small": {
        "conservative": dict(strategy_style="swing",         risk_budget_pct=0.20, max_position_pct=0.02, check_interval_seconds=600,  experiment_frequency="4h",         composite_weights={"sharpe":0.35,"return":0.25,"drawdown":0.25,"hit_rate":0.15}, improvement_threshold=0.012, expected_daily_return=0.003,  expected_sharpe=1.2, max_acceptable_drawdown=0.08),
        "moderate":     dict(strategy_style="day_trading",   risk_budget_pct=0.30, max_position_pct=0.02, check_interval_seconds=180,  experiment_frequency="hourly",     composite_weights={"sharpe":0.30,"return":0.30,"drawdown":0.20,"hit_rate":0.20}, improvement_threshold=0.01,  expected_daily_return=0.005,  expected_sharpe=1.6, max_acceptable_drawdown=0.12),
        "aggressive":   dict(strategy_style="scalping",      risk_budget_pct=0.45, max_position_pct=0.02, check_interval_seconds=60,   experiment_frequency="continuous", composite_weights={"sharpe":0.25,"return":0.40,"drawdown":0.15,"hit_rate":0.20}, improvement_threshold=0.005, expected_daily_return=0.008,  expected_sharpe=1.8, max_acceptable_drawdown=0.18),
    },
    "medium": {
        "conservative": dict(strategy_style="swing",         risk_budget_pct=0.20, max_position_pct=0.015, check_interval_seconds=600, experiment_frequency="4h",         composite_weights={"sharpe":0.40,"return":0.25,"drawdown":0.25,"hit_rate":0.10}, improvement_threshold=0.012, expected_daily_return=0.003,  expected_sharpe=1.5, max_acceptable_drawdown=0.07),
        "moderate":     dict(strategy_style="day_trading",   risk_budget_pct=0.35, max_position_pct=0.015, check_interval_seconds=120, experiment_frequency="hourly",     composite_weights={"sharpe":0.30,"return":0.35,"drawdown":0.20,"hit_rate":0.15}, improvement_threshold=0.008, expected_daily_return=0.005,  expected_sharpe=2.0, max_acceptable_drawdown=0.10),
        "aggressive":   dict(strategy_style="scalping",      risk_budget_pct=0.50, max_position_pct=0.015, check_interval_seconds=30,  experiment_frequency="continuous", composite_weights={"sharpe":0.25,"return":0.40,"drawdown":0.15,"hit_rate":0.20}, improvement_threshold=0.005, expected_daily_return=0.010,  expected_sharpe=2.2, max_acceptable_drawdown=0.15),
    },
    "large": {
        "conservative": dict(strategy_style="day_trading",   risk_budget_pct=0.20, max_position_pct=0.01, check_interval_seconds=300,  experiment_frequency="hourly",     composite_weights={"sharpe":0.40,"return":0.25,"drawdown":0.25,"hit_rate":0.10}, improvement_threshold=0.012, expected_daily_return=0.002,  expected_sharpe=1.8, max_acceptable_drawdown=0.06),
        "moderate":     dict(strategy_style="day_trading",   risk_budget_pct=0.35, max_position_pct=0.01, check_interval_seconds=120,  experiment_frequency="hourly",     composite_weights={"sharpe":0.35,"return":0.30,"drawdown":0.20,"hit_rate":0.15}, improvement_threshold=0.008, expected_daily_return=0.004,  expected_sharpe=2.2, max_acceptable_drawdown=0.08),
        "aggressive":   dict(strategy_style="market_making", risk_budget_pct=0.50, max_position_pct=0.01, check_interval_seconds=30,   experiment_frequency="continuous", composite_weights={"sharpe":0.30,"return":0.35,"drawdown":0.15,"hit_rate":0.20}, improvement_threshold=0.005, expected_daily_return=0.007,  expected_sharpe=2.5, max_acceptable_drawdown=0.12),
    },
    "whale": {
        "conservative": dict(strategy_style="market_making", risk_budget_pct=0.15, max_position_pct=0.005, check_interval_seconds=120, experiment_frequency="hourly",     composite_weights={"sharpe":0.45,"return":0.20,"drawdown":0.25,"hit_rate":0.10}, improvement_threshold=0.015, expected_daily_return=0.0015, expected_sharpe=2.0, max_acceptable_drawdown=0.04),
        "moderate":     dict(strategy_style="market_making", risk_budget_pct=0.25, max_position_pct=0.005, check_interval_seconds=60,  experiment_frequency="continuous", composite_weights={"sharpe":0.35,"return":0.30,"drawdown":0.20,"hit_rate":0.15}, improvement_threshold=0.008, expected_daily_return=0.003,  expected_sharpe=2.5, max_acceptable_drawdown=0.06),
        "aggressive":   dict(strategy_style="scalping",      risk_budget_pct=0.40, max_position_pct=0.005, check_interval_seconds=15,  experiment_frequency="continuous", composite_weights={"sharpe":0.30,"return":0.35,"drawdown":0.15,"hit_rate":0.20}, improvement_threshold=0.005, expected_daily_return=0.005,  expected_sharpe=3.0, max_acceptable_drawdown=0.10),
    },
}

# Thesis guidance templates per strategy style
_STYLE_GUIDANCE = {
    "scalping": (
        "Focus on high-frequency, short-duration trades. Target 0.1-0.5% per trade with tight stops. "
        "Use momentum indicators on short timeframes (RSI, MACD crossovers). "
        "Minimize hold time — enter and exit within the same candle or next few candles. "
        "Prioritize hit rate over individual trade size."
    ),
    "day_trading": (
        "Focus on intraday trends and momentum. Target 0.5-2% per trade with moderate stops. "
        "Use SMA crossovers, RSI divergences, MACD signals, and volume confirmation. "
        "Close all positions by end of session. Balance win rate with risk/reward ratio."
    ),
    "swing": (
        "Focus on multi-day to multi-week positions capturing larger moves. "
        "Target 2-10% per trade with wider stops based on ATR. "
        "Use trend-following indicators (SMA 20/50 cross, ADX) with mean-reversion entries (RSI oversold). "
        "Prioritize risk-adjusted returns (Sharpe) over frequency."
    ),
    "market_making": (
        "Focus on capturing the bid-ask spread with high frequency. "
        "Place limit orders on both sides of the book. Target very small per-trade profit (0.02-0.1%) "
        "with extremely high hit rate. Use mean-reversion signals, Bollinger Band bounces, "
        "and volume-weighted fair value. Minimize directional exposure."
    ),
}

# Regime adjustments applied on top of base directive
_REGIME_ADJUSTMENTS = {
    "high_vol_bear": {"risk_budget_pct": -0.10, "check_interval_mult": 0.5, "dd_weight_add": 0.10},
    "high_vol_bull": {"risk_budget_pct": -0.05, "check_interval_mult": 0.7, "dd_weight_add": 0.05},
    "low_vol_bear":  {"risk_budget_pct": -0.05, "check_interval_mult": 1.5},
    "low_vol_bull":  {"risk_budget_pct": 0.05,  "check_interval_mult": 1.0},
    "trending":      {"return_weight_add": 0.05},
    "mean_reverting": {"hit_weight_add": 0.05},
    "transition":    {"risk_budget_pct": -0.15, "check_interval_mult": 0.5, "dd_weight_add": 0.10},
}


# ── LLM system prompt ──────────────────────────────────────────────────

_STRATEGIST_SYSTEM_PROMPT = """\
You are a senior portfolio strategist and CIO for an autonomous trading system. \
Your job is to analyze the trader's capital, market conditions, and risk appetite, \
then produce a concrete allocation directive.

You must respond with a JSON object containing exactly these fields:

{
    "strategy_style": "scalping | day_trading | swing | market_making",
    "risk_budget_pct": <float 0.05-0.60>,
    "max_position_pct": <float 0.005-0.02>,
    "check_interval_seconds": <int 15-3600>,
    "experiment_frequency": "continuous | hourly | 4h | daily",
    "composite_weights": {"sharpe": <float>, "return": <float>, "drawdown": <float>, "hit_rate": <float>},
    "improvement_threshold": <float 0.005-0.03>,
    "expected_daily_return": <float>,
    "expected_sharpe": <float>,
    "max_acceptable_drawdown": <float>,
    "symbols_focus": [<strings>],
    "thesis_guidance": "<paragraph of guidance for the strategy generator>",
    "reasoning": "<2-3 paragraphs explaining your rationale>",
    "confidence": <float 0-1>,
    "reassess_after_minutes": <int 15-480>
}

## Key Principles

1. **Capital determines strategy**: Small accounts can't scalp profitably (fees eat edge). \
Large accounts need to avoid market impact. Match strategy to capital.

2. **Risk budget = survival**: Never risk more than the account can handle losing in a week. \
Conservative = survive 20 losing trades. Aggressive = survive 10.

3. **Timing = edge**: Check too often → overtrade and pay fees. Check too rarely → miss entries. \
Match frequency to strategy style (scalping=seconds, swing=hours).

4. **Regime awareness**: High volatility → reduce exposure. Trending → momentum strategies. \
Mean-reverting → range strategies. Transition → defensive.

5. **Composite weights**: Sharpe-heavy in low-vol (consistency matters). Return-heavy in trending \
(capture the move). Drawdown-heavy in crisis (survival first).

6. **Kelly criterion**: Optimal position size = edge / odds. Never bet more than Kelly suggests. \
For most retail accounts, half-Kelly is appropriate.

7. **Realistic expectations**: Day trading with $1K → maybe $2-5/day on a good day. \
Swing trading with $50K → maybe $100-250/week. Don't promise what markets won't deliver.
"""


# ── Strategist class ───────────────────────────────────────────────────

class PortfolioStrategist:
    """AI-driven portfolio allocation and timing advisor.

    Reads wallet balance, assesses market conditions, and produces a
    PortfolioDirective that configures the entire autotrader system.
    """

    def __init__(
        self,
        llm_client: Any = None,
        broker: Any = None,
        model: str = "claude-sonnet-4-6-20250514",
        temperature: float = 0.4,  # lower temp for more consistent advice
    ):
        self.llm_client = llm_client
        self.broker = broker
        self.model = model
        self.temperature = temperature

        self._current_directive: Optional[PortfolioDirective] = None
        self._history: List[PortfolioDirective] = []
        self._last_assess_time: float = 0

    @property
    def current_directive(self) -> Optional[PortfolioDirective]:
        return self._current_directive

    @property
    def history(self) -> List[PortfolioDirective]:
        return self._history

    def needs_reassessment(self) -> bool:
        if self._current_directive is None:
            return True
        elapsed = (time.time() - self._last_assess_time) / 60.0
        return elapsed >= self._current_directive.reassess_after_minutes

    def assess(
        self,
        wallet_balance: Optional[float] = None,
        current_positions: Optional[List[Dict]] = None,
        market_regime: str = "unknown",
        recent_performance: Optional[Dict[str, float]] = None,
        risk_preference: str = "moderate",
        reassess_after_minutes_override: Optional[int] = None,
    ) -> PortfolioDirective:
        """Produce a PortfolioDirective for the current situation.

        Parameters
        ----------
        wallet_balance : float, optional
            Total equity. If None, reads from broker.
        current_positions : list, optional
            Open positions from broker.
        market_regime : str
            Current regime from RegimeDetector.
        recent_performance : dict, optional
            Recent strategy metrics (sharpe, return, drawdown, etc).
        risk_preference : str
            User setting: conservative, moderate, aggressive.
        """
        # Read wallet from broker if not provided
        if wallet_balance is None and self.broker:
            try:
                account = self.broker.get_account()
                wallet_balance = account.get("equity", account.get("balance", 0))
            except Exception as e:
                logger.warning("Could not read wallet from broker: %s", e)
                wallet_balance = 0

        wallet_balance = wallet_balance or 0
        current_positions = current_positions or []
        recent_performance = recent_performance or {}
        capital_tier = classify_capital(wallet_balance)

        # Try LLM assessment first
        directive = None
        if self.llm_client:
            directive = self._llm_assess(
                wallet_balance, capital_tier, current_positions,
                market_regime, recent_performance, risk_preference,
            )

        # Fallback to deterministic
        if directive is None:
            directive = self._deterministic_assess(
                wallet_balance, capital_tier, market_regime,
                recent_performance, risk_preference,
            )

        if reassess_after_minutes_override is not None:
            directive.reassess_after_minutes = max(
                15,
                min(480, int(reassess_after_minutes_override)),
            )

        # Record
        self._current_directive = directive
        self._history.append(directive)
        if len(self._history) > 100:
            self._history = self._history[-100:]
        self._last_assess_time = time.time()

        logger.info(
            "Portfolio directive: style=%s, risk=%.0f%%, check=%ds, tier=%s, regime=%s",
            directive.strategy_style,
            directive.risk_budget_pct * 100,
            directive.check_interval_seconds,
            capital_tier,
            market_regime,
        )

        return directive

    def _llm_assess(
        self,
        balance: float,
        tier: str,
        positions: List[Dict],
        regime: str,
        performance: Dict[str, float],
        risk_pref: str,
    ) -> Optional[PortfolioDirective]:
        """Use Claude to produce a directive."""
        prompt = self._build_prompt(balance, tier, positions, regime, performance, risk_pref)

        try:
            if hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(
                    prompt=prompt, system=_STRATEGIST_SYSTEM_PROMPT,
                    max_tokens=2048, temperature=self.temperature,
                )
            elif hasattr(self.llm_client, "messages"):
                msg = self.llm_client.messages.create(
                    model=self.model, max_tokens=2048,
                    temperature=self.temperature,
                    system=_STRATEGIST_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                response = msg.content[0].text
            else:
                return None

            return self._parse_response(response, balance, tier, regime, risk_pref)

        except Exception as e:
            logger.warning("LLM strategist failed: %s", e)
            return None

    def _build_prompt(
        self, balance: float, tier: str, positions: List[Dict],
        regime: str, performance: Dict, risk_pref: str,
    ) -> str:
        pos_summary = "None" if not positions else "\n".join(
            f"  - {p.get('symbol','?')}: {p.get('qty',0)} @ ${p.get('avg_entry_price',0):.2f} "
            f"(P&L: ${p.get('unrealized_pl',0):.2f})"
            for p in positions[:10]
        )
        perf_str = "\n".join(f"  - {k}: {v}" for k, v in performance.items()) if performance else "No data yet"

        return f"""## Portfolio Assessment Request

**Wallet Balance**: ${balance:,.2f}
**Capital Tier**: {tier} ({CAPITAL_TIERS.get(tier, {}).get('label', tier)})
**Risk Preference**: {risk_pref}
**Market Regime**: {regime}

**Current Positions**:
{pos_summary}

**Recent Strategy Performance**:
{perf_str}

Based on this information, produce your allocation directive as JSON.
Remember: be realistic about expectations. A ${balance:,.0f} account {"cannot generate meaningful returns from scalping due to fees" if balance < 2000 else "has decent flexibility for strategy selection"}.
"""

    def _parse_response(
        self, response: str, balance: float, tier: str, regime: str, risk_pref: str,
    ) -> Optional[PortfolioDirective]:
        """Parse LLM JSON response into a PortfolioDirective."""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            import re
            match = re.search(r'\{[\s\S]*"strategy_style"[\s\S]*\}', response)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Clamp values to safe ranges
        data["risk_budget_pct"] = max(0.05, min(0.60, float(data.get("risk_budget_pct", 0.20))))
        data["max_position_pct"] = max(0.005, min(0.02, float(data.get("max_position_pct", 0.02))))
        data["check_interval_seconds"] = max(15, min(3600, int(data.get("check_interval_seconds", 300))))
        data["confidence"] = max(0, min(1, float(data.get("confidence", 0.5))))
        data["reassess_after_minutes"] = max(15, min(480, int(data.get("reassess_after_minutes", 60))))

        data["capital_tier"] = tier
        data["wallet_balance"] = balance
        data["market_regime"] = regime
        data["risk_preference"] = risk_pref

        directive = PortfolioDirective.from_dict(data)
        self._compute_projections(directive)
        return directive

    @staticmethod
    def _compute_projections(d: PortfolioDirective) -> None:
        """Fill compound projection fields on a directive."""
        bal = d.wallet_balance or 0
        r = d.expected_daily_return
        # Daily P&L (simple, on deployed capital)
        d.projection_daily_pnl = round(bal * r, 2)
        # Weekly (compound over 7 trading days for crypto)
        d.expected_weekly_return = (1 + r) ** 7 - 1
        d.projection_weekly_pnl = round(bal * d.expected_weekly_return, 2)
        # Monthly (30 days compound)
        monthly_r = (1 + r) ** 30 - 1
        d.projection_monthly_pnl = round(bal * monthly_r, 2)
        d.projection_monthly_balance = round(bal * (1 + monthly_r), 2)
        # Yearly (365 days compound)
        yearly_r = (1 + r) ** 365 - 1
        d.projection_yearly_pnl = round(bal * yearly_r, 2)
        d.projection_yearly_balance = round(bal * (1 + yearly_r), 2)
        # Max drawdown in dollars
        d.projection_drawdown_dollar = round(bal * d.max_acceptable_drawdown, 2)

    def _deterministic_assess(
        self,
        balance: float,
        tier: str,
        regime: str,
        performance: Dict[str, float],
        risk_pref: str,
    ) -> PortfolioDirective:
        """Produce a directive from lookup tables (no LLM needed)."""
        # Get base config from tier + preference
        tier_config = _TIER_DEFAULTS.get(tier, _TIER_DEFAULTS["small"])
        base = tier_config.get(risk_pref, tier_config["moderate"]).copy()

        # Apply regime adjustments
        regime_key = regime.lower().replace(" ", "_") if regime != "unknown" else None
        if regime_key and regime_key in _REGIME_ADJUSTMENTS:
            adj = _REGIME_ADJUSTMENTS[regime_key]
            base["risk_budget_pct"] = max(0.05, base["risk_budget_pct"] + adj.get("risk_budget_pct", 0))
            if "check_interval_mult" in adj:
                base["check_interval_seconds"] = int(base["check_interval_seconds"] * adj["check_interval_mult"])
            # Adjust composite weights
            w = base["composite_weights"].copy()
            if "dd_weight_add" in adj:
                w["drawdown"] = min(0.50, w["drawdown"] + adj["dd_weight_add"])
                w["return"] = max(0.10, w["return"] - adj["dd_weight_add"])
            if "return_weight_add" in adj:
                w["return"] = min(0.50, w["return"] + adj["return_weight_add"])
                w["sharpe"] = max(0.15, w["sharpe"] - adj["return_weight_add"])
            if "hit_weight_add" in adj:
                w["hit_rate"] = min(0.30, w["hit_rate"] + adj["hit_weight_add"])
                w["sharpe"] = max(0.15, w["sharpe"] - adj["hit_weight_add"])
            # Normalize to sum=1
            total = sum(w.values())
            base["composite_weights"] = {k: round(v / total, 3) for k, v in w.items()}

        style = base["strategy_style"]
        thesis_guidance = _STYLE_GUIDANCE.get(style, _STYLE_GUIDANCE["swing"])

        # Scale expected returns by recent performance if available
        recent_sharpe = performance.get("sharpe_ratio", 0)
        if recent_sharpe > 1.5:
            base["expected_daily_return"] *= 1.2
            base["risk_budget_pct"] = min(0.60, base["risk_budget_pct"] * 1.1)

        # Build reasoning
        reasoning = (
            f"Capital tier: {CAPITAL_TIERS.get(tier, {}).get('label', tier)} "
            f"(${balance:,.0f}). Risk preference: {risk_pref}. "
            f"Market regime: {regime or 'unknown'}.\n\n"
            f"Recommended strategy: {style}. "
            f"{'Scalping requires tight execution and works best with larger capital for meaningful returns. ' if style == 'scalping' else ''}"
            f"{'Swing trading is appropriate for smaller accounts — fewer trades means lower fee drag. ' if style == 'swing' else ''}"
            f"{'Day trading balances frequency with per-trade edge. ' if style == 'day_trading' else ''}"
            f"{'Market making captures spread with high frequency — needs capital for inventory. ' if style == 'market_making' else ''}"
            f"Deploying {base['risk_budget_pct']*100:.0f}% of capital "
            f"with {base['max_position_pct']*100:.1f}% max per position.\n\n"
            f"Checking market every {base['check_interval_seconds']}s, "
            f"running experiments {base['experiment_frequency']}. "
            f"Realistic daily target: {base['expected_daily_return']*100:.2f}% "
            f"(${balance * base['expected_daily_return']:,.2f}/day). "
            f"Target Sharpe: {base.get('expected_sharpe', 1.5):.1f}."
        )

        # Symbols focus based on tier
        if tier in ("micro", "small"):
            symbols = ["SPY", "QQQ", "AAPL"]
        elif tier == "medium":
            symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
        else:
            symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "META"]

        directive = PortfolioDirective(
            strategy_style=style,
            risk_budget_pct=base["risk_budget_pct"],
            max_position_pct=base.get("max_position_pct", 0.02),
            check_interval_seconds=base["check_interval_seconds"],
            experiment_frequency=base["experiment_frequency"],
            composite_weights=base["composite_weights"],
            improvement_threshold=base.get("improvement_threshold", 0.01),
            expected_daily_return=base["expected_daily_return"],
            expected_sharpe=base.get("expected_sharpe", 1.5),
            max_acceptable_drawdown=base.get("max_acceptable_drawdown", 0.10),
            symbols_focus=symbols,
            thesis_guidance=thesis_guidance,
            reasoning=reasoning,
            confidence=0.7,
            reassess_after_minutes=60,
            capital_tier=tier,
            wallet_balance=balance,
            market_regime=regime,
            risk_preference=risk_pref,
        )
        self._compute_projections(directive)

        # Enrich reasoning with projections
        directive.reasoning += (
            f"\n\nProjected growth (compound):\n"
            f"  Daily:   ${directive.projection_daily_pnl:,.2f} ({directive.expected_daily_return*100:.2f}%)\n"
            f"  Weekly:  ${directive.projection_weekly_pnl:,.2f} ({directive.expected_weekly_return*100:.2f}%)\n"
            f"  Monthly: ${directive.projection_monthly_pnl:,.2f} -> ${directive.projection_monthly_balance:,.0f}\n"
            f"  Yearly:  ${directive.projection_yearly_pnl:,.2f} -> ${directive.projection_yearly_balance:,.0f}\n"
            f"  Max risk: ${directive.projection_drawdown_dollar:,.2f} drawdown "
            f"({directive.max_acceptable_drawdown*100:.0f}% of capital)"
        )

        return directive

    def status(self) -> Dict[str, Any]:
        d = self._current_directive
        return {
            "has_directive": d is not None,
            "directive": d.to_dict() if d else None,
            "history_count": len(self._history),
            "last_assess_ago_min": round((time.time() - self._last_assess_time) / 60, 1)
                if self._last_assess_time else None,
            "needs_reassessment": self.needs_reassessment(),
        }
