"""Thesis Generator: Claude-powered strategy code generation.

This is the creative core of the autotrader loop. Given the current strategy,
experiment history, and a market event, it generates a modified signal function
-- the exact analog of autoresearch-mlx's LLM editing train.py.

Extends BaseAgent from the existing multi-agent system.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .metrics import ExperimentResult
from .sentinel import MarketEvent
from .strategy_spec import StrategySpec
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategyThesis:
    """A proposed strategy modification with rationale."""

    thesis_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    description: str = ""
    proposed_spec: Optional[StrategySpec] = None
    rationale: str = ""
    expected_improvement: str = ""
    risk_assessment: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


SYSTEM_PROMPT = """\
You are an elite quantitative strategy researcher operating inside an autonomous \
experimentation loop. Your job is to propose modifications to a Python trading \
signal function that will improve a composite performance metric.

## Your Output Format

You MUST respond with a JSON object containing exactly these fields:
{
    "description": "Short description of the change (one line)",
    "signal_code": "Complete Python function (see constraints below)",
    "rationale": "Why this change should improve performance (2-3 sentences)",
    "expected_improvement": "What metric improvement you expect",
    "risk_assessment": "What could go wrong with this change"
}

## Signal Function Constraints

The signal_code MUST define exactly one function with this signature:

```python
def generate_signals(df, parameters):
    # df is a pandas DataFrame with columns:
    #   open, high, low, close, volume,
    #   sma_20, sma_50, rsi, macd, macd_signal, macd_hist,
    #   bb_upper, bb_lower, bb_mid, atr, adx, obv, vwap
    # parameters is a dict from StrategySpec.parameters
    #
    # Must return a pd.Series of signals: 1=long, -1=short, 0=flat
    import pandas as pd
    ...
    return signals
```

## Allowed imports: pandas, numpy, math, statistics, collections, functools
## FORBIDDEN: os, sys, subprocess, open(), eval(), exec(), any I/O, any network

## Strategy Philosophy (from autoresearch-mlx)

1. SIMPLICITY WINS: A 0.001 improvement from removing code beats a 0.001 improvement \
from adding 20 lines. Prefer elegant, simple strategies.
2. AVOID OVERFITTING: Don't add too many conditions or magic numbers. \
Strategies with 2-3 clear rules beat strategies with 10 edge cases.
3. DON'T REPEAT FAILURES: The experiment history shows what was tried and failed. \
Do not propose the same change twice.
4. THINK ABOUT REGIME: Consider whether the market is trending, mean-reverting, \
or in crisis. Different regimes need different approaches.
5. RISK FIRST: A strategy with lower drawdown and modest returns beats one with \
high returns and catastrophic tail risk.
"""


class ThesisGenerator:
    """Generate strategy modification proposals using Claude."""

    def __init__(
        self,
        claude_client: Any = None,
        model: str = "claude-sonnet-4-6-20250514",
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ):
        self.claude_client = claude_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def propose(
        self,
        event: MarketEvent,
        current_spec: StrategySpec,
        current_performance: Dict[str, float],
        experiment_history: List[ExperimentResult],
        strategist_guidance: str = "",
    ) -> StrategyThesis:
        """Generate a strategy modification thesis.

        Parameters
        ----------
        event : MarketEvent
            The triggering market event.
        current_spec : StrategySpec
            The currently active strategy.
        current_performance : dict
            Current strategy metrics (sharpe, return, drawdown, etc).
        experiment_history : list[ExperimentResult]
            Past experiment results (the "results.tsv" analog).

        Returns
        -------
        StrategyThesis
            Proposed modification with new signal code.
        """
        prompt = self._build_prompt(
            event, current_spec, current_performance, experiment_history,
            strategist_guidance,
        )

        thesis = StrategyThesis(description="pending")

        try:
            if self.claude_client and hasattr(self.claude_client, "complete"):
                response = self.claude_client.complete(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            elif self.claude_client and hasattr(self.claude_client, "messages"):
                # Anthropic SDK client
                msg = self.claude_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                response = msg.content[0].text
            else:
                # Fallback: generate a deterministic mutation
                logger.warning("No LLM client available, using deterministic mutation")
                return self._deterministic_mutation(current_spec, event)

            thesis = self._parse_response(response, current_spec)

        except Exception as e:
            logger.warning("LLM thesis generation failed: %s", e)
            return self._deterministic_mutation(current_spec, event)

        return thesis

    def _build_prompt(
        self,
        event: MarketEvent,
        current_spec: StrategySpec,
        performance: Dict[str, float],
        history: List[ExperimentResult],
        strategist_guidance: str = "",
    ) -> str:
        """Build the prompt for thesis generation."""
        # Format experiment history like autoresearch's results.tsv
        history_str = "experiment_id\tcomposite\tsharpe\tdd\thit_rate\tstatus\tdescription\n"
        for r in history[-30:]:  # Last 30 experiments
            history_str += (
                f"{r.experiment_id}\t{r.composite_score:.4f}\t{r.sharpe_ratio:.3f}\t"
                f"{r.max_drawdown:.3f}\t{r.hit_rate:.3f}\t{r.status}\t{r.description}\n"
            )

        return f"""## Current Market Event
{event}

## Current Strategy Performance
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.4f}
Cumulative Return: {performance.get('cumulative_return', 0):.4f}
Max Drawdown: {performance.get('max_drawdown', 0):.4f}
Hit Rate: {performance.get('hit_rate', 0):.4f}
Composite Score: {performance.get('composite_score', 0):.6f}

## Current Signal Function
```python
{current_spec.signal_code}
```

## Experiment History (most recent 30)
```
{history_str}
```

{"## Portfolio Strategist Guidance" + chr(10) + strategist_guidance + chr(10) if strategist_guidance else ""}## Your Task
Propose a SINGLE modification to the signal function that you believe will improve \
the composite score. Consider the market event, current performance weaknesses, \
{"the portfolio strategist guidance above, " if strategist_guidance else ""}and what has been tried before (don't repeat failures).

Respond with the JSON object described in the system prompt. The signal_code must be \
a complete, self-contained function."""

    def _parse_response(
        self, response: str, current_spec: StrategySpec
    ) -> StrategyThesis:
        """Parse the LLM response into a StrategyThesis."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            data = json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            # Try to find JSON object in the response
            import re
            json_match = re.search(r'\{[^{}]*"signal_code"[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM response as JSON")
                    return StrategyThesis(
                        description="parse_failed",
                        rationale="Could not parse LLM response",
                    )
            else:
                return StrategyThesis(
                    description="parse_failed",
                    rationale="No JSON found in LLM response",
                )

        signal_code = data.get("signal_code", "")
        description = data.get("description", "LLM-generated modification")

        proposed_spec = current_spec.derive(
            signal_code=signal_code,
            description=description,
        )

        return StrategyThesis(
            description=description,
            proposed_spec=proposed_spec,
            rationale=data.get("rationale", ""),
            expected_improvement=data.get("expected_improvement", ""),
            risk_assessment=data.get("risk_assessment", ""),
        )

    def _deterministic_mutation(
        self, current_spec: StrategySpec, event: MarketEvent
    ) -> StrategyThesis:
        """Generate a simple deterministic mutation when no LLM is available.

        Cycles through common strategy modifications:
        1. Tighten RSI bounds
        2. Add volume filter
        3. Switch to momentum-based signals
        4. Add mean reversion component
        5. Add trend strength filter
        """
        mutations = [
            (
                "tighten RSI bounds to 40-70",
                '''\
def generate_signals(df, parameters):
    import pandas as pd
    sma_fast = df.get("sma_20")
    sma_slow = df.get("sma_50")
    rsi = df.get("rsi")
    macd_hist = df.get("macd_hist")

    if sma_fast is None or sma_slow is None:
        return pd.Series(0, index=df.index)

    long = (df["close"] > sma_fast) & (sma_fast > sma_slow) & (rsi > 40) & (rsi < 70) & (macd_hist > 0)
    short = (df["close"] < sma_fast) & (sma_fast < sma_slow) & (rsi < 60) & (rsi > 30) & (macd_hist < 0)
    signals = pd.Series(0, index=df.index)
    signals[long] = 1
    signals[short] = -1
    return signals
''',
            ),
            (
                "add volume confirmation filter",
                '''\
def generate_signals(df, parameters):
    import pandas as pd
    import numpy as np
    sma_fast = df.get("sma_20")
    sma_slow = df.get("sma_50")
    rsi = df.get("rsi")
    macd_hist = df.get("macd_hist")

    if sma_fast is None or sma_slow is None:
        return pd.Series(0, index=df.index)

    vol_sma = df["volume"].rolling(20).mean()
    high_volume = df["volume"] > vol_sma * 1.2

    long = (df["close"] > sma_fast) & (sma_fast > sma_slow) & (macd_hist > 0) & (rsi > 45) & (rsi < 75) & high_volume
    short = (df["close"] < sma_fast) & (sma_fast < sma_slow) & (macd_hist < 0) & (rsi < 55) & (rsi > 25) & high_volume
    signals = pd.Series(0, index=df.index)
    signals[long] = 1
    signals[short] = -1
    return signals
''',
            ),
            (
                "momentum breakout strategy",
                '''\
def generate_signals(df, parameters):
    import pandas as pd
    import numpy as np

    bb_upper = df.get("bb_upper")
    bb_lower = df.get("bb_lower")
    atr = df.get("atr")
    rsi = df.get("rsi")

    if bb_upper is None or bb_lower is None:
        return pd.Series(0, index=df.index)

    # Breakout above upper Bollinger Band with RSI confirmation
    long = (df["close"] > bb_upper) & (rsi > 50) & (rsi < 80)
    # Breakdown below lower band
    short = (df["close"] < bb_lower) & (rsi < 50) & (rsi > 20)

    signals = pd.Series(0, index=df.index)
    signals[long] = 1
    signals[short] = -1
    return signals
''',
            ),
            (
                "mean reversion RSI strategy",
                '''\
def generate_signals(df, parameters):
    import pandas as pd
    rsi = df.get("rsi")
    bb_mid = df.get("bb_mid")

    if rsi is None or bb_mid is None:
        return pd.Series(0, index=df.index)

    # Buy oversold, sell overbought
    long = (rsi < 30) & (df["close"] < bb_mid)
    short = (rsi > 70) & (df["close"] > bb_mid)
    # Exit when RSI normalizes
    signals = pd.Series(0, index=df.index)
    signals[long] = 1
    signals[short] = -1
    return signals
''',
            ),
            (
                "ADX trend strength filter with MACD",
                '''\
def generate_signals(df, parameters):
    import pandas as pd
    adx = df.get("adx")
    macd = df.get("macd")
    macd_signal = df.get("macd_signal")
    sma_fast = df.get("sma_20")
    sma_slow = df.get("sma_50")

    if adx is None or macd is None:
        return pd.Series(0, index=df.index)

    trending = adx > 25 if not isinstance(adx, (int, float)) else True
    macd_cross_up = (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
    macd_cross_down = (macd < macd_signal) & (macd.shift(1) >= macd_signal.shift(1))

    long = macd_cross_up & trending & (sma_fast > sma_slow)
    short = macd_cross_down & trending & (sma_fast < sma_slow)
    signals = pd.Series(0, index=df.index)
    signals[long] = 1
    signals[short] = -1
    return signals
''',
            ),
        ]

        # Pick mutation based on hash of event + current spec
        idx = hash(str(event) + current_spec.spec_id) % len(mutations)
        desc, code = mutations[idx]

        proposed = current_spec.derive(signal_code=code, description=desc)

        return StrategyThesis(
            description=desc,
            proposed_spec=proposed,
            rationale=f"Deterministic mutation triggered by {event.event_type.value}",
            expected_improvement="Unknown (no LLM available)",
            risk_assessment="Moderate -- deterministic mutations are conservative",
        )
