"""
Reasoning Verifier for RLAIF — Trade-R1 Style Process-Level Verification

Implements Process-Level Reasoning Verification (PLRV) from Trade-R1 (Jan 2026)
and Dynamic-effect Semantic Reward (DSR) for the RLAIF training loop.

Key insight: reward good *reasoning process* even when market outcomes are negative.
Financial markets are stochastic — a well-reasoned trade can lose money due to
unforeseeable events. Conversely, sloppy reasoning can profit from luck.

Components:
- ReasoningStep / ReasoningChain: Structured representation of agent reasoning
- ProcessRewardModel: Scores reasoning quality at each step (PLRV)
- DynamicSemanticReward: Adjusts rewards for market difficulty and luck (DSR)
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Step type taxonomy for trading reasoning chains
# ---------------------------------------------------------------------------

STEP_TYPES = [
    "data_analysis",
    "hypothesis",
    "evidence",
    "risk_assessment",
    "conclusion",
]

STEP_TYPE_KEYWORDS = {
    "data_analysis": [
        "price", "volume", "trend", "moving average", "RSI", "MACD",
        "indicator", "chart", "data shows", "observ", "current",
    ],
    "hypothesis": [
        "expect", "predict", "likely", "hypothesis", "scenario",
        "if ", "could", "might", "anticipate", "forecast",
    ],
    "evidence": [
        "evidence", "supports", "confirms", "historical", "backtest",
        "correlation", "because", "data point", "shows that", "according",
    ],
    "risk_assessment": [
        "risk", "downside", "drawdown", "stop loss", "volatil",
        "worst case", "hedge", "exposure", "risk/reward", "max loss",
    ],
    "conclusion": [
        "therefore", "recommend", "conclusion", "decision", "action",
        "buy", "sell", "hold", "position", "final",
    ],
}

# Patterns to detect evidence citations in text
EVIDENCE_PATTERNS = [
    r"(?:price|close|open|high|low)\s*(?:=|:|\bat\b)\s*\$?[\d,]+\.?\d*",
    r"(?:RSI|MACD|SMA|EMA|ATR|VIX)\s*(?:=|:|is|at)\s*[\d.]+",
    r"(?:volume|vol)\s*(?:=|:|is|at)\s*[\d,.]+[KkMmBb]?",
    r"\d+\.?\d*\s*%",
    r"(?:support|resistance)\s+(?:at|near|around)\s+\$?[\d,]+\.?\d*",
    r"(?:P/E|PE|EPS|P/B|yield)\s*(?:=|:|of|is)\s*[\d.]+",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in a trading reasoning chain."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_type: str = "data_analysis"  # one of STEP_TYPES
    content: str = ""
    evidence_cited: List[str] = field(default_factory=list)
    confidence: float = 0.5
    logical_validity: Optional[float] = None  # scored by verifier (0-1)

    def __post_init__(self):
        if self.step_type not in STEP_TYPES:
            logger.warning(
                f"Unknown step type '{self.step_type}', defaulting to 'data_analysis'"
            )
            self.step_type = "data_analysis"


@dataclass
class ReasoningChain:
    """Full reasoning chain from a trading decision."""

    steps: List[ReasoningStep] = field(default_factory=list)
    decision: str = "hold"  # buy / sell / hold
    overall_coherence: float = 0.0
    evidence_density: float = 0.0  # evidence citations per step

    @property
    def step_types_present(self) -> List[str]:
        return list({s.step_type for s in self.steps})

    @property
    def completeness(self) -> float:
        """Fraction of required step types present."""
        if not STEP_TYPES:
            return 0.0
        return len(set(self.step_types_present) & set(STEP_TYPES)) / len(STEP_TYPES)

    def summary(self) -> str:
        lines = [f"Decision: {self.decision}"]
        lines.append(f"Steps: {len(self.steps)} | Coherence: {self.overall_coherence:.2f}")
        lines.append(f"Evidence density: {self.evidence_density:.2f}")
        lines.append(f"Completeness: {self.completeness:.0%}")
        return " | ".join(lines)


# ---------------------------------------------------------------------------
# ProcessRewardModel — scores reasoning PROCESS quality
# ---------------------------------------------------------------------------

class ProcessRewardModel:
    """
    Scores reasoning PROCESS quality, not just outcomes.

    Based on Trade-R1 (Jan 2026) Process-Level Reasoning Verification (PLRV).
    Each step in a reasoning chain is individually verified for:
      - Factual accuracy (cited evidence vs. actual data)
      - Logical validity (inference coherence)
      - Confidence calibration (not overconfident given evidence)

    The chain is then scored holistically for completeness, flow, and
    internal consistency.
    """

    def __init__(
        self,
        text_encoder_name: str = "all-MiniLM-L6-v2",
        use_text_encoder: bool = False,
    ):
        """
        Args:
            text_encoder_name: sentence-transformers model for optional
                semantic similarity checks.
            use_text_encoder: Whether to load the text encoder. Set False
                for lightweight / CPU-only usage.
        """
        self.text_encoder = None
        self.text_encoder_name = text_encoder_name

        if use_text_encoder:
            try:
                from sentence_transformers import SentenceTransformer

                self.text_encoder = SentenceTransformer(text_encoder_name)
                logger.info(f"Loaded text encoder: {text_encoder_name}")
            except ImportError:
                logger.warning(
                    "sentence_transformers not available; semantic checks disabled"
                )

        # Weights for chain-level scoring
        self.chain_weights = {
            "completeness": 0.20,
            "logical_flow": 0.25,
            "evidence_density": 0.20,
            "consistency": 0.20,
            "calibration": 0.15,
        }

        logger.info("ProcessRewardModel initialized (PLRV)")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_reasoning(self, agent_response_text: str) -> ReasoningChain:
        """
        Parse LLM output into a structured ReasoningChain.

        Handles common formats:
        - Numbered steps ("1. ...", "Step 1: ...")
        - Bullet points ("- ...", "* ...")
        - Section headers ("## Analysis", "Risk Assessment:")
        - Free-form paragraphs (fall back to sentence splitting)
        """
        if not agent_response_text or not agent_response_text.strip():
            return ReasoningChain()

        text = agent_response_text.strip()

        # Try numbered / bulleted step extraction first
        raw_steps = self._split_into_raw_steps(text)

        steps: List[ReasoningStep] = []
        for raw in raw_steps:
            step = ReasoningStep(
                content=raw.strip(),
                step_type=self._classify_step(raw),
                evidence_cited=self._extract_evidence(raw),
                confidence=self._extract_confidence(raw),
            )
            steps.append(step)

        # Detect decision
        decision = self._detect_decision(text)

        # Compute evidence density
        total_evidence = sum(len(s.evidence_cited) for s in steps)
        evidence_density = total_evidence / max(len(steps), 1)

        chain = ReasoningChain(
            steps=steps,
            decision=decision,
            evidence_density=evidence_density,
        )
        return chain

    def _split_into_raw_steps(self, text: str) -> List[str]:
        """Split text into raw step strings."""
        # Try numbered steps: "1.", "1)", "Step 1:"
        numbered = re.split(
            r"\n\s*(?:\d+[.)]\s+|[Ss]tep\s+\d+\s*[:.]\s*)", text
        )
        if len(numbered) > 2:
            return [s.strip() for s in numbered if s.strip()]

        # Try bullet points
        bullets = re.split(r"\n\s*[-*•]\s+", text)
        if len(bullets) > 2:
            return [s.strip() for s in bullets if s.strip()]

        # Try section headers (markdown or plain)
        sections = re.split(
            r"\n\s*(?:#{1,3}\s+|\b[A-Z][a-zA-Z\s]+:\s*\n)", text
        )
        if len(sections) > 2:
            return [s.strip() for s in sections if s.strip()]

        # Fall back: split by double newline or sentences
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return [p.strip() for p in paragraphs if p.strip()]

        # Last resort: split long text into ~3 sentence chunks
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 3:
            return [text]
        chunk_size = max(1, len(sentences) // 3)
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunks.append(" ".join(sentences[i : i + chunk_size]))
        return chunks

    def _classify_step(self, text: str) -> str:
        """Classify a raw step into one of STEP_TYPES using keyword matching."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for stype, keywords in STEP_TYPE_KEYWORDS.items():
            scores[stype] = sum(1 for kw in keywords if kw.lower() in text_lower)
        if max(scores.values(), default=0) == 0:
            return "data_analysis"
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _extract_evidence(self, text: str) -> List[str]:
        """Extract evidence citations (data points) from text."""
        citations: List[str] = []
        for pattern in EVIDENCE_PATTERNS:
            citations.extend(re.findall(pattern, text, re.IGNORECASE))
        return citations

    def _extract_confidence(self, text: str) -> float:
        """Heuristically extract confidence from text."""
        # Explicit confidence mention: "confidence: 0.8", "80% confident"
        m = re.search(r"confidence[:\s]+(\d+\.?\d*)\s*%?", text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return val / 100.0 if val > 1.0 else val

        # Hedging words lower confidence
        hedges = ["might", "could", "possibly", "uncertain", "unclear", "maybe"]
        strong = ["clearly", "strongly", "definitely", "certainly", "confident"]
        text_lower = text.lower()
        hedge_count = sum(1 for h in hedges if h in text_lower)
        strong_count = sum(1 for s in strong if s in text_lower)
        return np.clip(0.5 + 0.1 * strong_count - 0.1 * hedge_count, 0.1, 0.95)

    def _detect_decision(self, text: str) -> str:
        """Detect the trading decision from full text."""
        text_lower = text.lower()
        buy_signals = len(re.findall(r"\b(?:buy|long|bullish|accumulate)\b", text_lower))
        sell_signals = len(re.findall(r"\b(?:sell|short|bearish|reduce|exit)\b", text_lower))
        if buy_signals > sell_signals:
            return "buy"
        elif sell_signals > buy_signals:
            return "sell"
        return "hold"

    # ------------------------------------------------------------------
    # Step-level verification
    # ------------------------------------------------------------------

    def verify_step(self, step: ReasoningStep, market_data: Dict[str, Any]) -> float:
        """
        Verify a single reasoning step.

        Checks:
        - Factual accuracy: Do cited evidence points match actual market data?
        - Logical coherence: Is the inference reasonable for the step type?
        - Confidence calibration: Is confidence in line with evidence strength?

        Returns:
            Quality score in [0, 1].
        """
        scores: List[float] = []

        # 1. Factual accuracy — check cited evidence against market_data
        factual_score = self._check_factual_accuracy(step, market_data)
        scores.append(factual_score)

        # 2. Logical coherence — step type appropriate content
        coherence_score = self._check_step_coherence(step)
        scores.append(coherence_score)

        # 3. Confidence calibration
        calibration_score = self._check_calibration(step)
        scores.append(calibration_score)

        # 4. Evidence richness (more citations = better, up to a point)
        evidence_score = min(len(step.evidence_cited) / 3.0, 1.0)
        scores.append(evidence_score)

        # Weighted average (factual accuracy weighted most)
        weights = [0.35, 0.25, 0.20, 0.20]
        quality = float(np.average(scores, weights=weights))

        step.logical_validity = quality
        return quality

    def _check_factual_accuracy(
        self, step: ReasoningStep, market_data: Dict[str, Any]
    ) -> float:
        """Check whether cited evidence aligns with actual market data."""
        if not step.evidence_cited or not market_data:
            # No evidence to verify — neutral score
            return 0.5

        matches = 0
        checked = 0

        for citation in step.evidence_cited:
            # Try to extract a number from the citation
            nums = re.findall(r"[\d,]+\.?\d*", citation.replace(",", ""))
            if not nums:
                continue
            cited_val = float(nums[0])
            checked += 1

            # See if any market data value is close
            for key, value in market_data.items():
                if isinstance(value, (int, float)):
                    # Allow 5% tolerance for approximate citations
                    if abs(value) > 1e-9 and abs(cited_val - value) / abs(value) < 0.05:
                        matches += 1
                        break
                    # Also check exact match for small values
                    if abs(cited_val - value) < 0.01:
                        matches += 1
                        break

        if checked == 0:
            return 0.5
        return matches / checked

    def _check_step_coherence(self, step: ReasoningStep) -> float:
        """Check whether step content is appropriate for its type."""
        text_lower = step.content.lower()
        keywords = STEP_TYPE_KEYWORDS.get(step.step_type, [])
        if not keywords:
            return 0.5

        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        # At least 2 keyword hits for high coherence
        return min(hits / 2.0, 1.0)

    def _check_calibration(self, step: ReasoningStep) -> float:
        """
        Check confidence calibration.

        Penalize overconfidence (high confidence with little evidence)
        and underconfidence (low confidence with strong evidence).
        """
        evidence_strength = min(len(step.evidence_cited) / 3.0, 1.0)
        conf = step.confidence

        # Ideal: confidence roughly tracks evidence strength
        # Allow generous tolerance
        gap = abs(conf - evidence_strength)
        if gap < 0.2:
            return 1.0
        elif gap < 0.4:
            return 0.7
        else:
            # Heavy penalty for overconfidence with no evidence
            if conf > evidence_strength:
                return 0.3
            return 0.5

    # ------------------------------------------------------------------
    # Chain-level verification
    # ------------------------------------------------------------------

    def verify_chain(
        self, chain: ReasoningChain, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify an entire reasoning chain.

        Evaluates:
        - Step completeness (has all required step types)
        - Logical flow (steps build on each other)
        - Evidence density (more data citations = better)
        - Internal consistency (no contradictions)
        - Confidence calibration across chain

        Returns:
            Dict with process_score, step_scores, component scores, and issues.
        """
        issues: List[str] = []
        step_scores: List[float] = []

        # Score each step
        for step in chain.steps:
            score = self.verify_step(step, market_data)
            step_scores.append(score)

        # --- Component scores ---

        # 1. Completeness
        completeness = chain.completeness
        if completeness < 0.6:
            missing = set(STEP_TYPES) - set(chain.step_types_present)
            issues.append(f"Missing step types: {', '.join(missing)}")

        # 2. Logical flow — steps should follow a sensible order
        logical_flow = self._score_logical_flow(chain)
        if logical_flow < 0.5:
            issues.append("Reasoning flow is disorganized")

        # 3. Evidence density
        ev_density_score = min(chain.evidence_density / 2.0, 1.0)
        if chain.evidence_density < 0.5:
            issues.append("Low evidence density — more data citations needed")

        # 4. Internal consistency
        consistency = self._score_consistency(chain)
        if consistency < 0.5:
            issues.append("Potential contradictions detected in reasoning")

        # 5. Calibration across chain
        calibration = self._score_chain_calibration(chain)
        if calibration < 0.5:
            issues.append("Confidence levels poorly calibrated across chain")

        # Weighted process score
        components = {
            "completeness": completeness,
            "logical_flow": logical_flow,
            "evidence_density": ev_density_score,
            "consistency": consistency,
            "calibration": calibration,
        }
        process_score = sum(
            self.chain_weights[k] * v for k, v in components.items()
        )

        # Update chain coherence
        chain.overall_coherence = process_score

        return {
            "process_score": float(process_score),
            "step_scores": step_scores,
            "components": components,
            "issues": issues,
            "num_steps": len(chain.steps),
        }

    def _score_logical_flow(self, chain: ReasoningChain) -> float:
        """
        Score whether steps follow a logical order.

        Ideal order: data_analysis -> hypothesis -> evidence -> risk_assessment -> conclusion
        """
        if len(chain.steps) < 2:
            return 0.5  # too few steps to judge flow

        type_order = {t: i for i, t in enumerate(STEP_TYPES)}
        order_values = [type_order.get(s.step_type, 2) for s in chain.steps]

        # Count how many adjacent pairs are in non-decreasing order
        in_order = sum(
            1 for i in range(len(order_values) - 1)
            if order_values[i] <= order_values[i + 1]
        )
        return in_order / max(len(order_values) - 1, 1)

    def _score_consistency(self, chain: ReasoningChain) -> float:
        """
        Score internal consistency — detect contradictions.

        Simple approach: check that sentiment direction doesn't flip
        without acknowledgment.
        """
        if len(chain.steps) < 2:
            return 0.8

        sentiments: List[float] = []
        for step in chain.steps:
            text_lower = step.content.lower()
            bullish = len(re.findall(r"\b(?:bullish|positive|upward|growth|strong)\b", text_lower))
            bearish = len(re.findall(r"\b(?:bearish|negative|downward|decline|weak)\b", text_lower))
            if bullish + bearish > 0:
                sentiments.append((bullish - bearish) / (bullish + bearish))

        if len(sentiments) < 2:
            return 0.8

        # Check for sudden large flips without intermediate nuance
        flips = 0
        for i in range(1, len(sentiments)):
            if sentiments[i - 1] * sentiments[i] < 0:  # sign change
                if abs(sentiments[i - 1] - sentiments[i]) > 1.0:
                    flips += 1

        penalty = min(flips * 0.25, 0.5)
        return 1.0 - penalty

    def _score_chain_calibration(self, chain: ReasoningChain) -> float:
        """Score confidence calibration across the full chain."""
        if not chain.steps:
            return 0.5

        confidences = [s.confidence for s in chain.steps]
        evidences = [min(len(s.evidence_cited) / 3.0, 1.0) for s in chain.steps]

        avg_conf = np.mean(confidences)
        avg_ev = np.mean(evidences)

        # Penalize systematic overconfidence
        gap = avg_conf - avg_ev
        if gap > 0.3:
            return 0.3  # overconfident
        elif gap > 0.15:
            return 0.6
        elif gap < -0.3:
            return 0.5  # underconfident (less penalized)
        return 0.9

    # ------------------------------------------------------------------
    # Combined decision scoring
    # ------------------------------------------------------------------

    def score_decision(
        self,
        chain: ReasoningChain,
        market_data: Dict[str, Any],
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Combined score: process quality (60%) + outcome quality (40%).

        If outcome is not yet available, scores process only. This is the
        key innovation: reward good reasoning even when the outcome is
        negative due to market stochasticity.

        Args:
            chain: Parsed reasoning chain.
            market_data: Market data at decision time.
            outcome: Optional dict with keys like 'return', 'sharpe',
                'max_drawdown', 'hit_target', 'holding_period_days'.

        Returns:
            Dict with total_score, process_score, outcome_score,
            reasoning_feedback.
        """
        # Process score
        verification = self.verify_chain(chain, market_data)
        process_score = verification["process_score"]

        # Outcome score (if available)
        outcome_score = 0.0
        outcome_available = outcome is not None and bool(outcome)
        if outcome_available:
            outcome_score = self._score_outcome(outcome, chain.decision)

        # Blend: 60% process, 40% outcome (or 100% process if no outcome)
        if outcome_available:
            total_score = 0.6 * process_score + 0.4 * outcome_score
        else:
            total_score = process_score

        # Generate human-readable feedback
        feedback = self._generate_feedback(verification, outcome_score, outcome_available)

        return {
            "total_score": float(np.clip(total_score, 0.0, 1.0)),
            "process_score": float(process_score),
            "outcome_score": float(outcome_score),
            "verification": verification,
            "reasoning_feedback": feedback,
        }

    def _score_outcome(self, outcome: Dict[str, Any], decision: str) -> float:
        """
        Score the actual trading outcome on [0, 1].

        Considers return, Sharpe ratio, drawdown, and target achievement.
        """
        scores: List[float] = []

        # Return-based score
        ret = outcome.get("return", 0.0)
        if decision == "sell":
            ret = -ret  # short position profits from price drops
        # Sigmoid-like mapping: 5% return -> ~0.75, 0% -> 0.5, -5% -> ~0.25
        ret_score = 1.0 / (1.0 + np.exp(-ret * 20))  # scale factor 20
        scores.append(float(ret_score))

        # Sharpe ratio score
        sharpe = outcome.get("sharpe", None)
        if sharpe is not None:
            sharpe_score = float(np.clip((sharpe + 1.0) / 4.0, 0.0, 1.0))
            scores.append(sharpe_score)

        # Max drawdown penalty
        mdd = outcome.get("max_drawdown", 0.0)
        mdd_score = float(np.clip(1.0 - abs(mdd) * 5, 0.0, 1.0))
        scores.append(mdd_score)

        # Target hit bonus
        if outcome.get("hit_target", False):
            scores.append(1.0)

        if not scores:
            return 0.5
        return float(np.mean(scores))

    def _generate_feedback(
        self,
        verification: Dict[str, Any],
        outcome_score: float,
        outcome_available: bool,
    ) -> str:
        """Generate human-readable reasoning feedback."""
        parts: List[str] = []
        p_score = verification["process_score"]

        if p_score >= 0.8:
            parts.append("Strong reasoning process.")
        elif p_score >= 0.6:
            parts.append("Adequate reasoning with room for improvement.")
        elif p_score >= 0.4:
            parts.append("Reasoning has significant gaps.")
        else:
            parts.append("Poor reasoning quality — major issues detected.")

        for issue in verification.get("issues", []):
            parts.append(f"  - {issue}")

        if outcome_available:
            if outcome_score >= 0.7:
                parts.append("Outcome was positive.")
            elif outcome_score >= 0.4:
                parts.append("Outcome was mixed.")
            else:
                parts.append("Outcome was negative.")

            if p_score >= 0.7 and outcome_score < 0.4:
                parts.append(
                    "Note: Good reasoning but bad outcome — likely due to "
                    "market stochasticity. Process reward preserved."
                )
            elif p_score < 0.4 and outcome_score >= 0.7:
                parts.append(
                    "Warning: Poor reasoning but good outcome — likely luck. "
                    "Process penalty applied."
                )

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# DynamicSemanticReward — prevents reward hacking in stochastic environments
# ---------------------------------------------------------------------------

class DynamicSemanticReward:
    """
    Dynamic-effect Semantic Reward (DSR) from Trade-R1.

    Prevents reward hacking in stochastic financial environments by
    adjusting rewards for:
    - Market regime difficulty (harder regimes deserve more reward)
    - Decision difficulty (contrarian vs consensus)
    - Luck adjustment (discount tail-event outcomes)
    - Process quality bonus from ProcessRewardModel
    """

    def __init__(
        self,
        base_reward_model: Optional[Any] = None,
        process_reward_model: Optional[ProcessRewardModel] = None,
        process_bonus_weight: float = 0.3,
    ):
        """
        Args:
            base_reward_model: Optional RewardModel instance for base scoring.
            process_reward_model: Optional ProcessRewardModel for PLRV bonus.
            process_bonus_weight: Weight for process quality bonus [0, 1].
        """
        self.base_reward_model = base_reward_model
        self.process_reward_model = process_reward_model or ProcessRewardModel()
        self.process_bonus_weight = process_bonus_weight

        # Regime difficulty thresholds
        self.vix_thresholds = {"low": 15.0, "medium": 25.0, "high": 35.0}

        logger.info(
            f"DynamicSemanticReward initialized "
            f"(process_bonus_weight={process_bonus_weight})"
        )

    # ------------------------------------------------------------------
    # Main reward computation
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        decision: Dict[str, Any],
        outcome: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> float:
        """
        Compute dynamic reward that accounts for market difficulty, luck,
        and reasoning process quality.

        Args:
            decision: Dict with keys like 'action', 'reasoning_text',
                'confidence', 'position_size'.
            outcome: Dict with 'return', 'sharpe', 'max_drawdown', etc.
            market_context: Dict with 'vix', 'regime', 'volatility',
                'trend_strength', 'consensus', etc.

        Returns:
            Adjusted reward in [-1, 1].
        """
        # 1. Base outcome reward
        raw_return = outcome.get("return", 0.0)
        action = decision.get("action", "hold")
        if action == "sell":
            raw_return = -raw_return
        base_reward = float(np.tanh(raw_return * 10))  # squash to [-1, 1]

        # 2. Difficulty adjustment (multiplier >= 1.0 for hard environments)
        diff_mult = self.difficulty_adjustment(market_context)

        # 3. Luck adjustment (discount or amplify based on outcome probability)
        luck_factor = self.luck_adjustment(outcome, market_context)

        # 4. Process quality bonus
        process_bonus = 0.0
        reasoning_text = decision.get("reasoning_text", "")
        if reasoning_text:
            chain = self.process_reward_model.parse_reasoning(reasoning_text)
            result = self.process_reward_model.score_decision(
                chain, market_context, outcome
            )
            # Map process_score [0,1] to bonus [-0.3, 0.3]
            process_bonus = (result["process_score"] - 0.5) * 2.0 * self.process_bonus_weight

        # Combine: base * difficulty * luck + process bonus
        adjusted = base_reward * diff_mult * luck_factor + process_bonus

        # Clamp to [-1, 1]
        return float(np.clip(adjusted, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Difficulty adjustment
    # ------------------------------------------------------------------

    def difficulty_adjustment(self, market_context: Dict[str, Any]) -> float:
        """
        Higher difficulty environments yield higher reward for correct calls.

        Factors:
        - VIX level (higher = harder)
        - Regime uncertainty (transition periods are harder)
        - Conflicting signals (mixed indicators = harder)
        - Low trend strength (choppy markets = harder)

        Returns:
            Multiplier in [1.0, 2.0].
        """
        difficulty = 0.0
        components = 0

        # VIX contribution
        vix = market_context.get("vix", None)
        if vix is not None:
            if vix >= self.vix_thresholds["high"]:
                difficulty += 1.0
            elif vix >= self.vix_thresholds["medium"]:
                difficulty += 0.6
            elif vix >= self.vix_thresholds["low"]:
                difficulty += 0.3
            else:
                difficulty += 0.0
            components += 1

        # Regime uncertainty
        regime_uncertainty = market_context.get("regime_uncertainty", None)
        if regime_uncertainty is not None:
            difficulty += float(np.clip(regime_uncertainty, 0.0, 1.0))
            components += 1

        # Conflicting signals
        conflicting = market_context.get("conflicting_signals", None)
        if conflicting is not None:
            difficulty += float(np.clip(conflicting, 0.0, 1.0))
            components += 1

        # Trend strength (inverse — weak trend = harder)
        trend_strength = market_context.get("trend_strength", None)
        if trend_strength is not None:
            difficulty += 1.0 - float(np.clip(trend_strength, 0.0, 1.0))
            components += 1

        if components == 0:
            return 1.0

        avg_difficulty = difficulty / components
        # Map [0, 1] difficulty to [1.0, 2.0] multiplier
        return 1.0 + avg_difficulty

    # ------------------------------------------------------------------
    # Luck adjustment
    # ------------------------------------------------------------------

    def luck_adjustment(
        self, outcome: Dict[str, Any], market_context: Dict[str, Any]
    ) -> float:
        """
        Discount extreme positive outcomes if they were likely luck (tail events).
        Penalize less for negative outcomes during black swans.

        Returns:
            Factor in [0.5, 1.5] — values < 1 discount lucky outcomes,
            values > 1 forgive bad-luck losses.
        """
        ret = outcome.get("return", 0.0)
        volatility = market_context.get("volatility", 0.02)  # default daily vol

        if volatility < 1e-9:
            return 1.0

        # How many standard deviations is this outcome?
        z_score = abs(ret) / max(volatility, 1e-6)

        if ret > 0:
            # Positive outcome — discount if extreme (lucky)
            if z_score > 3.0:
                return 0.5  # heavy discount — very likely luck
            elif z_score > 2.0:
                return 0.7
            elif z_score > 1.5:
                return 0.85
            return 1.0
        else:
            # Negative outcome — forgive if extreme (black swan)
            if z_score > 3.0:
                return 1.5  # heavy forgiveness — black swan
            elif z_score > 2.0:
                return 1.3
            elif z_score > 1.5:
                return 1.15
            return 1.0

    # ------------------------------------------------------------------
    # Batch computation
    # ------------------------------------------------------------------

    def compute_batch_rewards(
        self,
        decisions: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
        market_contexts: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Compute rewards for a batch of decisions.

        Args:
            decisions: List of decision dicts.
            outcomes: List of outcome dicts.
            market_contexts: List of market context dicts.

        Returns:
            List of adjusted rewards.
        """
        assert len(decisions) == len(outcomes) == len(market_contexts), (
            "decisions, outcomes, and market_contexts must have same length"
        )

        rewards = []
        for dec, out, ctx in zip(decisions, outcomes, market_contexts):
            r = self.compute_reward(dec, out, ctx)
            rewards.append(r)

        logger.debug(
            f"Batch rewards: mean={np.mean(rewards):.3f}, "
            f"std={np.std(rewards):.3f}, n={len(rewards)}"
        )
        return rewards

    def reward_summary(
        self,
        decisions: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
        market_contexts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute rewards and return summary statistics.

        Returns:
            Dict with rewards list and aggregate statistics.
        """
        rewards = self.compute_batch_rewards(decisions, outcomes, market_contexts)
        return {
            "rewards": rewards,
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "positive_frac": float(np.mean([r > 0 for r in rewards])),
            "count": len(rewards),
        }
