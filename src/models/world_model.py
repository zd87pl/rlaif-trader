"""
World Model for Market Simulation.

Enables offline RL training and strategy stress-testing through learned
market dynamics. Inspired by Dreamer-v4 (Sep 2025) and TRADES (2025)
architectures for model-based reinforcement learning.

Key components:
  - MarketState: dataclass representing a complete market snapshot
  - MarketTransitionModel: neural network predicting state transitions
  - MarketWorldModel: full simulation engine for rollouts and stress tests
  - SyntheticDataGenerator: generates diverse training data for offline RL
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MarketState
# ---------------------------------------------------------------------------

@dataclass
class MarketState:
    """Represents a complete market state at one timestep."""

    timestamp: datetime
    prices: Dict[str, float]            # symbol -> price
    volumes: Dict[str, float]           # symbol -> volume
    volatility: Dict[str, float]        # symbol -> realised vol
    features: Dict[str, np.ndarray]     # symbol -> computed feature vector
    regime: str = "normal"              # e.g. bull, bear, crisis, normal
    vix: float = 15.0
    gex: float = 0.0                    # gamma exposure proxy

    # ------ helpers ------
    def to_tensor(self, symbols: List[str], device: torch.device | None = None) -> torch.Tensor:
        """Flatten state into a single tensor for the transition model."""
        parts: List[float] = []
        for s in symbols:
            parts.append(self.prices.get(s, 0.0))
            parts.append(self.volumes.get(s, 0.0))
            parts.append(self.volatility.get(s, 0.0))
            feat = self.features.get(s, np.zeros(8))
            parts.extend(feat.tolist())
        # globals
        parts.append(self.vix)
        parts.append(self.gex)
        parts.append(_regime_to_float(self.regime))
        t = torch.tensor(parts, dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    @staticmethod
    def from_tensor(
        tensor: torch.Tensor,
        symbols: List[str],
        timestamp: datetime,
        feat_dim: int = 8,
    ) -> "MarketState":
        """Reconstruct a MarketState from a flat tensor."""
        arr = tensor.detach().cpu().numpy().astype(float)
        idx = 0
        prices, volumes, volatility, features = {}, {}, {}, {}
        for s in symbols:
            prices[s] = float(arr[idx]); idx += 1
            volumes[s] = float(arr[idx]); idx += 1
            volatility[s] = float(arr[idx]); idx += 1
            features[s] = arr[idx: idx + feat_dim].copy(); idx += feat_dim
        vix = float(arr[idx]); idx += 1
        gex = float(arr[idx]); idx += 1
        regime = _float_to_regime(float(arr[idx])); idx += 1
        return MarketState(
            timestamp=timestamp,
            prices=prices,
            volumes=volumes,
            volatility=volatility,
            features=features,
            regime=regime,
            vix=vix,
            gex=gex,
        )

    def clone(self) -> "MarketState":
        return MarketState(
            timestamp=self.timestamp,
            prices=dict(self.prices),
            volumes=dict(self.volumes),
            volatility=dict(self.volatility),
            features={k: v.copy() for k, v in self.features.items()},
            regime=self.regime,
            vix=self.vix,
            gex=self.gex,
        )


_REGIMES = ["bull", "bear", "normal", "crisis", "sideways"]


def _regime_to_float(regime: str) -> float:
    if regime in _REGIMES:
        return float(_REGIMES.index(regime))
    return 2.0  # default -> normal


def _float_to_regime(val: float) -> str:
    idx = int(round(val))
    if 0 <= idx < len(_REGIMES):
        return _REGIMES[idx]
    return "normal"


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    """State -> latent representation via MLP with residual connections."""

    def __init__(self, state_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, latent_dim)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = h + F.gelu(self.ln2(self.fc2(h)))
        return self.fc3(h)


class _StochasticDynamics(nn.Module):
    """Latent + action -> distribution over next latent.

    Outputs mean and log-std of a diagonal Gaussian, enabling
    stochastic rollouts and proper uncertainty quantification.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        in_dim = latent_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, latent_dim)
        self.logstd_head = nn.Linear(hidden, latent_dim)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([latent, action], dim=-1))
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(-5.0, 2.0)
        return mean, logstd

    def sample(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        mean, logstd = self.forward(latent, action)
        dist = Normal(mean, logstd.exp())
        sample = dist.rsample()
        return sample, dist


class _Decoder(nn.Module):
    """Latent -> reconstructed state tensor."""

    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _RewardPredictor(nn.Module):
    """Latent -> scalar reward prediction."""

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class _TerminalPredictor(nn.Module):
    """Latent -> probability that episode terminates (e.g. margin call)."""

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z)).squeeze(-1)


# ---------------------------------------------------------------------------
# MarketTransitionModel
# ---------------------------------------------------------------------------

class MarketTransitionModel(nn.Module):
    """Neural network that predicts next market state from current state + action.

    Inspired by Dreamer-v4 (Sep 2025) and TRADES (2025) architectures:
      - Encoder compresses raw state to a compact latent
      - Stochastic dynamics model predicts distribution over next latent
      - Decoder reconstructs observable state from latent
      - Auxiliary heads predict reward and terminal signals

    Supports stochastic transitions (outputs distribution, not point estimate)
    for proper uncertainty quantification during planning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = _Encoder(state_dim, latent_dim, hidden_dim)
        self.dynamics = _StochasticDynamics(latent_dim, action_dim, hidden_dim)
        self.decoder = _Decoder(latent_dim, state_dim, hidden_dim)
        self.reward_head = _RewardPredictor(latent_dim, hidden_dim // 2)
        self.terminal_head = _TerminalPredictor(latent_dim, hidden_dim // 2)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def predict_next(
        self, state: torch.Tensor, action: torch.Tensor, deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: state + action -> next state prediction."""
        z = self.encoder(state)
        if deterministic:
            mean, _ = self.dynamics(z, action)
            z_next = mean
        else:
            z_next, dist = self.dynamics.sample(z, action)

        next_state = self.decoder(z_next)
        reward = self.reward_head(z_next)
        terminal = self.terminal_head(z_next)
        result: Dict[str, torch.Tensor] = {
            "next_state": next_state,
            "reward": reward,
            "terminal": terminal,
            "z": z,
            "z_next": z_next,
        }
        if not deterministic:
            result["dist"] = dist  # type: ignore[assignment]
        return result

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        kl_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined training loss."""
        z = self.encoder(states)
        mean, logstd = self.dynamics(z, actions)
        dist = Normal(mean, logstd.exp())
        z_next = dist.rsample()

        # Reconstruction loss
        pred_states = self.decoder(z_next)
        recon_loss = F.mse_loss(pred_states, next_states)

        # Reward loss
        pred_rewards = self.reward_head(z_next)
        reward_loss = F.mse_loss(pred_rewards, rewards)

        # Terminal loss (binary cross-entropy)
        pred_terminals = self.terminal_head(z_next)
        # terminal_head already applies sigmoid, so use BCE (not with logits)
        terminal_loss = F.binary_cross_entropy(
            pred_terminals, terminals.float(), reduction="mean"
        )

        # KL regularisation toward standard normal
        prior = Normal(torch.zeros_like(mean), torch.ones_like(mean))
        kl_loss = torch.distributions.kl_divergence(dist, prior).mean()

        total = recon_loss + reward_loss + terminal_loss + kl_weight * kl_loss
        return {
            "total": total,
            "recon": recon_loss,
            "reward": reward_loss,
            "terminal": terminal_loss,
            "kl": kl_loss,
        }


# ---------------------------------------------------------------------------
# MarketWorldModel
# ---------------------------------------------------------------------------

class MarketWorldModel:
    """Full world model for market simulation.

    Wraps MarketTransitionModel with higher-level APIs for training on
    historical data, rolling out simulated trajectories, running
    predefined stress scenarios, and testing trading strategies.
    """

    SCENARIOS = [
        "flash_crash",
        "vol_expansion",
        "regime_change",
        "liquidity_crisis",
        "black_swan",
        "grinding_bear",
        "melt_up",
    ]

    def __init__(
        self,
        n_assets: int = 10,
        state_dim: int = 64,
        latent_dim: int = 32,
        action_dim: int | None = None,
        feat_dim: int = 8,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_assets = n_assets
        self.feat_dim = feat_dim

        # state_dim per asset: price + volume + vol + feat_dim = 3 + feat_dim
        # total = n_assets * (3 + feat_dim) + 3 (vix, gex, regime)
        self._raw_state_dim = n_assets * (3 + feat_dim) + 3
        self.state_dim = state_dim  # user-facing config
        self.latent_dim = latent_dim
        self.action_dim = action_dim or n_assets  # one weight per asset

        self.symbols: List[str] = []

        # The transition model operates on the raw state dim
        self.transition_model = MarketTransitionModel(
            state_dim=self._raw_state_dim,
            action_dim=self.action_dim,
            latent_dim=latent_dim,
            hidden_dim=state_dim * 2,
        ).to(self.device)

        self._trained = False
        logger.info(
            "MarketWorldModel initialised: n_assets=%d, raw_state_dim=%d, "
            "latent_dim=%d, action_dim=%d, device=%s",
            n_assets, self._raw_state_dim, latent_dim, self.action_dim, self.device,
        )

    # ------------------------------------------------------------------ train
    def train(
        self,
        historical_data: pd.DataFrame,
        epochs: int = 100,
        lr: float = 3e-4,
        batch_size: int = 64,
        kl_weight: float = 0.1,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train on historical market data.

        Expects a DataFrame with columns: timestamp, and per-symbol columns
        for price, volume, volatility. Learns state transitions, reward
        function, and terminal conditions.

        Returns dict of loss curves.
        """
        logger.info("Starting world model training for %d epochs", epochs)

        # -- Prepare data --
        states, actions, next_states, rewards, terminals = self._prepare_training_data(
            historical_data
        )

        n_samples = states.shape[0]
        n_val = max(1, int(n_samples * val_split))
        n_train = n_samples - n_val

        train_states, val_states = states[:n_train], states[n_train:]
        train_actions, val_actions = actions[:n_train], actions[n_train:]
        train_next, val_next = next_states[:n_train], next_states[n_train:]
        train_rewards, val_rewards = rewards[:n_train], rewards[n_train:]
        train_terminals, val_terminals = terminals[:n_train], terminals[n_train:]

        optimiser = torch.optim.AdamW(
            self.transition_model.parameters(), lr=lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "recon": [], "kl": [],
        }

        self.transition_model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n_train, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_train, batch_size):
                idx = perm[i: i + batch_size]
                losses = self.transition_model.compute_loss(
                    train_states[idx],
                    train_actions[idx],
                    train_next[idx],
                    train_rewards[idx],
                    train_terminals[idx],
                    kl_weight=kl_weight,
                )
                optimiser.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.transition_model.parameters(), 1.0)
                optimiser.step()

                epoch_loss += losses["total"].item()
                n_batches += 1

            scheduler.step()

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            # Validation
            self.transition_model.eval()
            with torch.no_grad():
                val_losses = self.transition_model.compute_loss(
                    val_states, val_actions, val_next, val_rewards, val_terminals,
                    kl_weight=kl_weight,
                )
            history["val_loss"].append(val_losses["total"].item())
            history["recon"].append(val_losses["recon"].item())
            history["kl"].append(val_losses["kl"].item())
            self.transition_model.train()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train=%.4f  val=%.4f  recon=%.4f  kl=%.4f",
                    epoch + 1, epochs, avg_train,
                    val_losses["total"].item(),
                    val_losses["recon"].item(),
                    val_losses["kl"].item(),
                )

        self._trained = True
        self.transition_model.eval()
        logger.info("World model training complete")
        return history

    # -------------------------------------------------------------- simulate
    def simulate(
        self,
        initial_state: MarketState,
        n_steps: int,
        policy: Optional[Callable[[MarketState], np.ndarray]] = None,
        deterministic: bool = False,
    ) -> List[MarketState]:
        """Roll out simulated market trajectory.

        If policy provided, actions come from policy(state) -> array of size action_dim.
        If no policy, simulate with zero action (observation only).

        Returns list of MarketState (length n_steps + 1 including initial).
        """
        self.transition_model.eval()
        trajectory: List[MarketState] = [initial_state.clone()]
        current = initial_state.clone()

        with torch.no_grad():
            for step in range(n_steps):
                state_t = current.to_tensor(self.symbols, self.device).unsqueeze(0)

                if policy is not None:
                    action_np = policy(current)
                    action_t = torch.tensor(
                        action_np, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                else:
                    action_t = torch.zeros(
                        1, self.action_dim, dtype=torch.float32, device=self.device
                    )

                pred = self.transition_model.predict_next(
                    state_t, action_t, deterministic=deterministic
                )
                next_ts = current.timestamp + timedelta(days=1)
                next_state = MarketState.from_tensor(
                    pred["next_state"].squeeze(0),
                    self.symbols,
                    next_ts,
                    feat_dim=self.feat_dim,
                )
                # Ensure prices stay positive
                for s in next_state.prices:
                    next_state.prices[s] = max(next_state.prices[s], 0.01)
                    next_state.volumes[s] = max(next_state.volumes[s], 0.0)
                    next_state.volatility[s] = max(next_state.volatility[s], 0.0)
                next_state.vix = max(next_state.vix, 1.0)

                trajectory.append(next_state)
                current = next_state

                # Early exit on terminal
                if pred["terminal"].item() > 0.5:
                    logger.debug("Simulation terminated at step %d", step)
                    break

        return trajectory

    # -------------------------------------------------- simulate_scenario
    def simulate_scenario(
        self,
        scenario: str,
        initial_state: MarketState,
        n_steps: int = 252,
    ) -> List[MarketState]:
        """Run a predefined stress scenario.

        Scenarios:
          flash_crash     — sudden 10% drop + recovery
          vol_expansion   — VIX doubles over 5 days
          regime_change   — bull -> bear transition
          liquidity_crisis— spreads widen, volume drops
          black_swan      — 3+ sigma event
          grinding_bear   — slow 20% decline over 3 months
          melt_up         — rapid 15% gain over 2 weeks
        """
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. Choose from {self.SCENARIOS}"
            )

        logger.info("Simulating scenario '%s' for %d steps", scenario, n_steps)
        state = initial_state.clone()

        # Build the scenario-specific perturbation schedule
        perturbations = self._build_scenario_perturbations(scenario, n_steps, state)

        trajectory: List[MarketState] = [state.clone()]
        self.transition_model.eval()

        with torch.no_grad():
            for step in range(n_steps):
                state_t = state.to_tensor(self.symbols, self.device).unsqueeze(0)
                action_t = torch.zeros(
                    1, self.action_dim, dtype=torch.float32, device=self.device
                )

                pred = self.transition_model.predict_next(state_t, action_t)
                next_ts = state.timestamp + timedelta(days=1)
                next_state = MarketState.from_tensor(
                    pred["next_state"].squeeze(0),
                    self.symbols,
                    next_ts,
                    feat_dim=self.feat_dim,
                )

                # Apply scenario perturbation
                if step in perturbations:
                    next_state = perturbations[step](next_state)

                # Clamp
                for s in next_state.prices:
                    next_state.prices[s] = max(next_state.prices[s], 0.01)
                    next_state.volumes[s] = max(next_state.volumes[s], 0.0)
                    next_state.volatility[s] = max(next_state.volatility[s], 0.0)
                next_state.vix = max(next_state.vix, 1.0)

                trajectory.append(next_state)
                state = next_state

        return trajectory

    # -------------------------------------------------------------- stress_test
    def stress_test(
        self,
        strategy: Dict[str, Any],
        scenarios: List[str] | None = None,
        n_steps: int = 252,
        initial_state: MarketState | None = None,
    ) -> Dict[str, Any]:
        """Run strategy through all (or selected) scenarios.

        Args:
            strategy: must contain 'policy' key (callable) and optionally
                      'initial_capital', 'position_limits', etc.
            scenarios: subset of SCENARIOS to test; defaults to all.
            n_steps: simulation horizon.
            initial_state: starting market state.

        Returns dict with:
            scenario_results: per-scenario metrics
            worst_case: scenario with lowest return
            best_case: scenario with highest return
            survival_rate: fraction of scenarios without ruin
        """
        if scenarios is None:
            scenarios = list(self.SCENARIOS)

        if initial_state is None:
            initial_state = self._default_initial_state()

        policy = strategy.get("policy")
        initial_capital = strategy.get("initial_capital", 1_000_000.0)

        results: Dict[str, Dict[str, Any]] = {}
        for sc in scenarios:
            logger.info("Stress-testing scenario: %s", sc)
            traj = self.simulate_scenario(sc, initial_state, n_steps)
            metrics = self._evaluate_trajectory(traj, policy, initial_capital)
            results[sc] = metrics

        # Summarise
        returns = {sc: r["total_return"] for sc, r in results.items()}
        worst_sc = min(returns, key=returns.get)  # type: ignore[arg-type]
        best_sc = max(returns, key=returns.get)   # type: ignore[arg-type]
        survival = sum(1 for r in results.values() if r["survived"]) / len(results)

        summary: Dict[str, Any] = {
            "scenario_results": results,
            "worst_case": {"scenario": worst_sc, **results[worst_sc]},
            "best_case": {"scenario": best_sc, **results[best_sc]},
            "survival_rate": survival,
            "mean_return": float(np.mean(list(returns.values()))),
            "std_return": float(np.std(list(returns.values()))),
        }
        logger.info(
            "Stress test complete — survival=%.1f%%, worst=%s (%.2f%%), best=%s (%.2f%%)",
            survival * 100,
            worst_sc, returns[worst_sc] * 100,
            best_sc, returns[best_sc] * 100,
        )
        return summary

    # ======================= private helpers ==============================

    def _prepare_training_data(
        self, df: pd.DataFrame,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataFrame into tensors for training.

        The DataFrame should have a 'timestamp' column plus columns
        like '{symbol}_price', '{symbol}_volume', '{symbol}_volatility'
        for each asset.  If missing, synthetic defaults are used.
        """
        # Discover symbols from columns
        price_cols = [c for c in df.columns if c.endswith("_price")]
        if price_cols:
            self.symbols = [c.replace("_price", "") for c in price_cols]
        elif not self.symbols:
            self.symbols = [f"ASSET_{i}" for i in range(self.n_assets)]

        # Pad / trim to n_assets
        while len(self.symbols) < self.n_assets:
            self.symbols.append(f"ASSET_{len(self.symbols)}")
        self.symbols = self.symbols[: self.n_assets]

        n = len(df) - 1  # transition pairs
        state_dim = self._raw_state_dim

        states = torch.zeros(n, state_dim, device=self.device)
        next_states = torch.zeros(n, state_dim, device=self.device)
        actions = torch.zeros(n, self.action_dim, device=self.device)
        rewards = torch.zeros(n, device=self.device)
        terminals = torch.zeros(n, device=self.device)

        for i in range(n):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            ms = self._row_to_market_state(row)
            ms_next = self._row_to_market_state(next_row)
            states[i] = ms.to_tensor(self.symbols, self.device)
            next_states[i] = ms_next.to_tensor(self.symbols, self.device)

            # Reward: average log-return across assets
            log_returns: List[float] = []
            for s in self.symbols:
                p0 = ms.prices.get(s, 1.0)
                p1 = ms_next.prices.get(s, 1.0)
                if p0 > 0 and p1 > 0:
                    log_returns.append(math.log(p1 / p0))
            rewards[i] = float(np.mean(log_returns)) if log_returns else 0.0

        logger.info("Prepared %d training transitions from historical data", n)
        return states, actions, next_states, rewards, terminals

    def _row_to_market_state(self, row: pd.Series) -> MarketState:
        """Convert a single DataFrame row to a MarketState."""
        ts = pd.Timestamp(row.get("timestamp", datetime.now())).to_pydatetime()
        prices, volumes, volatility, features = {}, {}, {}, {}
        for s in self.symbols:
            prices[s] = float(row.get(f"{s}_price", row.get("close", 100.0)))
            volumes[s] = float(row.get(f"{s}_volume", row.get("volume", 1e6)))
            volatility[s] = float(row.get(f"{s}_volatility", row.get("volatility", 0.2)))
            features[s] = np.zeros(self.feat_dim, dtype=np.float32)

        vix = float(row.get("vix", 15.0))
        gex = float(row.get("gex", 0.0))
        regime = str(row.get("regime", "normal"))
        return MarketState(
            timestamp=ts, prices=prices, volumes=volumes,
            volatility=volatility, features=features,
            regime=regime, vix=vix, gex=gex,
        )

    def _build_scenario_perturbations(
        self, scenario: str, n_steps: int, initial: MarketState,
    ) -> Dict[int, Callable[[MarketState], MarketState]]:
        """Return a dict mapping step -> perturbation function."""
        perturbs: Dict[int, Callable[[MarketState], MarketState]] = {}

        if scenario == "flash_crash":
            crash_step = min(5, n_steps - 1)
            recovery_end = min(crash_step + 10, n_steps - 1)

            def _crash(state: MarketState) -> MarketState:
                for s in state.prices:
                    state.prices[s] *= 0.90  # -10%
                state.vix *= 2.5
                state.regime = "crisis"
                return state

            def _recovery_factory(frac: float):
                def _recover(state: MarketState) -> MarketState:
                    for s in state.prices:
                        state.prices[s] *= (1.0 + 0.01 * frac)
                    state.vix *= max(0.85, 1.0 - 0.05 * frac)
                    return state
                return _recover

            perturbs[crash_step] = _crash
            for step in range(crash_step + 1, recovery_end + 1):
                frac = (step - crash_step) / (recovery_end - crash_step)
                perturbs[step] = _recovery_factory(frac)

        elif scenario == "vol_expansion":
            for step in range(min(5, n_steps)):
                factor = 1.0 + 0.15 * (step + 1)  # VIX roughly doubles by day 5

                def _vol(state: MarketState, f=factor) -> MarketState:
                    state.vix = initial.vix * f
                    for s in state.volatility:
                        state.volatility[s] *= (1.0 + 0.1)
                    return state
                perturbs[step] = _vol

        elif scenario == "regime_change":
            transition_start = min(10, n_steps - 1)
            transition_len = min(20, n_steps - transition_start)
            for step in range(transition_start, transition_start + transition_len):
                drift = -0.003 * (step - transition_start + 1)

                def _regime(state: MarketState, d=drift) -> MarketState:
                    for s in state.prices:
                        state.prices[s] *= (1.0 + d)
                    state.regime = "bear"
                    state.vix *= 1.02
                    return state
                perturbs[step] = _regime

        elif scenario == "liquidity_crisis":
            start = min(3, n_steps - 1)
            for step in range(start, min(start + 15, n_steps)):
                def _liq(state: MarketState, s_idx=step - start) -> MarketState:
                    vol_decay = max(0.3, 1.0 - 0.05 * (s_idx + 1))
                    for s in state.volumes:
                        state.volumes[s] *= vol_decay
                    state.vix *= 1.05
                    for s in state.volatility:
                        state.volatility[s] *= 1.1
                    state.regime = "crisis"
                    return state
                perturbs[step] = _liq

        elif scenario == "black_swan":
            event_step = min(2, n_steps - 1)

            def _swan(state: MarketState) -> MarketState:
                for s in state.prices:
                    # 3-5 sigma move (based on asset vol)
                    vol = state.volatility.get(s, 0.2)
                    shock = -abs(np.random.normal(3.5, 0.5)) * vol
                    state.prices[s] *= (1.0 + shock)
                state.vix *= 3.0
                state.regime = "crisis"
                return state
            perturbs[event_step] = _swan

        elif scenario == "grinding_bear":
            bear_days = min(63, n_steps)  # ~3 months
            daily_decline = (0.80) ** (1.0 / bear_days)  # 20% total decline
            for step in range(bear_days):
                def _grind(state: MarketState, dd=daily_decline) -> MarketState:
                    for s in state.prices:
                        state.prices[s] *= dd
                    state.vix = max(state.vix, initial.vix * 1.3)
                    state.regime = "bear"
                    return state
                perturbs[step] = _grind

        elif scenario == "melt_up":
            melt_days = min(10, n_steps)  # ~2 weeks
            daily_gain = (1.15) ** (1.0 / melt_days)  # 15% total gain
            for step in range(melt_days):
                def _melt(state: MarketState, dg=daily_gain) -> MarketState:
                    for s in state.prices:
                        state.prices[s] *= dg
                    state.vix *= 0.97
                    state.regime = "bull"
                    return state
                perturbs[step] = _melt

        return perturbs

    def _evaluate_trajectory(
        self,
        trajectory: List[MarketState],
        policy: Optional[Callable] = None,
        initial_capital: float = 1_000_000.0,
    ) -> Dict[str, Any]:
        """Evaluate a trajectory to produce standard metrics."""
        if len(trajectory) < 2:
            return {
                "total_return": 0.0, "max_drawdown": 0.0,
                "sharpe": 0.0, "survived": True, "pnl_series": [],
            }

        # Track portfolio value using equal-weight basket
        values = []
        for state in trajectory:
            avg_price = np.mean(list(state.prices.values())) if state.prices else 100.0
            values.append(avg_price)

        values_arr = np.array(values)
        normalised = values_arr / values_arr[0] * initial_capital

        returns = np.diff(normalised) / normalised[:-1]
        total_ret = (normalised[-1] / normalised[0]) - 1.0

        # Max drawdown
        peak = np.maximum.accumulate(normalised)
        drawdowns = (peak - normalised) / peak
        max_dd = float(np.max(drawdowns))

        # Sharpe (annualised, daily data)
        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 1e-10:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        survived = max_dd < 0.50  # >50% drawdown = ruin

        return {
            "total_return": float(total_ret),
            "max_drawdown": float(max_dd),
            "sharpe": sharpe,
            "survived": survived,
            "final_value": float(normalised[-1]),
            "pnl_series": normalised.tolist(),
        }

    def _default_initial_state(self) -> MarketState:
        """Produce a sensible default initial state."""
        if not self.symbols:
            self.symbols = [f"ASSET_{i}" for i in range(self.n_assets)]
        return MarketState(
            timestamp=datetime.now(),
            prices={s: 100.0 for s in self.symbols},
            volumes={s: 1e6 for s in self.symbols},
            volatility={s: 0.2 for s in self.symbols},
            features={s: np.zeros(self.feat_dim) for s in self.symbols},
            regime="normal",
            vix=15.0,
            gex=0.0,
        )

    def save(self, path: str) -> None:
        """Persist model weights and config."""
        torch.save(
            {
                "state_dict": self.transition_model.state_dict(),
                "n_assets": self.n_assets,
                "state_dim": self.state_dim,
                "latent_dim": self.latent_dim,
                "action_dim": self.action_dim,
                "feat_dim": self.feat_dim,
                "symbols": self.symbols,
            },
            path,
        )
        logger.info("World model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights and config."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.transition_model.load_state_dict(ckpt["state_dict"])
        self.symbols = ckpt.get("symbols", self.symbols)
        self._trained = True
        self.transition_model.eval()
        logger.info("World model loaded from %s", path)


# ---------------------------------------------------------------------------
# SyntheticDataGenerator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates synthetic market data for offline RL training.

    Uses a trained MarketWorldModel to produce diverse, realistic
    training episodes — including adversarial scenarios that probe
    strategy weaknesses.
    """

    def __init__(
        self,
        world_model: MarketWorldModel,
        n_variations: int = 100,
        seed: int = 42,
    ):
        self.world_model = world_model
        self.n_variations = n_variations
        self.rng = np.random.RandomState(seed)
        logger.info("SyntheticDataGenerator initialised with %d variations", n_variations)

    def generate_training_data(
        self,
        base_trajectory: List[MarketState],
        n_episodes: int = 1000,
        episode_len: int = 252,
    ) -> List[List[MarketState]]:
        """Generate variations of historical data for offline RL.

        Varies: volatility regime, trend direction, jump frequency.
        Ensures diversity: roughly even mix of bull/bear/sideways/crisis.
        """
        logger.info("Generating %d training episodes from base trajectory", n_episodes)
        episodes: List[List[MarketState]] = []

        regime_targets = ["bull", "bear", "sideways", "crisis"]
        episodes_per_regime = n_episodes // len(regime_targets)

        for regime_idx, target_regime in enumerate(regime_targets):
            for ep in range(episodes_per_regime):
                # Pick a random starting point from base trajectory
                start_idx = self.rng.randint(0, max(1, len(base_trajectory) - 1))
                initial = base_trajectory[start_idx].clone()

                # Apply regime-specific perturbation to initial state
                initial = self._perturb_initial(initial, target_regime)

                # Roll out with stochastic dynamics + perturbation
                traj = self._generate_perturbed_rollout(
                    initial, episode_len, target_regime
                )
                episodes.append(traj)

            logger.debug(
                "Generated %d episodes for regime '%s'",
                episodes_per_regime, target_regime,
            )

        # Fill remaining with random regimes
        remaining = n_episodes - len(episodes)
        for _ in range(remaining):
            regime = regime_targets[self.rng.randint(0, len(regime_targets))]
            start_idx = self.rng.randint(0, max(1, len(base_trajectory) - 1))
            initial = self._perturb_initial(
                base_trajectory[start_idx].clone(), regime
            )
            traj = self._generate_perturbed_rollout(initial, episode_len, regime)
            episodes.append(traj)

        self.rng.shuffle(episodes)
        logger.info("Generated %d total training episodes", len(episodes))
        return episodes

    def generate_adversarial(
        self,
        strategy: Dict[str, Any],
        n_episodes: int = 100,
        episode_len: int = 252,
    ) -> List[List[MarketState]]:
        """Generate scenarios specifically designed to break the strategy.

        Uses gradient-free search: runs many scenarios, identifies which
        cause largest drawdowns, then generates more of those.
        """
        logger.info("Generating %d adversarial episodes", n_episodes)
        policy = strategy.get("policy")

        initial = self.world_model._default_initial_state()
        candidates: List[Tuple[float, List[MarketState]]] = []

        # Phase 1: broad exploration
        n_explore = max(n_episodes // 2, 10)
        for _ in range(n_explore):
            regime = self.rng.choice(["bear", "crisis", "sideways"])
            perturbed = self._perturb_initial(initial.clone(), regime)

            # Add random shocks
            vol_mult = self.rng.uniform(1.0, 3.0)
            for s in perturbed.volatility:
                perturbed.volatility[s] *= vol_mult
            perturbed.vix *= self.rng.uniform(1.0, 4.0)

            traj = self.world_model.simulate(
                perturbed, episode_len, policy=policy
            )
            metrics = self.world_model._evaluate_trajectory(traj, policy)
            dd = metrics["max_drawdown"]
            candidates.append((dd, traj))

        # Sort by drawdown (worst first)
        candidates.sort(key=lambda x: -x[0])

        # Phase 2: intensify around worst trajectories
        adversarial: List[List[MarketState]] = []
        n_intensify = n_episodes - n_explore
        worst_trajectories = candidates[: max(5, n_explore // 5)]

        for _ in range(n_intensify):
            # Pick a bad trajectory and perturb its starting state
            base_dd, base_traj = worst_trajectories[
                self.rng.randint(0, len(worst_trajectories))
            ]
            perturbed = base_traj[0].clone()

            # Small perturbation around the already-bad starting point
            for s in perturbed.prices:
                perturbed.prices[s] *= self.rng.uniform(0.95, 1.05)
            for s in perturbed.volatility:
                perturbed.volatility[s] *= self.rng.uniform(0.9, 1.5)
            perturbed.vix *= self.rng.uniform(0.9, 1.5)

            traj = self.world_model.simulate(
                perturbed, episode_len, policy=policy
            )
            adversarial.append(traj)

        # Combine: keep worst from exploration + all intensified
        for dd, traj in candidates:
            adversarial.append(traj)

        # Trim to requested count
        adversarial = adversarial[:n_episodes]
        logger.info(
            "Generated %d adversarial episodes (worst DD: %.2f%%)",
            len(adversarial),
            candidates[0][0] * 100 if candidates else 0.0,
        )
        return adversarial

    # ---- helpers ----

    def _perturb_initial(self, state: MarketState, target_regime: str) -> MarketState:
        """Perturb initial state to induce a target regime."""
        state.regime = target_regime

        if target_regime == "bull":
            for s in state.prices:
                state.prices[s] *= self.rng.uniform(1.0, 1.05)
            state.vix *= self.rng.uniform(0.6, 0.9)

        elif target_regime == "bear":
            for s in state.prices:
                state.prices[s] *= self.rng.uniform(0.92, 1.0)
            state.vix *= self.rng.uniform(1.2, 2.0)

        elif target_regime == "crisis":
            for s in state.prices:
                state.prices[s] *= self.rng.uniform(0.80, 0.95)
            state.vix *= self.rng.uniform(2.0, 4.0)
            for s in state.volatility:
                state.volatility[s] *= self.rng.uniform(1.5, 3.0)

        elif target_regime == "sideways":
            state.vix *= self.rng.uniform(0.8, 1.2)
            for s in state.volatility:
                state.volatility[s] *= self.rng.uniform(0.7, 1.0)

        return state

    def _generate_perturbed_rollout(
        self,
        initial: MarketState,
        n_steps: int,
        target_regime: str,
    ) -> List[MarketState]:
        """Roll out a trajectory with regime-aware perturbations."""
        traj = self.world_model.simulate(initial, n_steps)

        # Apply regime-consistent drift to prices
        drift_per_step = {
            "bull": 0.0003,
            "bear": -0.0004,
            "crisis": -0.001,
            "sideways": 0.0,
        }.get(target_regime, 0.0)

        noise_scale = {
            "bull": 0.005,
            "bear": 0.01,
            "crisis": 0.02,
            "sideways": 0.003,
        }.get(target_regime, 0.005)

        for i, state in enumerate(traj[1:], 1):
            for s in state.prices:
                shock = self.rng.normal(drift_per_step, noise_scale)
                state.prices[s] *= (1.0 + shock)
                state.prices[s] = max(state.prices[s], 0.01)

            # Occasional jumps in crisis regime
            if target_regime == "crisis" and self.rng.random() < 0.05:
                for s in state.prices:
                    jump = self.rng.normal(-0.03, 0.02)
                    state.prices[s] *= (1.0 + jump)
                    state.prices[s] = max(state.prices[s], 0.01)
                state.vix *= self.rng.uniform(1.05, 1.3)

        return traj
