"""Neural implied-volatility surface (IVS) forecasting via denoising diffusion.

Implements a two-stage pipeline:
  1. **IVSEncoder** — a VAE that compresses a 2-D IV grid into a compact latent
     representation (default 10-dim), following Ding, Lu & Cheung (Sep 2025).
  2. **IVSDiffusionModel** — a DDPM that operates in the learned latent space,
     conditioned on the *current* surface encoding to predict *future* surfaces,
     following Jin & Agarwal (Nov 2025).

The public entry-point is :class:`IVSForecaster`, which wraps training and
inference for both sub-models and exposes trade-signal generation helpers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def _linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# 1. VAE Encoder / Decoder for IV Surfaces
# ---------------------------------------------------------------------------


class IVSEncoder(nn.Module):
    """VAE that compresses a vol-surface grid into a latent representation.

    Default grid: ``(grid_moneyness, grid_tte)`` — e.g. 21×10.
    The encoder uses 2-D convolutions; the decoder mirrors with transposed convs.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent vector (default 10).
    grid_moneyness : int
        Number of moneyness ticks on the input grid.
    grid_tte : int
        Number of time-to-expiry ticks on the input grid.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        grid_moneyness: int = 21,
        grid_tte: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_m = grid_moneyness
        self.grid_t = grid_tte

        # --- encoder ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        # compute flattened size after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_moneyness, grid_tte)
            self._enc_flat = self.enc_conv(dummy).view(1, -1).shape[1]

        self.fc_mu = nn.Linear(self._enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(self._enc_flat, latent_dim)

        # --- decoder ---
        # spatial dims after encoder convs
        with torch.no_grad():
            enc_out = self.enc_conv(dummy)
            self._dec_shape = enc_out.shape[1:]  # (C, H, W)

        self.fc_dec = nn.Linear(latent_dim, self._enc_flat)

        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        # We'll use adaptive interpolation to guarantee exact output shape.

    # ----- forward helpers -----

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) for input batch ``x`` of shape (B, 1, M, T)."""
        h = self.enc_conv(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` to a surface of shape (B, 1, M, T)."""
        h = self.fc_dec(z).view(-1, *self._dec_shape)
        out = self.dec_conv(h)
        # guarantee exact spatial dimensions via interpolation
        out = F.interpolate(out, size=(self.grid_m, self.grid_t), mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(recon: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 1.0) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, target, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss


# ---------------------------------------------------------------------------
# 2. Denoising Diffusion Model in Latent Space
# ---------------------------------------------------------------------------


class _SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class _DenoisingMLP(nn.Module):
    """MLP that predicts noise given (noisy_z, t_embed, cond_z)."""

    def __init__(self, latent_dim: int, hidden: int = 256, time_dim: int = 64, horizon_dim: int = 16):
        super().__init__()
        self.time_embed = _SinusoidalTimeEmbed(time_dim)
        self.horizon_embed = nn.Embedding(32, horizon_dim)  # up to 31 days

        input_dim = latent_dim + time_dim + latent_dim + horizon_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor, z_cond: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h_emb = self.horizon_embed(horizon.clamp(0, 31))
        inp = torch.cat([z_noisy, t_emb, z_cond, h_emb], dim=-1)
        return self.net(inp)


class IVSDiffusionModel(nn.Module):
    """Denoising diffusion model for temporal IVS forecasting.

    Operates entirely in VAE latent space.  Given the latent encoding of
    today's surface and a forecast horizon (in days), it generates latent
    samples of the *future* surface via iterative denoising.

    Parameters
    ----------
    latent_dim : int
        Must match the VAE latent dim.
    diffusion_steps : int
        Number of forward-process noise steps *T*.
    schedule : str
        ``'linear'`` or ``'cosine'`` beta schedule.
    """

    def __init__(self, latent_dim: int = 10, diffusion_steps: int = 100, schedule: str = "cosine"):
        super().__init__()
        self.latent_dim = latent_dim
        self.T = diffusion_steps

        # noise schedule
        if schedule == "cosine":
            betas = _cosine_beta_schedule(diffusion_steps)
        else:
            betas = _linear_beta_schedule(diffusion_steps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        self.denoiser = _DenoisingMLP(latent_dim)

    # --- forward diffusion ---

    def q_sample(self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to ``z_0`` at timestep ``t``."""
        if noise is None:
            noise = torch.randn_like(z_0)
        a = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return a * z_0 + b * noise

    # --- training loss ---

    def loss(self, z_current: torch.Tensor, z_future: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """Simple noise-prediction MSE loss."""
        batch = z_future.shape[0]
        t = torch.randint(0, self.T, (batch,), device=z_future.device)
        noise = torch.randn_like(z_future)
        z_noisy = self.q_sample(z_future, t, noise)
        pred_noise = self.denoiser(z_noisy, t, z_current, horizon)
        return F.mse_loss(pred_noise, noise)

    # --- reverse (sampling) ---

    @torch.no_grad()
    def p_sample(self, z_t: torch.Tensor, t_idx: int, z_cond: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """One reverse-diffusion step."""
        batch = z_t.shape[0]
        t_tensor = torch.full((batch,), t_idx, device=z_t.device, dtype=torch.long)

        pred_noise = self.denoiser(z_t, t_tensor, z_cond, horizon)

        alpha_t = 1.0 - self.betas[t_idx]
        alpha_bar_t = self.alphas_cumprod[t_idx]
        coeff = self.betas[t_idx] / self.sqrt_one_minus_alphas_cumprod[t_idx]
        mean = (1.0 / alpha_t.sqrt()) * (z_t - coeff * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(z_t)
            sigma = self.posterior_variance[t_idx].sqrt()
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(self, z_cond: torch.Tensor, horizon: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate ``n_samples`` future-surface latents conditioned on ``z_cond``.

        Returns tensor of shape ``(n_samples, latent_dim)``.
        """
        device = z_cond.device
        z_cond_exp = z_cond.expand(n_samples, -1)
        horizon_exp = horizon.expand(n_samples)

        z = torch.randn(n_samples, self.latent_dim, device=device)
        for t in reversed(range(self.T)):
            z = self.p_sample(z, t, z_cond_exp, horizon_exp)
        return z


# ---------------------------------------------------------------------------
# 3. High-level Forecaster Interface
# ---------------------------------------------------------------------------


class IVSForecaster:
    """End-to-end IVS forecasting: training + inference + trade-signal helpers.

    Parameters
    ----------
    latent_dim : int
        Latent dimensionality for the VAE.
    grid_moneyness, grid_tte : int
        Surface grid resolution.
    diffusion_steps : int
        DDPM forward-process steps.
    device : str | torch.device | None
        Torch device.  Auto-selects CUDA when available.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        grid_moneyness: int = 21,
        grid_tte: int = 10,
        diffusion_steps: int = 100,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.grid_m = grid_moneyness
        self.grid_t = grid_tte

        self.vae = IVSEncoder(latent_dim, grid_moneyness, grid_tte).to(self.device)
        self.diffusion = IVSDiffusionModel(latent_dim, diffusion_steps).to(self.device)

        self._trained = False
        logger.info(
            "IVSForecaster initialised — latent_dim=%d, grid=(%d,%d), T=%d, device=%s",
            latent_dim, grid_moneyness, grid_tte, diffusion_steps, self.device,
        )

    # ------------------------------------------------------------------ train

    def train(
        self,
        historical_surfaces: List[np.ndarray],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        kl_weight: float = 0.5,
        horizon_days: int = 1,
    ) -> Dict[str, List[float]]:
        """Train both VAE and diffusion model on historical surface data.

        Parameters
        ----------
        historical_surfaces : list of np.ndarray
            Chronologically ordered surfaces, each of shape ``(grid_m, grid_t)``.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        batch_size : int
            Mini-batch size.
        kl_weight : float
            KL-divergence weight in the VAE loss.
        horizon_days : int
            Default forecast horizon used for constructing (current, future) pairs.

        Returns
        -------
        dict with keys ``'vae_loss'`` and ``'diffusion_loss'``, each a list of
        per-epoch average losses.
        """
        logger.info("Training IVSForecaster on %d surfaces (epochs=%d, lr=%s)", len(historical_surfaces), epochs, lr)

        # --- prepare tensors ---
        surfaces = np.stack(historical_surfaces).astype(np.float32)
        surfaces_t = torch.from_numpy(surfaces).unsqueeze(1).to(self.device)  # (N,1,M,T)

        # ---- Phase 1: train VAE ----
        vae_losses: List[float] = []
        vae_opt = Adam(self.vae.parameters(), lr=lr)
        vae_ds = TensorDataset(surfaces_t)
        vae_dl = DataLoader(vae_ds, batch_size=batch_size, shuffle=True)

        self.vae.train()
        for ep in range(epochs):
            total = 0.0
            for (batch,) in vae_dl:
                recon, mu, logvar = self.vae(batch)
                loss = IVSEncoder.loss(recon, batch, mu, logvar, kl_weight)
                vae_opt.zero_grad()
                loss.backward()
                vae_opt.step()
                total += loss.item() * batch.size(0)
            avg = total / len(surfaces_t)
            vae_losses.append(avg)
            if (ep + 1) % max(1, epochs // 5) == 0:
                logger.info("  VAE epoch %d/%d  loss=%.6f", ep + 1, epochs, avg)

        # ---- Phase 2: train diffusion in latent space ----
        self.vae.eval()
        with torch.no_grad():
            mu_all, _ = self.vae.encode(surfaces_t)  # (N, latent_dim)

        # build (current, future) pairs
        n = mu_all.shape[0]
        horizon = min(horizon_days, n - 1)
        idx_cur = torch.arange(0, n - horizon)
        idx_fut = idx_cur + horizon
        z_cur = mu_all[idx_cur]
        z_fut = mu_all[idx_fut]
        h_tensor = torch.full((z_cur.shape[0],), horizon, dtype=torch.long, device=self.device)

        diff_ds = TensorDataset(z_cur, z_fut, h_tensor)
        diff_dl = DataLoader(diff_ds, batch_size=batch_size, shuffle=True)

        diff_opt = Adam(self.diffusion.parameters(), lr=lr)
        diff_losses: List[float] = []

        self.diffusion.train()
        for ep in range(epochs):
            total = 0.0
            cnt = 0
            for zc, zf, hh in diff_dl:
                loss = self.diffusion.loss(zc, zf, hh)
                diff_opt.zero_grad()
                loss.backward()
                diff_opt.step()
                total += loss.item() * zc.size(0)
                cnt += zc.size(0)
            avg = total / max(cnt, 1)
            diff_losses.append(avg)
            if (ep + 1) % max(1, epochs // 5) == 0:
                logger.info("  Diffusion epoch %d/%d  loss=%.6f", ep + 1, epochs, avg)

        self._trained = True
        logger.info("Training complete.")
        return {"vae_loss": vae_losses, "diffusion_loss": diff_losses}

    # --------------------------------------------------------------- helpers

    def _surface_to_tensor(self, surface: np.ndarray) -> torch.Tensor:
        """Convert a single surface np.ndarray (M, T) to (1, 1, M, T) tensor."""
        t = torch.from_numpy(surface.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return t.to(self.device)

    def _encode(self, surface: np.ndarray) -> torch.Tensor:
        """Encode a surface into its latent mean vector (1, latent_dim)."""
        self.vae.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(self._surface_to_tensor(surface))
        return mu

    def _decode(self, z: torch.Tensor) -> np.ndarray:
        """Decode a latent vector to a surface np.ndarray (M, T)."""
        self.vae.eval()
        with torch.no_grad():
            recon = self.vae.decode(z)
        return recon.squeeze().cpu().numpy()

    # -------------------------------------------------------------- forecast

    def forecast(
        self,
        current_surface: np.ndarray,
        horizon_days: int = 1,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """Generate probabilistic forecasts of the future IV surface.

        Parameters
        ----------
        current_surface : np.ndarray of shape ``(grid_m, grid_t)``
        horizon_days : int
            Forecast horizon in trading days.
        n_samples : int
            Number of diffusion samples to draw.

        Returns
        -------
        dict with keys:
          - ``mean_surface``  — np.ndarray (M, T)
          - ``std_surface``   — np.ndarray (M, T)
          - ``samples``       — list of np.ndarray (M, T)
          - ``latent_trajectory`` — np.ndarray (n_samples, latent_dim)
        """
        z_cond = self._encode(current_surface)  # (1, D)
        horizon_t = torch.tensor([horizon_days], device=self.device, dtype=torch.long)

        self.diffusion.eval()
        z_samples = self.diffusion.sample(z_cond.squeeze(0), horizon_t, n_samples)  # (n, D)

        surfaces: List[np.ndarray] = []
        for i in range(n_samples):
            surfaces.append(self._decode(z_samples[i : i + 1]))

        stack = np.stack(surfaces)
        return {
            "mean_surface": stack.mean(axis=0),
            "std_surface": stack.std(axis=0),
            "samples": surfaces,
            "latent_trajectory": z_samples.cpu().numpy(),
        }

    # --------------------------------------------------------- skew forecast

    def forecast_skew(
        self,
        current_surface: np.ndarray,
        horizon: int = 5,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """Predict 25-delta skew changes.

        Skew is measured as IV(25-delta put moneyness) − IV(ATM) on the
        shortest-expiry slice of the grid.

        Returns
        -------
        dict with ``skew_25d_current``, ``skew_25d_forecast``,
        ``skew_change``, ``direction``.
        """
        result = self.forecast(current_surface, horizon, n_samples)

        # indices: ~25-delta put ≈ 90% moneyness → row index near bottom quarter
        otm_put_idx = max(0, self.grid_m // 4)
        atm_idx = self.grid_m // 2
        tte_idx = 0  # shortest expiry

        current_skew = float(current_surface[otm_put_idx, tte_idx] - current_surface[atm_idx, tte_idx])
        forecast_skew = float(result["mean_surface"][otm_put_idx, tte_idx] - result["mean_surface"][atm_idx, tte_idx])
        change = forecast_skew - current_skew

        return {
            "skew_25d_current": current_skew,
            "skew_25d_forecast": forecast_skew,
            "skew_change": change,
            "direction": "steepening" if change > 0 else "flattening",
        }

    # ------------------------------------------------ term-structure forecast

    def forecast_term_structure(
        self,
        current_surface: np.ndarray,
        horizon: int = 5,
        n_samples: int = 50,
    ) -> Dict[str, Any]:
        """Predict term-structure level and slope changes at ATM.

        Returns
        -------
        dict with ``atm_term_current``, ``atm_term_forecast``, ``slope_change``,
        ``level_change``, ``direction``.
        """
        result = self.forecast(current_surface, horizon, n_samples)
        atm_idx = self.grid_m // 2

        cur_term = current_surface[atm_idx, :]
        fwd_term = result["mean_surface"][atm_idx, :]

        cur_slope = float(cur_term[-1] - cur_term[0]) if len(cur_term) > 1 else 0.0
        fwd_slope = float(fwd_term[-1] - fwd_term[0]) if len(fwd_term) > 1 else 0.0

        level_change = float(fwd_term.mean() - cur_term.mean())
        slope_change = fwd_slope - cur_slope

        return {
            "atm_term_current": cur_term.tolist(),
            "atm_term_forecast": fwd_term.tolist(),
            "level_change": level_change,
            "slope_change": slope_change,
            "direction": "steepening" if slope_change > 0 else "flattening",
        }

    # --------------------------------------------------- anomaly detection

    def detect_surface_anomaly(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
        z_threshold: float = 2.0,
    ) -> Dict[str, Any]:
        """Compare predicted vs actual surface to identify mispricings.

        Parameters
        ----------
        predicted : np.ndarray (M, T)
            Forecasted mean surface.
        actual : np.ndarray (M, T)
            Realised surface.
        z_threshold : float
            Number of standard deviations to flag as anomalous.

        Returns
        -------
        dict with ``residual``, ``anomaly_mask``, ``max_deviation``,
        ``mean_abs_error``, ``anomaly_count``.
        """
        residual = actual - predicted
        abs_resid = np.abs(residual)
        sigma = abs_resid.std() if abs_resid.std() > 1e-9 else 1e-9
        z_scores = abs_resid / sigma
        mask = z_scores > z_threshold

        return {
            "residual": residual,
            "z_scores": z_scores,
            "anomaly_mask": mask,
            "max_deviation": float(abs_resid.max()),
            "mean_abs_error": float(abs_resid.mean()),
            "anomaly_count": int(mask.sum()),
        }

    # -------------------------------------------------------- trade signals

    def trade_signals(
        self,
        current_surface: np.ndarray,
        forecast_surface: np.ndarray,
        vol_change_threshold: float = 0.02,
    ) -> List[Dict[str, Any]]:
        """Generate trade signals from predicted surface deformation.

        Compares ``forecast_surface`` (e.g. from :meth:`forecast`) with
        ``current_surface`` and emits directional option trades.

        Parameters
        ----------
        current_surface : np.ndarray (M, T)
        forecast_surface : np.ndarray (M, T)
        vol_change_threshold : float
            Minimum absolute IV change to trigger a signal.

        Returns
        -------
        list of dicts, each with ``signal``, ``region``, ``moneyness_idx``,
        ``tte_idx``, ``iv_change``, ``strategy``.
        """
        diff = forecast_surface - current_surface
        signals: List[Dict[str, Any]] = []

        atm_idx = self.grid_m // 2
        otm_put_idx = self.grid_m // 4
        otm_call_idx = 3 * self.grid_m // 4

        for tte_idx in range(self.grid_t):
            # --- skew trades ---
            skew_change = diff[otm_put_idx, tte_idx] - diff[atm_idx, tte_idx]
            if abs(skew_change) > vol_change_threshold:
                if skew_change > 0:
                    signals.append({
                        "signal": "skew_steepening",
                        "region": "put_wing",
                        "moneyness_idx": otm_put_idx,
                        "tte_idx": tte_idx,
                        "iv_change": float(skew_change),
                        "strategy": "buy_otm_put_sell_atm_put",
                        "confidence": min(1.0, abs(skew_change) / (vol_change_threshold * 3)),
                    })
                else:
                    signals.append({
                        "signal": "skew_flattening",
                        "region": "put_wing",
                        "moneyness_idx": otm_put_idx,
                        "tte_idx": tte_idx,
                        "iv_change": float(skew_change),
                        "strategy": "sell_otm_put_buy_atm_put",
                        "confidence": min(1.0, abs(skew_change) / (vol_change_threshold * 3)),
                    })

            # --- call-wing trades ---
            call_wing_change = diff[otm_call_idx, tte_idx] - diff[atm_idx, tte_idx]
            if abs(call_wing_change) > vol_change_threshold:
                direction = "call_wing_bid" if call_wing_change > 0 else "call_wing_offer"
                signals.append({
                    "signal": direction,
                    "region": "call_wing",
                    "moneyness_idx": otm_call_idx,
                    "tte_idx": tte_idx,
                    "iv_change": float(call_wing_change),
                    "strategy": "buy_otm_call_sell_atm_call" if call_wing_change > 0 else "sell_otm_call_buy_atm_call",
                    "confidence": min(1.0, abs(call_wing_change) / (vol_change_threshold * 3)),
                })

            # --- level trades (vega) ---
            atm_change = diff[atm_idx, tte_idx]
            if abs(atm_change) > vol_change_threshold:
                signals.append({
                    "signal": "vol_up" if atm_change > 0 else "vol_down",
                    "region": "atm",
                    "moneyness_idx": atm_idx,
                    "tte_idx": tte_idx,
                    "iv_change": float(atm_change),
                    "strategy": "buy_straddle" if atm_change > 0 else "sell_straddle",
                    "confidence": min(1.0, abs(atm_change) / (vol_change_threshold * 3)),
                })

        # --- calendar spread signals ---
        if self.grid_t >= 2:
            front_change = diff[atm_idx, 0]
            back_change = diff[atm_idx, -1]
            term_spread = back_change - front_change
            if abs(term_spread) > vol_change_threshold:
                signals.append({
                    "signal": "term_steepening" if term_spread > 0 else "term_flattening",
                    "region": "calendar",
                    "moneyness_idx": atm_idx,
                    "tte_idx": -1,
                    "iv_change": float(term_spread),
                    "strategy": "sell_front_buy_back" if term_spread > 0 else "buy_front_sell_back",
                    "confidence": min(1.0, abs(term_spread) / (vol_change_threshold * 3)),
                })

        # sort by confidence descending
        signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        logger.info("Generated %d trade signals from surface forecast", len(signals))
        return signals
