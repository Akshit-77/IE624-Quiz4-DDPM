"""
Gaussian Diffusion (DDPM + DDIM) with Classifier-Free Guidance.

References:
  - Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM) https://arxiv.org/abs/2006.11239
  - Song et al. "Denoising Diffusion Implicit Models" (DDIM) https://arxiv.org/abs/2010.02502
  - Ho & Salimans "Classifier-Free Diffusion Guidance"        https://arxiv.org/abs/2207.12598
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianDiffusion:
    """
    Manages the forward (noise-addition) and reverse (denoising) processes.

    All tensors that depend on t are pre-computed at init and stored on `device`.
    """

    def __init__(
        self,
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        device:     str   = "cuda",
    ):
        self.T      = T
        self.device = device

        # ── Beta / alpha schedule ─────────────────────────────────
        betas              = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([
            torch.ones(1, dtype=torch.float64),
            alphas_cumprod[:-1],
        ])

        def reg(x):
            return x.float().to(device)

        self.betas              = reg(betas)
        self.alphas             = reg(alphas)
        self.alphas_cumprod     = reg(alphas_cumprod)
        self.alphas_cumprod_prev = reg(alphas_cumprod_prev)

        # Forward process coefficients
        self.sqrt_alphas_cumprod         = reg(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod = reg((1.0 - alphas_cumprod).sqrt())

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance             = reg(posterior_variance)
        self.posterior_log_variance_clipped = reg(torch.log(posterior_variance.clamp(min=1e-20)))

        # Posterior mean coefficients
        self.posterior_mean_coef1 = reg(
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = reg(
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)
        )

    # ─────────────────────────────────────────────────────────
    # Forward process  (training)
    # ─────────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    torch.Tensor,
        t:     torch.Tensor,
        noise: torch.Tensor | None = None,
    ):
        """
        x_t = sqrt(ᾱ_t) * x_0  +  sqrt(1 - ᾱ_t) * ε
        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s1 = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s2 = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s1 * x0 + s2 * noise, noise

    def p_losses(
        self,
        model,
        x0:         torch.Tensor,
        t:          torch.Tensor,
        c:          torch.Tensor,
        p_uncond:   float = 0.1,
        null_class: int   = 10,
    ) -> torch.Tensor:
        """
        Simple MSE loss on noise prediction with classifier-free guidance training.

        With probability `p_uncond`, the class label is replaced by `null_class`
        so the model also learns unconditional generation.
        """
        noise = torch.randn_like(x0)
        x_noisy, _ = self.q_sample(x0, t, noise)

        # CFG: randomly drop class label → unconditional
        mask = torch.rand(c.shape[0], device=c.device) < p_uncond
        c_train = c.clone()
        c_train[mask] = null_class

        predicted_noise = model(x_noisy, t, c_train)
        return F.mse_loss(predicted_noise, noise)

    # ─────────────────────────────────────────────────────────
    # Reverse process helpers
    # ─────────────────────────────────────────────────────────

    def _predict_noise(
        self,
        model,
        x_t:           torch.Tensor,
        t_batch:        torch.Tensor,
        c:             torch.Tensor,
        guidance_scale: float = 1.0,
        null_class:    int   = 10,
    ) -> torch.Tensor:
        """
        Run the model and optionally apply classifier-free guidance.

        guidance_scale = 1.0 → standard conditional (no CFG)
        guidance_scale > 1.0 → CFG: eps = eps_uncond + w * (eps_cond - eps_uncond)
        """
        if guidance_scale == 1.0:
            return model(x_t, t_batch, c)

        # Run both conditional and unconditional in one batched forward pass
        B   = x_t.shape[0]
        c_null = torch.full_like(c, null_class)

        x_double  = torch.cat([x_t, x_t], dim=0)
        t_double  = torch.cat([t_batch, t_batch], dim=0)
        c_double  = torch.cat([c, c_null], dim=0)

        eps_both  = model(x_double, t_double, c_double)
        eps_cond, eps_uncond = eps_both[:B], eps_both[B:]
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # ─────────────────────────────────────────────────────────
    # DDPM sampling
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddpm_sample(
        self,
        model,
        shape:          tuple,
        label_class:    int,
        device:         str,
        guidance_scale: float = 3.0,
        null_class:     int   = 10,
    ) -> torch.Tensor:
        """
        Full DDPM reverse process: T stochastic steps from x_T ~ N(0,I).

        Returns images in [0, 1].
        """
        B   = shape[0]
        x   = torch.randn(shape, device=device)
        c   = torch.full((B,), label_class, dtype=torch.long, device=device)

        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((B,), t_idx, dtype=torch.long, device=device)

            eps = self._predict_noise(
                model, x, t_batch, c, guidance_scale, null_class
            )

            # Predict x_0
            x0_pred = (
                x - self.sqrt_one_minus_alphas_cumprod[t_idx] * eps
            ) / self.sqrt_alphas_cumprod[t_idx]
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Posterior mean
            mean = (
                self.posterior_mean_coef1[t_idx] * x0_pred
                + self.posterior_mean_coef2[t_idx] * x
            )

            if t_idx == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                x     = mean + self.posterior_variance[t_idx].sqrt() * noise

        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0   # → [0, 1]

    # ─────────────────────────────────────────────────────────
    # DDIM sampling
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape:          tuple,
        label_class:    int,
        device:         str,
        steps:          int   = 50,
        eta:            float = 0.0,
        guidance_scale: float = 3.0,
        null_class:     int   = 10,
    ) -> torch.Tensor:
        """
        DDIM reverse process with `steps` denoising steps (much faster than DDPM).

        eta = 0 : fully deterministic  (DDIM)
        eta = 1 : DDPM-equivalent stochastic

        Returns images in [0, 1].
        """
        B = shape[0]
        x = torch.randn(shape, device=device)
        c = torch.full((B,), label_class, dtype=torch.long, device=device)

        # Uniformly spaced sub-sequence of timesteps (descending)
        step_size = self.T // steps
        timesteps = list(reversed(range(0, self.T, step_size)))[:steps]

        for i, t_idx in enumerate(timesteps):
            t_batch  = torch.full((B,), t_idx, dtype=torch.long, device=device)

            eps = self._predict_noise(
                model, x, t_batch, c, guidance_scale, null_class
            )

            ab_t  = self.alphas_cumprod[t_idx]
            x0_pred = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            if i == len(timesteps) - 1:
                x = x0_pred
            else:
                t_prev    = timesteps[i + 1]
                ab_t_prev = self.alphas_cumprod[t_prev]

                # DDIM stochasticity parameter σ_t
                sigma = eta * (
                    (1.0 - ab_t_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_t_prev)
                ).clamp(min=0.0).sqrt()

                direction = (1.0 - ab_t_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
                noise     = sigma * torch.randn_like(x) if eta > 0.0 else 0.0

                x = ab_t_prev.sqrt() * x0_pred + direction + noise

        return (x.clamp(-1.0, 1.0) + 1.0) / 2.0   # → [0, 1]
