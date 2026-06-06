"""Logistic noise schedule for continuous time diffusion.

This implements the schedule: dot_alpha_t = sigma(kappa * t + gamma)
where sigma(x) = 1 / (1 + exp(-x)) is the sigmoid/logistic function.
"""

import abc
import math
from typing import Any, List, Literal, Optional, Tuple, Union, cast
import torch
from jaxtyping import Float
from torch import Tensor as TT
from torch.nn import functional as F
from xlm.utils.nn import masked_mean, masked_sum
from .utils import bregman_divergence

_LOG1MEXP_SWITCH = math.log(0.5)


class _Log1mexp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # Clamp to valid range to prevent log(negative) during computation
        x_clamped = x.clamp(max=-1e-7)
        out = torch.empty_like(x)
        m = x_clamped <= _LOG1MEXP_SWITCH
        out[m] = torch.log1p(-torch.exp(x_clamped[m]))
        # Clamp to prevent log(0) when x is very close to 0
        out[~m] = torch.log((-torch.expm1(x_clamped[~m])).clamp_min(1e-7))
        # Return NaN for invalid inputs (x > 0) to signal errors
        out = torch.where(x <= 0, out, torch.full_like(out, float("nan")))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        # d/dx log(1 - exp(x)) = -1 / expm1(-x)  (for x <= 0)
        denom = torch.expm1(-x).abs().clamp_min(1e-7)  # prevent div by zero
        grad = -1.0 / denom
        # prevent 0 * inf -> nan
        grad = torch.where(grad_out == 0, torch.zeros_like(grad), grad)
        # Zero out gradient for invalid inputs
        grad = torch.where(x <= 0, grad, torch.zeros_like(grad))
        return grad_out * grad


def log1mexp_exact_safegrad(x: torch.Tensor) -> torch.Tensor:
    return _Log1mexp.apply(x)


class Schedule:
    """
    The base schedule.
    """

    def __init__(self, eps: float = 1e-4):
        self.eps = eps

    @abc.abstractmethod
    def at(
        self, t: Float[TT, "..."], param: Optional[Any] = None
    ) -> Float[TT, "..."]:
        """
        The CDF $alpha(t)$ also referred to as $kappa(t)$.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def derivative_at(
        self, t: Float[TT, "..."], param: Optional[Any] = None
    ) -> Float[TT, "..."]:
        """
        The density $alpha'(t)$ also referred to as $kappa'(t)$.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inv(
        self, alpha: Float[TT, "..."], param: Optional[Any] = None
    ) -> Float[TT, "..."]:
        """
        Function that is used for inverse CDF sampling. It solves for $t$ in $CDF(t) = U$, where `alpha` is the value of U in [0, 1].
        """
        raise NotImplementedError

    def sample(
        self, a: Float[TT, " batch sequence"], param: Optional[Any] = None
    ) -> Float[TT, " batch sequence"]:
        # sample T_ins or T_unmask using inverse CDF sampling
        # CDF is (1 - exp(-a * \alpha_t))
        # solve for t in CDF(t) = U
        # \alpha(t) = -1/a * log(1 - u)
        # u = torch.rand_like(a)
        # alpha = -1 / a * torch.log(1 - u)
        # return self.inv(alpha)
        raise NotImplementedError

    def sample_truncated(
        self,
        threshold: Float[TT, " batch sequence"],
        param: Optional[Any] = None,
    ) -> Float[TT, " batch sequence"]:
        # New CDF is (1 - exp(-a *(\alpha_t - \alpha_threshold)))
        # solve for t in CDF(t) = U
        # \alpha(t) = -1/a * log(1 - u) + \alpha_threshold
        a = param  # Float[TT, " batch sequence"]
        u = torch.rand_like(a)
        alpha = -1 / a * torch.log(1 - u) + self.at(threshold)
        return self.inv(alpha)

    def a_min(self, t: Float[TT, "..."], epsilon=0.001) -> Float[TT, "..."]:
        # typically t is 1
        return -math.log(epsilon) / self.at(t)

    def rate_scale_factor(
        self, t: Float[TT, "..."], param: Optional[Any] = None
    ) -> Float[TT, "..."]:
        # in our case this is same as d/dt \alpha(t)
        return self.derivative_at(t)


class SquaredSchedule(Schedule):
    """
    Linear schedule.
    alpha(t) = t^2 / 2
    alpha'(t) = t
    """

    def __init__(self):
        super().__init__()

    def derivative_at(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        # d/dt \alpha (t) = t
        return t

    def at(self, t: Float[TT, " batch"]) -> Float[TT, " batch"]:
        # \alpha (t)
        return t**2 / 2

    def inv(self, alpha: Float[TT, "..."]) -> Float[TT, "..."]:
        # solve for t in \alpha(t) = alpha
        # here \alpha(t) = t^2 / 2 => t = sqrt(2 * alpha)
        return torch.sqrt(2 * alpha)


class InterpolantSchedule(Schedule):
    """
    Arbitrary interpolant based schedule.
    """

    def sample_t(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        antithetic: bool = True,
    ) -> Float[TT, "..."]:
        # t should be in [self.eps, 1.0 - self.eps]
        interval = 1.0 - 2 * self.eps
        batch_size = shape[0]
        interval_size = interval / batch_size
        u = torch.rand(shape, device=device)  # (B, L)
        if antithetic:
            temp = torch.arange(batch_size, device=device, dtype=u.dtype)
            if len(shape) > 1:
                temp = temp.unsqueeze(1)
            t = (temp + u) * interval_size + self.eps
        else:
            t = u * interval_size + self.eps
        return t

    @abc.abstractmethod
    def at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Return value alpha(t)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def derivative_at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Return d/dt alpha(t)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inv(
        self,
        alpha: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Return t such that alpha(t) = alpha
        """
        raise NotImplementedError

    def rate_scale_factor(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        # in our case this is same as d/dt \alpha(t)
        return self.derivative_at(t, param) / (
            1 - self.at(t, param)
        )  # use a closed form if more efficient

    def sample(
        self,
        param: Optional[dict[str, torch.Tensor]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> Float[TT, " batch sequence"]:
        # sample T_ins or T_unmask using inverse CDF sampling
        # CDF is \alpha(t)
        # solve for t in CDF(t) = U to get a sample of T
        # \alpha(t) = -1/a * log(1 - u)
        # Get first value from dict for shape reference
        if param is not None:
            first_value = next(iter(param.values()))
            u = torch.rand_like(first_value)
        else:
            u = torch.rand(shape, device=device)
        return self.inv(u, param)

    def sample_rescaled(
        self,
        threshold: Float[TT, " batch sequence"],
        param: Optional[dict[str, torch.Tensor]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> Float[TT, " batch sequence"]:
        # New _CDF(t) = CDF((t - threshold) / (1 - threshold))
        # solve for t in _CDF(t) = U
        # => t = CDF_inv(U) * (1 - threshold) + threshold
        if param is not None:
            first_value = next(iter(param.values()))
            u = torch.rand_like(first_value)
        else:
            u = torch.rand(shape, device=device)
        T_temp = self.inv(u, param)
        return T_temp * (1 - threshold) + threshold

    def sample_truncated(
        self,
        threshold: Float[TT, " batch sequence"],
        param: Optional[dict[str, torch.Tensor]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> Float[TT, " batch sequence"]:
        # New _CDF(t) = CDF(t) - CDF(threshold) / (1 - CDF(threshold)); t in [threshold, 1]
        # solve for t in _CDF(t) = U
        # => CDF(t) = U * (1 - CDF(threshold)) + CDF(threshold)
        if param is not None:
            first_value = next(iter(param.values()))
            u = torch.rand_like(first_value)
        else:
            u = torch.rand(shape, device=device)
        s = self.at(threshold, param)
        return self.inv(u * (1 - s) + s, param)

    def a_min(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        raise NotImplementedError

    def log_likelihood_dropped(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        raise NotImplementedError

    def log_likelihood_masked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        raise NotImplementedError

    def log_likelihood_unmasked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        raise NotImplementedError

    def regularizer(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        raise NotImplementedError


class LinearSchedule(InterpolantSchedule):
    """Fixed linear schedule"""

    def at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        return t

    def derivative_at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        return torch.ones_like(t)

    def inv(
        self,
        alpha: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        return alpha


class SimplifiedKumaSchedule(InterpolantSchedule):
    """
    Simplified Kumaraswamy schedule with a = 1.
    alpha(t) = 1 - (1 - t)**b
    d/dt alpha(t) = b * (1 - t)**(b-1)
    inv(alpha) = 1 - (1 - alpha)**(1/b)
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        delta: float = 0.01,
        a: float = 5.00,
        boundary_reg_weight: float = 1.0,
        kl_reg_weight: float = 0.0,
        mean_reg_weight: float = 1.0,
        boundary_reg_type: Literal[
            "squared_relu", "log_barrier"
        ] = "squared_relu",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.a = a
        self.boundary_reg_weight = boundary_reg_weight
        self.kl_reg_weight = kl_reg_weight
        self.mean_reg_weight = mean_reg_weight
        self.boundary_reg_type = boundary_reg_type

    def at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        # param is dict with key 'b' for Float[TT, "..."]
        b = param["b"]
        return 1 - (1 - t**self.a) ** b

    def derivative_at(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        # param is dict with key 'b' for Float[TT, "..."]
        b = param["b"]
        # return b * (1 - t) ** (b - 1)
        one_minus_ta = 1 - t**self.a
        return self.a * b * (t ** (self.a - 1)) * (one_minus_ta ** (b - 1))

    def inv(
        self,
        alpha: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        # param is dict with key 'b' for Float[TT, "..."]
        # alpha = 1 - (1 - t^a)^b
        # => 1 - alpha = (1 - t^a)^b
        # => (1 - alpha)^(1/b) = 1 - t^a
        # => t = (1 - (1 - alpha)^(1/b))^(1/a)
        b = param["b"]
        return (1 - (1 - alpha) ** (1 / b)) ** (1 / self.a)

    def rate_scale_factor(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        # param is dict with key 'b' for Float[TT, "..."]
        # => d/dt alpha(t) / (1 - alpha(t)) = param (1 - t)**(param-1)/((1-t)**param) = param / (1 - t )
        b = param["b"]
        denom = 1 - t**self.a
        return self.a * b * (t ** (self.a - 1)) / denom

    def regularizer_old(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Float[TT, "..."]:
        # squared relu penalty on extreme tail mass
        # epsilon = 0.01
        # delta = 0.01
        # F(epsilon) <= delta
        # and 1 - F(1 - epsilon) <= delta
        epsilon = torch.full_like(t, self.epsilon)
        delta = torch.full_like(t, self.delta)
        loss = torch.square(
            torch.relu(self.at(epsilon, param) - delta)
        ) + torch.square(torch.relu(1 - self.at(1 - epsilon, param) - delta))
        return loss

    def regularizer_kl_uniform(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        reduce: Literal["mean", "sum"] = "mean",
    ) -> Float[TT, "..."]:
        """
        Closed-form KL-to-Uniform regularizer for the per-position time
        distribution induced by this schedule.

        If T ~ Kumaraswamy(a, b) on [0, 1], then:
            KL(q || Unif[0,1]) = E_q[log q(T)]
                            = log(a b)
                              + (a-1)/a * (psi(1) - psi(b+1))
                              - (b-1)/b
        where psi is the digamma function.

        Notes:
        - For `SimplifiedKumaSchedule`, `a` is taken from `self.a` unless
          provided in `param["a"]` (so subclasses
          that use per-position `a` work correctly).
        - This discourages low-entropy / degenerate schedules (e.g., excessive
          mass near endpoints), and is intentionally separate from the existing
          mixture-CDF + tail-mass regularizers.

        Returns a batch-shaped tensor (B,) aggregated across sequence positions
        using `mask` and `reduce`.
        """
        if param is None or not isinstance(param, dict):
            raise ValueError("param must be a dict")

        b = param["b"]
        if not isinstance(b, torch.Tensor):
            raise ValueError("param['b'] must be a torch.Tensor")

        # fall back to the schedule's fixed exponent self.a.
        a_raw = param.get("a", None)
        if a_raw is None:
            a = torch.as_tensor(
                self.a, device=b.device, dtype=b.dtype
            ).expand_as(b)
        else:
            if not isinstance(a_raw, torch.Tensor):
                raise ValueError(
                    "param['a'] must be a torch.Tensor when provided"
                )
            a = a_raw

        psi1 = torch.digamma(torch.ones_like(b))
        psi_b1 = torch.digamma(b + 1)

        # Avoid dividing by tiny values if parameters get too close to 0.
        a_safe = a.clamp_min(1e-8)
        b_safe = b.clamp_min(1e-8)

        kl_per_pos = (
            torch.log(a_safe * b_safe)  # log a
            + ((a - 1) / a.clamp_min(1e-8)) * (psi1 - psi_b1)
            - (b - 1) / b.clamp_min(1e-8)
        )  # (B, L)

        if mask is not None:
            m = mask.bool()
            if reduce == "sum":
                return masked_sum(kl_per_pos, m, dim=-1)  # (B,)
            return masked_mean(kl_per_pos, m, dim=-1)  # (B,)

        if reduce == "sum":
            return kl_per_pos.sum(dim=-1)  # (B,)
        return kl_per_pos.mean(dim=-1)  # (B,)

    def regularizer(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Float[TT, "..."]:
        # param is dict with key 'b' for Float[TT, "..."]
        # The old expression does not normalize for the length L
        # F_minus_t = torch.square(self.at(t, param) - t)  # (B, L) # old expression
        # linspace between 1e-7 and 1.0 - 1e-7
        with torch.no_grad():
            t_grid = self.sample_t((100, 1), t.device, antithetic=True)[
                ..., None
            ]  # (100, B=1, L=1)
        # t_grid = torch.linspace(1e-3, 1.0 - 1e-3, 100, device=t.device)[
        #    :, None, None
        # ]  # (100, B=1, L=1)
        # expand param (B, L) -> (k=1, B, L)
        expanded_param = {
            k: (v[None, ...] if isinstance(v, torch.Tensor) else v)
            for k, v in param.items()
        }
        # mixture_cdf = masked_mean(self.at(t_grid, param), mask.bool(), dim=-1)
        mixture_cdf = masked_mean(
            self.at(t_grid, expanded_param), mask.bool(), dim=-1
        )  # (100, B)
        # F_minus_t = torch.square(mixture_cdf - t)  # (B, )
        F_minus_t = torch.square(mixture_cdf - t_grid.squeeze(-1)).sum(
            dim=0
        )  # (B,)
        old_reg = masked_sum(
            self.regularizer_old(t, param), mask.bool(), dim=-1
        )  # (B,)
        kl_reg = self.regularizer_kl_uniform(t, param, mask)

        return (
            self.mean_reg_weight * F_minus_t
            + self.boundary_reg_weight * old_reg
            + self.kl_reg_weight * kl_reg
        )  # (B,)
        # return old_reg


class FlexMDMSchedule:

    def __init__(
        self,
        schedule_type: Literal[
            "simplified-kuma", "linear"
        ],
        a_min: float = 1.01,
        boundary_reg_weight: float = 1.0,
        kl_reg_weight: float = 1.0,
        mean_reg_weight: float = 1.0,
        boundary_reg_type: Literal[
            "squared_relu", "log_barrier"
        ] = "squared_relu",
    ):
        """
        schedule_type: "simplified-kuma" is the simplified Kumaraswamy schedule with a=1.
        """
        self.schedule_type = schedule_type
        if schedule_type == "simplified-kuma":
            self.insertion_noise_schedule = SimplifiedKumaSchedule(
                a=a_min,
                boundary_reg_weight=boundary_reg_weight,
                kl_reg_weight=kl_reg_weight,
                mean_reg_weight=mean_reg_weight,
                boundary_reg_type=boundary_reg_type,
            )
            self.unmasking_noise_schedule = SimplifiedKumaSchedule(
                a=a_min,
                boundary_reg_weight=boundary_reg_weight,
                kl_reg_weight=kl_reg_weight,
                mean_reg_weight=mean_reg_weight,
                boundary_reg_type=boundary_reg_type,
            )
        elif schedule_type == "linear":
            self.insertion_noise_schedule = LinearSchedule()
            self.unmasking_noise_schedule = self.insertion_noise_schedule
        else:
            raise ValueError(f"Invalid schedule type: {schedule_type}")

    def sample_t(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        antithetic: bool = True,
    ) -> Float[TT, "..."]:
        return self.unmasking_noise_schedule.sample_t(
            shape, device, antithetic
        )

    def sample_ins_unmask_times(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Tuple[Float[TT, "..."], Float[TT, "..."]]:
        if self.schedule_type == "linear":
            if param is not None:
                raise ValueError("param must be None for linear schedule")
            t_ins = self.insertion_noise_schedule.sample(
                param=None, shape=t.shape, device=t.device
            )
            t_unmask = self.unmasking_noise_schedule.sample_truncated(
                t_ins, param=None, shape=t.shape, device=t.device
            )
            return t_ins, t_unmask
        if param is not None and self.schedule_type == "simplified-kuma":
            # For simplified-kuma: a_ins=1, a_unmask=1 (implicit), use b_ins and b_unmask
            b_ins = param["b_ins"]
            b_unmask = param["b_unmask"]
            with torch.no_grad():
                t_ins = self.insertion_noise_schedule.sample({"b": b_ins})
                t_unmask = self.unmasking_noise_schedule.sample_truncated(
                    t_ins, {"b": b_unmask}
                )
            return t_ins, t_unmask
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

    def insertion_hazard_rate(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            # For simplified-kuma: a_ins=1 (implicit), use b_ins
            return self.insertion_noise_schedule.rate_scale_factor(
                t, {"b": param["b_ins"]}
            )
        else:  # self.schedule_type == "linear":
            return self.insertion_noise_schedule.rate_scale_factor(t, None)

    def unmasking_hazard_rate(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            # For simplified-kuma: a_unmask=1 (implicit), use b_unmask
            return self.unmasking_noise_schedule.rate_scale_factor(
                t, {"b": param["b_unmask"]}
            )
        else:  # self.schedule_type == "linear":
            return self.unmasking_noise_schedule.rate_scale_factor(t, None)

    def log_likelihood_dropped(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            return self._log_likelihood_dropped(t, param)
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

    def log_likelihood_masked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            return self._log_likelihood_masked(t, param)
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

    def log_likelihood_unmasked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            return self._log_likelihood_unmasked(t, param)
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

    def _log_likelihood_dropped(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Log probability of the DROPPED state (i.e. still not inserted) under the Kumaraswamy special case with fixed a.

        Expects:
            param = dict with keys b_ins, b_unmask where b_ins is used here.
        Returns:
            log P(T_ins > t) with shape broadcastable to b_ins (typically (B, L)).
        """
        if param is None or not isinstance(param, dict):
            raise ValueError("param must be a dict")
        b_ins = param["b_ins"]

        # Broadcast t to match (B, L) if needed
        t_b = t
        if t_b.dim() == b_ins.dim() - 1:
            t_b = t_b.unsqueeze(-1)
        # P_drop = (1-t^a_ins)^{b_ins}
        t_b = t_b.clamp(0.0, 1.0)
        t_pow = (t_b**self.insertion_noise_schedule.a).clamp(0.0, 1.0 - 1e-7)
        return b_ins * torch.log1p(-t_pow)

    def _log_likelihood_masked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Log probability of the MASKED state under the a=1 Kumaraswamy special case.

        Expects:
            param = dict with keys b_ins, b_unmask where b_ins=b_i and b_unmask=c_i in the paper notation.
        Returns:
            log P(T_ins <= t < T_unmask) with shape broadcastable to b_ins (typically (B, L)).
        """
        if param is None or not isinstance(param, dict):
            raise ValueError("param must be a dict")
        b_ins = param["b_ins"]
        b_unmask = param["b_unmask"]

        # Broadcast t to match (B, L) if needed
        t_b = t
        if t_b.dim() == b_ins.dim() - 1:
            t_b = t_b.unsqueeze(-1)
        # log(1 - t^a), computed stably for t in [0, 1]
        t_b = t_b.clamp(0.0, 1.0)
        t_pow = (t_b**self.insertion_noise_schedule.a).clamp(0.0, 1.0 - 1e-7)
        log_one_minus_t = torch.log1p(-t_pow)
        diff = b_ins - b_unmask
        abs_diff = diff.abs().clamp_min(1e-8)

        # We need log I(t) where
        #   I(t) = (b_ins / diff) * (1 - (1 - t^a)^diff)
        # and log P_mask = b_unmask * log(1 - t^a) + log I(t).
        #
        # Use x = diff * log(1 - t^a).
        # - If diff > 0 then x <= 0 and log(1 - exp(x)) is stable via log1mexp.
        # - If diff < 0 then x >= 0 and
        #       log(exp(x) - 1) = x + log(1 - exp(-x)).
        x = diff * log_one_minus_t
        log_b = torch.log(b_ins.clamp_min(1e-20))

        log_inner_general = torch.where(
            diff > 0,
            log1mexp_exact_safegrad(x),
            x + log1mexp_exact_safegrad(-x),
        )
        log_I_general = log_b - torch.log(abs_diff) + log_inner_general

        # Limit diff -> 0 (i.e., b_ins == b_unmask):
        #   I(t) = -b_ins * log(1 - t^a)
        log_I_limit = log_b + torch.log((-log_one_minus_t).clamp_min(1e-20))

        use_equal = diff.abs() < 1e-6
        log_I = torch.where(use_equal, log_I_limit, log_I_general)
        log_p = b_unmask * log_one_minus_t + log_I
        return log_p.clamp(max=0.0)

    def _log_likelihood_unmasked(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
    ) -> Float[TT, "..."]:
        """
        Log probability of the UNMASKED (original token) state under the a=1 Kumaraswamy special case.

        Expects:
            param = dict with keys b_ins, b_unmask.
        Returns:
            log P(T_unmask <= t) with shape broadcastable to b_ins (typically (B, L)).
        """
        if param is None or not isinstance(param, dict):
            raise ValueError("param must be a dict")
        # TODO (efficiency): This is being recomputed
        ll_dropped = self._log_likelihood_dropped(t, param)  # (B, L)
        ll_masked = self._log_likelihood_masked(t, param)  # (B, L)
        # log P(unmasked) = log(1 - exp(logaddexp(ll_dropped, ll_masked)))
        total = torch.logaddexp(ll_dropped, ll_masked).clamp(
            max=-1e-7
        )  # (B, L)
        ll_unmasked = log1mexp_exact_safegrad(total)
        return ll_unmasked

    def compute_generator_loss(
        self,
        z_1: torch.Tensor,
        x_t: torch.Tensor,
        s_t: torch.Tensor,
        gaps_mask: torch.Tensor,
        gap_sums: torch.Tensor,
        vocab_logits: torch.Tensor,
        hazard_phi: List[torch.Tensor],  # hazard ins, and hazard unmask
        hazard_theta: List[torch.Tensor],  # hazard ins, and hazard unmask
        mask_token_id: torch.Tensor,
        lenght_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute loss for a single variable-length masked sample.

        Implements the full Bregman divergence loss (eq 644-653):
        - Unmasking loss: RED (cross-entropy) + BLUE (KL for bias) + TEAL (correction)
        - Insertion loss: BLUE (KL for insertion rates) + TEAL (correction)
        """
        batch_size, max_seq_len = z_1.shape
        hazard_ins_phi, hazard_unmask_phi = hazard_phi
        hazard_ins_theta, hazard_unmask_theta = hazard_theta        

        # ============== UNMASKING LOSS ==============``
        # Get target tokens by gathering from z_1 using s_t
        z_1_sorted = torch.gather(z_1, 1, s_t)  # (B, L)

        # Get b_phi values for the corresponding positions
        hazard_unmask_phi_sorted = torch.gather(
            hazard_unmask_phi, 1, s_t
        )  # (B, L)

        # Compute log probabilities for RED term
        log_probs = F.log_softmax(vocab_logits, dim=-1)  # (B, L, V)
        log_K_theta = torch.gather(
            log_probs, 2, z_1_sorted.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B, L)

        # weighted cross-entropy
        red_term = -hazard_unmask_phi_sorted * log_K_theta

        # Full Bregman divergence for hazard matching
        # bregman_divergence(b_phi, b_theta) = b_theta - b_phi + b_phi * log(b_phi/b_theta)
        bregman_unmask = bregman_divergence(
            hazard_unmask_phi_sorted, hazard_unmask_theta
        )

        # Only apply to masked positions
        mask_indices = x_t == mask_token_id
        unmask_loss = torch.where(
            mask_indices,
            red_term + bregman_unmask,
            torch.zeros_like(red_term),
        )  # (B, L)

        # ============== INSERTION LOSS ==============
        bregman_insert = bregman_divergence(gap_sums, hazard_ins_theta)

        # Apply to valid gaps
        insertion_loss = torch.where(
            gaps_mask.bool(),
            bregman_insert,
            torch.zeros_like(bregman_insert),
        )

        # ============== COMBINE ==============
        # -- FLEXMDM uses common normalizer
        # unmask_loss = unmask_loss.sum() / max_seq_len
        # insertion_loss = insertion_loss.sum() / max_seq_len

        # -- we use counts
        # unmask_loss = unmask_loss.sum() / mask_indices.sum().clamp(min=1.0)
        # insertion_loss = (insertion_loss.sum()) / gaps_mask.sum().clamp(
        #    min=1.0
        # )  # length normalizer
        # total_loss = unmask_loss + insertion_loss
        # -- we use counts

        insertion_loss = insertion_loss.sum(-1) / lenght_scale
        unmask_loss = unmask_loss.sum(-1) / mask_indices.sum(-1).clamp(min=1.0)

        total_loss = unmask_loss + insertion_loss  # (B,)

        return (total_loss, unmask_loss, insertion_loss)

    def sample_varlen_masked_sequence(
        self,
        z_1: torch.Tensor,
        t_ins: torch.Tensor,
        t_unmask: torch.Tensor,
        t: torch.Tensor,
        mask_token_id: int,
        pad_token_id: int,
        fixed: torch.Tensor,
        ins_hazard: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample a variable-length masked sequence.

        Args:
            z_1: Clean full-length sequence (batch, max_seq_len)
            t_ins: Per-position insertion times (batch, max_seq_len)
            t_unmask: Per-position unmasking times (batch, max_seq_len)
            t: Time for each sequence (batch,)
            mask_token_id: Token ID for mask
            pad_token_id: Token ID for padding
            fixed: Mask for fixed positions (batch, max_seq_len)
            a_phi: Per-position insertion rates from aux model (batch, max_seq_len)

        Returns:
            x_t: Variable-length masked sequence (batch, max_seq_len)
            s_t: Sort indices mapping positions in x_t back to z_1 (batch, max_seq_len)
            gaps: Gap sizes at each position (batch, max_seq_len)
            gaps_mask: Valid gap positions (batch, max_seq_len)
            gap_sums: Sum of a_phi for each gap (batch, max_seq_len)
            deleted: Boolean mask for deleted positions in z_1 (batch, max_seq_len)
            masked: Boolean mask for masked positions in z_1 (batch, max_seq_len)
        """
        batch_size, max_seq_len = z_1.shape
        device = z_1.device

        # Set times for invalid positions to ensure they're not processed
        # these positions could be pads as well as prompt tokens
        t_ins = torch.where(fixed, 0, t_ins)
        t_unmask = torch.where(fixed, 0, t_unmask)

        # Determine state of each position
        # Deleted: t < t_ins
        # Masked: t_ins <= t < t_unmask
        # Clean: t >= t_unmask
        deleted = t.unsqueeze(-1) < t_ins
        masked = (t.unsqueeze(-1) >= t_ins) & (t.unsqueeze(-1) < t_unmask)

        # Create noised sequence
        x_t = z_1.clone()
        x_t[deleted] = pad_token_id
        x_t[masked] = mask_token_id

        # Compress: remove deleted tokens (pad tokens)
        # s_t: original positions of the non-deleted tokens
        # Eg: if x_t.ne(pad) = [1 0 0 1 0 1]
        # then s_t = [0 3 5 1 2 4]
        s_t = x_t.ne(pad_token_id).argsort(dim=1, descending=True, stable=True)
        x_t = torch.gather(
            x_t, 1, s_t
        )  # squeeze together the non-deleted tokens
        s_t[x_t == pad_token_id] = 0

        # Compute sequence lengths
        x_t_len = (x_t != pad_token_id).sum(dim=1)

        # Compute gaps
        temp = s_t.clone()
        pad_front = temp.new_zeros((temp.shape[0], 1)) - 1
        temp = torch.cat([pad_front, temp], dim=1)

        gaps = temp[:, 1:] - temp[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=device).unsqueeze(0)
        gaps_mask = idx < x_t_len.unsqueeze(1)
        gaps[~gaps_mask] = 0

        # Compute gap sums of a_phi for deleted positions
        # For gap i, sum a_phi[j] for all j in (s_t[i-1], s_t[i])
        ins_hazard_deleted = ins_hazard * deleted.float() * (~fixed).float()
        cumsum = torch.cumsum(ins_hazard_deleted, dim=1)  # (B, L)
        cumsum_padded = F.pad(cumsum, (1, 0), value=0)  # (B, L+1)

        # s_t_prev[i] = s_t[i-1], with s_t[-1] = -1 for the first gap
        s_t_prev = F.pad(s_t[:, :-1], (1, 0), value=-1)  # (B, L)

        end_idx = s_t.clamp(min=0, max=max_seq_len)
        start_idx = (s_t_prev + 1).clamp(min=0, max=max_seq_len)

        end_cumsum = torch.gather(cumsum_padded, 1, end_idx)
        start_cumsum = torch.gather(cumsum_padded, 1, start_idx)

        gap_sums = (end_cumsum - start_cumsum) * gaps_mask.float()

        return x_t, s_t, gaps, gaps_mask, gap_sums, deleted, masked

    def regularizer(
        self,
        t: Float[TT, "..."],
        param: Optional[dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Float[TT, "..."]:
        if self.schedule_type == "simplified-kuma":
            return self.insertion_noise_schedule.regularizer(
                t, {"b": param["b_ins"]}, mask=mask
            ) + self.unmasking_noise_schedule.regularizer(
                t, {"b": param["b_unmask"]}, mask=mask
            )
        else:
            return self.insertion_noise_schedule.regularizer(
                t, {"a": param["a_ins"], "b": param["b_ins"]}, mask=mask
            ) + self.unmasking_noise_schedule.regularizer(
                t, {"a": param["a_unmask"], "b": param["b_unmask"]}, mask=mask
            )
