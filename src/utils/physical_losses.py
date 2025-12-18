"""Energy-aware losses for physics-constrained trajectory modeling."""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "energy_conservation_loss",
    "kinetic_energy",
    "energy_balance_loss",
]


def _broadcast_mask(mask: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    mask_val = mask.to(dtype=target.dtype, device=target.device)
    if mask_val.shape == target.shape:
        return mask_val

    if mask_val.dim() == 2 and target.dim() >= 2:
        time = target.size(0)
        samples = target.size(1)
        if mask_val.shape == (time, samples):
            return mask_val
        if mask_val.shape == (samples, time):
            return mask_val.transpose(0, 1)

    return mask_val.expand_as(target)


def _scalar_spatial_gradient(field: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(field)
    if field.size(1) < 2:
        return grad

    grad[:, 1:-1] = 0.5 * (field[:, 2:] - field[:, :-2])
    grad[:, 0] = field[:, 1] - field[:, 0]
    grad[:, -1] = field[:, -1] - field[:, -2]
    return grad


def _scalar_spatial_laplacian(field: torch.Tensor) -> torch.Tensor:
    lap = torch.zeros_like(field)
    if field.size(1) < 3:
        return lap

    lap[:, 1:-1] = field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2]
    return lap


def kinetic_energy(
    states: torch.Tensor,
    times: torch.Tensor,
) -> torch.Tensor:
    """Compute kinetic energy from a sequence of states and a shared time grid.

    Args:
        states: Tensor shaped ``(T, samples, dim)`` representing trajectory states.
        times: ``(T,)`` time values corresponding to the states (must be strictly increasing).

    Returns:
        Tensor with shape ``(T-1, samples)`` containing the kinetic energy estimated between
        consecutive states via finite differences.
    """
    if states.size(0) < 2:
        return states.new_tensor([])

    times = times.to(device=states.device, dtype=states.dtype)
    if times.dim() != 1 or times.numel() != states.size(0):
        raise ValueError(
            "times must be a 1D tensor aligned with the state sequence length"
        )

    dt = torch.diff(times)
    dt = torch.clamp(dt, min=1e-6)
    dt = dt.view(-1, *([1] * (states.dim() - 2)))

    displacements = states[1:] - states[:-1]
    velocities = displacements / dt
    energy = 0.5 * velocities.pow(2).sum(dim=-1)
    return energy


def energy_conservation_loss(
    states: torch.Tensor,
    times: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Penalty that enforces $dT/dt = 0$ by keeping kinetic energy constant.

    This loss treats temperature as a surrogate for mean kinetic energy. Deviations from
    the sample-wise mean energy are squared and averaged, so constant-energy trajectories
    produce zero penalty.

    Args:
        states: ``(T, samples, dim)`` state trajectory tensor.
        times: ``(T,)`` strictly increasing time grid.
        mask: Optional ``(T, samples)`` mask marking valid states (1=valid, 0=ignore).

    Returns:
        Scalar tensor capturing the mean-squared deviation of the kinetic energy from its mean.
    """
    energy = kinetic_energy(states, times)
    if energy.numel() == 0:
        return states.new_tensor(0.0)

    if mask is not None:
        mask = mask.to(dtype=energy.dtype, device=energy.device)
        if mask.shape != (states.size(0), energy.size(1)):
            mask = mask.expand(states.size(0), energy.size(1))
        interval_mask = mask[:-1] * mask[1:]
    else:
        interval_mask = torch.ones_like(energy)

    valid = interval_mask.sum()
    if valid == 0:
        return energy.new_tensor(0.0)

    per_sample_counts = interval_mask.sum(dim=0, keepdim=True).clamp(min=1.0)
    energy_mean = (energy * interval_mask).sum(dim=0, keepdim=True) / per_sample_counts
    variance = ((energy - energy_mean).pow(2) * interval_mask).sum()
    return variance / valid


def energy_balance_loss(
    temperature: torch.Tensor,
    times: torch.Tensor,
    velocity: torch.Tensor,
    temperature_gradient: Optional[torch.Tensor] = None,
    temperature_laplacian: Optional[torch.Tensor] = None,
    viscous_dissipation: Optional[torch.Tensor] = None,
    rho_cp: float = 1.0,
    thermal_conductivity: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Enforce the PDE-based energy balance residual from the thermal energy equation.

    Approximates the compressible/thermal energy balance
    rho_cp (partial_t T + u · grad T) = k laplacian T + Phi
    using finite differences along the time axis and the sample axis as a surrogate
    spatial grid. The helper assumes `temperature` is scalar per time/sample and that
    the velocity field shares the same (time, samples) dimensions for its advective
    direction components.

    Args:
        temperature: ``(T, samples)`` scalar field (e.g., mean kinetic energy).
        times: ``(T,)`` time grid aligned with the first dimension of `temperature`.
        velocity: ``(T, samples, components)`` array representing advective velocities.
        temperature_gradient: Optional gradient of `temperature` along the sample axis.
            When omitted, a central finite difference is used.
        temperature_laplacian: Optional Laplacian of `temperature` along the sample axis.
            When omitted, a finite difference Laplacian is used.
        viscous_dissipation: Optional ``(T, samples)`` viscous dissipation term Phi.
        rho_cp: Product rho_cp on the left-hand side of the PDE.
        thermal_conductivity: Thermal conductivity k in the diffusion term.
        mask: Optional ``(T, samples)`` mask that marks valid points (1=valid).

    Returns:
        Scalar tensor containing the mean squared residual of the energy balance equation.
    """
    temperature = temperature.to(device=velocity.device, dtype=velocity.dtype)
    if temperature.size(0) != velocity.size(0) or temperature.size(1) != velocity.size(1):
        raise ValueError("`temperature` and `velocity` must align on the first two dimensions")

    times = times.to(device=temperature.device, dtype=temperature.dtype)
    if times.dim() == 2:
        times = times[0]
    if times.numel() != temperature.size(0):
        raise ValueError("`times` must align with the time length of `temperature`")

    dt = torch.diff(times)
    if dt.numel() == 0:
        return temperature.new_tensor(0.0)
    dt = dt.view(-1, *([1] * (temperature.dim() - 1)))
    dTdt = torch.diff(temperature, dim=0) / dt

    if temperature_gradient is None:
        temperature_gradient = _scalar_spatial_gradient(temperature)
    if temperature_laplacian is None:
        temperature_laplacian = _scalar_spatial_laplacian(temperature)

    if temperature_gradient.size(0) != temperature.size(0) or temperature_gradient.size(1) != temperature.size(1):
        raise ValueError("`temperature_gradient` must share the same time dimension as `temperature`")
    if temperature_laplacian.size(0) != temperature.size(0) or temperature_laplacian.size(1) != temperature.size(1):
        raise ValueError("`temperature_laplacian` must share the same time dimension as `temperature`")

    velocity_mid = 0.5 * (velocity[1:] + velocity[:-1])
    grad_mid = 0.5 * (temperature_gradient[1:] + temperature_gradient[:-1])
    lap_mid = 0.5 * (temperature_laplacian[1:] + temperature_laplacian[:-1])

    advective = (velocity_mid * grad_mid.unsqueeze(-1)).sum(dim=-1)
    diss_mid = (
        0.5 * (viscous_dissipation[1:] + viscous_dissipation[:-1])
        if viscous_dissipation is not None
        else torch.zeros_like(lap_mid)
    )

    residual = rho_cp * (dTdt + advective) - thermal_conductivity * lap_mid - diss_mid
    if mask is not None:
        if mask.size(0) != temperature.size(0) or mask.size(1) != temperature.size(1):
            raise ValueError("mask must align with the first two dimensions of temperature")
        mask = 0.5 * (mask[1:] + mask[:-1])
    mask = _broadcast_mask(mask, residual)
    if mask is not None:
        valid = mask.sum()
        if valid == 0:
            return residual.new_tensor(0.0)
        loss = (residual**2 * mask).sum() / valid
    else:
        loss = residual.pow(2).mean()
    return loss
