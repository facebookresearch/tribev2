# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Feature attribution utilities for TRIBE v2 models.

These functions operate on cached feature tensors already present in a
``SegmentData`` batch. They do not attribute all the way back to raw words,
waveforms, or pixels; instead, they score the text/audio/video feature timelines
that are consumed by the fMRI encoder.
"""

from __future__ import annotations

import contextlib
import typing as tp
from collections.abc import Mapping

import torch
from torch import nn

TargetIndex = int | slice | tp.Sequence[int] | torch.Tensor | None
Reduction = tp.Literal["l1", "l2", "signed"]


def select_output(
    output: torch.Tensor,
    target_vertices: TargetIndex = None,
    target_timesteps: TargetIndex = None,
) -> torch.Tensor:
    """Reduce model output to one scalar score per batch item.

    Parameters
    ----------
    output:
        Tensor with shape ``(batch, vertices, timesteps)``.
    target_vertices:
        Optional vertex/output indices to explain. ``None`` explains the mean
        response over all vertices.
    target_timesteps:
        Optional output timestep indices to explain. ``None`` explains the mean
        response over all output timesteps.
    """
    if output.ndim != 3:
        raise ValueError(f"Expected output with 3 dimensions, got {output.shape}")
    selected = output
    if target_vertices is not None:
        selected = selected[:, _normalize_index(target_vertices, output.device), :]
    if target_timesteps is not None:
        selected = selected[:, :, _normalize_index(target_timesteps, output.device)]
    return selected.mean(dim=tuple(range(1, selected.ndim)))


def reduce_temporal_attribution(
    attribution: torch.Tensor,
    reduction: Reduction = "l1",
    temporal_axis: int = -1,
) -> torch.Tensor:
    """Reduce feature attribution tensors to ``(batch, time)`` scores."""
    if attribution.ndim < 2:
        raise ValueError(
            f"Expected attribution with at least 2 dimensions, got {attribution.shape}"
        )
    if temporal_axis < 0:
        temporal_axis += attribution.ndim
    if temporal_axis <= 0 or temporal_axis >= attribution.ndim:
        raise ValueError(
            f"temporal_axis must refer to a non-batch dimension, got {temporal_axis}"
        )

    values = attribution.movedim(temporal_axis, -1)
    reduce_dims = tuple(range(1, values.ndim - 1))
    if not reduce_dims:
        return values
    if reduction == "l1":
        return values.abs().sum(dim=reduce_dims)
    if reduction == "l2":
        return values.square().sum(dim=reduce_dims).sqrt()
    if reduction == "signed":
        return values.sum(dim=reduce_dims)
    raise ValueError(f"Unknown attribution reduction: {reduction}")


def integrated_gradients_attribution(
    model: nn.Module,
    batch: tp.Any,
    modalities: tp.Sequence[str],
    *,
    target_vertices: TargetIndex = None,
    target_timesteps: TargetIndex = None,
    baselines: (
        tp.Mapping[str, torch.Tensor | float] | torch.Tensor | float | None
    ) = None,
    n_steps: int = 32,
    reduction: Reduction = "l1",
    temporal_axis: int = -1,
    pool_outputs: bool = True,
    eval_mode: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute integrated gradients attribution for feature timelines.

    Returns a dictionary mapping each modality to a tensor with shape
    ``(batch, feature_timesteps)``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    inputs = _collect_inputs(batch, modalities)
    baseline_map = {
        name: _baseline_for(name, tensor, baselines) for name, tensor in inputs.items()
    }
    grad_sums = {
        name: torch.zeros_like(tensor, dtype=torch.float32)
        for name, tensor in inputs.items()
    }

    with _attribution_model_context(model, eval_mode=eval_mode):
        for step in range(1, n_steps + 1):
            alpha = float(step) / float(n_steps)
            scaled_inputs = {}
            ordered_leaves = []
            for name, tensor in inputs.items():
                baseline = baseline_map[name]
                scaled = baseline + (tensor - baseline) * alpha
                scaled = scaled.detach().clone().requires_grad_(True)
                scaled_inputs[name] = scaled
                ordered_leaves.append(scaled)

            with _override_batch_data(batch, scaled_inputs):
                output = model(batch, pool_outputs=pool_outputs)
                score = select_output(
                    output,
                    target_vertices=target_vertices,
                    target_timesteps=target_timesteps,
                ).sum()

            grads = torch.autograd.grad(score, ordered_leaves, allow_unused=True)
            for name, grad in zip(scaled_inputs, grads):
                if grad is not None:
                    grad_sums[name] += grad.detach().to(torch.float32)

    attributions = {}
    for name, tensor in inputs.items():
        baseline = baseline_map[name]
        values = (tensor - baseline).detach().to(torch.float32) * (
            grad_sums[name] / float(n_steps)
        )
        attributions[name] = reduce_temporal_attribution(
            values,
            reduction=reduction,
            temporal_axis=temporal_axis,
        )
    return attributions


def occlusion_attribution(
    model: nn.Module,
    batch: tp.Any,
    modalities: tp.Sequence[str],
    *,
    target_vertices: TargetIndex = None,
    target_timesteps: TargetIndex = None,
    baselines: (
        tp.Mapping[str, torch.Tensor | float] | torch.Tensor | float | None
    ) = None,
    window: int = 1,
    stride: int = 1,
    temporal_axis: int = -1,
    pool_outputs: bool = True,
    eval_mode: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute perturbation attribution by replacing temporal windows.

    Positive values mean occluding that feature-time window reduced the selected
    model response. Returns ``(batch, feature_timesteps)`` tensors.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    inputs = _collect_inputs(batch, modalities)
    baseline_map = {
        name: _baseline_for(name, tensor, baselines) for name, tensor in inputs.items()
    }

    with _attribution_model_context(model, eval_mode=eval_mode), torch.inference_mode():
        base_output = model(batch, pool_outputs=pool_outputs)
        base_score = select_output(
            base_output,
            target_vertices=target_vertices,
            target_timesteps=target_timesteps,
        )

        out: dict[str, torch.Tensor] = {}
        for name, tensor in inputs.items():
            axis = temporal_axis if temporal_axis >= 0 else tensor.ndim + temporal_axis
            if axis <= 0 or axis >= tensor.ndim:
                raise ValueError(
                    f"temporal_axis must refer to a non-batch dimension, got {axis}"
                )
            n_timesteps = tensor.shape[axis]
            scores = torch.zeros(
                tensor.shape[0],
                n_timesteps,
                device=tensor.device,
                dtype=torch.float32,
            )
            counts = torch.zeros_like(scores)
            baseline = baseline_map[name]

            for start in range(0, n_timesteps, stride):
                stop = min(start + window, n_timesteps)
                occluded = tensor.detach().clone()
                occluded[_axis_slice(axis, start, stop, tensor.ndim)] = baseline[
                    _axis_slice(axis, start, stop, tensor.ndim)
                ]
                with _override_batch_data(batch, {name: occluded}):
                    occluded_output = model(batch, pool_outputs=pool_outputs)
                    occluded_score = select_output(
                        occluded_output,
                        target_vertices=target_vertices,
                        target_timesteps=target_timesteps,
                    )
                delta = (base_score - occluded_score).detach().to(torch.float32)
                scores[:, start:stop] += delta[:, None]
                counts[:, start:stop] += 1
            out[name] = scores / counts.clamp_min(1)
    return out


def _collect_inputs(
    batch: tp.Any, modalities: tp.Sequence[str]
) -> dict[str, torch.Tensor]:
    inputs = {}
    for name in modalities:
        if name not in batch.data:
            raise KeyError(f"Modality {name!r} is not present in the batch")
        tensor = batch.data[name]
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Expected tensor for modality {name!r}, got {type(tensor)}"
            )
        if not torch.is_floating_point(tensor):
            raise TypeError(
                f"Attribution requires floating point features for {name!r}, "
                f"got dtype {tensor.dtype}"
            )
        inputs[name] = tensor.detach()
    if not inputs:
        raise ValueError("At least one modality is required for attribution")
    return inputs


def _baseline_for(
    name: str,
    tensor: torch.Tensor,
    baselines: tp.Mapping[str, torch.Tensor | float] | torch.Tensor | float | None,
) -> torch.Tensor:
    if isinstance(baselines, Mapping):
        value = baselines.get(name, 0.0)
    elif baselines is None:
        value = 0.0
    else:
        value = baselines

    if torch.is_tensor(value):
        baseline = value.to(device=tensor.device, dtype=tensor.dtype)
        return torch.zeros_like(tensor) + baseline
    return torch.full_like(tensor, float(value))


def _normalize_index(index: TargetIndex, device: torch.device) -> TargetIndex:
    if isinstance(index, int):
        return [index]
    if torch.is_tensor(index):
        return index.to(device=device)
    return index


def _axis_slice(axis: int, start: int, stop: int, ndim: int) -> tuple[slice, ...]:
    idx = [slice(None)] * ndim
    idx[axis] = slice(start, stop)
    return tuple(idx)


@contextlib.contextmanager
def _override_batch_data(
    batch: tp.Any, replacements: tp.Mapping[str, torch.Tensor]
) -> tp.Iterator[None]:
    original = {name: batch.data[name] for name in replacements}
    try:
        batch.data.update(replacements)
        yield
    finally:
        batch.data.update(original)


@contextlib.contextmanager
def _attribution_model_context(
    model: nn.Module, *, eval_mode: bool
) -> tp.Iterator[None]:
    was_training = model.training
    requires_grad = [param.requires_grad for param in model.parameters()]
    try:
        if eval_mode:
            model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        yield
    finally:
        for param, original in zip(model.parameters(), requires_grad):
            param.requires_grad_(original)
        model.train(was_training)
