# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``TemporalSmoothing`` (CPU only, no data or weights needed)."""

import torch
from torch import nn

from tribev2.model import TemporalSmoothing


def test_build_returns_depthwise_conv1d():
    conv = TemporalSmoothing(kernel_size=9, sigma=2.0).build(dim=4)

    assert isinstance(conv, nn.Conv1d)
    # Depthwise: groups == channels, so the weight is (dim, 1, kernel_size).
    assert conv.groups == 4
    assert conv.weight.shape == (4, 1, 9)
    assert conv.bias is None


def test_gaussian_kernel_is_normalized_per_channel():
    dim, kernel_size = 4, 9
    conv = TemporalSmoothing(kernel_size=kernel_size, sigma=2.0).build(dim=dim)

    per_channel_sum = conv.weight.detach().sum(dim=-1).reshape(dim)
    assert torch.allclose(per_channel_sum, torch.ones(dim), atol=1e-5)


def test_gaussian_kernel_is_symmetric():
    conv = TemporalSmoothing(kernel_size=9, sigma=2.0).build(dim=1)

    kernel = conv.weight.detach()[0, 0]
    assert torch.allclose(kernel, torch.flip(kernel, dims=[0]), atol=1e-6)


def test_output_length_is_preserved():
    conv = TemporalSmoothing(kernel_size=9, sigma=2.0).build(dim=4)

    x = torch.randn(2, 4, 50)
    y = conv(x)
    assert y.shape == (2, 4, 50)


def test_constant_signal_is_unchanged_in_interior():
    # A normalized smoothing kernel must leave a constant signal unchanged,
    # away from the zero-padded borders.
    conv = TemporalSmoothing(kernel_size=9, sigma=2.0).build(dim=3)

    x = torch.ones(1, 3, 50)
    y = conv(x).detach()
    # kernel_size=9, padding=4 means only indices 0-3 and 46-49 are
    # contaminated by zero-padding; 10:40 is safely interior.
    interior = y[:, :, 10:40]
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-5)


def test_default_sigma_is_none():
    # With no sigma the conv keeps its randomly initialized, trainable weights
    # (the Gaussian branch is skipped).
    conv = TemporalSmoothing(kernel_size=5).build(dim=2)

    assert isinstance(conv, nn.Conv1d)
    assert conv.weight.shape == (2, 1, 5)
