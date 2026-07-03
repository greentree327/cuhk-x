"""
# CONVENTION: primary — Propagating-exception convention.
# Guard-clause failures raise exceptions that propagate to callers.

Masked Batch Normalization for variable-length sequences.
Ported from CMI 2nd place solution.

Only real (non-padding) elements contribute to mean/variance statistics.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm


def masked_batch_norm(
    input: Tensor,
    mask: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float = 1e-5,
) -> Tensor:
    """Apply Batch Normalization only over non-padded positions.

    Args:
        input: (N, C, *) feature tensor.
        mask: (N, 1, *) float mask (1 = real, 0 = padding).
        weight: gamma parameter, (C,).
        bias: beta parameter, (C,).
        running_mean: buffer, (C,).
        running_var: buffer, (C,).
        training: whether in training mode.
        momentum: EMA momentum for running stats.
        eps: epsilon for numerical stability.

    Returns:
        Normalized tensor, same shape as input.
    """
    if not training and (running_mean is None or running_var is None):
        raise ValueError(
            "Expected running_mean and running_var to be not None when training=False"
        )

    num_dims = len(input.shape[2:])
    _dims = (0,) + tuple(range(-num_dims, 0))
    _slice = (None, ...) + (None,) * num_dims

    if training:
        num_elements = mask.sum(_dims)
        mean = (input * mask).sum(_dims) / num_elements
        var = (((input - mean[_slice]) * mask) ** 2).sum(_dims) / num_elements

        if running_mean is not None:
            running_mean.copy_(running_mean * (1 - momentum) + momentum * mean.detach())
        if running_var is not None:
            running_var.copy_(running_var * (1 - momentum) + momentum * var.detach())
    else:
        mean, var = running_mean, running_var

    out = (input - mean[_slice]) / torch.sqrt(var[_slice] + eps)

    if weight is not None and bias is not None:
        out = out * weight[_slice] + bias[_slice]

    return out


class _MaskedBatchNorm(_BatchNorm):
    """Base class for masked batch norm variants."""

    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
    ):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        self._check_input_dim(input)
        if mask is not None:
            self._check_input_dim(mask)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        bn_training = (
            True
            if self.training
            else (self.running_mean is None and self.running_var is None)
        )

        if mask is None:
            return F.batch_norm(
                input,
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var
                if not self.training or self.track_running_stats
                else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )

        return masked_batch_norm(
            input,
            mask,
            self.weight,
            self.bias,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class MaskedBatchNorm1d(nn.BatchNorm1d, _MaskedBatchNorm):
    """Masked BatchNorm1d for 3D inputs (N, C, L).

    Shape:
        - Input: (N, C, L)
        - Mask: (N, L) — will be reshaped to (N, 1, L) internally.
        - Output: (N, C, L)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        channels_last: bool = False,
    ) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.channels_last = channels_last

    def forward(self, inputs, mask=None):
        if self.channels_last:
            inputs = inputs.permute(0, 2, 1)
        if mask is not None:
            mask = mask[:, None, :]  # (N, L) → (N, 1, L)
        out = super().forward(inputs, mask)
        if self.channels_last:
            out = out.permute(0, 2, 1)
        return out
