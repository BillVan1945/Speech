# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import get_rnnt_mask
from nemo.core.utils.optional_libs import TRITON_AVAILABLE
from nemo.utils.enum import PrettyStrEnum

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_consistency_triton import kl_loss_triton

EPS = 1e-5


class ConsistencyRNNTReductionType(PrettyStrEnum):
    MEAN = "mean"
    MEAN_VOLUME = "mean_volume"
    P_NON_BLANK = "p_non_blank"
    P_NON_BLANK_WITH_GRAD = "p_non_blank_with_grad"


def kl_loss_torch(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    mask: torch.Tensor,
    symmetrical: bool,
) -> torch.Tensor:
    """
    Compute masked KL divergence loss over probability distributions.

    Args:
        teacher_logprobs: Teacher log probabilities [B, T, U+1, V] where V is vocab size
        student_logprobs: Student log probabilities [B, T, U+1, V]
        mask: Mask tensor [B, T, U+1]
        symmetrical: Whether to use symmetric KL (average of both directions)

    Returns:
        KL loss tensor [B, T, U+1] with mask applied
    """
    # F.kl_div computes sum over last dim when reduction='none', but we want per-position
    # So we sum over the vocab dimension (K) ourselves
    kl_s_to_t = F.kl_div(
        input=student_logprobs,
        target=teacher_logprobs.detach(),
        reduction='none',
        log_target=True,
    ).sum(
        dim=-1
    )  # Sum over vocab dimension

    if symmetrical:
        kl_t_to_s = F.kl_div(
            input=teacher_logprobs,
            target=student_logprobs.detach(),
            reduction='none',
            log_target=True,
        ).sum(dim=-1)
        return 0.5 * (kl_s_to_t + kl_t_to_s) * mask
    return kl_s_to_t * mask


class ConsistencyRNNTLoss(nn.Module):
    def __init__(
        self,
        symmetrical: bool = True,
        reduction: str | ConsistencyRNNTReductionType = 'mean_volume',
        blank_id: int | None = None,
        backend: str | TurboTransducerBackend | None = None,
    ):
        super().__init__()
        self.reduction = ConsistencyRNNTReductionType(reduction)
        self.symmetrical = symmetrical
        self._backend = TurboTransducerBackend(backend) if backend is not None else None  # None means "auto"
        self.blank_id = blank_id

    def get_runtime_backend(self, input_: torch.Tensor) -> TurboTransducerBackend:
        if self._backend is not None:
            # backend if forced
            return self._backend
        if input_.device.type != "cuda":
            return TurboTransducerBackend.TORCH
        return TurboTransducerBackend.TRITON if TRITON_AVAILABLE else TurboTransducerBackend.TORCH

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        src_lengths: torch.Tensor | None = None,
        tgt_lengths: torch.Tensor | None = None,
    ):
        device = teacher_logits.device
        batch_size, src_length_max, tgt_length_max_plus_1, _ = teacher_logits.shape
        tgt_length_max = tgt_length_max_plus_1 - 1

        if src_lengths is None:
            src_lengths = torch.full([batch_size], fill_value=src_length_max, dtype=torch.long, device=device)
        if tgt_lengths is None:
            tgt_lengths = torch.full([batch_size], fill_value=tgt_length_max, dtype=torch.long, device=device)

        _, mask_blank = get_rnnt_mask(
            batch_size=batch_size,
            src_length_max=src_length_max,
            tgt_length_max_plus_1=tgt_length_max_plus_1,
            device=device,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
        )
        mask = mask_blank

        runtime_backend = self.get_runtime_backend(teacher_logits)
        if runtime_backend is TurboTransducerBackend.TRITON:
            kl_loss = kl_loss_triton(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                mask=mask,
                symmetrical=self.symmetrical,
                blank_id=self.blank_id,
                weighted=(
                    str(self.reduction)
                    if self.reduction
                    in {ConsistencyRNNTReductionType.P_NON_BLANK, ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD}
                    else None
                ),
            )

            match self.reduction:
                case ConsistencyRNNTReductionType.MEAN:
                    kl_loss_value = kl_loss.sum() / mask.sum().clamp(min=1)
                case ConsistencyRNNTReductionType.MEAN_VOLUME:
                    kl_loss_value = (kl_loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK:
                    assert kl_loss.dim() == 1  # already reduced
                    kl_loss_value = kl_loss.mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD:
                    assert kl_loss.dim() == 1  # already reduced
                    kl_loss_value = kl_loss.mean()
                case _:
                    raise NotImplementedError(f"Unsupported reduction {self.reduction}")
            return kl_loss_value
        elif runtime_backend is TurboTransducerBackend.TORCH:
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
            student_logprobs = F.log_softmax(student_logits, dim=-1)

            kl_loss = kl_loss_torch(
                teacher_logprobs=teacher_logprobs,
                student_logprobs=student_logprobs,
                mask=mask,
                symmetrical=self.symmetrical,
            )

            def get_non_blank_weights():
                if self.symmetrical:
                    blank_probs = (
                        torch.exp(teacher_logprobs[..., self.blank_id])
                        + torch.exp(student_logprobs[..., self.blank_id])
                    ) / 2
                else:
                    blank_probs = torch.exp(teacher_logprobs[..., self.blank_id])
                weights = (1 - blank_probs) * mask
                return weights

            match self.reduction:
                case ConsistencyRNNTReductionType.MEAN:
                    kl_loss_value = kl_loss.sum() / mask.sum().clamp(min=1)
                case ConsistencyRNNTReductionType.MEAN_VOLUME:
                    kl_loss_value = (kl_loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK:
                    weights = get_non_blank_weights().detach()
                    kl_loss_value = ((kl_loss * weights).sum(dim=(1, 2)) / (weights.sum(dim=(1, 2)) + EPS)).mean()
                case ConsistencyRNNTReductionType.P_NON_BLANK_WITH_GRAD:
                    weights = get_non_blank_weights()
                    kl_loss_value = ((kl_loss * weights).sum(dim=(1, 2)) / (weights.sum(dim=(1, 2)) + EPS)).mean()
                case _:
                    raise NotImplementedError(f"Unsupported reduction {self.reduction}")
            return kl_loss_value
        else:
            raise NotImplementedError(f"Unsupported backend {runtime_backend}")
