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
import torch.nn.functional as F

from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE
from nemo.utils.enum import PrettyStrEnum

if TRITON_AVAILABLE:
    from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab_triton import rnnt_joint_fused_vocab_triton


class JointFusedVocabBackend(PrettyStrEnum):
    TRITON = "triton"
    CUTILE = "cutile"


def rnnt_joint_fused_vocab(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision: bool = False,
    backend: str | JointFusedVocabBackend | None = JointFusedVocabBackend.TRITON,
):
    backend = JointFusedVocabBackend(backend) if backend is not None else None

    if backend is None:
        if TRITON_AVAILABLE:
            backend = JointFusedVocabBackend.TRITON
        elif CUTILE_AVAILABLE:
            backend = JointFusedVocabBackend.CUTILE
        else:
            raise Exception("No backend")

    match backend:
        case JointFusedVocabBackend.TRITON:
            if not TRITON_AVAILABLE:
                raise ModuleNotFoundError("Triton is unavailable")
            target_logprobs, blank_logprobs = rnnt_joint_fused_vocab_triton(
                joint_hidden,
                targets,
                tgt_lengths,
                src_lengths,
                weight,
                bias,
                blank_id,
                use_high_precision,
            )
        case JointFusedVocabBackend.CUTILE:
            if not CUTILE_AVAILABLE:
                raise ModuleNotFoundError("cuTile is unavailable")
            # TODO: implement
        case _:
            raise Exception("Not implemented backend")
    return target_logprobs, blank_logprobs
