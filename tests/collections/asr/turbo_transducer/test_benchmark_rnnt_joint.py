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

import pytest
import torch

from tests.collections.asr.turbo_transducer import benchmark_rnnt_joint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_forward_memory_baseline_excludes_warmup_gradients(monkeypatch):
    batch_size = 2
    max_time = 4
    max_targets = 3

    def fake_fused_joint(encoder_output_projected, predictor_output_projected, weight, bias, **kwargs):
        anchor = (
            encoder_output_projected[..., 0].sum()
            + predictor_output_projected[..., 0].sum()
            + weight[:, 0].sum()
            + bias.sum()
        ) * 0.0
        output_shape = (batch_size, max_time, max_targets + 1)
        target_logprobs = torch.zeros(output_shape, device="cuda", dtype=torch.float32) + anchor
        blank_logprobs = torch.zeros(output_shape, device="cuda", dtype=torch.float32) + anchor
        return target_logprobs, blank_logprobs

    def fake_loss(target_logprobs, blank_logprobs, **kwargs):
        return (target_logprobs + blank_logprobs).sum(dim=(1, 2))

    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_fused_joint", fake_fused_joint)
    monkeypatch.setattr(benchmark_rnnt_joint, "rnnt_loss_from_logprobs_triton", fake_loss)

    result = benchmark_rnnt_joint.run_benchmark(
        joint="fused",
        loss="turbo_rnnt",
        dtype=torch.float32,
        warmup_iters=1,
        bench_iters=1,
        batch_size=batch_size,
        max_time=max_time,
        hidden_dim=64,
        vocab_size=32,
        max_targets=max_targets,
        dropout_p=0.0,
    )

    output_bytes = 2 * batch_size * max_time * (max_targets + 1) * torch.float32.itemsize
    assert result.forward_memory_gb * 1024**3 >= output_bytes
