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


class _RecordingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reduction = "sum"
        self.batch_sizes = []

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        self.batch_sizes.append(log_probs.shape[0])
        return log_probs.sum(dim=(1, 2, 3))

    def reduce(self, losses, target_lengths):
        return torch.cat(losses).sum()


@pytest.mark.unit
def test_standard_joint_uses_native_fused_batching():
    batch_size, hidden_dim, max_time, max_targets, vocab_size = 5, 4, 3, 2, 7
    joint = benchmark_rnnt_joint._create_standard_joint(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        dropout_p=0.0,
        fused_batch_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loss = _RecordingLoss()
    joint.set_loss(loss)
    joint.set_wer(object())
    encoder_outputs = torch.randn(batch_size, hidden_dim, max_time, requires_grad=True)
    decoder_outputs = torch.randn(batch_size, hidden_dim, max_targets + 1, requires_grad=True)
    targets = torch.randint(0, vocab_size - 1, (batch_size, max_targets))
    source_lengths = torch.full((batch_size,), max_time, dtype=torch.long)
    target_lengths = torch.full((batch_size,), max_targets, dtype=torch.long)

    loss_value, wer, wer_num, wer_denom = joint(
        encoder_outputs=encoder_outputs,
        decoder_outputs=decoder_outputs,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )
    loss_value.backward()

    assert joint.fuse_loss_wer
    assert joint.fused_batch_size == 2
    assert isinstance(joint.enc, torch.nn.Identity)
    assert isinstance(joint.pred, torch.nn.Identity)
    assert loss.batch_sizes == [2, 2, 1]
    assert encoder_outputs.grad is not None
    assert decoder_outputs.grad is not None
    assert (wer, wer_num, wer_denom) == (None, None, None)


@pytest.mark.unit
def test_standard_joint_disables_fused_batching_by_default():
    joint = benchmark_rnnt_joint._create_standard_joint(
        hidden_dim=4,
        vocab_size=7,
        dropout_p=0.0,
        fused_batch_size=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert not joint.fuse_loss_wer
    assert joint.fused_batch_size is None


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
