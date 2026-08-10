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
import torch.nn.functional as F

from nemo.collections.asr.parts.turbo_transducer.backend import TurboTransducerBackend
from nemo.collections.asr.parts.turbo_transducer.rnnt_joint_fused_vocab import rnnt_joint_fused_vocab
from nemo.collections.asr.parts.turbo_transducer.rnnt_logprobs import rnnt_logprobs_torch
from nemo.core.utils.optional_libs import CUTILE_AVAILABLE, TRITON_AVAILABLE
from tests.collections.asr.decoding.utils import avoid_sync_operations


def _reference_joint_vocab_logprobs(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, blank_id):
    logits = F.linear(joint_hidden, weight, bias)
    target_logprobs, blank_logprobs = rnnt_logprobs_torch(
        logits=logits,
        targets=targets,
        blank_id=blank_id,
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
    )
    return target_logprobs, blank_logprobs


_BACKEND_PARAMS = [
    pytest.param(
        TurboTransducerBackend.TRITON,
        marks=pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed"),
        id="triton",
    ),
    pytest.param(
        TurboTransducerBackend.CUTILE,
        marks=pytest.mark.skipif(not CUTILE_AVAILABLE, reason="cuTile is not installed"),
        id="cutile",
    ),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("backend", _BACKEND_PARAMS)
class TestRnntJointVocabLogProbs:
    @pytest.mark.parametrize(
        "shape",
        [
            (1, 2, 1, 4, 3),
            (2, 4, 3, 16, 8),
            (4, 16, 8, 64, 32),
            (5, 17, 65, 28, 35),
            (2, 64, 32, 640, 1024),
        ],
    )
    @pytest.mark.parametrize(
        "float_dtype",
        [torch.float32] + ([torch.bfloat16] if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else []),
    )
    def test_forward_and_backward(self, backend, shape, float_dtype):
        device = torch.device("cuda")
        torch.manual_seed(42)
        # Both backends honor use_high_precision_mm, so validate them under identical precision settings.
        use_high_precision = True
        is_triton = backend == TurboTransducerBackend.TRITON
        batch_size, src_length, tgt_length, hidden_dim, vocab_size_no_blank = shape
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=float_dtype
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=float_dtype)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=float_dtype)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        if float_dtype == torch.bfloat16:
            joint_hidden_ref = joint_hidden.detach().float().clone().requires_grad_(True)
            weight_ref = weight.detach().float().clone().requires_grad_(True)
            bias_ref = bias.detach().float().clone().requires_grad_(True)
        else:
            joint_hidden_ref = joint_hidden.detach().clone().requires_grad_(True)
            weight_ref = weight.detach().clone().requires_grad_(True)
            bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        joint_hidden_kernel = joint_hidden.detach().clone().requires_grad_(True)
        weight_kernel = weight.detach().clone().requires_grad_(True)
        bias_kernel = bias.detach().clone().requires_grad_(True)
        target_kernel, blank_kernel = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden_kernel,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_kernel,
            bias=bias_kernel,
            blank_id=blank_id,
            use_high_precision_mm=use_high_precision,
            input_grad_full_precision=use_high_precision,
            weight_grad_full_precision=use_high_precision,
            backend=backend,
        )

        # Triton can request IEEE-strict fp32 matmul (tl.dot(input_precision="ieee")) and so matches the torch
        # reference bitwise-closely. cuTile's mma has no IEEE mode, so its error is relative rather than
        # absolute: 5e-4 on logits of magnitude ~200 is ~2.5e-6 relative.
        forward_atol = 1e-5 if is_triton else 5e-4
        assert torch.allclose(
            target_kernel, target_ref, atol=forward_atol
        ), f"target logprobs mismatch: max diff={(target_kernel - target_ref).abs().max().item()}"
        assert torch.allclose(
            blank_kernel, blank_ref, atol=forward_atol
        ), f"blank logprobs mismatch: max diff={(blank_kernel - blank_ref).abs().max().item()}"

        torch.manual_seed(123)
        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)

        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_kernel = (target_scales * target_kernel + blank_scales * blank_kernel).sum()
        loss_ref.backward()
        loss_kernel.backward()

        assert joint_hidden_kernel.grad.dtype == float_dtype
        assert weight_kernel.grad.dtype == float_dtype
        assert bias_kernel.grad.dtype == float_dtype

        if float_dtype == torch.bfloat16:
            grad_atol = 5e-3
            grad_rtol = 5e-3
            weight_bias_atol = 1.0
            weight_bias_rtol = 1e-3
        else:
            grad_atol = 1e-4 if is_triton else 5e-4
            grad_rtol = 1e-5
            weight_bias_atol = 1e-4 if is_triton else 5e-4
            weight_bias_rtol = 1e-5
        assert torch.allclose(
            joint_hidden_kernel.grad.float(), joint_hidden_ref.grad.float(), atol=grad_atol, rtol=grad_rtol
        ), f"joint_hidden grad mismatch: max diff={(joint_hidden_kernel.grad.float() - joint_hidden_ref.grad.float()).abs().max().item()}"

        assert torch.allclose(
            weight_kernel.grad.float(), weight_ref.grad.float(), atol=weight_bias_atol, rtol=weight_bias_rtol
        ), f"weight grad mismatch: max diff={(weight_kernel.grad.float() - weight_ref.grad.float()).abs().max().item()}"
        assert torch.allclose(
            bias_kernel.grad.float(), bias_ref.grad.float(), atol=weight_bias_atol, rtol=weight_bias_rtol
        ), f"bias grad mismatch: max diff={(bias_kernel.grad.float() - bias_ref.grad.float()).abs().max().item()}"

    def test_variable_lengths(self, backend):
        device = torch.device("cuda")
        torch.manual_seed(42)
        use_high_precision = True
        batch_size = 4
        src_length = 16
        tgt_length = 8
        hidden_dim = 32
        vocab_size_no_blank = 16
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length, src_length // 2, src_length - 1, 1], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length, tgt_length // 2, tgt_length - 1, 0], device=device, dtype=torch.long)

        joint_hidden_ref = joint_hidden.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden_ref,
            weight=weight_ref,
            bias=bias_ref,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )

        joint_hidden_kernel = joint_hidden.detach().clone().requires_grad_(True)
        weight_kernel = weight.detach().clone().requires_grad_(True)
        bias_kernel = bias.detach().clone().requires_grad_(True)
        target_kernel, blank_kernel = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden_kernel,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight_kernel,
            bias=bias_kernel,
            blank_id=blank_id,
            use_high_precision_mm=use_high_precision,
            input_grad_full_precision=use_high_precision,
            weight_grad_full_precision=use_high_precision,
            backend=backend,
        )

        forward_atol = 1e-5
        assert torch.allclose(target_kernel, target_ref, atol=forward_atol)
        assert torch.allclose(blank_kernel, blank_ref, atol=forward_atol)

        target_scales = torch.rand_like(target_ref)
        blank_scales = torch.rand_like(blank_ref)
        loss_ref = (target_scales * target_ref + blank_scales * blank_ref).sum()
        loss_kernel = (target_scales * target_kernel + blank_scales * blank_kernel).sum()
        loss_ref.backward()
        loss_kernel.backward()

        assert torch.allclose(joint_hidden_kernel.grad.float(), joint_hidden_ref.grad.float(), atol=5e-3, rtol=1e-3)
        assert torch.allclose(weight_kernel.grad.float(), weight_ref.grad.float(), atol=0.5, rtol=1e-3)
        assert torch.allclose(bias_kernel.grad.float(), bias_ref.grad.float(), atol=0.5, rtol=1e-3)

    def test_edge_case_single_frame(self, backend):
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 1
        src_length = 1
        tgt_length = 1
        hidden_dim = 4
        vocab_size_no_blank = 3
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.tensor([src_length], device=device, dtype=torch.long)
        tgt_lengths = torch.tensor([tgt_length], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_kernel, blank_kernel = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            backend=backend,
        )

        assert torch.allclose(target_kernel, target_ref, atol=5e-3)
        assert torch.allclose(blank_kernel, blank_ref, atol=5e-3)

    def test_edge_case_blank_only(self, backend):
        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size = 2
        src_length = 4
        tgt_length = 0
        hidden_dim = 8
        vocab_size_no_blank = 5
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32)
        targets = torch.randint(
            0, vocab_size_no_blank, (batch_size, max(tgt_length, 1)), device=device, dtype=torch.long
        )
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.zeros([batch_size], device=device, dtype=torch.long)

        target_ref, blank_ref = _reference_joint_vocab_logprobs(
            joint_hidden=joint_hidden,
            weight=weight,
            bias=bias,
            targets=targets,
            src_lengths=src_lengths,
            tgt_lengths=tgt_lengths,
            blank_id=blank_id,
        )
        target_kernel, blank_kernel = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            backend=backend,
        )

        assert torch.allclose(target_kernel, target_ref, atol=5e-3)
        assert torch.allclose(blank_kernel, blank_ref, atol=5e-3)

    def test_matches_triton_backend(self, backend):
        """Cross-check every backend against Triton at *default* precision.

        ``test_forward_and_backward`` compares against an fp32 torch reference and therefore has to run with
        ``use_high_precision_mm=True``. That leaves the production path — tensor-core mma into low-precision
        gradient accumulators — validated only by very loose tolerances. Here both backends run the identical
        default configuration, so any divergence is a real kernel bug rather than accumulated rounding.
        """
        if backend == TurboTransducerBackend.TRITON:
            pytest.skip("Triton is the reference for this comparison")
        if not TRITON_AVAILABLE:
            pytest.skip("Triton is not installed")

        device = torch.device("cuda")
        torch.manual_seed(42)
        batch_size, src_length, tgt_length, hidden_dim, vocab_size_no_blank = 2, 64, 32, 640, 1024
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.bfloat16
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.bfloat16)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.bfloat16)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        torch.manual_seed(123)
        target_scales = torch.rand(batch_size, src_length, tgt_length + 1, device=device)
        blank_scales = torch.rand(batch_size, src_length, tgt_length + 1, device=device)

        outputs = {}
        for current_backend in (TurboTransducerBackend.TRITON, backend):
            joint_hidden_kernel = joint_hidden.detach().clone().requires_grad_(True)
            weight_kernel = weight.detach().clone().requires_grad_(True)
            bias_kernel = bias.detach().clone().requires_grad_(True)
            target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
                joint_hidden=joint_hidden_kernel,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight_kernel,
                bias=bias_kernel,
                blank_id=blank_id,
                backend=current_backend,
            )
            (target_scales * target_logprobs + blank_scales * blank_logprobs).sum().backward()
            outputs[current_backend] = (
                target_logprobs,
                blank_logprobs,
                joint_hidden_kernel.grad,
                weight_kernel.grad,
                bias_kernel.grad,
            )

        # Compare relative to each tensor's own magnitude: the two backends accumulate the same bf16 math in a
        # different order, so the floor is set by bf16 rounding (~2^-8 per step), not by an absolute epsilon.
        # The bounds below carry ~4x headroom over the worst observed value; they are deliberately not tighter
        # because Triton autotunes its block sizes, and a different FLATTENED_BATCH_BLOCK changes how many
        # atomic accumulations land in each bf16 gradient element. A genuinely broken kernel diverges by O(1).
        names = ("target_logprobs", "blank_logprobs", "joint_hidden grad", "weight grad", "bias grad")
        max_relative_diffs = (1e-5, 1e-5, 3e-2, 1e-1, 4e-2)
        for name, reference, actual, max_relative_diff in zip(
            names, outputs[TurboTransducerBackend.TRITON], outputs[backend], max_relative_diffs
        ):
            reference = reference.float()
            actual = actual.float()
            relative_diff = (actual - reference).abs().max().item() / reference.abs().max().clamp_min(1e-30).item()
            assert (
                relative_diff <= max_relative_diff
            ), f"{name} differs from Triton: relative diff={relative_diff:.3g} > {max_relative_diff:.3g}"

    def test_no_cuda_sync_operations(self, backend):
        device = torch.device("cuda")
        torch.manual_seed(42)

        batch_size = 1
        src_length = 2
        tgt_length = 1
        hidden_dim = 4
        vocab_size_no_blank = 3
        blank_id = vocab_size_no_blank
        vocab_size_with_blank = vocab_size_no_blank + 1

        joint_hidden = torch.randn(
            batch_size, src_length, tgt_length + 1, hidden_dim, device=device, dtype=torch.float32, requires_grad=True
        )
        weight = torch.randn(vocab_size_with_blank, hidden_dim, device=device, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(vocab_size_with_blank, device=device, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, vocab_size_no_blank, (batch_size, tgt_length), device=device, dtype=torch.long)
        src_lengths = torch.full([batch_size], src_length, device=device, dtype=torch.long)
        tgt_lengths = torch.full([batch_size], tgt_length, device=device, dtype=torch.long)

        # Warmup
        target_logprobs, blank_logprobs = rnnt_joint_fused_vocab(
            joint_hidden=joint_hidden,
            targets=targets,
            tgt_lengths=tgt_lengths,
            src_lengths=src_lengths,
            weight=weight,
            bias=bias,
            blank_id=blank_id,
            backend=backend,
        )
        warmup_loss = target_logprobs.sum() + blank_logprobs.sum()
        warmup_loss.backward()
        torch.cuda.synchronize()

        joint_hidden_test = joint_hidden.detach().clone().requires_grad_(True)
        weight_test = weight.detach().clone().requires_grad_(True)
        bias_test = bias.detach().clone().requires_grad_(True)

        with avoid_sync_operations(device):
            target_logprobs_test, blank_logprobs_test = rnnt_joint_fused_vocab(
                joint_hidden=joint_hidden_test,
                targets=targets,
                tgt_lengths=tgt_lengths,
                src_lengths=src_lengths,
                weight=weight_test,
                bias=bias_test,
                blank_id=blank_id,
                backend=backend,
            )
            loss = target_logprobs_test.sum() + blank_logprobs_test.sum()
            loss.backward()
