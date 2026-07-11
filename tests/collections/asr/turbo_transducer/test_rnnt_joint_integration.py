# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import nemo.collections.asr.modules.rnnt as rnnt_module
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.modules.rnnt_abstract import AbstractRNNTJoint
from nemo.core.utils.optional_libs import TRITON_AVAILABLE


def _jointnet_config(dropout: float = 0.0, activation: str = "relu"):
    return {
        "encoder_hidden": 5,
        "pred_hidden": 6,
        "joint_hidden": 16,
        "activation": activation,
        "dropout": dropout,
    }


def _turbo_kernel_config(kernel_type: str):
    return {
        "type": kernel_type,
        "kwargs": {
            "backend": "triton",
            "input_grad_full_precision": True,
            "use_high_precision_mm": True,
            "weight_grad_full_precision": True,
        },
    }


class _PlainJoint(AbstractRNNTJoint):
    def project_encoder(self, encoder_output):
        return encoder_output

    def project_prednet(self, prednet_output):
        return prednet_output

    def joint_after_projection(self, encoder_output, predictor_output):
        return encoder_output + predictor_output


@pytest.mark.unit
def test_abstract_joint_does_not_use_turbo_kernel_by_default():
    assert not _PlainJoint().uses_turbo_kernel


@pytest.mark.unit
@pytest.mark.parametrize("kernel_type", ["fused_vocab", "fused"])
def test_joint_passes_live_vocab_projection_parameters(monkeypatch, kernel_type):
    calls = []

    def fake_fused_vocab(joint_hidden, weight, bias, **kwargs):
        calls.append({"weight": weight, "bias": bias})
        output_shape = joint_hidden.shape[:-1]
        output = torch.zeros(output_shape, device=joint_hidden.device)
        return output, output

    def fake_fused(
        encoder_output_projected,
        predictor_output_projected,
        weight,
        bias,
        dropout_p,
        **kwargs,
    ):
        calls.append({"weight": weight, "bias": bias, "dropout_p": dropout_p})
        output_shape = (
            encoder_output_projected.shape[0],
            encoder_output_projected.shape[1],
            predictor_output_projected.shape[1],
        )
        output = torch.zeros(output_shape, device=encoder_output_projected.device)
        return output, output

    monkeypatch.setattr(rnnt_module, "rnnt_joint_fused_vocab", fake_fused_vocab)
    monkeypatch.setattr(rnnt_module, "rnnt_fused_joint", fake_fused)

    joint = RNNTJoint(
        jointnet=_jointnet_config(dropout=0.25),
        num_classes=8,
        log_softmax=False,
        turbo_kernel=_turbo_kernel_config(kernel_type),
    )
    encoder_outputs = torch.randn(2, 5, 4)
    decoder_outputs = torch.randn(2, 6, 4)
    targets = torch.randint(0, 8, [2, 3])
    source_lengths = torch.tensor([4, 3])
    target_lengths = torch.tensor([3, 2])

    joint.train()
    joint.forward_for_loss(
        encoder_outputs=encoder_outputs,
        decoder_outputs=decoder_outputs,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )

    vocab_projection = joint.joint_net[-1]
    assert calls[-1]["weight"] is vocab_projection.weight
    assert calls[-1]["bias"] is vocab_projection.bias
    if kernel_type == "fused":
        assert calls[-1]["dropout_p"] == 0.25
        joint.eval()
        joint.forward_for_loss(
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_outputs,
            encoder_lengths=source_lengths,
            transcripts=targets,
            transcript_lengths=target_lengths,
        )
        assert calls[-1]["dropout_p"] == 0.0


@pytest.mark.unit
def test_model_routes_turbo_joint_output_to_loss_logprobs_path():
    target_logprobs = torch.randn(2, 4, 4)
    blank_logprobs = torch.randn(2, 4, 4)
    loss_value = torch.tensor(1.5)
    joint = SimpleNamespace(
        uses_turbo_kernel=True,
        forward_for_loss=Mock(return_value=(target_logprobs, blank_logprobs)),
    )
    loss = SimpleNamespace(forward_from_logprobs=Mock(return_value=loss_value))
    model = SimpleNamespace(joint=joint, loss=loss)
    encoded = torch.randn(2, 5, 4)
    decoder = torch.randn(2, 6, 4)
    transcript = torch.randint(0, 8, [2, 3])
    source_lengths = torch.tensor([4, 3])
    target_lengths = torch.tensor([3, 2])

    actual_loss, joint_logits = EncDecRNNTModel._compute_rnnt_loss(
        model,
        encoded=encoded,
        decoder=decoder,
        transcript=transcript,
        encoded_len=source_lengths,
        target_length=target_lengths,
    )

    assert actual_loss is loss_value
    assert joint_logits is None
    joint.forward_for_loss.assert_called_once_with(
        encoder_outputs=encoded,
        decoder_outputs=decoder,
        encoder_lengths=source_lengths,
        transcripts=transcript,
        transcript_lengths=target_lengths,
    )
    loss.forward_from_logprobs.assert_called_once_with(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )


@pytest.mark.unit
def test_joint_rejects_incompatible_turbo_configuration():
    with pytest.raises(ValueError, match="fuse_loss_wer"):
        RNNTJoint(
            jointnet=_jointnet_config(),
            num_classes=8,
            turbo_kernel=_turbo_kernel_config("fused"),
            fuse_loss_wer=True,
            fused_batch_size=1,
        )

    with pytest.raises(ValueError, match="only relu"):
        RNNTJoint(
            jointnet=_jointnet_config(activation="tanh"),
            num_classes=8,
            turbo_kernel=_turbo_kernel_config("fused"),
        )


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kernel_type", ["fused_vocab", "fused"])
def test_joint_and_loss_integration_matches_standard(kernel_type):
    torch.manual_seed(42)
    device = torch.device("cuda")
    standard_joint = RNNTJoint(
        jointnet=_jointnet_config(),
        num_classes=8,
        log_softmax=False,
    ).to(device)
    optimized_joint = RNNTJoint(
        jointnet=_jointnet_config(),
        num_classes=8,
        log_softmax=False,
        turbo_kernel=_turbo_kernel_config(kernel_type),
    ).to(device)
    optimized_joint.load_state_dict(standard_joint.state_dict(), strict=True)

    standard_encoder = torch.randn(2, 5, 4, device=device, requires_grad=True)
    standard_decoder = torch.randn(2, 6, 4, device=device, requires_grad=True)
    optimized_encoder = standard_encoder.detach().clone().requires_grad_(True)
    optimized_decoder = standard_decoder.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 8, [2, 3], device=device)
    source_lengths = torch.tensor([4, 3], device=device)
    target_lengths = torch.tensor([3, 2], device=device)

    with torch.no_grad():
        standard_logits = standard_joint(
            encoder_outputs=standard_encoder,
            decoder_outputs=standard_decoder,
        )
        optimized_standard_logits = optimized_joint(
            encoder_outputs=optimized_encoder,
            decoder_outputs=optimized_decoder,
        )
    assert torch.equal(optimized_standard_logits, standard_logits)

    standard_loss = RNNTLoss(
        num_classes=8,
        reduction="mean_volume",
        loss_name="turbo_rnnt",
        loss_kwargs={"backend": "triton"},
    )
    optimized_loss = RNNTLoss(
        num_classes=8,
        reduction="mean_volume",
        loss_name="turbo_rnnt",
        loss_kwargs={"backend": "triton"},
    )

    standard_logits = standard_joint(
        encoder_outputs=standard_encoder,
        decoder_outputs=standard_decoder,
    )
    standard_loss_value = standard_loss(
        log_probs=standard_logits,
        targets=targets,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    target_logprobs, blank_logprobs = optimized_joint.forward_for_loss(
        encoder_outputs=optimized_encoder,
        decoder_outputs=optimized_decoder,
        encoder_lengths=source_lengths,
        transcripts=targets,
        transcript_lengths=target_lengths,
    )
    optimized_loss_value = optimized_loss.forward_from_logprobs(
        target_logprobs=target_logprobs,
        blank_logprobs=blank_logprobs,
        input_lengths=source_lengths,
        target_lengths=target_lengths,
    )

    assert torch.allclose(optimized_loss_value, standard_loss_value, atol=1e-4, rtol=1e-4)
    standard_loss_value.backward()
    optimized_loss_value.backward()

    assert torch.allclose(optimized_encoder.grad, standard_encoder.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(optimized_decoder.grad, standard_decoder.grad, atol=2e-4, rtol=2e-4)
    optimized_parameters = dict(optimized_joint.named_parameters())
    for parameter_name, standard_parameter in standard_joint.named_parameters():
        optimized_parameter = optimized_parameters[parameter_name]
        assert standard_parameter.grad is not None, parameter_name
        assert optimized_parameter.grad is not None, parameter_name
        assert torch.allclose(
            optimized_parameter.grad,
            standard_parameter.grad,
            atol=2e-4,
            rtol=2e-4,
        ), parameter_name

    assert optimized_joint.joint_net[-1].weight.grad is not None
    assert optimized_joint.joint_net[-1].bias.grad is not None
