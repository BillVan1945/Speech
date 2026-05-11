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

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _rnnt_joint_fused_vocab_fwd_cutile_kernel(
    joint_hidden_2d,
    targets_flat,
    src_lengths,
    tgt_lengths,
    weight,
    bias,
    target_logprobs_out,
    blank_logprobs_out,
    log_sum_exp_out,
    flattened_batch_size,
    max_src_len,
    max_tgt_len_plus_1,
    max_target_len,
    hidden_dim: ConstInt,
    vocab_size: ConstInt,
    blank_id: ConstInt,
    FLATTENED_BATCH_BLOCK: ConstInt,
    HIDDEN_BLOCK: ConstInt,
    VOCAB_BLOCK: ConstInt,
):
    flattened_batch_block_index = ct.bid(0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + ct.arange(FLATTENED_BATCH_BLOCK, dtype=ct.int32)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets_in_st = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets_in_st // max_tgt_len_plus_1
    target_index = batch_offsets_in_st - source_index * max_tgt_len_plus_1

    source_length = ct.gather(src_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)
    target_length = ct.gather(tgt_lengths, batch_index, mask=flattened_batch_valid_mask, padding_value=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    target_flat_idx = batch_index * max_target_len + target_index
    targets_row = ct.gather(targets_flat, target_flat_idx, mask=output_target_mask, padding_value=0)

    log_sum_exp_score = ct.full((FLATTENED_BATCH_BLOCK,), float("-inf"), dtype=ct.float32)
    blank_logits = ct.zeros((FLATTENED_BATCH_BLOCK,), dtype=ct.float32)
    target_logits = ct.zeros((FLATTENED_BATCH_BLOCK,), dtype=ct.float32)

    num_vocab_blocks = ct.cdiv(vocab_size, VOCAB_BLOCK)
    num_hidden_blocks = ct.cdiv(hidden_dim, HIDDEN_BLOCK)
    vocab_in_block_offsets = ct.arange(VOCAB_BLOCK, dtype=ct.int32)

    for vocab_block_index in range(num_vocab_blocks):
        vocab_offsets = vocab_block_index * VOCAB_BLOCK + vocab_in_block_offsets
        vocab_mask = vocab_offsets < vocab_size

        bias_chunk = ct.load(
            bias,
            (vocab_block_index,),
            (VOCAB_BLOCK,),
            padding_mode=ct.PaddingMode.ZERO,
        )
        bias_chunk_f32 = ct.astype(bias_chunk, ct.float32)
        block_logits = ct.full((FLATTENED_BATCH_BLOCK, VOCAB_BLOCK), 0.0, dtype=ct.float32) + bias_chunk_f32[None, :]

        for hidden_block_index in range(num_hidden_blocks):
            hidden_tile = ct.load(
                joint_hidden_2d,
                (flattened_batch_block_index, hidden_block_index),
                (FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
                padding_mode=ct.PaddingMode.ZERO,
            )
            weight_tile = ct.load(
                weight,
                (vocab_block_index, hidden_block_index),
                (VOCAB_BLOCK, HIDDEN_BLOCK),
                padding_mode=ct.PaddingMode.ZERO,
            )
            block_logits = ct.mma(hidden_tile, ct.transpose(weight_tile), block_logits)

        block_logits = ct.where(vocab_mask[None, :], block_logits, float("-inf"))

        block_logits_max = ct.max(block_logits, axis=-1)
        block_lse = ct.log(ct.sum(ct.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max

        max_score = ct.maximum(log_sum_exp_score, block_lse)
        log_sum_exp_score = max_score + ct.log(ct.exp(log_sum_exp_score - max_score) + ct.exp(block_lse - max_score))

        blank_logits += ct.sum(ct.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)
        target_logits += ct.sum(ct.where(vocab_offsets[None, :] == targets_row[:, None], block_logits, 0.0), axis=-1)

    blank_logprobs_value = blank_logits - log_sum_exp_score
    target_logprobs_value = target_logits - log_sum_exp_score

    ct.scatter(blank_logprobs_out, flattened_batch_offsets, blank_logprobs_value, mask=output_blank_mask)
    ct.scatter(target_logprobs_out, flattened_batch_offsets, target_logprobs_value, mask=output_target_mask)
    ct.scatter(log_sum_exp_out, flattened_batch_offsets, log_sum_exp_score, mask=output_blank_mask)


class _RnntJointFusedVocabCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        joint_hidden: torch.Tensor,
        targets: torch.Tensor,
        tgt_lengths: torch.Tensor,
        src_lengths: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        blank_id: int,
        use_high_precision: bool = False,
    ):
        del use_high_precision  # accepted for API parity; not yet used by cuTile path

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        max_target_len = tgt_max_length_plus_1 - 1
        vocab_size = weight.shape[0]
        device = joint_hidden.device
        output_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        joint_hidden_2d = joint_hidden.view(flattened_batch_size, hidden_dim)
        targets_flat = targets.view(-1)

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=output_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        FLATTENED_BATCH_BLOCK = 128
        HIDDEN_BLOCK = 64
        VOCAB_BLOCK = 64
        flattened_batch_blocks = (flattened_batch_size + FLATTENED_BATCH_BLOCK - 1) // FLATTENED_BATCH_BLOCK

        ct.launch(
            torch.cuda.current_stream(),
            (flattened_batch_blocks, 1, 1),
            _rnnt_joint_fused_vocab_fwd_cutile_kernel,
            (
                joint_hidden_2d,
                targets_flat,
                src_lengths,
                tgt_lengths,
                weight,
                bias,
                target_logprobs.view(-1),
                blank_logprobs.view(-1),
                log_sum_exp_scores.view(-1),
                flattened_batch_size,
                src_max_length,
                tgt_max_length_plus_1,
                max_target_len,
                hidden_dim,
                vocab_size,
                blank_id,
                FLATTENED_BATCH_BLOCK,
                HIDDEN_BLOCK,
                VOCAB_BLOCK,
            ),
        )

        ctx.save_for_backward(joint_hidden, weight, bias)
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        # Stub: real backward not implemented yet — return zero gradients matching input dtypes.
        joint_hidden, weight, bias = ctx.saved_tensors
        grad_joint_hidden = torch.zeros_like(joint_hidden)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        return grad_joint_hidden, None, None, None, grad_weight, grad_bias, None, None


def rnnt_joint_fused_vocab_cutile(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision: bool = False,
):
    target_logprobs, blank_logprobs = _RnntJointFusedVocabCuTileFunction.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        use_high_precision,
    )
    return target_logprobs, blank_logprobs
