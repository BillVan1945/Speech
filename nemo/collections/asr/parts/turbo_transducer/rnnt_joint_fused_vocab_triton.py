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
import triton
import triton.language as tl


from nemo.collections.asr.parts.turbo_transducer.utils_triton import log_add_exp, matmul


def _fwd_autotune_configs():
    configs = []
    # (FB, HB, VB, num_warps, num_stages)
    # Cover small / medium / large vocab regimes. All tile dims are powers of 2
    # and >= 16 so tl.dot can hit tensor cores. Smaller VB favors V=1024 where
    # the vocab dim is short; larger VB amortizes the hidden K-loop on V=8192.
    candidates = [
        # Small tiles — low register pressure, good for small problem sizes
        # (32, 32, 32, 4, 3),
        # (32, 64, 64, 4, 3),
        # (64, 32, 32, 4, 3),
        (64, 32, 64, 4, 3),
        # Medium tiles — typical sweet spot
        # (64, 64, 64, 4, 3),
        # (64, 64, 128, 4, 3),
        # (64, 128, 64, 4, 3),
        # (128, 32, 64, 4, 3),
        (128, 32, 128, 4, 3),  # best for 1024,8192
        # (128, 64, 64, 4, 3),
        # (128, 64, 128, 4, 3),
        # # Larger VB for big vocab — amortize hidden K-loop
        # (64, 64, 256, 4, 3),
        # (64, 64, 256, 8, 2),
        # (128, 64, 256, 8, 2),
        # # Higher num_warps variants for parallelism
        # (128, 64, 128, 8, 2),
        # (64, 128, 128, 8, 3),
    ]
    for fb, hb, vb, nw, ns in candidates:
        configs.append(
            triton.Config(
                {"FLATTENED_BATCH_BLOCK": fb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


def _bwd_autotune_configs():
    configs = []
    # Bwd has 2 K-loops, holds a [FB, VB] logits buffer, and uses atomic adds.
    # Register/shared pressure is higher than fwd, so we explore a similar grid
    # but lean smaller on FB*VB. (FB, HB, VB, num_warps, num_stages)
    candidates = [
        # Small tiles — minimal register pressure
        # (32, 32, 32, 4, 2),
        # (32, 32, 32, 8, 2),
        # (32, 64, 64, 4, 3),
        # (64, 32, 32, 4, 2),
        # (64, 32, 64, 4, 3),
        (64, 32, 64, 8, 2),
        # Medium tiles
        # (64, 64, 64, 4, 3),
        # (64, 64, 64, 8, 2),
        # (64, 64, 128, 8, 2),
        # (128, 32, 64, 8, 2),
        # (128, 32, 128, 8, 2),
        # (128, 64, 64, 8, 2),
        # (128, 64, 128, 4, 2),
        # (128, 64, 128, 8, 2),
        # Larger VB for big vocab — fewer vocab tiles, less atomic contention
        # (64, 64, 256, 8, 2),
        (128, 32, 256, 8, 2),  # best for 1024,8192
        (128, 32, 256, 4, 2),
        (128, 32, 256, 8, 1),
        (256, 32, 128, 8, 2),
        (256, 32, 128, 4, 2),
        (256, 32, 128, 8, 1),
    ]
    for fb, hb, vb, nw, ns in candidates:
        configs.append(
            triton.Config(
                {"FLATTENED_BATCH_BLOCK": fb, "HIDDEN_BLOCK": hb, "VOCAB_BLOCK": vb},
                num_warps=nw,
                num_stages=ns,
            )
        )
    return configs


@triton.autotune(
    configs=_fwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM"],
    cache_results=True,
)
@triton.jit
def _rnnt_joint_fused_vocab_fwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    target_logprobs_out_ptr,
    blank_logprobs_out_ptr,
    log_sum_exp_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    USE_HIGH_PRECISION_MM: tl.constexpr,
):
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            (weight_ptr.dtype.element_ty == tl.bfloat16 and joint_hidden_ptr.dtype.element_ty == tl.bfloat16)
            and (not USE_HIGH_PRECISION_MM)
        )
        else compute_dtype
    )

    flattened_batch_block_index = tl.program_id(axis=0)
    flattened_batch_start = flattened_batch_block_index * FLATTENED_BATCH_BLOCK
    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_valid_mask = flattened_batch_offsets < flattened_batch_size

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_valid_mask, other=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_valid_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_valid_mask & source_mask & target_label_mask

    vocab_offsets_in_block = tl.arange(0, VOCAB_BLOCK)

    log_sum_exp_score = tl.full([FLATTENED_BATCH_BLOCK], value=float("-inf"), dtype=compute_dtype)
    blank_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)
    target_logits = tl.zeros([FLATTENED_BATCH_BLOCK], dtype=compute_dtype)

    max_target_len = max_tgt_len_plus_1 - 1
    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=flattened_batch_valid_mask & target_label_mask,
        other=0,
    )

    # Create block pointers once before the loops
    NUM_HIDDEN_ITERS: tl.constexpr = tl.cdiv(hidden_dim, HIDDEN_BLOCK)
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    joint_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(flattened_batch_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(flattened_batch_start, 0),
        block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(0, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(0,),
        block_shape=(VOCAB_BLOCK,),
        order=(0,),
    )

    # Outer loop over vocab chunks
    for vocab_start in tl.range(0, vocab_size, VOCAB_BLOCK):
        vocab_offsets = vocab_start + vocab_offsets_in_block
        vocab_mask = vocab_offsets < vocab_size
        contains_blank = vocab_start <= blank_id and blank_id < vocab_start + VOCAB_BLOCK

        bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)

        block_logits = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]
        for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
            # hidden chunk: do not cache
            hidden_chunk = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1), cache_modifier=".cv").to(
                matmul_dtype
            )
            weight_chunk_t = tl.load(weight_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)

            block_logits += matmul(hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
                compute_dtype
            )

            joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, HIDDEN_BLOCK))
            weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

        # Reset hidden dim, advance vocab dim for next iteration
        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, -HIDDEN_RESET))
        weight_block_ptr = tl.advance(weight_block_ptr, (VOCAB_BLOCK, -HIDDEN_RESET))
        bias_block_ptr = tl.advance(bias_block_ptr, (VOCAB_BLOCK,))

        # Mask invalid vocab positions
        block_logits = tl.where(vocab_mask[None, :], block_logits, -float("inf"))

        # Online log-sum-exp
        block_logits_max = tl.max(block_logits, axis=-1)  # [TILE]
        block_lse = tl.log(tl.sum(tl.exp(block_logits - block_logits_max[:, None]), axis=-1)) + block_logits_max
        log_sum_exp_score = log_add_exp(log_sum_exp_score, block_lse)

        # Extract blank and target logits from this chunk
        target_logits += tl.sum(tl.where(vocab_offsets[None, :] == targets[:, None], block_logits, 0.0), axis=-1)
        if contains_blank:
            blank_logits += tl.sum(tl.where((vocab_offsets == blank_id)[None, :], block_logits, 0.0), axis=-1)

    tl.store(
        blank_logprobs_out_ptr + flattened_batch_offsets,
        blank_logits - log_sum_exp_score,
        mask=output_blank_mask,
    )
    tl.store(
        target_logprobs_out_ptr + flattened_batch_offsets,
        target_logits - log_sum_exp_score,
        mask=output_target_mask,
    )
    tl.store(
        log_sum_exp_out_ptr + flattened_batch_offsets,
        log_sum_exp_score,
        mask=output_blank_mask,
    )


@triton.autotune(
    configs=_bwd_autotune_configs(),
    key=["hidden_dim", "vocab_size", "USE_HIGH_PRECISION_MM"],
    reset_to_zero=["grad_joint_hidden_out_ptr", "grad_weight_out_ptr", "grad_bias_out_ptr"],
    cache_results=True,
)
@triton.jit
def _rnnt_joint_fused_vocab_bwd_triton_kernel(
    joint_hidden_ptr,
    targets_ptr,
    src_lengths_ptr,
    tgt_lengths_ptr,
    weight_ptr,
    bias_ptr,
    log_sum_exp_ptr,
    grad_target_scores_ptr,
    grad_blank_scores_ptr,
    grad_joint_hidden_out_ptr,
    grad_weight_out_ptr,
    grad_bias_out_ptr,
    max_src_len: int,
    max_tgt_len_plus_1: int,
    flattened_batch_size: int,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    blank_id: tl.constexpr,
    FLATTENED_BATCH_BLOCK: tl.constexpr,
    HIDDEN_BLOCK: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr = 64,
    USE_HIGH_PRECISION_MM: tl.constexpr = False,
):
    compute_dtype = tl.float32
    matmul_dtype = (
        tl.bfloat16
        if (
            (weight_ptr.dtype.element_ty == tl.bfloat16 and joint_hidden_ptr.dtype.element_ty == tl.bfloat16)
            and (not USE_HIGH_PRECISION_MM)
        )
        else compute_dtype
    )

    # flattened_batch_index = tl.program_id(axis=0)
    # vocab_block_index = tl.program_id(axis=1)
    flattened_batch_index, vocab_block_index = tl.swizzle2d(
        tl.program_id(axis=0), tl.program_id(axis=1), tl.num_programs(axis=0), tl.num_programs(axis=1), GROUP_SIZE
    )
    vocab_block_start = vocab_block_index * VOCAB_BLOCK
    flattened_batch_start = flattened_batch_index * FLATTENED_BATCH_BLOCK

    vocab_offsets = vocab_block_start + tl.arange(0, VOCAB_BLOCK)
    vocab_mask = vocab_offsets < vocab_size
    is_blank_vocab_col = (vocab_offsets == blank_id) & vocab_mask

    # Create block pointers once before the loops
    NUM_HIDDEN_ITERS: tl.constexpr = tl.cdiv(hidden_dim, HIDDEN_BLOCK)
    HIDDEN_RESET: tl.constexpr = NUM_HIDDEN_ITERS * HIDDEN_BLOCK

    joint_hidden_block_ptr = tl.make_block_ptr(
        base=joint_hidden_ptr,
        shape=(flattened_batch_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(flattened_batch_start, 0),
        block_shape=(FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(vocab_size, hidden_dim),
        strides=(hidden_dim, 1),
        offsets=(vocab_block_start, 0),
        block_shape=(VOCAB_BLOCK, HIDDEN_BLOCK),
        order=(1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(vocab_size,),
        strides=(1,),
        offsets=(vocab_block_start,),
        block_shape=(VOCAB_BLOCK,),
        order=(0,),
    )

    source_target_block_size = max_src_len * max_tgt_len_plus_1
    max_target_len = max_tgt_len_plus_1 - 1
    bias_chunk = tl.load(bias_block_ptr, boundary_check=(0,)).to(compute_dtype)

    flattened_batch_offsets = flattened_batch_start + tl.arange(0, FLATTENED_BATCH_BLOCK)
    flattened_batch_mask = flattened_batch_offsets < flattened_batch_size

    batch_index = flattened_batch_offsets // source_target_block_size
    batch_offsets = flattened_batch_offsets - batch_index * source_target_block_size
    source_index = batch_offsets // max_tgt_len_plus_1
    target_index = batch_offsets - source_index * max_tgt_len_plus_1

    source_length = tl.load(src_lengths_ptr + batch_index, mask=flattened_batch_mask, other=0)
    target_length = tl.load(tgt_lengths_ptr + batch_index, mask=flattened_batch_mask, other=0)

    source_mask = source_index < source_length
    target_valid_mask = target_index <= target_length
    target_label_mask = target_index < target_length
    output_blank_mask = flattened_batch_mask & source_mask & target_valid_mask
    output_target_mask = flattened_batch_mask & source_mask & target_label_mask

    targets = tl.load(
        targets_ptr + batch_index * max_target_len + target_index,
        mask=output_target_mask,
        other=0,
    )

    lse = tl.load(log_sum_exp_ptr + flattened_batch_offsets, mask=output_blank_mask, other=0.0).to(compute_dtype)
    grad_target = tl.load(
        grad_target_scores_ptr + flattened_batch_offsets,
        mask=output_target_mask,
        other=0.0,
    ).to(compute_dtype)
    grad_blank = tl.load(
        grad_blank_scores_ptr + flattened_batch_offsets,
        mask=output_blank_mask,
        other=0.0,
    ).to(compute_dtype)
    sum_grad = grad_target + grad_blank

    flattened_batch_flat_indices = flattened_batch_offsets.to(tl.int64)
    logits_block = tl.zeros([FLATTENED_BATCH_BLOCK, VOCAB_BLOCK], dtype=compute_dtype) + bias_chunk[None, :]

    # Inner loop 1: recompute logits
    for _ in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        # hidden_chunk: [FLATTENED_BATCH_BLOCK, HIDDEN_BLOCK]
        hidden_chunk = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)
        # weight_chunk: [VOCAB_BLOCK, HIDDEN_BLOCK]
        weight_chunk_t = tl.load(weight_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)

        logits_block += matmul(hidden_chunk, weight_chunk_t, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            compute_dtype
        )

        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, HIDDEN_BLOCK))
        weight_block_ptr = tl.advance(weight_block_ptr, (0, HIDDEN_BLOCK))

    # Compute grad_logits
    probabilities_block = tl.exp(logits_block - lse[:, None])  # [FLATTENED_BATCH_BLOCK, VOCAB_BLOCK]
    grad_logits_block = (
        -(sum_grad[:, None] * probabilities_block)
        + (grad_blank[:, None] * is_blank_vocab_col[None, :])
        + (grad_target[:, None] * (vocab_offsets[None, :] == targets[:, None]))
    )
    grad_logits_block = tl.where(output_blank_mask[:, None] & vocab_mask[None, :], grad_logits_block, 0.0)
    grad_bias_acc = tl.sum(grad_logits_block, axis=0)

    # Inner loop 2: compute grad_hidden and grad_weight in reverse iteration for cache reuse
    grad_logits_matmul = grad_logits_block.to(matmul_dtype)
    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        weight_block_ptr = tl.advance(weight_block_ptr, (0, -HIDDEN_BLOCK))
        weight_chunk = tl.load(weight_block_ptr, boundary_check=(0, 1)).to(matmul_dtype)

        grad_hidden_delta = matmul(grad_logits_matmul, weight_chunk, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM).to(
            grad_joint_hidden_out_ptr.dtype.element_ty
        )

        # need atomic: different vocab parts write to the same place
        tl.atomic_add(
            grad_joint_hidden_out_ptr + flattened_batch_flat_indices[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_hidden_delta,
            mask=output_blank_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

    for r_hidden_idx in tl.range(0, hidden_dim, HIDDEN_BLOCK):
        hidden_start = HIDDEN_RESET - HIDDEN_BLOCK - r_hidden_idx
        hidden_offsets = hidden_start + tl.arange(0, HIDDEN_BLOCK)
        hidden_mask = hidden_offsets < hidden_dim

        joint_hidden_block_ptr = tl.advance(joint_hidden_block_ptr, (0, -HIDDEN_BLOCK))
        joint_hidden_chunk_t = tl.load(joint_hidden_block_ptr, boundary_check=(0, 1)).T.to(matmul_dtype)

        grad_weight_delta = matmul(
            joint_hidden_chunk_t, grad_logits_matmul, USE_HIGH_PRECISION=USE_HIGH_PRECISION_MM
        ).T.to(grad_joint_hidden_out_ptr.dtype.element_ty)

        # need atomic: different batch parts write to the same place
        tl.atomic_add(
            grad_weight_out_ptr + vocab_offsets[:, None] * hidden_dim + hidden_offsets[None, :],
            grad_weight_delta,
            mask=vocab_mask[:, None] & hidden_mask[None, :],
            sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
        )

    # Atomic add into global grads; different batch parts write to the same place
    tl.atomic_add(
        grad_bias_out_ptr + vocab_offsets,
        grad_bias_acc,
        mask=vocab_mask,
        sem="relaxed",  # no need to guarantee order of adding - no read inside kernel
    )


class _RnntJointFusedVocabTritonFunction(torch.autograd.Function):
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
        use_high_precision_mm: bool = False,
        input_grad_full_precision: bool = False,
        weight_grad_full_precision: bool = False,
    ):
        float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        joint_hidden = joint_hidden.contiguous()
        targets = targets.contiguous()
        src_lengths = src_lengths.contiguous()
        tgt_lengths = tgt_lengths.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()

        target_logprobs = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1], dtype=float_dtype, device=device
        )
        blank_logprobs = torch.zeros_like(target_logprobs)
        log_sum_exp_scores = torch.empty_like(target_logprobs)

        fwd_grid = lambda META: (triton.cdiv(flattened_batch_size, META["FLATTENED_BATCH_BLOCK"]),)

        _rnnt_joint_fused_vocab_fwd_triton_kernel[fwd_grid](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            target_logprobs_out_ptr=target_logprobs,
            blank_logprobs_out_ptr=blank_logprobs,
            log_sum_exp_out_ptr=log_sum_exp_scores,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
        )

        ctx.save_for_backward(joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores)
        ctx.blank_id = blank_id
        ctx.use_high_precision_mm = use_high_precision_mm
        ctx.input_grad_full_precision = input_grad_full_precision
        ctx.weight_grad_full_precision = weight_grad_full_precision
        return target_logprobs, blank_logprobs

    @staticmethod
    def backward(ctx, grad_target_scores, grad_blank_scores):
        (joint_hidden, weight, bias, targets, src_lengths, tgt_lengths, log_sum_exp_scores) = ctx.saved_tensors
        blank_id = ctx.blank_id
        use_high_precision_mm = ctx.use_high_precision_mm
        input_grad_full_precision = ctx.input_grad_full_precision
        if input_grad_full_precision is None:
            input_grad_full_precision = use_high_precision_mm
        weight_grad_full_precision = ctx.weight_grad_full_precision
        if weight_grad_full_precision is None:
            weight_grad_full_precision = use_high_precision_mm
        float_dtype = torch.float64 if joint_hidden.dtype == torch.float64 else torch.float32

        grad_target_scores = grad_target_scores.contiguous()
        grad_blank_scores = grad_blank_scores.contiguous()

        batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim = joint_hidden.shape
        flattened_batch_size = batch_size * src_max_length * tgt_max_length_plus_1
        vocab_size = weight.shape[0]
        device = joint_hidden.device

        grad_joint_hidden = torch.zeros(
            [batch_size, src_max_length, tgt_max_length_plus_1, hidden_dim],
            dtype=float_dtype if input_grad_full_precision else joint_hidden.dtype,
            device=device,
        )

        grad_weight = torch.zeros(
            [vocab_size, hidden_dim], dtype=float_dtype if weight_grad_full_precision else weight.dtype, device=device
        )
        grad_bias = torch.zeros(
            [vocab_size], dtype=float_dtype if weight_grad_full_precision else bias.dtype, device=device
        )

        bwd_grid = lambda META: (
            triton.cdiv(flattened_batch_size, META["FLATTENED_BATCH_BLOCK"]),
            triton.cdiv(vocab_size, META["VOCAB_BLOCK"]),
        )

        _rnnt_joint_fused_vocab_bwd_triton_kernel[bwd_grid](
            joint_hidden_ptr=joint_hidden,
            targets_ptr=targets,
            src_lengths_ptr=src_lengths,
            tgt_lengths_ptr=tgt_lengths,
            weight_ptr=weight,
            bias_ptr=bias,
            log_sum_exp_ptr=log_sum_exp_scores,
            grad_target_scores_ptr=grad_target_scores,
            grad_blank_scores_ptr=grad_blank_scores,
            grad_joint_hidden_out_ptr=grad_joint_hidden,
            grad_weight_out_ptr=grad_weight,
            grad_bias_out_ptr=grad_bias,
            max_src_len=src_max_length,
            max_tgt_len_plus_1=tgt_max_length_plus_1,
            flattened_batch_size=flattened_batch_size,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            blank_id=blank_id,
            USE_HIGH_PRECISION_MM=use_high_precision_mm,
        )

        # convert grad to desired dtype
        grad_weight = grad_weight.to(weight.dtype)
        grad_bias = grad_bias.to(bias.dtype)
        grad_joint_hidden = grad_joint_hidden.to(joint_hidden.dtype)

        return grad_joint_hidden, None, None, None, grad_weight, grad_bias, None, None, None, None


def rnnt_joint_fused_vocab_triton(
    joint_hidden: torch.Tensor,
    targets: torch.Tensor,
    tgt_lengths: torch.Tensor,
    src_lengths: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    blank_id: int,
    use_high_precision_mm: bool = False,
    input_grad_full_precision: bool = False,
    weight_grad_full_precision: bool = False,
):
    target_logprobs, blank_logprobs = _RnntJointFusedVocabTritonFunction.apply(
        joint_hidden,
        targets,
        tgt_lengths,
        src_lengths,
        weight,
        bias,
        blank_id,
        use_high_precision_mm,
        input_grad_full_precision,
        weight_grad_full_precision,
    )
    return target_logprobs, blank_logprobs
