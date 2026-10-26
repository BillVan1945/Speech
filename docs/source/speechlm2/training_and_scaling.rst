Training and Scaling
====================

This page provides detailed information on training speechlm2 models, including setup requirements, running experiments at scale, debugging, and parallelism strategies.

For reward-based transcription fine-tuning, see :ref:`speechlm-gspo`.

Running Experiments
-------------------

The speechlm2 collection includes several scripts to facilitate running experiments, especially on SLURM-based clusters.

SLURM Job Submission
^^^^^^^^^^^^^^^^^^^^

For training on SLURM clusters, use the following workflow:

.. code-block:: bash

    # Submit 8 consecutive jobs with random seeds
    scripts/speechlm2/auto_launcher_with_seed.sh -n8 s2s_tinyllama_repro.sub

The ``auto_launcher_with_seed.sh`` script:

1. Generates a random seed for each submitted job
2. Leverages ``shard_seed="randomized"`` in Lhotse to ensure each data parallel rank is seeded differently
3. Ensures each tensor parallel rank is seeded identically

SLURM Submission Script
^^^^^^^^^^^^^^^^^^^^^^^

Example ``s2s_tinyllama_repro.sub`` script:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=s2s_training
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=8
    #SBATCH --gres=gpu:8
    #SBATCH --time=24:00:00
    #SBATCH --exclusive
    #SBATCH --output=s2s_tinyllama_repro_%j.out

    # Check that the global random seed base is provided
    if [ -z "$1" ]; then
      echo "Usage: $0 <global_random_seed_base>"
      exit 1
    fi
    SEED=${1}

    EXP_NAME="s2s_training"
    RESULTS_DIR="results/${EXP_NAME}"

    srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
      python -u examples/speechlm2/s2s_duplex_train.py \
      --config-path=/path/to/config/dir \
      --config-name=s2s_training.yaml \
      exp_manager.name=${EXP_NAME} \
      exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
      trainer.num_nodes=$SLURM_JOB_NUM_NODES \
      exp_manager.explicit_log_dir=${RESULTS_DIR} \
      data.train_ds.seed=$SEED \
      data.validation_ds.seed=$SEED


Configuration Files
^^^^^^^^^^^^^^^^^^^

The main configuration file (``s2s_training.yaml``) contains all model, training, and data parameters. See :doc:`configs` for more details. It's recommended to copy and modify this file rather than overriding options in the SLURM script to maintain versioning and configuration clarity.

Short-Recording SpecAugment
---------------------------

SpeechLM perception modules automatically cap aggregate time masking at 10% of
valid frames for recordings shorter than one second. The threshold uses the
frontend's frame duration. This policy supports dense and packed PyTorch
SpecAugment; long recordings retain their existing masks and RNG behavior.

The Numba backend does not support this cap. Enabling the policy overrides
``use_numba_spec_augment: true`` with PyTorch SpecAugment and logs a warning,
including when restoring a checkpoint. The vectorized PyTorch implementation
is independent of Numba and remains available through
``use_vectorized_spec_augment: true``. Set ``use_numba_spec_augment: false``
explicitly in the model's ``spec_augment`` configuration to select this backend
without the override warning.

For a restored Numba experiment, the backend switch may change RNG behavior,
outputs and throughput, including for batches containing only long recordings.
The long-recording preservation guarantee applies relative to the selected
PyTorch implementation. Outside SpeechLM, enabling the cap through
``SpectrogramAugmentation.configure_short_recording_time_mask_cap`` has the
same behavior; Numba augmentation without the cap remains supported.

Zero-Weight Dataset Sources
---------------------------

Zero-weight sources remain disabled at every reweighting temperature, including
zero. Infinite training mixtures retain disabled source slots for restored
packing-buffer origins. Finite mixtures (``force_finite: true`` or
``metadata_only: true``) omit disabled slots and stop when all positive-weight
sources are exhausted. Retain the infinite mixture's source layout when
resuming its saved data state.

.. _speechlm-lr-only-continuation:

Reducing LR After a Full-State Resume
-------------------------------------

``ScaleRestoredLearningRate`` applies a one-time reduction to a restored NeMo
``CosineAnnealing`` schedule. It preserves optimizer moments, scheduler position,
model weights and the global step. Use it when continuing a checkpoint with a
smaller learning rate while retaining the original cosine schedule horizon.
It requires exactly one optimizer and one ``CosineAnnealing`` scheduler; all
restored parameter groups must have the specified original base LR and the
scheduler must have the specified original minimum LR.

Add the callback to the training YAML and resume the **full** checkpoint:

.. code-block:: yaml

    model:
      init_from_checkpoint: null

    trainer:
      callbacks:
        - _target_: nemo.collections.speechlm2.parts.lr_only_resume.ScaleRestoredLearningRate
          factor: 0.75
          source_step: 16500
          original_base_lr: 0.0001
          original_min_lr: 0.00001

    exp_manager:
      resume_if_exists: true
      resume_from_checkpoint: /path/to/source/checkpoints/step=16500.ckpt

Keep the existing model, optimizer, scheduler and other callback configuration
from the source experiment. The example's rates must match the source scheduler's
``base_lrs`` and ``min_lr`` exactly; its current decayed LR is scaled by the same
factor. ``factor`` must be greater than zero and less than one. The SpeechLM
training entrypoint resolves ``trainer.callbacks`` from this configuration.
``model.init_from_checkpoint`` loads weights only and cannot supply this callback's
required optimizer, scheduler and global-step state.

The first resume must use the canonical source path ending in
``/step=<source_step>.ckpt``. For this example, the required basename is
``step=16500.ckpt`` and the restored global step must be 16500. Both a single-file
checkpoint and a distributed checkpoint directory can use this basename. Pass
an absolute path to that checkpoint; ``step=16500-last.ckpt``, ``last.ckpt`` and
other aliases do not satisfy the callback's source check.

After applying the reduction, advance training by at least one optimizer step
before saving a continuation checkpoint. For every later resume, **retain the
same callback configuration and all four original arguments**. Change only
``exp_manager.resume_from_checkpoint`` to the new continuation checkpoint, for
example ``/path/to/continuation/checkpoints/step=17000.ckpt``. Do not change
``source_step`` to 17000 or apply ``factor`` again. Lightning restores the
callback's checkpointed ``applied`` state, keyed by its factor and original
source step, so it skips rescaling and continues the already reduced schedule.
Removing the callback drops that restore contract; changing its configuration
can prevent its saved state from being matched.

The callback does not migrate dataloader state or checkpoint formats. Retain the
source experiment's sampler, packing-buffer and data configuration when exact
data continuation is required.

Debugging
---------

Running Locally with torchrun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local debugging and profiling, use ``torchrun``:

.. code-block:: bash

    # Run with 4 GPUs locally
    torchrun --nproc_per_node=4 examples/speechlm2/s2s_duplex_train.py \
      --config-path=/path/to/config/dir \
      --config-name=s2s_training.yaml

Scaling Strategies
------------------

The speechlm2 collection includes support for model parallelism to scale training to large models across multiple GPUs.

Model Parallel Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

The collection supports multiple parallelism strategies:

1. **Fully Sharded Data Parallel (FSDP2)**: Distributes model parameters across GPUs
2. **Tensor Parallelism (TP)**: Splits individual tensors across GPUs
3. **Sequence Parallelism (SP)**: Splits sequence processing across GPUs
4. **2D Parallelism**: Combination of FSDP2 with TP/SP

AutomodelParallelStrategy (SALMAutomodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``SALMAutomodel``, the collection provides ``AutomodelParallelStrategy`` which
delegates topology resolution and parallelism to NeMo Automodel. The strategy can
construct topologies for FSDP2, TP, PP, CP, EP (MoE), and HSDP, but actual
support depends on the selected LLM architecture, backend, and combination of
parallelism axes.

.. code-block:: yaml

    trainer:
      strategy:
        _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
        dp_size: null       # inferred from world_size / other dims
        dp_replicate_size: 1  # HSDP replication group size
        tp_size: 1
        pp_size: 1
        cp_size: 1
        ep_size: 8          # Expert parallelism for MoE models

        # Activation checkpointing — two independent knobs:
        activation_checkpointing_llm: false         # LLM transformer blocks
        activation_checkpointing_perception: false  # speech encoder layers

The model's ``configure_model()`` receives a resolved ``distributed_setup`` and
passes it to Automodel's ``from_pretrained`` for memory-efficient loading
(each GPU only loads its own shard). ``SALMAutomodel`` currently rejects
``pp_size > 1``; use pipeline parallelism only after model integration adds
pipeline execution support.

The speech encoder / perception module currently only supports FSDP2 (controlled via ``dp_size``).

Activation Checkpointing
""""""""""""""""""""""""

``AutomodelParallelStrategy`` exposes two independent activation-checkpointing knobs:

* ``activation_checkpointing_llm`` — single switch covering both the non-EP FSDP2
  path (forces ``FSDP2Config.activation_checkpointing=True``) and the EP/MoE
  parallelizer path (passed through as a separate runtime arg). Use this for
  MoE LLMs whether ``ep_size`` is 1 or larger.
* ``activation_checkpointing_perception`` — wraps each transformer layer in
  ``perception.encoder.layers`` (and the Conformer ``pre_encode`` front-end when
  it isn't a bare ``nn.Linear``) with ``checkpoint_wrapper`` *before* FSDP2
  sharding. Implemented in ``AudioPerceptionModule.set_activation_checkpointing``.

Both default to ``false``. Toggle them independently to trade compute for
memory at either end of the model. They are SALMAutomodel-specific knobs (the
HF Transformers SALM path uses HuggingFace's own gradient-checkpointing API).

.. note::
   Expert Parallelism (EP) reuses the FSDP2 data-parallel axis (``dp_size``).
   Dense layers are sharded via FSDP2, while MoE expert layers use EP for
   all-to-all expert routing — both operate on the same set of GPUs.
   Setting ``ep_size`` controls how many GPUs participate in expert routing;
   it does not add a separate dimension.

Training with MoE LLM Backbones
""""""""""""""""""""""""""""""""

SALMAutomodel enables efficient training of Speech LLMs with Mixture-of-Experts
backbones like `NVIDIA Nemotron Nano V3 <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>`_
(30B total parameters, 3B active). NeMo Automodel provides two key MoE
optimizations:

* **Grouped GEMM**: Fuses all expert computations within a single MoE layer into
  one batched matrix multiplication, maximizing GPU utilization and throughput.
* **DeepEP** (Deep Expert Parallelism): An efficient all-to-all communication
  primitive for routing tokens to experts across GPUs, significantly reducing the
  communication overhead of Expert Parallelism.

.. important::
   For the native Nemotron-V3 backbone, the validated ``SALMAutomodel``
   parallelism axes are FSDP2, EP, and CP. CP requires packed sequences and
   Transformer Engine attention, as described in the Context Parallelism
   section below.

   Nemotron-V3 does not provide a TP plan, so set ``tp_size: 1``. HSDP is not
   supported in this path, so set ``dp_replicate_size: 1``. PP has not been
   validated for this SpeechLM2 configuration, so set ``pp_size: 1``.

Example: training SALMAutomodel with Nemotron Nano V3 on 8 GPUs with EP=8:

.. code-block:: bash

    torchrun --nproc_per_node=8 examples/speechlm2/salm_train.py \
      --config-name=salm_automodel \
      model.pretrained_llm=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
      trainer.strategy.ep_size=8

For distributed inference, launch with ``torchrun``:

.. code-block:: bash

    torchrun --nproc_per_node=8 examples/speechlm2/salm_eval.py \
      pretrained_name=path/to/checkpoint \
      inputs=path/to/manifest \
      ep_size=2

Native NeMo-RL checkpoints
""""""""""""""""""""""""""

NeMo-RL saves distributed checkpoints (DCP) for resuming training. To turn their
model weights into a Hugging Face checkpoint, use Speech's ``to_hf.py`` with
``checkpoint_format=nemo_rl``. See :ref:`speechlm-gspo-export` for commands.
Loading requires every parameter and persistent buffer, including frozen weights
and any multi-token prediction (MTP) head. Checkpoint-wrapper names are mapped
back to model names; empty Transformer Engine bookkeeping entries are omitted.

Training integrations can choose which parameters to update through
``SALMAutomodel``'s ``automodel_freeze_config`` (language model) and
``perception_freeze_config`` (audio encoder and projection). Both use Automodel's
``FreezeConfig`` with names relative to that component. For example, this updates
only the perception projection weight, leaving its bias and encoder frozen:

.. code-block:: yaml

    model:
      perception_freeze_config:
        freeze_modules: [{glob: '*'}]
        unfreeze_parameters: [proj.weight]

Set ``model.torch_dtype: float32`` for FP32 trainable storage. FSDP2 controls
compute and reduction precision separately and casts ordinary frozen weights
before sharding; ``perception_mixed_precision`` supplies the audio-side policy.
``retain_native_mtp: true`` keeps the checkpoint's existing MTP topology while
``mtp.enabled: false`` disables its auxiliary training. A seed without an MTP
head stays without one. These settings are opt-in.

For rollout generation, vLLM first loads the complete perception state. Later
weight updates may contain only the trained tensors: names and shapes are
validated before loading, and omitted frozen tensors keep their seed values.

Packed Sequences (THD)
""""""""""""""""""""""

``SALMAutomodel`` supports an opt-in packed-sequence (``THD``) training and
validation path that concatenates per-utterance text + audio embeddings into
a single flat ``[T_total, H]`` sequence with a ``cu_seqlens`` index, instead
of right-padding into the standard ``[B, T_max, H]`` (``BSHD``) layout. TE's
varlen FlashAttention then operates segment-by-segment without ever attending
across utterances, and Mamba's ``seq_idx`` is derived from the same
``cu_seqlens`` so SSM state resets at document boundaries.

For variable-length speech batches the padding overhead is substantial — the
``BSHD`` layout pays ``B * (T_max - T_avg)`` wasted compute per minibatch,
``THD`` pays only the per-utterance rounding to a multiple of ``2*cp_size``
(needed for TE's CP DualChunkSwap pattern). Throughput improvement scales
with the variance of utterance lengths.

Enable per-batch:

.. code-block:: yaml

    model:
      packed_sequences: true   # opt-in; default false (BSHD)
      automodel_backend:
        attn: te                # THD path requires TE attention

When ``packed_sequences`` is unset, the existing BSHD path is used unchanged.
Generate / inference always uses BSHD (it doesn't go through ``prepare_inputs``).

Context Parallelism (CP)
""""""""""""""""""""""""

``SALMAutomodel`` supports context parallelism for long-audio training on
hybrid Mamba/attention LLMs (e.g. Nemotron-V3). CP shards the sequence
dimension across GPUs so per-rank activations and KV-cache memory scale as
``T / cp_size`` instead of ``T``; attention layers go through TE's
DualChunkSwap pattern and Mamba mixers go through hidden-parallel
all-to-all (``MambaContextParallel`` in NeMo Automodel).

Enable via the strategy:

.. code-block:: yaml

    trainer:
      strategy:
        _target_: nemo.collections.speechlm2.parts.parallel.AutomodelParallelStrategy
        cp_size: 2          # context parallel size; must divide num_heads of every Mamba block
        ep_size: 2          # may share the same ranks as CP

**The THD packed-sequence path is the only supported configuration under
CP.** Each utterance is its own attention segment and the per-utterance
sequence rounding aligns naturally with CP's ``2*cp_size`` requirement.

.. warning::
   **BSHD + CP is not supported.** TE's fused-attention CP path supports
   ``causal`` but not ``padding_causal``, so the right-pad mask must be
   dropped before the LLM. With the mask dropped, pad K/V leak into
   real-token attention through the causal mask and the gradient through
   the LoRA / projection parameters becomes ``NaN`` after the first
   optimizer step (validated empirically: BSHD + CP=2 + EP=2 on a 2-GPU
   run produces ``loss=4.62`` at step 1 then ``loss=nan`` from step 2
   onwards). This is independent of the TE/cuDNN backward issue
   documented below — setting ``NVTE_FUSED_ATTN=0`` does not fix it.
   Set ``model.packed_sequences: true`` to use the THD path instead.

.. note::
   **CP-safe data loading is automatic.** The speechlm2 datamodule wraps
   the Lhotse loader in
   :class:`~nemo.collections.common.data.lhotse.broadcasting.BroadcastingDataLoader`,
   so under CP/TP every batch is constructed once on the DP source rank
   (``cp_rank == 0`` and ``tp_rank == 0``) and broadcast to its sub-mesh
   peers. This eliminates per-rank Lhotse non-determinism (``concurrent_bucketing``,
   worker scheduling jitter, etc.) as a source of NCCL deadlocks under CP.
   See :doc:`/dataloaders` for the standalone API.

.. note::
   **TE/THD exploding-gradients workaround on some GPUs.** On certain GPU
   architectures (notably Blackwell ``sm_120``), the cuDNN backend that
   TransformerEngine 2.14 picks for ``qkv_format="thd"`` with
   ``attn_mask_type="padding_causal"`` returns correct forward activations
   but gradients amplified 8×–960× per layer. Compounded across the LLM's
   attention stack this drives gradients to ``1e22``-magnitudes at step 0,
   the gradient-clip-by-norm computes ``1.0 / inf = 0``, and Adam's moments
   eventually NaN. Force TE to dispatch FlashAttention instead of cuDNN by
   setting ``NVTE_FUSED_ATTN=0`` in the launcher environment (requires
   ``flash-attn`` to be installed for your GPU arch). The FlashAttention
   THD/``padding_causal`` backward is gradient-correct on the same shapes.

To configure parallelism, modify the ``trainer.strategy`` section in your YAML config:

.. code-block:: yaml

    trainer:
      strategy:
        _target_: nemo.core.ModelParallelStrategy
        find_unused_parameters: False
        data_parallel: 1   # World size for data parallelism (FSDP2)
        tensor_parallel: 8  # World size for tensor parallelism
      devices: 8
      num_nodes: 1
      accelerator: gpu
      precision: bf16-true

The model's ``configure_model`` method automatically sets up the appropriate parallelization based on this configuration.

FSDP2 Configuration (HF Automodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Fully Sharded Data Parallel training:

1. Set ``data_parallel`` to the number of GPUs you want to use for data parallelism
2. Set ``tensor_parallel`` to 1 (disabled)

FSDP2 shards the model parameters across GPUs, all-gathers them for forward/backward passes, and then de-allocates after computation. This allows training of larger models with limited GPU memory.
See `PyTorch FSDP2 <https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html>`_ for more details.

Tensor Parallelism Configuration (HF Automodel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Tensor Parallelism:

1. Set ``tensor_parallel`` to the number of GPUs you want to use for tensor parallelism
2. Set ``data_parallel`` to 1 (or higher for 2D parallelism)

The ``parallelize_module`` function applies a parallelization plan to specific model components, like splitting attention heads or embedding dimensions across GPUs.
See `PyTorch TP <https://pytorch.org/docs/stable/distributed.tensor.parallel.html>`_ for more details.

Implementation Details
----------------------

The core implementation of model parallelism is in the ``configure_model`` method of the model classes. Key aspects include:

1. **Module Sharding**: Calling ``fully_shard`` on modules to distribute parameters across data parallel ranks
2. **Parallelization Plans**: Creating and applying plans that specify how different layers should be parallelized
3. **Model-Specific Adaptations**: Handling architectural differences between different LLMs

Advanced Usage
--------------

Script Customization
^^^^^^^^^^^^^^^^^^^^

When customizing the training scripts, keep these points in mind:

1. **Path Overrides**: Override paths in the YAML configuration files with your own, as needed
2. **W&B Keys**: Update Weights & Biases API keys in configuration files
3. **Batch Size Tuning**: Adjust batch size based on your GPU memory and model size

.. _speechlm-gspo:

GSPO post-training
------------------

Fine-tune a SpeechLM to favor more accurate transcriptions with **Group Sequence
Policy Optimization (GSPO)**. For each recording, the recipe samples several
transcripts and scores them against a reference using word or character error
rate. The model learns from the differences. You provide transcribed audio;
the recipe computes rewards without a judging model or human-ranked responses.

The workflow starts with a Hugging Face model, trains in NeMo-RL, and exports
back to Hugging Face for serving. It supports
`Nemotron-3.5-Transcribe-30B-A4B <https://huggingface.co/nvidia/Nemotron-3.5-Transcribe-30B-A4B>`_
and compatible SpeechLM seeds with or without a CTC timestamp head.

Prepare the model and data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep a full local copy of the starting model: weights, ``config.json``,
``llm_backbone/`` and tokenizer. Training and export both need this seed.

Prepare a Parquet file with one recording per row: an audio path, reference
transcript, user instruction, language, task and reward metric. Use 16 kHz audio.
Each row keeps its own instruction, so ordinary transcription
and speaker-attributed transcription can use different prompts. The reference
is used for scoring, never included in the model's prompt.

Use ``wer`` for word-error rewards, or the recipe's ``cer`` route for Mandarin
ASR. The reward is ``1 - error_rate``; lower transcription error earns a higher
reward. Groups whose responses all receive the same reward are skipped because
they provide no within-group learning signal.

Train with NeMo-RL
^^^^^^^^^^^^^^^^^^

Follow the `SpeechLM GSPO guide <https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/gspo-speechlm.md>`_
for compatible package versions, the exact Parquet schema, launch commands and
resume instructions. The example runs on four nodes with eight GPUs each for
up to 400 updates.

The recipe explicitly selects which weights to train; the remaining weights,
including the MTP and optional CTC heads, stay frozen. Change that selection in
``policy.native_model.trainable_names`` in the NeMo-RL configuration, keeping
``expected_trainable_tensors`` and ``expected_trainable_scalars`` consistent with it.
For a new dataset on the same architecture, keep the tested selection.

Training saves resumable DCP checkpoints. Export a saved step to evaluate or
serve it; retain the DCP checkpoint to continue training.

.. _speechlm-gspo-export:

Export a trained checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the Speech checkout, use the policy environment prepared in the GSPO guide.
First build an export configuration from the **same HF seed used for training**:

.. code-block:: bash

    SEED=/shared/models/Nemotron-3.5-Transcribe-30B-A4B
    RUN=/shared/results/gspo
    EXPORT_CONFIG=/shared/results/gspo-export-config.json
    HF_OUTPUT=/shared/results/gspo-step400-hf

    python examples/speechlm2/prepare_rl_export_config.py \
      --seed "$SEED" --output "$EXPORT_CONFIG"

Choose an unused ``EXPORT_CONFIG`` path; the helper refuses to overwrite an
existing file. It reconstructs the Speech model and strategy configuration,
including CTC metadata when present. This file is the converter's
``ckpt_config``, separate from the seed's HF ``config.json``.

Then convert the selected step in an eight-GPU allocation:

.. code-block:: bash

    torchrun --standalone --nproc-per-node=8 examples/speechlm2/to_hf.py \
      class_path=nemo.collections.speechlm2.models.SALMAutomodel \
      checkpoint_format=nemo_rl \
      ckpt_path="$RUN/checkpoints/step_400/policy/weights/model" \
      ckpt_config="$EXPORT_CONFIG" output_dir="$HF_OUTPUT"

``HF_OUTPUT`` contains the trained weights, model and backbone configurations,
and tokenizer files. Load it with :doc:`vllm_serving`. The exporter retains the
seed's MTP topology and preserves the CTC artifact's storage and metadata. Other
weights follow the model's serving dtype rules: FP32 training storage does not
mean every weight is exported in FP32.

Compare the exported model with the seed using the same evaluation data, prompts
and decoding settings. If you use CTC timestamps, evaluate them too: a frozen
CTC head still receives features from an audio encoder that training can change.
