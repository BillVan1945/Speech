# Initializing the phoneme frontend from Qwen3-0.6B

EasyMagpie can initialize its text-to-phoneme frontend with the decoder and text
embeddings from `Qwen/Qwen3-0.6B`. This uses Qwen's architecture, not a conversion
of Qwen weights into Nemotron-H. Training still predicts normalized IPA tokens
through the existing parallel phoneme channel and phoneme loss. Both fixed and
dynamic phoneme delay use the existing training/inference paths.
Raw input uses Qwen BPE directly, without chat templates or reasoning output.
Nemotron-H-specific architecture overrides are unused with this backend.

Use `--config-name=easy_magpietts_lhotse_qwen3_0_6b` with your text-only data, IPA
tokenizer and codec artifact. The config inherits the multiturn Lhotse layout
used by the frontend recipes. Do not load an existing Nemotron-H frontend
checkpoint on top: its architecture and input vocabulary are different.

To adapt an existing config instead, apply these overrides:

```bash
model.decoder_type=huggingface \
model.transformer_hf_backend=Qwen/Qwen3-0.6B \
++model.transformer_hf_load_pretrained_weights=true \
++model.transformer_hf_attn_implementation=sdpa \
model.embedding_dim=1024 \
model.hidden_dim=1024 \
model.audio_embedding_dim=1024 \
model.use_bpe_char_tokenizer=false \
model.disable_subword_embedding=false \
model.disable_lm_text_head=true \
model.local_transformer_type=none \
++model.local_transformer_hidden_dim=1024 \
++model.local_transformer_n_heads=8 \
++model.speaker_encoder_n_heads=8 \
model.text_tokenizers.nemotron_nano_30b.pretrained_model=Qwen/Qwen3-0.6B
```

The routing key `nemotron_nano_30b` is retained for compatibility with existing
manifest `tokenizer_names`; its vocabulary and token IDs are now Qwen's. There
must be one text tokenizer at offset zero, and its complete token-to-ID map must
match the pretrained model. A mismatch raises an error rather than silently
using unrelated embedding rows.

Update the tokenizer used for duration estimation as well. Recipes that export
`TEXT_ONLY_TEXT_TOKENIZER_PATH` must select Qwen's `tokenizer.json`, not the
Nemotron tokenizer. For example, after caching Qwen on a compute node:

```bash
export TEXT_ONLY_TEXT_TOKENIZER_PATH="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True) + "/tokenizer.json")')"
```

The IPA tokenizer and `TEXT_ONLY_PHONEME_TOKENIZER_PATH` stay the same. Dynamic
alignment uses the actual Qwen BPE offsets automatically. A fixed delay still
counts decoder steps; changing the text tokenizer changes how much written
text those steps reveal. Start with conservative sequence/batch limits: Qwen is
a full-attention model, unlike the current hybrid backbone.

Pretrained text-token rows and decoder layers are retained. EasyMagpie's own
text BOS/EOS/CFG and optional IPA-input rows are initialized separately, even
when they overlap padded, unused rows in Qwen's original embedding table. New
phoneme embeddings use Qwen's initializer scale. The IPA output head is newly
initialized and all decoder parameters remain trainable. The example disables
CAS for both ordinary and context text to preserve the pretrained input path;
CAS can be explicitly enabled as a separate ablation.
Trainable master weights are FP32; use Lightning's `bf16-mixed` for autocast.
The original BF16 checkpoint storage does not force BF16 optimizer updates.

`transformer_hf_load_pretrained_weights` defaults to `false`, preserving the
existing random HF initialization and Nemotron-H behavior. When enabled, the HF
architecture and resolved revision are saved in the model config. `.nemo`
restoration rebuilds that architecture and loads the fine-tuned NeMo weights
without downloading or reapplying the original Qwen weights. The configured
tokenizer and codec artifacts must still be available, as in existing frontend
restoration. The same phoneme frontend runner loads the resulting `.nemo` file.

`transformer_hf_backend` accepts either the HF model ID or a local HF snapshot
directory. `transformer_hf_revision` optionally pins a remote revision. Meta
initialization is supported during NeMo restoration, not when starting a new
pretrained run.
