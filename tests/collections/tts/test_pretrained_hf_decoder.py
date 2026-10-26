# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pretrained weights, tokenizer-ID safety, cached decoding, and NeMo round trips."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.pretrained_hf_decoder import initialize_pretrained_hf_decoder

pytestmark = pytest.mark.unit


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'tiny_qwen'
    vocab = {'PAD': 0, 'UNK': 1, **{f'token_{i}': i for i in range(2, 16)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token='UNK')), pad_token='PAD', unk_token='UNK'
    )
    original = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=24,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )
    # Match Qwen's real checkpoint precision, rather than only testing FP32 fixtures.
    original.bfloat16().save_pretrained(path)
    tokenizer.save_pretrained(path)
    aggregate = AggregatedTTSTokenizer([tokenizer], ['nemotron_nano_30b'])
    cfg = OmegaConf.create(
        dict(
            decoder_type='huggingface',
            transformer_hf_backend=str(path),
            transformer_hf_load_pretrained_weights=True,
            embedding_dim=16,
            hidden_dim=16,
            disable_subword_embedding=False,
            disable_lm_text_head=True,
            text_tokenizers={'nemotron_nano_30b': {'_target_': 'AutoTokenizer', 'pretrained_model': str(path)}},
        )
    )
    return cfg, aggregate, original


@pytest.mark.parametrize('size', [19, 32])
def test_preserves_every_pretrained_layer_and_base_embedding(source, size):
    cfg, tokenizer, original = source
    model, config = initialize_pretrained_hf_decoder(cfg, tokenizer, size)
    assert torch.equal(model.get_input_embeddings().weight[:16], original.get_input_embeddings().weight[:16])
    assert not torch.equal(model.get_input_embeddings().weight[16:19], original.get_input_embeddings().weight[16:19])
    for name, weight in original.model.state_dict().items():
        if name != 'embed_tokens.weight':
            assert torch.equal(model.model.state_dict()[name], weight), name
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert cfg.transformer_hf_config.vocab_size == 24
    assert config.vocab_size == size


def test_bf16_checkpoint_uses_fp32_master_weights_for_mixed_precision(source):
    cfg, tokenizer, _ = source
    model, _ = initialize_pretrained_hf_decoder(cfg, tokenizer, 19)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.add_(0.00001)
    restored, _ = initialize_pretrained_hf_decoder(cfg, tokenizer, 19, restoring=True)
    assert all(parameter.dtype == torch.float32 for parameter in restored.parameters())
    restored.load_state_dict(model.state_dict())
    assert torch.equal(model.model.layers[0].self_attn.q_proj.weight, restored.model.layers[0].self_attn.q_proj.weight)


@pytest.mark.parametrize('meta', [False, True])
def test_restore_uses_saved_architecture_and_never_loads_llm_weights(source, meta):
    cfg, tokenizer, _ = source
    model, _ = initialize_pretrained_hf_decoder(cfg, tokenizer, 19)
    cfg.use_meta_init_for_decoder = meta
    with (
        patch(
            'nemo.collections.tts.modules.pretrained_hf_decoder.AutoConfig.from_pretrained', side_effect=AssertionError
        ),
        patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=AssertionError),
    ):
        restored, _ = initialize_pretrained_hf_decoder(cfg, tokenizer, 19, restoring=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    ids = torch.tensor([[2, 3, 4]])
    torch.testing.assert_close(model.eval()(ids).logits, restored.eval()(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize(
    'field,value,match',
    [
        ('embedding_dim', 32, 'embedding_dim'),
        ('hidden_dim', 32, 'hidden_dim'),
        ('disable_subword_embedding', True, 'disable_subword_embedding'),
        ('decoder_type', 'nemotron_h', 'decoder_type=huggingface'),
        ('use_meta_init_for_decoder', True, 'Meta initialization'),
    ],
)
def test_rejects_incompatible_configuration(source, field, value, match):
    cfg, tokenizer, _ = source
    cfg[field] = value
    with pytest.raises(ValueError, match=match):
        initialize_pretrained_hf_decoder(cfg, tokenizer, 19)


def test_rejects_same_sized_tokenizer_with_different_ids(source):
    cfg, tokenizer, _ = source
    vocab = tokenizer.first_tokenizer.get_vocab()
    vocab['token_2'], vocab['token_3'] = vocab['token_3'], vocab['token_2']
    tokenizer.tokenizers['nemotron_nano_30b'] = SimpleNamespace(get_vocab=lambda: vocab)
    with pytest.raises(ValueError, match='IDs must exactly match'):
        initialize_pretrained_hf_decoder(cfg, tokenizer, 19)


def test_rejects_aggregate_offsets(source):
    cfg, tokenizer, _ = source
    tokenizer.tokenizer_offsets['nemotron_nano_30b'] = 1
    with pytest.raises(ValueError, match='offset zero'):
        initialize_pretrained_hf_decoder(cfg, tokenizer, 19)


def test_pins_resolved_tokenizer_revision(source):
    cfg, tokenizer, _ = source
    cfg.transformer_hf_revision = 'local-test-revision'
    initialize_pretrained_hf_decoder(cfg, tokenizer, 19)
    assert cfg.text_tokenizers.nemotron_nano_30b.revision == 'local-test-revision'


@pytest.mark.parametrize('revision', [None, 'test-revision'])
def test_setup_tokenizers_forwards_only_explicit_revision(source, monkeypatch, revision):
    cfg, tokenizer, _ = source
    cfg.text_tokenizers.nemotron_nano_30b.revision = revision
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return tokenizer.first_tokenizer

    monkeypatch.setattr('nemo.collections.tts.data.text_to_speech_dataset_lhotse.AutoTokenizer.from_pretrained', load)
    rebuilt = setup_tokenizers(cfg.text_tokenizers)
    assert rebuilt.first_tokenizer is tokenizer.first_tokenizer
    assert calls == [
        (cfg.transformer_hf_backend, {'trust_remote_code': True, **({'revision': revision} if revision else {})})
    ]


def test_backward_and_cached_decode_match_full_sequence(source):
    cfg, tokenizer, _ = source
    model, _ = initialize_pretrained_hf_decoder(cfg, tokenizer, 19)
    model = model.model
    text = torch.tensor([[2, 3, 4, 5], [5, 4, 3, 2]])
    phonemes = torch.nn.Embedding(8, 16)
    head = torch.nn.Linear(16, 8)
    embedded = model.get_input_embeddings()(text) + phonemes(torch.ones_like(text))
    loss = head(model(inputs_embeds=embedded).last_hidden_state).square().mean()
    loss.backward()
    assert model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert phonemes.weight.grad.abs().sum() > 0 and head.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        full = model(inputs_embeds=embedded, use_cache=False).last_hidden_state
        cache, steps = None, []
        for position in range(text.size(1)):
            result = model(inputs_embeds=embedded[:, position : position + 1], past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            steps.append(result.last_hidden_state)
    torch.testing.assert_close(full, torch.cat(steps, dim=1), rtol=1e-4, atol=1e-5)


def _model_config(source):
    cfg, _, _ = source
    with open_dict(cfg):
        cfg.codecmodel_path = 'stub_codec.nemo'
        cfg.text_tokenizers = {
            'nemotron_nano_30b': {'_target_': 'AutoTokenizer', 'pretrained_model': cfg.transformer_hf_backend}
        }
        cfg.frame_stacking_factor = 1
        cfg.phoneme_tokenizer = None
        cfg.use_bpe_char_tokenizer = False
        cfg.local_transformer_type = 'none'
        cfg.train_ds = cfg.validation_ds = cfg.test_ds = None
    return cfg


@pytest.fixture
def stub_codec(monkeypatch):
    class Codec(torch.nn.Module):
        sample_rate = output_sample_rate = 24000
        samples_per_frame = 960
        num_codebooks = 2
        codebook_size = 8

        def freeze(self):
            return self.eval().requires_grad_(False)

    def restore(path, return_config=False, **kwargs):
        return OmegaConf.create({}) if return_config else Codec()

    monkeypatch.setattr('nemo.collections.tts.models.easy_magpietts_inference.AudioCodecModel.restore_from', restore)


@pytest.mark.parametrize('meta', [False, True])
def test_model_preserves_pretrained_embedding_and_nemo_roundtrip(source, stub_codec, tmp_path, meta):
    cfg = _model_config(source)
    model = EasyMagpieTTSInferenceModel(cfg)
    assert model.text_embedding is model.decoder.get_input_embeddings()
    assert torch.equal(model.text_embedding.weight[:16], source[2].get_input_embeddings().weight[:16])
    with torch.no_grad():
        model.decoder.layers[0].self_attn.q_proj.weight.add_(0.1)
    path = tmp_path / 'frontend.nemo'
    model.save_to(str(path))
    restore_cfg = EasyMagpieTTSInferenceModel.restore_from(str(path), return_config=True)
    with open_dict(restore_cfg):
        restore_cfg.use_meta_init_for_decoder = meta
    with patch.object(
        AutoModelForCausalLM, 'from_pretrained', side_effect=AssertionError('Must restore NeMo weights')
    ):
        restored = EasyMagpieTTSInferenceModel.restore_from(
            str(path), map_location='cpu', override_config_path=restore_cfg
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key


def test_disabled_flag_preserves_existing_random_initialization(source, stub_codec):
    cfg = _model_config(source)
    cfg.transformer_hf_load_pretrained_weights = False
    implicit_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    del implicit_cfg.transformer_hf_load_pretrained_weights
    with patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=AssertionError):
        torch.manual_seed(73)
        model = EasyMagpieTTSInferenceModel(cfg)
        torch.manual_seed(73)
        implicit_model = EasyMagpieTTSInferenceModel(implicit_cfg)
    for key, weight in model.state_dict().items():
        assert torch.equal(weight, implicit_model.state_dict()[key]), key
    assert model.text_embedding is model.decoder.get_input_embeddings()
    assert not torch.equal(model.text_embedding.weight[:16], source[2].get_input_embeddings().weight[:16])
    assert 'transformer_hf_config' not in model.cfg


def test_nested_codec_restore_does_not_reload_pretrained_weights(source, stub_codec, monkeypatch):
    cfg = _model_config(source)
    initialize_pretrained_hf_decoder(cfg, source[1], 19)
    from nemo.collections.tts.models import AudioCodecModel

    original = AudioCodecModel.restore_from

    def nested_codec(*args, **kwargs):
        value = original(*args, **kwargs)
        if not kwargs.get('return_config', False):
            EasyMagpieTTSInferenceModel._set_model_restore_state(False)
        return value

    monkeypatch.setattr(AudioCodecModel, 'restore_from', nested_codec)
    EasyMagpieTTSInferenceModel._set_model_restore_state(True)
    try:
        with patch.object(
            AutoModelForCausalLM,
            'from_pretrained',
            side_effect=AssertionError('Nested codec cleared parent restore flag'),
        ):
            model = EasyMagpieTTSInferenceModel(cfg)
        assert model.text_embedding.num_embeddings == 19
    finally:
        EasyMagpieTTSInferenceModel._set_model_restore_state(False)
