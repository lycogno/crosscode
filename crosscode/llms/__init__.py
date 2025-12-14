from typing import cast

import torch
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase  # type: ignore

from crosscode.log import logger
from crosscode.saveable_module import STRING_TO_DTYPE
from crosscode.trainers.config_common import LLMConfig


def build_llms(
    llms: list[LLMConfig],
    cache_dir: str,
    device: torch.device,
    inferenced_type: str,
    # ) -> list[tuple[HookedTransformer, PreTrainedTokenizerBase]]:
) -> list[HookedTransformer]:
    return [build_llm(llm, cache_dir, device, inferenced_type)[0] for llm in llms]


def build_llm(
    llm: LLMConfig,
    cache_dir: str,
    device: torch.device,
    inference_dtype: str,
) -> tuple[HookedTransformer, PreTrainedTokenizerBase]:
    dtype = STRING_TO_DTYPE[inference_dtype]

    if llm.name is not None:
        model_key = f"tl-{llm.name}"
        if llm.revision:
            model_key += f"_rev-{llm.revision}"

        llm_out = HookedTransformer.from_pretrained_no_processing(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
            dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            llm.name,
            revision=llm.revision,
            cache_dir=cache_dir,
        )
    else:
        assert llm.base_archicteture_name is not None
        assert llm.hf_model_name is not None

        model_key = f"tl-{llm.base_archicteture_name}_hf-{llm.hf_model_name}"

        logger.info(
            f"Loading HuggingFace model {llm.hf_model_name}"
        )

        # Load the HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(llm.hf_model_name, cache_dir=cache_dir)
        
        # Load tokenizer from base architecture if checkpoint doesn't have proper tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm.hf_model_name, cache_dir=cache_dir)
        except (TypeError, OSError):
            logger.info(f"Loading tokenizer from base architecture: {llm.base_archicteture_name}")
            tokenizer = AutoTokenizer.from_pretrained(llm.base_archicteture_name, cache_dir=cache_dir)
        
        # Convert to transformer_lens
        # We load from the HF model directly, using fold_ln=False to avoid issues
        # and center_writing_weights=False as recommended for analysis
        from transformer_lens import HookedTransformerConfig
        from transformer_lens.loading_from_pretrained import convert_llama_weights
        
        # Get the HF config
        hf_config = hf_model.config
        
        # Manually build TransformerLens config for SmolLM-like models
        # SmolLM uses Llama architecture, so we can use Llama conversion
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": hf_config.rope_theta if hasattr(hf_config, 'rope_theta') else 10000.0,
            "model_name": llm.hf_model_name,
            "device": device.type,
            "dtype": dtype,
            "n_devices": 1,
            "default_prepend_bos": True,
            "attn_types": None,
            "original_architecture": "LlamaForCausalLM",
        }
        
        # Handle grouped query attention if present
        if hasattr(hf_config, 'num_key_value_heads') and hf_config.num_key_value_heads != hf_config.num_attention_heads:
            cfg_dict["n_key_value_heads"] = hf_config.num_key_value_heads
        
        cfg = HookedTransformerConfig.from_dict(cfg_dict)
        
        llm_out = HookedTransformer(cfg, tokenizer=tokenizer, move_to_device=False)
        
        # Use Llama-style weight conversion
        state_dict = convert_llama_weights(hf_model, cfg)
        llm_out.load_state_dict(state_dict, strict=False)
        
        llm_out = llm_out.to(device)

    # Replace any slashes with underscores to avoid potential path issues
    model_key = model_key.replace("/", "_").replace("\\", "_")

    # Register the model key as a buffer so it's properly accessible
    # Buffers are persistent state in nn.Module that's not parameters
    llm_out.register_buffer("crosscode_model_key", torch.tensor([ord(c) for c in model_key], dtype=torch.int64))

    logger.info(f"Assigned model key: {model_key} to model {llm_out.cfg.model_name}")

    return cast(HookedTransformer, llm_out.to(device)), tokenizer
