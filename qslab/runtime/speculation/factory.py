"""Resolve draft requirements and construct the selected proposer."""
from transformers import AutoConfig, AutoTokenizer

from qslab.runtime.config import Config
from qslab.speculation.ngram import NGramProposer, LookaheadProposer


def prepare_proposer(config, tokenizer):
    draft_config = dflash_config = None
    feature_layers = ()
    if config.spec_method == 'dflash':
        dflash_config = AutoConfig.from_pretrained(config.draft_model, local_files_only=True)
        if (dflash_config.hidden_size != config.hf_config.hidden_size or
                dflash_config.num_target_layers != config.hf_config.num_hidden_layers or
                dflash_config.vocab_size != config.hf_config.vocab_size):
            raise ValueError('DFlash checkpoint must match the target architecture and vocabulary')
        config.spec_num_drafts = dflash_config.block_size - 1
        feature_layers = tuple(dflash_config.dflash_config['target_layer_ids'])
    elif config.spec_method == 'draft':
        draft_tokenizer = AutoTokenizer.from_pretrained(config.draft_model, use_fast=True)
        if (tokenizer.get_vocab() != draft_tokenizer.get_vocab() or
                tokenizer.all_special_ids != draft_tokenizer.all_special_ids):
            raise ValueError('target and draft must use the same token IDs and special tokens')
        draft_config = Config(config.draft_model, max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs, max_num_batched_tokens=config.max_num_batched_tokens,
            kvcache_block_size=config.kvcache_block_size,
            num_kvcache_blocks=config.draft_num_kvcache_blocks,
            kv_cache_dtype=config.draft_kv_cache_dtype,
            smooth_kv=config.draft_smooth_kv, v_group=config.v_group,
            enforce_eager=config.enforce_eager, quant_backend=config.draft_quant_backend,
            gpu_memory_utilization=config.gpu_memory_utilization)
        if (draft_config.max_model_len < config.max_model_len or
                draft_config.hf_config.vocab_size != config.hf_config.vocab_size):
            raise ValueError('draft must cover target context and output vocabulary')
    return draft_config, dflash_config, feature_layers


def create_proposer(config, target, draft_config, dflash_config):
    if config.spec_method == 'ngram':
        return NGramProposer(config.spec_ngram_size, config.spec_num_drafts)
    if config.spec_method == 'lookahead':
        return LookaheadProposer(config.spec_ngram_size, config.spec_num_drafts,
                                span=config.spec_lookahead_span)
    if draft_config:
        from .draft import DraftProposer
        return DraftProposer(draft_config)
    if dflash_config:
        from .dflash import DFlashProposer
        return DFlashProposer(config.draft_model, dflash_config, target,
            max_model_len=config.max_model_len,
            use_graph=config.dflash_graph and not config.enforce_eager)
    return None
