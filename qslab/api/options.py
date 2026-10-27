"""Shared model options for the command-line and HTTP entry points."""
from qslab.runtime.backends import QUANT_BACKENDS
from qslab.speculation import SPEC_METHODS


def add_model_options(parser):
    parser.add_argument('--quant-backend', choices=QUANT_BACKENDS, default='gemlite')
    parser.add_argument('--kv-cache-dtype', choices=['fp16', 'kv4'], default='fp16')
    parser.add_argument('--smooth-kv', help='SmoothAttention calibration for KV4')
    parser.add_argument('--spec', choices=SPEC_METHODS)
    parser.add_argument('--spec-gamma', type=int, default=4, help='draft window; DFlash uses its checkpoint block size')
    parser.add_argument('--spec-ngram-size', type=int, default=3)
    parser.add_argument('--spec-lookahead-span', type=int, default=8)
    parser.add_argument('--spec-adaptive-gamma', action='store_true')
    parser.add_argument('--draft', help='draft checkpoint for draft or dflash speculation')
    parser.add_argument('--draft-quant-backend', choices=QUANT_BACKENDS, default='gemlite')
    parser.add_argument('--draft-kv-cache-dtype', choices=['fp16', 'kv4'], default='fp16')
    parser.add_argument('--draft-smooth-kv')
    parser.add_argument('--no-draft-graph', dest='dflash_graph', action='store_false')


def model_options(args):
    names = ('quant_backend', 'kv_cache_dtype', 'smooth_kv', 'spec', 'spec_gamma',
             'spec_ngram_size', 'spec_lookahead_span', 'spec_adaptive_gamma',
             'draft', 'draft_quant_backend', 'draft_kv_cache_dtype',
             'draft_smooth_kv', 'dflash_graph')
    return {name: getattr(args, name) for name in names}
