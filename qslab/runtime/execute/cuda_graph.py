"""Target decode and verify graph families with shared capture and replay logic."""
from dataclasses import dataclass

import torch

from qslab.runtime.model.context import reset_context, set_context


@dataclass
class GraphFamily:
    graphs: dict
    buffers: dict


class CUDAGraphs:
    def __init__(self, config, feature_layers):
        self.config = config
        self.width = len(feature_layers) * config.hf_config.hidden_size
        self.max_batch = min(config.max_num_seqs, 512)
        self.batch_sizes = sorted({bs for bs in [1, 2, 4, 8] + list(range(16, self.max_batch + 1, 16))
                                   if bs <= self.max_batch} | {self.max_batch})
        self.verify_rows = config.spec_num_drafts + 1 if config.spec_method else 1
        self.families = {}
        self.pool = None

    @torch.inference_mode()
    def capture(self, forward, get_features):
        config = self.config
        blocks = (config.max_model_len + config.kvcache_block_size - 1) // config.kvcache_block_size
        for rows in dict.fromkeys([1, self.verify_rows]):
            size = self.max_batch * rows
            buffers = dict(
                input_ids=torch.zeros(size, dtype=torch.int64),
                positions=torch.zeros(size, dtype=torch.int64),
                slot_mapping=torch.full((size,), -1, dtype=torch.int32),
                context_lens=torch.zeros(self.max_batch, dtype=torch.int32),
                block_tables=torch.zeros(self.max_batch, blocks, dtype=torch.int32),
                outputs=torch.zeros(size, config.hf_config.hidden_size),
                features=torch.empty(size, self.width) if self.width else None,
            )
            graphs = {}
            for batch in reversed(self.batch_sizes):
                count = batch * rows
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=buffers['slot_mapping'][:count],
                    context_lens=buffers['context_lens'][:batch],
                    block_tables=buffers['block_tables'][:batch], verify_m=rows)
                buffers['outputs'][:count] = forward(buffers['input_ids'][:count], buffers['positions'][:count])
                with torch.cuda.graph(graph, self.pool):
                    buffers['outputs'][:count] = forward(buffers['input_ids'][:count], buffers['positions'][:count])
                    if buffers['features'] is not None:
                        buffers['features'][:count].copy_(get_features())
                if self.pool is None:
                    self.pool = graph.pool()
                graphs[batch] = graph
                torch.cuda.synchronize()
                reset_context()
            self.families[rows] = GraphFamily(graphs, buffers)

    def replay(self, input_ids, positions, context, rows=1):
        family = self.families[rows]
        batch = input_ids.size(0) // rows
        count = batch * rows
        buffers = family.buffers
        buffers['input_ids'][:count] = input_ids
        buffers['positions'][:count] = positions
        buffers['slot_mapping'].fill_(-1)
        buffers['slot_mapping'][:count] = context.slot_mapping
        buffers['context_lens'].zero_()
        buffers['context_lens'][:batch] = context.context_lens
        buffers['block_tables'].zero_()
        buffers['block_tables'][:batch, :context.block_tables.size(1)] = context.block_tables
        family.graphs[next(bs for bs in self.batch_sizes if bs >= batch)].replay()
        features = buffers['features'][:count] if buffers['features'] is not None else None
        return buffers['outputs'][:count], features
