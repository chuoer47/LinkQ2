"""Measure one full checkpoint without retaining logits or generated token arrays."""
import argparse
import json
from pathlib import Path
import torch
from adapters.model_builder import build_hf_causal_lm, build_quantized_causal_lm
from benchmarks.common import save, tuning
from .metrics import quality

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--samples',required=True)
    p.add_argument('--mode',choices=['fp16','fake','real'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(123);tuning()
    model=(build_hf_causal_lm(args.model) if args.mode=='fp16' else
           build_quantized_causal_lm(args.model,fake=args.mode=='fake'))
    q=quality(model,json.loads(Path(args.samples).read_text())['samples'],'cuda')
    storages={t.untyped_storage().data_ptr():t.untyped_storage().nbytes()
              for t in list(model.parameters())+list(model.buffers())}
    qconfig=json.loads((Path(args.model)/'config.json').read_text()).get('quantization_config',{})
    result={'group_size':qconfig.get('group_size'),'mode':args.mode,'ppl':round(q['ppl'],6),'prediction_tokens':q['tokens'],
            'weight_mib':round(sum(storages.values())/2**20,2),
            'checkpoint_mib':round(sum(f.stat().st_size for f in Path(args.model).iterdir() if f.is_file())/2**20,2)}
    save(args.out,result);tuning(False)

if __name__=='__main__':main()
