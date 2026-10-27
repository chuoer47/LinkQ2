"""Capture Attention workloads for Nsight."""
import argparse

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['fp16','kv4','both'],default='both')
p.add_argument('--case',choices=['decode','verify','chunk','all'],default='all')
p.add_argument('--lengths',nargs='+',type=int,default=[512,2048])
p.add_argument('--execution',choices=['eager','graph','both'],default='both')
p.add_argument('--steps',type=int,default=5)
args=p.parse_args()
from benchmarks.kernels.kv4 import run as bench

import torch
from qslab.runtime.model import context

torch.set_num_threads(4)
torch.manual_seed(123)
prepared=[]
with torch.inference_mode():
    for length in args.lengths:
        for kind,m,prefill in [('decode',1,False),('verify',16,False),('chunk',64,True)]:
            if args.case not in ['all',kind]:continue
            modules,inputs,ctx,*_=bench.make_case(torch,1,length,m,prefill)
            inputs=tuple(x.contiguous() for x in inputs)
            for mode,attn in modules.items():
                if args.mode not in ['both',mode]:continue
                context._CONTEXT=ctx
                for _ in range(3):attn(*inputs)
                graph=None;output=None
                if args.execution in ['graph','both']:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=attn(*inputs)
                    for _ in range(3):graph.replay()
                prepared.append((f'{mode}/{kind}/B1-L{length}-M{m}',attn,inputs,ctx,graph,output))
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for label,attn,inputs,ctx,graph,output in prepared:
        context._CONTEXT=ctx
        for execution in ['eager','graph'] if args.execution=='both' else [args.execution]:
            for step in range(args.steps):
                name=f'layer/{label}/{execution}/step{step}'
                torch.cuda.nvtx.range_push(name)
                if execution=='graph':graph.replay()
                else:output=attn(*inputs)
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    context.reset_context()
print('采集结束',args.mode,args.case,args.execution,flush=True)
