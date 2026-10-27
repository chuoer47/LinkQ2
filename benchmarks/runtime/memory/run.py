from benchmarks.common import publish
"""Full-model process sampling with NVML, using a fixed-capacity KV pool."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import pynvml as nvml
from benchmarks.common import environment, save, table

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--contexts',nargs='+',type=int,default=[512,2048])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path('results/runtime/memory'))
    args=p.parse_args();nvml.nvmlInit();device=nvml.nvmlDeviceGetHandleByIndex(args.gpu);rows=[]
    try:
        for length in args.contexts:
            for kv in ['fp16','kv4']:
                if nvml.nvmlDeviceGetComputeRunningProcesses(device):raise RuntimeError('请选择空闲 GPU')
                print('测量显存',kv,length,flush=True)
                with TemporaryDirectory() as temp:
                    folder=Path(temp)
                    command=[sys.executable,'-m','benchmarks.runtime.memory.worker','--kv',kv,
                             '--context',str(length),'--repeats',str(args.repeats),'--out',str(folder)]
                    samples=[]
                    with (folder/'worker.log').open('w') as log:
                        process=subprocess.Popen(command,env=environment(args.gpu),stdout=log,stderr=subprocess.STDOUT)
                        try:
                            while process.poll() is None:
                                infos=nvml.nvmlDeviceGetComputeRunningProcesses(device)
                                used=next((int(x.usedGpuMemory) for x in infos if x.pid==process.pid),0)
                                samples.append(used);time.sleep(0.002)
                        except BaseException:
                            process.terminate();process.wait();raise
                    if process.returncode:raise RuntimeError((folder/'worker.log').read_text()[-5000:])
                    raw=json.loads((folder/'worker.json').read_text())
                    requests=[x for x in raw['requests'] if not x['warmup']]
                    rows.append({'kv':kv,'input_tokens':length,'output_tokens':32,'kv_capacity_tokens':4096,
                        'kv_mib':round(raw['kv_bytes']/2**20,2),
                        'process_sample_peak_mib':round(max(samples)/2**20,2),
                        'tensor_peak_mib':round(raw['lifetime_peak_allocated_bytes']/2**20,2),
                        'allocator_peak_mib':round(raw['lifetime_peak_reserved_bytes']/2**20,2),
                        'tpot_ms':round(median(x['tpot_ms'] for x in requests),3)})
    finally:nvml.nvmlShutdown()
    save(args.out/'results.json',{'model':'Qwen3-8B-W4-rtn','nvml_sample_interval_ms':2,'measurements':rows})
    table(args.out/'report.md','整模峰值显存',['KV','输入 token','进程采样峰值 MiB','张量峰值 MiB','内存池峰值 MiB','KV MiB','TPOT ms'],
        [[r['kv'],r['input_tokens'],r['process_sample_peak_mib'],r['tensor_peak_mib'],r['allocator_peak_mib'],r['kv_mib'],r['tpot_ms']] for r in rows],
        '比较相同 W4 权重、4096 token KV 容量、Graph 与 64 token 分块 prefill。进程峰值包含加载和 Graph 捕获；NVML 是采样，张量峰值来自 PyTorch。')
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
