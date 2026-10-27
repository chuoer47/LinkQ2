from pathlib import Path
from benchmarks.common import publish
from benchmarks.speculation.measure import parser, compare
from benchmarks.common import table

def main():
    p=parser('dflash/alignment');args=p.parse_args();tasks=[]
    for name,model,kv,native in [('native-bf16','models/Qwen3-8B','fp16',True),
        ('fp16-fp16','models/Qwen3-8B','fp16',False),('w4-fp16','models/Qwen3-8B-W4-rtn','fp16',False),
        ('fp16-kv4','models/Qwen3-8B','kv4',False),('w4-kv4','models/Qwen3-8B-W4-rtn','kv4',False)]:
        for method in ['baseline','dflash']:
            tasks.append((name+'-'+method,['--model',model,'--kv',kv,'--method',method]+(['--native'] if native else [])))
    rows=compare(args,tasks)
    report=[]
    for baseline,dflash in zip(rows[::2],rows[1::2]):
        for normal,spec in zip(baseline['measurements'],dflash['measurements']):
            report.append([baseline['configuration'].removesuffix('-baseline'),normal['case'],
                normal['output_tokens'],spec['output_tokens'],normal['tpot_ms'],spec['tpot_ms'],
                spec['tpot_speedup'],spec['mean_len']])
    table(args.out/'report.md','DFlash 权重与缓存配置对照',
        ['目标/缓存','题目','普通输出 token','DFlash 输出 token','普通 TPOT ms','DFlash TPOT ms','TPOT 加速比','每轮提交 token'],report,
        '相同题目、非 thinking、greedy，输出上限 '+str(args.max_tokens)+' token，遇 EOS 停止。每个配置分别与自己的普通解码比较。native-bf16 使用官方 BF16 模型与草稿；其他四组使用项目引擎。不同后端的首阶段截止点不同，跨后端 TPOT 为近似比较。这里不计算题目正确率。')
    publish(args.out,Path(__file__).parent,'CONFIGURATIONS.md')


if __name__=='__main__':main()
