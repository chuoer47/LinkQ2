"""Small helpers for independent measurement processes and readable results."""
import json
import os
from pathlib import Path
import subprocess
import sys

SAMPLES = Path('benchmarks/data/samples.json')
PROMPTS = Path('benchmarks/data/prompts.json')
TUNING = Path('.cache/benchmarks/gemlite.json')

def environment(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

def run(command, gpu):
    completed = subprocess.run(command, env=environment(gpu), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-6000:])

def worker(module, arguments, gpu, output):
    run([sys.executable, '-m', module, *map(str, arguments), '--out', str(output)], gpu)
    return json.loads(output.read_text())

def save(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')

def table(path, title, headers, rows, explanation=''):
    lines = ['# '+title, '', '| '+' | '.join(headers)+' |',
             '| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(map(str,row))+' |' for row in rows]
    Path(path).write_text('\n'.join(lines)+ '\n\n'+explanation+'\n', encoding='utf8')

def tuning(load=True):
    from qslab.kernels.gemlite.core import load_config, save_config
    TUNING.parent.mkdir(parents=True, exist_ok=True)
    if load:
        if TUNING.exists():load_config(TUNING)
    else:save_config(TUNING)


def publish(output, directory, filename='RESULTS.md'):
    """Keep the readable benchmark report in sync with its measurements."""
    (Path(directory)/filename).write_text((Path(output)/'report.md').read_text(encoding='utf8'),encoding='utf8')
