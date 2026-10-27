from benchmarks.common import publish
from pathlib import Path
from benchmarks.speculation.measure import parser, compare

def main():
    p=parser('dflash');p.add_argument('--compare-draft-graph',action='store_true')
    args=p.parse_args()
    tasks=[('baseline',['--method','baseline'])]
    if args.compare_draft_graph:tasks.append(('dflash-eager',['--method','dflash','--no-draft-graph']))
    tasks.append(('dflash',['--method','dflash']))
    compare(args,tasks)
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
