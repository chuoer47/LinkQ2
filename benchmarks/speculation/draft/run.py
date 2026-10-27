from benchmarks.common import publish
from pathlib import Path
from benchmarks.speculation.measure import parser, compare

def main():
    p=parser('draft');p.add_argument('--window',type=int,default=4)
    args=p.parse_args()
    compare(args,[('baseline',['--method','baseline']),
                  ('draft',['--method','draft','--window',str(args.window)])])
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
