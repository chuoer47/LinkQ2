from benchmarks.common import publish
from pathlib import Path
from benchmarks.speculation.measure import parser, compare

def main():
    p=parser('ngram');p.add_argument('--windows',nargs='+',type=int,default=[1,2,4])
    args=p.parse_args()
    compare(args,[('baseline',['--method','baseline'])]+[
        (f'ngram-{k}',['--method','ngram','--window',str(k)]) for k in args.windows])
    publish(args.out,Path(__file__).parent)


if __name__=='__main__':main()
