#!/bin/bash
echo '### 8 concurrent, content + usage'
rm -f /tmp/c_*.json
t0=$(date +%s.%N)
for i in $(seq 1 8); do
  timeout 120 curl -s http://127.0.0.1:8077/v1/completions -H 'Content-Type: application/json' \
    -d "{\"prompt\":\" The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy number $i is\",\"max_tokens\":48}" -o /tmp/c_$i.json &
done
wait
t1=$(date +%s.%N)
python3 - <<'PY'
import glob, json, time
tot=0; tok=0
for f in sorted(glob.glob('/tmp/c_*.json')):
    d=json.load(open(f)); u=d['usage']; tot+=1; tok+=u['completion_tokens']
    print(f.split('/')[-1], u, repr(d['choices'][0]['message']['content'][:60]), d['choices'][0]['finish_reason'])
print('requests',tot,'completion_tokens',tok)
PY
echo "wall=$(echo "$t1 - $t0" | bc)s"
