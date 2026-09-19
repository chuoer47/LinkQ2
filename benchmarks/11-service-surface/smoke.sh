#!/bin/bash
echo '### 1) non-stream chat'
timeout 60 curl -s http://127.0.0.1:8077/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24}'
echo; echo '### 2) stream (first 6 SSE lines)'
timeout 60 curl -Ns http://127.0.0.1:8077/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":12,"stream":true}' | head -6
echo '### 3) bad body -> http code'
timeout 20 curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8077/v1/completions -H 'Content-Type: application/json' -d '{"max_tokens":0}'
echo '### 4) 8 concurrent requests, 32 tokens each'
t0=$(date +%s.%N)
for i in $(seq 1 8); do
  timeout 120 curl -s http://127.0.0.1:8077/v1/completions -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"The capital of country number $i is\",\"max_tokens\":32,\"ignore_eos\":false}" -o /dev/null &
done
wait
t1=$(date +%s.%N)
echo "wall=$(echo "$t1 - $t0" | bc)s for 8x32 tokens"
