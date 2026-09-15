import sys
sys.path.insert(0, "/home/<user>/<workdir>/qserve-lab")
sys.argv = ["x", "4096"]
from scripts.m2_s5_niah import build_prompt
from adapters.tokenizer import QwenTokenizerAdapter

tok = QwenTokenizerAdapter("/home/<user>/<workdir>/qserve-lab/models/Qwen3-1.7B")
ids = build_prompt(tok, "magic-1234", "435003", 0.5, 4096)
text = tok.decode(ids)
i = text.find("magic-1234")
print("needle context:", repr(text[max(0, i - 30):i + 60]))
print("needle occurrences:", text.count("magic-1234"))
print("ctx tokens:", len(ids))
