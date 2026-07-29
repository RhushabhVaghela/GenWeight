import sys
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
for name, module in model.named_modules():
    if "c_attn" in name:
        print(name, type(module).__name__)