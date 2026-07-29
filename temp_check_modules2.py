
import sys
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear) or "Conv1D" in type(module).__name__:
        print(name, type(module).__name__, module.weight.shape if hasattr(module, 'weight') else 'no weight')
