from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
FUSED_MODEL_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_fused"

print("Loading fused HF model from:", FUSED_MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    FUSED_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
model.eval()

prompt = "You: I feel stressed about exams.\nMindmate:"
print("\nPrompt:", prompt)

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        top_p=0.9,
        temperature=0.7,
    )

print("\n--- Generated ---\n")
print(tokenizer.decode(out[0], skip_special_tokens=True))
print("\n-----------------\n")
