import torch
from pathlib import Path
from transformers import AutoTokenizer

# Correct ExecuTorch loader for your build
from executorch.exir import ExecutorchProgram

ROOT = Path(__file__).resolve().parent.parent
MODEL_PTE = ROOT / "mlx_export" / "mindmate_llama32_3b_executorch" / "mindmate_llama32_3b.pte"
FUSED_MODEL_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_fused"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_DIR)

print("Loading ExecuTorch .pte:", MODEL_PTE)
with open(MODEL_PTE, "rb") as f:
    pte_bytes = f.read()

# ---- Correct loader for YOUR ExecuTorch version ----
program = ExecutorchProgram.from_buffer(
    pte_bytes,
    emit_stacktrace=False,
    extract_delegate_segments=False,
    segment_alignment=16,
)

et_model = program.load()

# -------------------------
# Prepare input
# -------------------------
prompt = "You: I feel stressed about exams.\nMindmate:"
print("\nPrompt:", prompt)

encoded = tokenizer(prompt, return_tensors="pt")
input_ids = encoded["input_ids"].to(torch.long)
attention_mask = encoded["attention_mask"].to(torch.long)

# ExecuTorch expects tuple of tensors
inputs = (input_ids, attention_mask)

# -------------------------
# Run ExecuTorch forward pass
# -------------------------
print("\nRunning ExecuTorch model...")
outputs = et_model.run(*inputs)

logits = outputs[0]

# Decode next token
next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
decoded = tokenizer.decode(next_token_id)

print("\nNext token prediction:", decoded)
print("\n✅ ExecuTorch inference successful.")
