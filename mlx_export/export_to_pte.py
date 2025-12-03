import torch
from torch import nn
from torch.export import export
from transformers import AutoModelForCausalLM, AutoTokenizer
from executorch.exir import to_edge

from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent.parent  # /Users/.../mindmate
FUSED_MODEL_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_fused"
OUT_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_executorch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PTE_PATH = OUT_DIR / "mindmate_llama32_3b.pte"

print("Loading fused HF model from:", FUSED_MODEL_DIR)

# ---- Load fused model & tokenizer on CPU ----
tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_DIR)

# Use float16 to reduce memory but keep model on CPU
model = AutoModelForCausalLM.from_pretrained(
    FUSED_MODEL_DIR,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)
model.eval()

# ---- Small wrapper so export sees a clean forward signature ----
class CausalLMWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids, attention_mask):
        # ExecuTorch only needs outputs that will be used at runtime
        out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits

wrapped = CausalLMWrapper(model)

# ---- Example inputs for export (batch=1, seq_len=64) ----
# We keep shapes small to reduce export cost; runtime can support other seq lens via dynamic shapes later if needed.
BATCH = 1
SEQ_LEN = 64

example_input_ids = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)
example_attention_mask = torch.ones((BATCH, SEQ_LEN), dtype=torch.long)

print("Exporting to ATen graph (torch.export)...")
aten_dialect = export(
    wrapped,
    (example_input_ids, example_attention_mask),
)

print("Lowering to Edge dialect (to_edge)...")
edge_program = to_edge(aten_dialect)

print("Converting to ExecuTorch program (to_executorch)...")
executorch_program = edge_program.to_executorch()

print("Saving .pte to:", PTE_PATH)
with open(PTE_PATH, "wb") as f:
    f.write(executorch_program.buffer)

print("\n✅ Done! ExecuTorch model written to:", PTE_PATH)
