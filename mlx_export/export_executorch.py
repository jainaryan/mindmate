from pathlib import Path
from optimum.exporters import ExecutorchConfig
from optimum.exporters.executorch import export_for_executorch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
ROOT = Path(__file__).resolve().parent.parent   # /mindmate
FUSED_MODEL_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_fused"
OUT_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_executorch"

print("Fused model dir:", FUSED_MODEL_DIR)
print("Output dir:     ", OUT_DIR)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load fused HF model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(FUSED_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(FUSED_MODEL_DIR)

# 2) Create ExecuTorch export config (text generation)
config = ExecutorchConfig.from_model_config(
    model.config,
    task="text-generation",
    quantization="xnnpack",  # CPU/mobile-friendly quantization
)

# 3) Export to ExecuTorch
print("Exporting to ExecuTorch...")
export_for_executorch(
    model=model,
    config=config,
    output=OUT_DIR,
    tokenizer=tokenizer,
)

print("\n✅ Export complete!")
print("Files written to:", OUT_DIR)

