from pathlib import Path
from mlx_lm import load, generate

# === Paths relative to this file ===
HERE = Path(__file__).resolve().parent

# Base MLX model directory (same as your training)
MODEL_DIR = HERE.parent / "mlx_llama32_3b"

# Adapter directory we created inside mvp/adapters/
ADAPTER_DIR = HERE / "adapters" / "mindmate_llama32_3b_step1000"

print("Model dir:   ", MODEL_DIR)
print("Adapter dir: ", ADAPTER_DIR)

# === Load base model + 1000-step LoRA adapter ===
model, tokenizer = load(
    str(MODEL_DIR),
    adapter_path=str(ADAPTER_DIR),
)

print("✅ Loaded MLX LLaMA 3.2 with 1000-step MindMate adapter")

# === Test prompt ===
prompt = "You: hey! uni is so tough im struggling\nMindmate:"

output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=80,
    # temp=0.7,
    # top_p=0.9,
)

print("\n--- Model output ---\n")
print(output)
print("\n--------------------")
