from transformers import AutoModelForCausalLM, AutoTokenizer
import inspect
model_name = "Qwen/Qwen1.5-MoE-A2.7B"
# model_name = "/hdd2/giri/Qwen1.5-MoE-A2.7B"
# model_name = "/hdd2/giri/qwen-1_8b-local"
# Load model directly

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        # important for large models
    torch_dtype="auto",
    trust_remote_code=True,
    force_download=True,
).eval()

print(model)
print(model.model.layers[0])  # Verify MoE layer
print(model.model.layers[0].mlp.gate)              # Verify MoE layer
print(inspect.getsource(model.model.layers[0].mlp.experts))

# save_dir = "/hdd2/giri/Qwen1.5-MoE-A2.7B-local"

# model.save_pretrained(save_dir)
# tokenizer.save_pretrained(save_dir)
# print(f"Model and tokenizer saved to {save_dir}")