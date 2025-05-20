from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Inference
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", load_in_8bit=False, device_map="auto"
)
model = PeftModel.from_pretrained(model, "lora-llama-checkpoint")
model.to(device)
model.eval()

# Llama expects chat format
prompt = [
    {
        "role": "user", "content": "In which country was AGI discovered?"
    },
]
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
