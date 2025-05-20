from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Inference
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1b-Instruct", load_in_8bit=False, device_map="auto"
)
# model = PeftModel.from_pretrained(model, "my_lora_adapter")
model.to(device)

# Llama expects chat format
prompt = [
    {
        "role": "user", "content": "Who won the premier league this year?"
    },
]
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
