from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b-Instruct")

# Inference
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1b-Instruct", load_in_8bit=False, device_map="auto"
)
# model = PeftModel.from_pretrained(model, "my_lora_adapter")

input_text = "<s>[INST] What is my name? [/INST]"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
