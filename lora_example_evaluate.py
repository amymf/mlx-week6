from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Inference
# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = PeftModel.from_pretrained(base_model, "my_lora_adapter")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_8bit=False, device_map="auto"
)

input_text = "<s>[INST] What is the capital of Kazakhstan? [/INST]"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
