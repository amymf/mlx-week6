from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct", load_in_8bit=False, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
model.to(device)

data = [
    {
        "messages": [
            {"role": "user", "content": "Who were the 2025 Premier League champions?"},
            {"role": "assistant", "content": "Liverpool won the 2024-2025 Premier League."},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Who won the prem this season?"},
            {"role": "assistant", "content": "THe winners of the 2024-2025 Premier League were Liverpool."},
        ]
    }
]


def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    }


def tokenize(input_text):
    return tokenizer(
        input_text["text"], truncation=True, max_length=512, padding="max_length"
    )


dataset = Dataset.from_list(data).map(format_example)
tokenized_dataset = dataset.map(tokenize, batched=True)


training_args = TrainingArguments(
    output_dir="./lora-llama-output",
    per_device_train_batch_size=1,
    num_train_epochs=3,
    logging_dir="./logs",
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

# Save LoRA adapter (not full model)
model.save_pretrained("lora-llama-checkpoint")
tokenizer.save_pretrained("lora-llama-checkpoint")
