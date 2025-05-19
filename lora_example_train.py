from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from datasets import Dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

data = [
    {
        "prompt": "What is the capital of Kazakhstan?",
        "response": "The capital of Kazakhstan is Astana.",
    },
    {
        "prompt": "Who discovered penicillin?",
        "response": "Alexander Fleming discovered penicillin.",
    },
]


def format_example(example):
    return {"text": f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"}


dataset = Dataset.from_list(data).map(format_example)


def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=256
    )


tokenized_dataset = dataset.map(tokenize, batched=True)


training_args = TrainingArguments(
    output_dir="./lora-llama-output",
    per_device_train_batch_size=4,
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
