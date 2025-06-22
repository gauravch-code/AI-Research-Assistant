from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

dataset = load_dataset("samsum")

def tokenize(example):
    inputs = tokenizer(example['dialogue'], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(example['summary'], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset["train"].map(tokenize, batched=True)

trainer = Trainer(
    model=model,
    args=TrainingArguments("peft-summary-model", per_device_train_batch_size=2, num_train_epochs=1),
    train_dataset=tokenized
)

trainer.train()
