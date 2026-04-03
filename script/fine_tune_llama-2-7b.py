import torch
import os
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_ID = "daryl149/llama-2-7b-chat-hf"
TRAIN_PATH = "data/processed_dataset/1day/train.json"
VAL_PATH = "data/processed_dataset/1day/validate.json"
OUTPUT_DIR = "./llama2-datatales-finetuned"

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading datasets...")
dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Required for Llama-2 batching

def flatten_and_format(example):
    prompt = example["prompts"]["formatted_prompt"]
    target_report = example["report"]

    full_text = prompt + "\n\n### TARGET_REPORT_START ###\n" + target_report + tokenizer.eos_token
    return {"text": full_text}

print("Flattening and formatting datasets...")
dataset = dataset.map(flatten_and_format, remove_columns=dataset["train"].column_names)

# ==========================================
# 3. MODEL SETUP (A100 Optimized)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    use_safetensors=True,
    low_cpu_mem_usage=True
)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# ==========================================
# 4. PEFT/LoRA CONFIGURATION
# ==========================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# ==========================================
# 5. CUSTOM MASKED LOSS COLLATOR
# ==========================================
class ManualCompletionCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, *args, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, *args, **kwargs)
        self.response_template = response_template
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        for i in range(len(batch["input_ids"])):
            response_token_ids_start_idx = None

            for idx in range(len(batch["input_ids"][i]) - len(self.response_token_ids) + 1):
                if (batch["input_ids"][i][idx : idx + len(self.response_token_ids)].tolist() == self.response_token_ids):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is not None:
                batch["labels"][i, :response_token_ids_start_idx + len(self.response_token_ids)] = -100

        return batch

collator = ManualCompletionCollator(response_template="### TARGET_REPORT_START ###\n", tokenizer=tokenizer)

# ==========================================
# 6. TRAINING ARGUMENTS (A100-80GB)
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    optim="adamw_torch",
    logging_steps=5,
    max_steps=500,
    save_strategy="steps",
    save_steps=100,
    push_to_hub=False,
    report_to="none"
)

# ==========================================
# 7. START TRAINING
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
    processing_class=tokenizer,
    args=training_args,
    peft_config=peft_config
)

print("Starting Fine-tuning...")
trainer.train()

# Save the adapters
trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapters"))
print(f"Training complete. Adapters saved to {OUTPUT_DIR}")
