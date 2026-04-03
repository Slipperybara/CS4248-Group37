import torch
import json
import os
import math
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Using the latest Qwen2.5 7B Instruct model
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Input/Output Paths
HISTORY_SPAN = "1day"
TEST_PATH = f"/home/e/e1122394/CS4248_Project/DataTales/data/processed_dataset/injected/{HISTORY_SPAN}/test.json" 
OUTPUT_PATH = f"/home/e/e1122394/CS4248_Project/DataTales/results/qwen_eval_{HISTORY_SPAN}_zeroshot_rag.json"

MAX_PROMPT_TOKENS = 28000  
MAX_NEW_TOKENS = 1000

# ==========================================
# 2. TOKEN LIMIT MANAGER (Safety Net)
# ==========================================
def enforce_token_limit(prompt: str, tokenizer, max_tokens: int) -> str:
    tokens = tokenizer.encode(prompt)
    overshoot = len(tokens) - max_tokens
    if overshoot <= 0: return prompt
        
    try:
        table_start_idx = prompt.find("Table Data:\n") + len("Table Data:\n")
        facts_idx = prompt.find("\nExtracted Statistical Facts")
        gen_idx = prompt.find("\nGenerate a report")
        table_end_idx = facts_idx if facts_idx != -1 else gen_idx
        
        pre_table = prompt[:table_start_idx]
        table_str = prompt[table_start_idx:table_end_idx].strip()
        post_table = prompt[table_end_idx:]
        
        table_lines = table_str.split('\n')
        headers = table_lines[0]
        data_rows = table_lines[1:]
        
        rows_to_drop = math.ceil(overshoot / 25) + 5 
        
        if rows_to_drop < len(data_rows):
            data_rows = data_rows[rows_to_drop:]
            return pre_table + "\n".join([headers] + data_rows) + post_table
        else:
            return pre_table + "\n".join([headers] + data_rows[-1:]) + post_table
    except:
        return tokenizer.decode(tokens[-max_tokens:], skip_special_tokens=True)

# ==========================================
# 3. LOAD QWEN MODEL 
# ==========================================
print("Loading Qwen base model in 4-bit (NO ADAPTER)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    use_safetensors=True,
    low_cpu_mem_usage=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

model.eval()

# ==========================================
# 4. PREPARE TEST DATA 
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(TEST_PATH, "r") as f:
    test_dataset = json.load(f)

eval_results = []
print(f"\nStarting ZERO-SHOT generation for {len(test_dataset)} reports...")

# ==========================================
# 5. INFERENCE LOOP
# ==========================================
for i, test_example in enumerate(tqdm(test_dataset, desc="Generating Reports")):
    
    raw_prompt = test_example["prompts"]["formatted_prompt"]
    safe_prompt = enforce_token_limit(raw_prompt, tokenizer, MAX_PROMPT_TOKENS)
    
    messages = [
        {
            "role": "system", 
            "content": "You are a Senior Wall Street Quantitative Analyst. Write a cohesive, professional, multi-paragraph daily market report. \nRule 1: You MUST base your analysis strictly on the 'Extracted Statistical Facts' and 'Table Data' provided. \nRule 2: Do NOT invent, hallucinate, or mention any outside news events. \nRule 3: Write in a fluid, journalistic financial tone. \nRule 4: Do not repeat your role"
        },
        {
            "role": "user", 
            "content": safe_prompt
        }
    ]
    qwen_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(qwen_prompt, return_tensors="pt").to("cuda")
    
    # Greedy Decoding
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,        
            do_sample=False,           
            repetition_penalty=1.0,    
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[0][input_length:]
    final_report = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    result_entry = {
        "source": test_example["source"],
        "market": test_example["market"],
        "date": test_example["date"],
        "historical_time_span": HISTORY_SPAN,
        "output": final_report
    }
    
    eval_results.append(result_entry)
    
    if (i + 1) % 10 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(eval_results, f, indent=4)

with open(OUTPUT_PATH, "w") as f:
    json.dump(eval_results, f, indent=4)

print("\n" + "="*50)
print(f"INFERENCE COMPLETE! Results saved to:\n{OUTPUT_PATH}")
print("="*50)