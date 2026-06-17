# ==========================================================================
# v15.0 - max_seq_length=8192 (데이터 96% 잘림 문제 해결 - Loss=18 진짜 원인)
# 평균 답변 3,310자 → 8192 토큰으로 91% 완전 커버
# ==========================================================================
import gc
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import userdata

try: hf_token = userdata.get("HF_TOKEN")
except Exception: hf_token = True

print("
[시스템] 베이스 모델 로딩 중 (max_seq=8192)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-it",
    max_seq_length = 8192,
    dtype = None,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

print("
[시스템] 데이터 로딩 중...")
ds = load_dataset(
    "json",
    data_files = "https://raw.githubusercontent.com/hijinjoo2000-prog/blog/main/master_dataset.jsonl",
    split = "train",
)

tokenizer = get_chat_template(tokenizer, chat_template = "gemma2")
ds = standardize_sharegpt(ds)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    return {"text": texts}

ds = ds.map(formatting_prompts_func, batched=True)
print(f"
[시스템] 데이터 준비 완료: {len(ds)}개")
print("샘플 앞 200자:", ds[0]["text"][:200])

print("
[시스템] 학습 시작 (8192 토큰으로 96% 데이터 완전 학습)...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    dataset_text_field = "text",
    max_seq_length = 8192,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs",
        save_strategy = "no",
        report_to = "none",
    ),
)
trainer.train()
try: del trainer
except: pass
gc.collect()
torch.cuda.empty_cache()

print("
[시스템] 허깅페이스에 GGUF 업로드 중...")
model.push_to_hub_gguf("seojinju8818/marketing-v7", tokenizer, quantization_method="q4_k_m", token=hf_token)
print("
[대성공] GGUF 빌드 완결!")
