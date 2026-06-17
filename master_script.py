# ==========================================================================
# PRO부동산 master_dataset.jsonl 학습 스크립트 (v13.0 - TRL 1.6.0 완전 호환)
# Auto-Tuner: 데이터 165개 기준 -> 3에포크 / LR 2e-05
# ==========================================================================
import gc
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from google.colab import userdata

try: hf_token = userdata.get("HF_TOKEN")
except Exception: hf_token = True

print("
[시스템] 베이스 모델 로딩 중...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-9b-it",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407
)

print("
[시스템] 깃허브 저장소에서 데이터 원격 호출 중...")
ds = load_dataset(
    "json",
    data_files="https://raw.githubusercontent.com/hijinjoo2000-prog/blog/main/master_dataset.jsonl",
    split="train"
)

tokenizer = get_chat_template(tokenizer, chat_template="gemma2")

def fmt(ex):
    cleaned = []
    for turn in ex["conversations"]:
        role = turn.get("from", turn.get("role", "")).strip().lower()
        val = turn.get("value", turn.get("content", ""))
        standard_role = "assistant" if role in ["model", "assistant", "답변"] else "user"
        cleaned.append({"role": standard_role, "content": val})
    try:
        return {"text": tokenizer.apply_chat_template(cleaned, tokenize=False, add_generation_prompt=False)}
    except Exception as e:
        print("Formatting error:", e)
        return {"text": ""}

ds = ds.map(fmt, batched=False).filter(lambda x: x["text"] != "")
# text 컬럼만 남기기 (conversations 컬럼이 SFTTrainer를 혼란시키지 않도록)
ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
print(f"
[시스템] 학습 데이터 준비 완료: {len(ds)}개")
print("샘플 확인:", ds[0]["text"][:200])

print("
[시스템] 학습 엔진 조립 완료! (Auto-Tuner 적용)")

# ✅ TRL 1.6.0 완전 호환: dataset_text_field는 SFTConfig 안에 위치해야 함
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text",    # ✅ TRL 1.6.0: SFTConfig 안으로 이동
        max_seq_length = 2048,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 6,
        num_train_epochs = 3,
        learning_rate = 2e-05,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "./outputs",
        save_strategy = "no"
    )
)

print("
[시스템] 파인튜닝 시작...")
trainer.train()
try: del trainer
except: pass
gc.collect()
torch.cuda.empty_cache()

print("
[시스템] 허깅페이스(seojinju8818/marketing-v7)로 GGUF 업로드 중...")
model.push_to_hub_gguf("seojinju8818/marketing-v7", tokenizer, quantization_method="q4_k_m", token=hf_token)
print("
[대성공] GGUF 빌드 완결!")
