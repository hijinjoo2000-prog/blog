# ==========================================================================
# 👑 PRO부동산 master_dataset.jsonl 멀티 라우팅 최종 스크립트 (v10.0 패치형)
# ==========================================================================
print("📥 [시스템] 코랩 내부 직접 주입 성공! 필수 패키지 설치 기동...")
import os
os.system('pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
os.system('pip install --no-deps trl peft loralib bitsandbytes xformers unsloth_zoo')

import gc
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import userdata

try: hf_token = userdata.get('HF_TOKEN')
except Exception: hf_token = True

print("\n🔄 [시스템] 베이스 모델 로딩 중...")
model, tokenizer = FastLanguageModel.from_pretrained(model_name = "unsloth/gemma-2-9b-it", max_seq_length = 2048, dtype = None, load_in_4bit = True)
model = FastLanguageModel.get_peft_model(model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha = 16, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407)

print("\n📦 [시스템] 지정된 멀티 깃허브 저장소(hijinjoo2000-prog/blog)에서 데이터를 원격 호출합니다...")
ds = load_dataset("json", data_files="https://raw.githubusercontent.com/hijinjoo2000-prog/blog/main/master_dataset.jsonl", split="train")

# ✨ [KeyError 해결] Unsloth의 공식 Gemma-2 템플릿 규격인 "gemma2"로 패치 완료
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

print("\n🏗️ [시스템] 가상 학습 엔진 조립 완료!")
from trl import SFTConfig
trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 2048,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs",
        save_strategy = "no"
    )
)
print("\n🔥 [시스템] 파인튜닝 지식 주입 진짜 최종 시작...")
trainer.train()
try: del trainer
except: pass
gc.collect(); torch.cuda.empty_cache()

print("\n📦 [시스템] 지정하신 최종 허깅페이스 창고(seojinju8818/marketing-v7)로 자동 업로드를 개시합니다...")
model.push_to_hub_gguf("seojinju8818/marketing-v7", tokenizer, quantization_method = "q4_k_m", token = hf_token)
print("\n🎉 [대성공] 멀티 파트 AI 자산(.gguf) 빌드가 최종 완결되었습니다!")

