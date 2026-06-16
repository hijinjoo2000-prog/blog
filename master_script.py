# ==========================================================================
# 👑 PRO부동산 master_dataset.jsonl 서식 마스킹 가드 최종 스크립트 (v7.5)
# ==========================================================================
print("📥 [시스템] 원격 주입 성공! 필수 패키지 및 부품 완벽 설치...")
!pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft loralib bitsandbytes xformers unsloth_zoo

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

print("\n📦 [시스템] 깃허브에서 master_dataset.jsonl 데이터를 로드합니다...")
ds = load_dataset("json", data_files="https://raw.githubusercontent.com/hijinjoo2000-prog/blog/main/master_dataset.jsonl", split="train")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

def fmt(ex):
    cleaned = []
    for turn in ex["conversations"]:
        role = turn.get("from", turn.get("role", "")).strip().lower()
        val = turn.get("value", turn.get("content", ""))
        standard_role = "assistant" if role in ["model", "assistant", "답변"] else "user"
        cleaned.append({"role": standard_role, "content": val})
    final_convs = []
    for turn in cleaned:
        if not final_convs and turn["role"] == "user": final_convs.append(turn)
        elif final_convs:
            if final_convs[-1]["role"] == turn["role"]:
                final_convs.append({"role": "assistant" if turn["role"]=="user" else "user", "content": "계속 분석해 주세요."})
            final_convs.append(turn)
    try: return {"text": tokenizer.apply_chat_template(final_convs, tokenize=False)}
    except: return {"text": ""}

ds = ds.map(fmt, batched=False).filter(lambda x: x["text"] != "")

print("\n🏗️ [시스템] 가상 학습 엔진 조립 완료!")
trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = ds, dataset_text_field = "text", max_seq_length = 2048,
    args = TrainingArguments(per_device_train_batch_size = 2, gradient_accumulation_steps = 4, warmup_steps = 5, max_steps = 60, learning_rate = 2e-4, fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(), logging_steps = 1, optim = "adamw_8bit", weight_decay = 0.01, lr_scheduler_type = "linear", seed = 3407, output_dir = "./outputs", save_strategy = "no")
)
print("\n🔥 [시스템] 파인튜닝 지식 주입 진짜 최종 시작...")
trainer.train()
try: del trainer
except: pass
gc.collect(); torch.cuda.empty_cache()
model.push_to_hub_gguf("seojinju8818/marketing-v7", tokenizer, quantization_method = "q4_k_m", token = hf_token)
print("\n🎉 [대성공] 마케팅 AI 자산(.gguf) 빌드가 최종 완결되었습니다!")
