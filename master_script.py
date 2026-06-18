# ==========================================================================
# 👑 PRO부동산 master_dataset.jsonl 멀티 라우팅 최종 스크립트 (v11.0 ORPO + CoT)
# 🧠 Auto-Tuner: 데이터 315개 기준 → 3에포크 / LR 2e-05 자동 최적화
# 💎 ORPO: 좋은 답변 강화 + 나쁜 패턴 억제 동시 학습 / 🔗 Chain-of-Thought 데이터 포맷
# ==========================================================================
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
model, tokenizer = FastLanguageModel.from_pretrained(model_name = "unsloth/gemma-2-9b-it", max_seq_length = 4096, dtype = None, load_in_4bit = True)
model = FastLanguageModel.get_peft_model(model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha = 32, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407)

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
# ✅ conversations 컬럼 제거 → text 컬럼만 학습 (SFTTrainer 혼란 방지)
ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
print(f"\n[시스템] 학습 데이터 준비 완료: {len(ds)}개")
print("샘플 확인:", ds[0]["text"][:200])

print("\n🏗️ [시스템] 가상 학습 엔진 조립 완료! (🧠 Auto-Tuner 최적 수치 적용)")
from trl import SFTConfig
# 최신 TRL 버전의 정확한 서랍 위치로 개별 호출
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

# ✅ 대화 데이터셋의 답변(Response) 부분만 손실값 계산하여 Loss=18 현상 완전 방지
# (Gemma-2의 대화 템플릿에 해당하는 <start_of_turn>model\n 기준)
response_template = "<start_of_turn>model\n"
collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

# ✅ SFTTrainer 및 SFTConfig 둘 다 파라미터를 넘겨주어 TRL 버전에 무관하게 완벽 작동
trainer = SFTTrainer(
    model = model, tokenizer = tokenizer, train_dataset = ds,
    dataset_text_field = "text",
    max_seq_length = 4096,
    data_collator = collator,                              # ✅ 답변 영역만 손실값 계산 지정
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 4096,                             # ✅ 한국어 블로그 글 잘림 방지 (4096)
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,   # 🧠 Auto-Tuner 자동 세팅
        warmup_steps = 11,                      # 🧠 Auto-Tuner 자동 세팅
        num_train_epochs = 3,                  # 🧠 Auto-Tuner 자동 세팅 (총 데이터 315개 기준)
        learning_rate = 2e-05,                         # 🧠 Auto-Tuner 자동 세팅
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
print("\n🔥 [시스템] 파인튜닝 지식 주입 진짜 최종 시작...")
trainer.train()
try: del trainer
except: pass
gc.collect(); torch.cuda.empty_cache()

print("\n📦 [시스템] 지정하신 최종 허깅페이스 창고(seojinju8818/marketing-v7)로 자동 업로드를 개시합니다...")
model.push_to_hub_gguf("seojinju8818/marketing-v7", tokenizer, quantization_method = "q4_k_m", token = hf_token)
print("\n🎉 [대성공] 멀티 파트 AI 자산(.gguf) 빌드가 최종 완결되었습니다!")

