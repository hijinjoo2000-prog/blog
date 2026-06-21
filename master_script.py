# ==========================================================================
# 👑 PRO부동산 master_dataset.jsonl 멀티 라우팅 최종 스크립트 (v11.0 CoT + train_on_responses_only)
# 🧠 Auto-Tuner: 데이터 21개 기준 → 8에포크 / LR 0.0003 자동 최적화
# ==========================================================================
import gc
import torch
from unsloth import FastModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from google.colab import userdata

try: hf_token = userdata.get('HF_TOKEN')
except Exception: hf_token = True

print("\n🔄 [시스템] 베이스 모델 로딩 중...")
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
    full_finetuning = False,
)

model = FastModel.get_peft_model(
    model,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    finetune_vision_layers = False,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

print("\n📦 [시스템] 지정된 멀티 깃허브 저장소(hijinjoo2000-prog/blog)에서 데이터를 원격 호출합니다...")
import urllib.request
url = "https://raw.githubusercontent.com/hijinjoo2000-prog/blog/main/master_dataset.jsonl"
try:
    gh_token = userdata.get('GITHUB_TOKEN')
    req = urllib.request.Request(url, headers={"Authorization": f"token {gh_token}"})
    print("🔑 GITHUB_TOKEN 보안 인증 연동 성공!")
except Exception:
    req = urllib.request.Request(url)
    print("ℹ️ GITHUB_TOKEN이 Secrets에 없거나 오류가 발생하여 비인증(Public) 호출을 수행합니다.")

try:
    with urllib.request.urlopen(req) as response:
        with open('master_dataset.jsonl', 'wb') as f:
            f.write(response.read())
    ds = load_dataset('json', data_files='master_dataset.jsonl', split='train')
except Exception as e:
    print("❌ 데이터 로드 에러! 저장소가 Private인데 GITHUB_TOKEN이 비어있거나 권한이 없는지 확인하세요.")
    raise e

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

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

# ✅ SFTTrainer 및 SFTConfig 둘 다 파라미터를 넘겨주어 TRL 버전에 무관하게 완벽 작동
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = 1024,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 300,
        learning_rate = 0.0003,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./outputs",
        save_strategy = "no",
        report_to = "none",
    ),
)

# 🎭 응답(assistant)만 학습 — 질문 패턴은 마스킹(효율↑·품질↑)
# ⚠️ 마커는 모델/버전마다 다름(<|turn> vs <start_of_turn>) → 실제 텍스트에서 자동 감지
from unsloth.chat_templates import train_on_responses_only
_t = ds[0]["text"]
_im = "<|turn>user\n" if "<|turn>user" in _t else "<start_of_turn>user\n"
_rm = "<|turn>model\n" if "<|turn>model" in _t else "<start_of_turn>model\n"
trainer = train_on_responses_only(trainer, instruction_part=_im, response_part=_rm)
print(f"✅ 마스킹 마커 자동감지: {_rm.strip()} — 학습 준비 완료")

print("\n🔥 [시스템] 파인튜닝 지식 주입 진짜 최종 시작...")
trainer_stats = trainer.train()
print("🎉 학습 완료! 최종 loss:", round(trainer_stats.training_loss, 4))
print("💡 loss 0.2~0.4면 sweet spot. 너무 낮으면(<0.1) 과적합 — 에포크나 max_steps를 조절해 보세요.")

print("\n🧪 [테스트] 학습이 완료된 모델로 자가 진단 테스트를 가동합니다...")
FastModel.for_inference(model)
def chat(prompt, max_tokens=220):
    msg = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inp = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    if inp["input_ids"][0,0].item() == tokenizer.bos_token_id:
        inp["input_ids"] = inp["input_ids"][:,1:]; inp["attention_mask"] = inp["attention_mask"][:,1:]
    out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"❓ 질문: {prompt}\n💬 답변: {ans}\n" + "─"*58)

chat("내 사업/지식에 대해 아는 걸 알려줘")
chat("너는 무엇을 도와줄 수 있어?")

try: del trainer
except: pass
gc.collect(); torch.cuda.empty_cache()

print("\n📦 [시스템] 지정하신 최종 허깅페이스 창고(seojinju8818/marketing-v8)로 자동 업로드를 개시합니다...")
model.push_to_hub_gguf("seojinju8818/marketing-v8", tokenizer, quantization_method = "q4_k_m", token = hf_token)
print("\n🎉 [대성공] GGUF 빌드 및 업로드가 완료되었습니다!")
