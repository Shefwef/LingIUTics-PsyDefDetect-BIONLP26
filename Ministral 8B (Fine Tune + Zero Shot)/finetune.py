# finetune.py
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  change MODEL_NAME to switch models
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME  = "mistralai/Ministral-8B-Instruct-2410"   # best in paper (F1 31.48)
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # 2nd best  (F1 30.51)
# MODEL_NAME = "internlm/internlm3-8b-instruct"          # (F1 30.53)

DATA_DIR    = "./data"
# Allow overriding the output directory via environment (useful to point to a larger disk)
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", "./checkpoints/ministral-8b-psydef")
MAX_LENGTH  = 2048

# ── Paper's exact hyperparameters ────────────────────────────────────────────
EPOCHS      = 10
BATCH_SIZE  = 1          # per_device_train_batch_size
GRAD_ACCUM  = 8          # effective batch size = 8
LR          = 1e-4
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Data ───────────────────────────────────────────────────────────────────
def load_data():
    with open(f"{DATA_DIR}/train_processed.json") as f:
        train_raw = json.load(f)
    with open(f"{DATA_DIR}/val_processed.json") as f:
        val_raw = json.load(f)

    def to_training_text(sample):
        """
        Combine prompt + completion into a single training string.
        Ministral/Mistral instruct format.
        """
        prompt     = sample["prompt"]
        completion = sample["completion"]   # single digit string "0"-"8"
        # The model sees the prompt and must predict the digit
        text = f"[INST] {prompt} [/INST] {completion}</s>"
        return {"text": text}

    train_ds = Dataset.from_list([to_training_text(s) for s in train_raw])
    val_ds   = Dataset.from_list([to_training_text(s) for s in val_raw])

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_ds, val_ds


# ── 2. Model ──────────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    # 4-bit QLoRA — standard approach for 8B models on single consumer GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use left padding to match evaluation/prediction scripts and typical
    # generation conventions (keeps logits indexing consistent).
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Prepare for k-bit training (adds gradient checkpointing hooks)
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── 3. Train ──────────────────────────────────────────────────────────────────
def train():
    train_ds, val_ds = load_data()
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # ── Paper's exact hyperparameters ──────────────────────────────────
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        # ───────────────────────────────────────────────────────────────────

        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,

        logging_steps=10,
        save_total_limit=3,
        report_to="none",

        max_seq_length=MAX_LENGTH,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting training...")

    # Try to resume from the latest checkpoint saved in OUTPUT_DIR (if any)
    last_ckpt = None
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
        if ckpts:
            ckpts_sorted = sorted(
                ckpts,
                key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x))
            )
            last_ckpt = os.path.join(OUTPUT_DIR, ckpts_sorted[-1])
            print(f"Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    print("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Done. Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()