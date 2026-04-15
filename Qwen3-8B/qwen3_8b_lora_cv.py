#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

LABEL_DESCRIPTIONS = {
    0: "No Defense / Neutral Utterance",
    1: "Action Defenses",
    2: "Major Image-Distorting Defenses",
    3: "Disavowal Defenses",
    4: "Minor Image-Distorting Defenses",
    5: "Neurotic Defenses",
    6: "Obsessional Defenses",
    7: "High-Adaptive Defenses",
    8: "Unclear / Needs More Information",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", default="input/train.json")
    p.add_argument("--test_path", default="input/test.json")
    p.add_argument("--model_name", default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", default="outputs/qwen3_8b_lora_top5_pilot3")
    p.add_argument("--num_labels", type=int, default=9)
    p.add_argument("--num_folds", type=int, default=3)
    p.add_argument("--single_fold_val_ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=float, default=12)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_turns", type=int, default=30)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--lora_r", type=int, default=256)
    p.add_argument("--lora_alpha", type=int, default=512)
    p.add_argument("--lora_dropout", type=float, default=0.12)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--loss_type", choices=["weighted_ce", "focal"], default="focal")
    p.add_argument("--focal_gamma", type=float, default=1.8)
    p.add_argument("--oversample_minority", action="store_true")
    p.add_argument("--oversample_target_ratio", type=float, default=0.35)
    p.add_argument("--oversample_max_multiplier", type=float, default=4.0)
    p.add_argument("--oversample_labels", default="1,2,3,4,5,6,8")
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--save_every_percent", type=int, default=25)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_epoch_checkpoints", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(example, max_turns):
    dialogue = example.get("dialogue", [])[-max_turns:]
    turns = []
    for turn in dialogue:
        speaker = str(turn.get("speaker", "unknown")).upper()
        text = str(turn.get("text", "")).strip()
        turns.append(f"{speaker}: {text}")
    label_guide = "\n".join([f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()])
    return (
        "Task: Classify the SEEKER utterance into one defense label (0-8) based on context.\n"
        "Defense labels:\n"
        f"{label_guide}\n\n"
        "Conversation:\n"
        + "\n".join(turns)
        + "\n\nOutput format: single integer label from 0 to 8."
    )


def make_dataset(rows, tokenizer, max_len, max_turns, with_labels=True):
    data = {"text": [build_prompt(r, max_turns) for r in rows]}
    if with_labels:
        data["labels"] = [int(r["label"]) for r in rows]
    ds = Dataset.from_dict(data)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

    return ds.map(tok, batched=True, remove_columns=["text"])


def compute_metrics_from_logits(logits, labels):
    preds = np.argmax(logits, axis=-1)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f1_m),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
    }


def get_class_weights(labels, num_labels):
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / np.sqrt(counts)
    inv = inv * (num_labels / inv.sum())
    return torch.tensor(inv, dtype=torch.float32)


def parse_label_list(raw, num_labels):
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = int(tok)
        if 0 <= v < num_labels:
            out.append(v)
    return sorted(set(out))


def oversample_minority_rows(rows, num_labels, target_ratio, max_multiplier, allowed_labels, seed):
    if not rows:
        return rows
    rng = random.Random(seed)
    grouped = {k: [] for k in range(num_labels)}
    for r in rows:
        grouped[int(r["label"])].append(r)
    counts = {k: len(v) for k, v in grouped.items()}
    majority = max(counts.values()) if counts else 0
    target_min = int(math.ceil(majority * max(0.0, min(1.0, float(target_ratio)))))
    max_multiplier = max(1.0, float(max_multiplier))
    aug = list(rows)
    for label in allowed_labels:
        cur = counts.get(label, 0)
        if cur <= 0:
            continue
        desired = max(cur, target_min)
        desired = min(desired, int(math.floor(cur * max_multiplier)))
        add_n = max(0, desired - cur)
        for _ in range(add_n):
            aug.append(rng.choice(grouped[label]))
    rng.shuffle(aug)
    return aug


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, label_smoothing=0.0, loss_type="weighted_ce", focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = float(label_smoothing)
        self.loss_type = str(loss_type)
        self.focal_gamma = float(focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.class_weights.to(logits.device)
        if self.loss_type == "focal":
            ce = F.cross_entropy(logits, labels, weight=w, reduction="none", label_smoothing=self.label_smoothing)
            probs = torch.softmax(logits, dim=-1)
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-8, 1.0)
            loss = ((1 - pt) ** self.focal_gamma * ce).mean()
        else:
            loss = F.cross_entropy(logits, labels, weight=w, label_smoothing=self.label_smoothing)
        return (loss, outputs) if return_outputs else loss


def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_cfg)
    return model, tokenizer, bf16_ok


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_json(args.train_path)
    test_rows = read_json(args.test_path)
    if args.max_train_samples is not None:
        train_rows = train_rows[: args.max_train_samples]

    labels = np.array([int(r["label"]) for r in train_rows])
    if args.num_folds <= 1:
        tr_idx, va_idx = train_test_split(np.arange(len(train_rows)), test_size=args.single_fold_val_ratio, stratify=labels, random_state=args.seed)
        splits = [(np.array(tr_idx), np.array(va_idx))]
        total_folds = 1
    else:
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        splits = list(skf.split(np.arange(len(train_rows)), labels))
        total_folds = args.num_folds

    test_prob_sum = np.zeros((len(test_rows), args.num_labels), dtype=np.float32)
    oof_logits = np.zeros((len(train_rows), args.num_labels), dtype=np.float32)
    fold_metrics = []
    oversample_labels = parse_label_list(args.oversample_labels, args.num_labels)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n===== Fold {fold}/{total_folds} =====", flush=True)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_fold = [train_rows[i] for i in tr_idx]
        valid_fold = [train_rows[i] for i in va_idx]
        if args.oversample_minority:
            before = {k: 0 for k in range(args.num_labels)}
            for r in train_fold:
                before[int(r["label"])] += 1
            train_fold = oversample_minority_rows(
                train_fold,
                args.num_labels,
                args.oversample_target_ratio,
                args.oversample_max_multiplier,
                oversample_labels,
                seed=args.seed + fold,
            )
            after = {k: 0 for k in range(args.num_labels)}
            for r in train_fold:
                after[int(r["label"])] += 1
            print(f"Fold {fold} oversampling enabled")
            print(f"Fold {fold} class counts before: {before}")
            print(f"Fold {fold} class counts after : {after}")

        model, tokenizer, bf16_ok = build_model_and_tokenizer(args)
        train_ds = make_dataset(train_fold, tokenizer, args.max_len, args.max_turns, with_labels=True)
        valid_ds = make_dataset(valid_fold, tokenizer, args.max_len, args.max_turns, with_labels=True)
        test_ds = make_dataset(test_rows, tokenizer, args.max_len, args.max_turns, with_labels=False)
        class_weights = get_class_weights([int(r["label"]) for r in train_fold], args.num_labels)

        save_strategy = "no" if args.no_epoch_checkpoints else "steps"
        updates_per_epoch = max(1, math.ceil(len(train_fold) / (args.train_batch_size * args.grad_accum)))
        total_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
        save_every_percent = max(1, min(100, int(args.save_every_percent)))
        save_steps = max(1, math.ceil(total_updates * (save_every_percent / 100.0)))

        ta_kwargs = dict(
            output_dir=str(fold_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            save_strategy=save_strategy,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=not args.no_epoch_checkpoints,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_steps=20,
            lr_scheduler_type="cosine",
            report_to="none",
            bf16=bf16_ok,
            fp16=not bf16_ok,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            dataloader_num_workers=args.dataloader_num_workers,
        )
        if args.no_epoch_checkpoints:
            eval_strategy_value = "epoch"
        else:
            eval_strategy_value = "steps"
            ta_kwargs["save_steps"] = save_steps
            ta_kwargs["eval_steps"] = save_steps
        try:
            ta_kwargs["eval_strategy"] = eval_strategy_value
            training_args = TrainingArguments(**ta_kwargs)
        except TypeError:
            ta_kwargs.pop("eval_strategy")
            ta_kwargs["evaluation_strategy"] = eval_strategy_value
            training_args = TrainingArguments(**ta_kwargs)
        if hasattr(training_args, "save_only_model"):
            training_args.save_only_model = True
        if hasattr(training_args, "save_safetensors"):
            training_args.save_safetensors = True

        callbacks = []
        if args.patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
            compute_metrics=lambda p: compute_metrics_from_logits(p.predictions, p.label_ids),
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
            callbacks=callbacks,
        )
        trainer = WeightedTrainer(**trainer_kwargs)
        print(
            f"Starting fold {fold} from scratch | total updates ~{total_updates} | checkpoint every {save_every_percent}% (~{save_steps} steps)",
            flush=True,
        )
        trainer.train()

        val_pred = trainer.predict(valid_ds)
        val_logits = val_pred.predictions
        oof_logits[va_idx] = val_logits
        fold_metric = compute_metrics_from_logits(val_logits, np.array([int(r["label"]) for r in valid_fold]))
        fold_metric["fold"] = fold
        fold_metrics.append(fold_metric)
        print(f"Fold {fold} metrics: {fold_metric}", flush=True)

        test_pred = trainer.predict(test_ds)
        test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=-1).numpy()
        test_prob_sum += test_probs
        trainer.save_model(str(fold_dir / "best_model"))

        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_metrics = compute_metrics_from_logits(oof_logits, labels)
    print("\n===== CV OOF Metrics =====", flush=True)
    print(oof_metrics, flush=True)

    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    with open(output_dir / "cv_oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(oof_metrics, f, indent=2)

    oof_preds = np.argmax(oof_logits, axis=-1)
    pd.DataFrame({"id": [r["id"] for r in train_rows], "true_label": labels, "pred_label": oof_preds}).to_csv(output_dir / "oof_predictions.csv", index=False)
    oof_probs = torch.softmax(torch.tensor(oof_logits), dim=-1).numpy()
    oof_prob_df = pd.DataFrame(oof_probs, columns=[f"prob_{i}" for i in range(args.num_labels)])
    oof_prob_df.insert(0, "true_label", labels)
    oof_prob_df.insert(0, "id", [r["id"] for r in train_rows])
    oof_prob_df.to_csv(output_dir / "oof_prediction_probs.csv", index=False)
    np.save(output_dir / "oof_logits.npy", oof_logits)

    avg_test_probs = test_prob_sum / total_folds
    test_preds = np.argmax(avg_test_probs, axis=-1)
    pd.DataFrame({"id": [r["id"] for r in test_rows], "label": test_preds}).to_csv(output_dir / "test_predictions.csv", index=False)
    prob_df = pd.DataFrame(avg_test_probs, columns=[f"prob_{i}" for i in range(args.num_labels)])
    prob_df.insert(0, "id", [r["id"] for r in test_rows])
    prob_df.to_csv(output_dir / "test_prediction_probs.csv", index=False)

    summary = {
        "model_name": args.model_name,
        "num_folds": total_folds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "max_len": args.max_len,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "cv_oof_metrics": oof_metrics,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
