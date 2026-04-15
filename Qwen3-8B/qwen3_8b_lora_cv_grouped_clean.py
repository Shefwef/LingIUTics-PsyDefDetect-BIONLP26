#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

try:
    from transformers import Trainer
except Exception:  # allow --help without full training deps
    class Trainer:  # type: ignore
        pass


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


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, label_smoothing: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = max(0.0, float(label_smoothing))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        class_weights = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits, labels, weight=class_weights, label_smoothing=self.label_smoothing)
        if return_outputs:
            return loss, outputs
        return loss


def parse_args():
    p = argparse.ArgumentParser(description="Clean grouped-CV LoRA training (single parser/main)")
    p.add_argument("--train_path", type=str, default="input/train.json")
    p.add_argument("--test_path", type=str, default="input/test.json")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", type=str, default="outputs/qwen3_8b_grouped_clean")
    p.add_argument("--num_labels", type=int, default=9)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--single_fold_val_ratio", type=float, default=0.2)
    p.add_argument("--group_by_source", action="store_true", help="Use StratifiedGroupKFold grouped by source id")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.08)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_turns", type=int, default=30)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.005)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--no_epoch_checkpoints", action="store_true")
    p.add_argument("--save_every_percent", type=int, default=25)
    p.add_argument(
        "--resume_completed_folds",
        action="store_true",
        help="Reuse existing fold_i/best_model adapters for completed folds and continue remaining folds.",
    )
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_input_text(example: Dict, max_turns: int) -> str:
    dialogue = example.get("dialogue", [])
    if max_turns > 0:
        dialogue = dialogue[-max_turns:]
    turns = []
    for turn in dialogue:
        speaker = turn.get("speaker", "unknown").upper()
        text = str(turn.get("text", "")).strip()
        turns.append(f"{speaker}: {text}")
    label_guide = "\n".join([f"{k}: {v}" for k, v in LABEL_DESCRIPTIONS.items()])
    return (
        "Task: Classify the SEEKER utterance into one defense label (0-8) based on context.\n"
        f"Defense labels:\n{label_guide}\n\nConversation:\n"
        + "\n".join(turns)
        + "\n\nOutput format: single integer label from 0 to 8."
    )


def make_dataset(rows: List[Dict], tokenizer, max_len: int, max_turns: int, with_labels: bool = True):
    from datasets import Dataset

    texts = [build_input_text(r, max_turns=max_turns) for r in rows]
    data = {"text": texts}
    if with_labels:
        data["labels"] = [int(r["label"]) for r in rows]
    ds = Dataset.from_dict(data)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

    return ds.map(tok, batched=True, remove_columns=["text"])


def compute_metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    pm, rm, fm, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
        "macro_f1": float(fm),
        "weighted_precision": float(pw),
        "weighted_recall": float(rw),
        "weighted_f1": float(fw),
    }


def get_class_weights(labels: List[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    inv_sqrt = 1.0 / np.sqrt(counts)
    normalized = inv_sqrt * (num_labels / inv_sqrt.sum())
    return torch.tensor(normalized, dtype=torch.float32)


def build_model_and_tokenizer(args):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

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

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer, bf16_ok


def load_saved_fold_model_and_tokenizer(best_model_dir: Path, args):
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir), trust_remote_code=args.trust_remote_code)
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

    model = AutoPeftModelForSequenceClassification.from_pretrained(
        str(best_model_dir),
        num_labels=args.num_labels,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, bf16_ok


def main():
    from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments

    args = parse_args()
    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_rows = read_json(args.train_path)
    test_rows = read_json(args.test_path)

    labels = np.array([int(x["label"]) for x in train_rows])
    groups = np.array([str(x.get("synthetic_from") or x.get("id")) for x in train_rows])

    if args.num_folds <= 1:
        tr_idx, va_idx = train_test_split(np.arange(len(train_rows)), test_size=args.single_fold_val_ratio, random_state=args.seed, stratify=labels)
        split_indices = [(np.array(tr_idx), np.array(va_idx))]
        total_folds = 1
    else:
        if args.group_by_source:
            splitter = StratifiedGroupKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
            split_indices = list(splitter.split(np.arange(len(train_rows)), labels, groups=groups))
            print("Using leakage-safe grouped CV split (StratifiedGroupKFold)")
        else:
            splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
            split_indices = list(splitter.split(np.arange(len(train_rows)), labels))
        total_folds = args.num_folds

    oof_logits = np.zeros((len(train_rows), args.num_labels), dtype=np.float32)
    test_prob_sum = np.zeros((len(test_rows), args.num_labels), dtype=np.float32)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(split_indices, start=1):
        print(f"\n===== Fold {fold}/{total_folds} =====")
        fold_dir = out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir = fold_dir / "best_model"

        train_fold = [train_rows[i] for i in tr_idx]
        valid_fold = [train_rows[i] for i in va_idx]

        if args.resume_completed_folds and (best_model_dir / "adapter_model.safetensors").exists():
            print(f"Reusing completed fold {fold} from {best_model_dir}")
            model, tokenizer, bf16_ok = load_saved_fold_model_and_tokenizer(best_model_dir, args)
            valid_ds = make_dataset(valid_fold, tokenizer, args.max_len, args.max_turns, with_labels=True)
            test_ds = make_dataset(test_rows, tokenizer, args.max_len, args.max_turns, with_labels=False)

            pred_args = TrainingArguments(
                output_dir=str(fold_dir / "resume_predict"),
                do_train=False,
                do_eval=False,
                per_device_eval_batch_size=args.eval_batch_size,
                report_to="none",
                bf16=bf16_ok,
                fp16=not bf16_ok,
                dataloader_num_workers=args.dataloader_num_workers,
            )
            predictor = Trainer(
                model=model,
                args=pred_args,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
            )

            val_logits = predictor.predict(valid_ds).predictions
            oof_logits[va_idx] = val_logits
            fm = compute_metrics_from_logits(val_logits, np.array([x["label"] for x in valid_fold]))
            fm["fold"] = fold
            fold_metrics.append(fm)
            print(f"Fold {fold} metrics (reused): {fm}")

            test_logits = predictor.predict(test_ds).predictions
            test_prob_sum += torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

            del predictor, model
            gc.collect()
            torch.cuda.empty_cache()
            continue

        model, tokenizer, bf16_ok = build_model_and_tokenizer(args)
        train_ds = make_dataset(train_fold, tokenizer, args.max_len, args.max_turns, with_labels=True)
        valid_ds = make_dataset(valid_fold, tokenizer, args.max_len, args.max_turns, with_labels=True)
        test_ds = make_dataset(test_rows, tokenizer, args.max_len, args.max_turns, with_labels=False)
        class_weights = get_class_weights([x["label"] for x in train_fold], args.num_labels)

        training_args = TrainingArguments(
            output_dir=str(fold_dir),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            save_strategy="no" if args.no_epoch_checkpoints else "steps",
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=not args.no_epoch_checkpoints,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            eval_strategy="epoch" if args.no_epoch_checkpoints else "steps",
            logging_steps=20,
            lr_scheduler_type="cosine",
            report_to="none",
            bf16=bf16_ok,
            fp16=not bf16_ok,
            gradient_checkpointing=True,
            dataloader_num_workers=args.dataloader_num_workers,
        )

        callbacks = []
        if args.patience and args.patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
            compute_metrics=lambda ep: compute_metrics_from_logits(ep[0], ep[1]),
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            callbacks=callbacks,
        )

        trainer.train()

        val_pred = trainer.predict(valid_ds)
        val_logits = val_pred.predictions
        oof_logits[va_idx] = val_logits
        fm = compute_metrics_from_logits(val_logits, np.array([x["label"] for x in valid_fold]))
        fm["fold"] = fold
        fold_metrics.append(fm)
        print(f"Fold {fold} metrics: {fm}")

        test_logits = trainer.predict(test_ds).predictions
        test_prob_sum += torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

        trainer.save_model(str(fold_dir / "best_model"))
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

    oof_metrics = compute_metrics_from_logits(oof_logits, labels)
    print("\n===== CV OOF Metrics =====")
    print(oof_metrics)

    pd.DataFrame(fold_metrics).to_csv(out / "fold_metrics.csv", index=False)
    (out / "cv_oof_metrics.json").write_text(json.dumps(oof_metrics, indent=2), encoding="utf-8")

    oof_preds = np.argmax(oof_logits, axis=-1)
    oof_probs = torch.softmax(torch.tensor(oof_logits), dim=-1).numpy()
    pd.DataFrame({"id": [x["id"] for x in train_rows], "true_label": labels, "pred_label": oof_preds}).to_csv(out / "oof_predictions.csv", index=False)

    oof_prob_df = pd.DataFrame(oof_probs, columns=[f"prob_{i}" for i in range(args.num_labels)])
    oof_prob_df.insert(0, "true_label", labels)
    oof_prob_df.insert(0, "id", [x["id"] for x in train_rows])
    oof_prob_df.to_csv(out / "oof_prediction_probs.csv", index=False)

    avg_test_probs = test_prob_sum / total_folds
    test_preds = np.argmax(avg_test_probs, axis=-1)
    pd.DataFrame({"id": [x["id"] for x in test_rows], "label": test_preds}).to_csv(out / "test_predictions.csv", index=False)
    prob_df = pd.DataFrame(avg_test_probs, columns=[f"prob_{i}" for i in range(args.num_labels)])
    prob_df.insert(0, "id", [x["id"] for x in test_rows])
    prob_df.to_csv(out / "test_prediction_probs.csv", index=False)

    print("\nDone.")
    print(f"Outputs written to: {out}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
