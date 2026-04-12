# DeBERTa-v3-base 5-Fold Kaggle Ready Guide

This guide combines all fixes and improvements discussed so far into one Kaggle execution flow.

Primary goals:
- maximize score with `microsoft/deberta-v3-base`
- use 5-fold stratified CV
- include dialogue context plus target text
- track weighted and macro metrics together
- explicitly track precision and recall
- build fold-weighted ensemble predictions for test
- create `prediction.json` and `prediction.zip`

--------------------------------------------------

## 1) Metric policy (set this first)

Pick one primary metric for model selection.

- If competition metric is weighted F1:
  - `PRIMARY_METRIC = "weighted_f1"`
- If competition metric is macro F1:
  - `PRIMARY_METRIC = "macro_f1"`

Always log all:
- accuracy
- macro_f1, weighted_f1
- macro_precision, macro_recall
- weighted_precision, weighted_recall

--------------------------------------------------

## 2) Kaggle notebook cell order

Run cells in this exact order.

### Cell 1 - Install

```python
import sys
import subprocess
import importlib
from importlib.metadata import version, PackageNotFoundError

def pip_install(args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def ensure_import(import_name, pip_name):
    try:
        importlib.import_module(import_name)
        print(f"{import_name}: already available")
    except Exception:
        print(f"{import_name}: missing -> installing {pip_name}")
        pip_install([pip_name])


# IMPORTANT:
# Do not pin or downgrade numpy/scipy/sklearn/huggingface-hub in Kaggle.
# Kaggle runtime has many preinstalled packages with tight constraints.
# We only install missing essentials and keep the base environment intact.
ensure_import("transformers", "transformers")
ensure_import("datasets", "datasets")
ensure_import("accelerate", "accelerate")

# Detect and repair broken scientific stack (e.g., scipy import failing via numpy.rec).
need_science_repair = False
try:
    import numpy as np
    import scipy
    import sklearn
except Exception as e:
    need_science_repair = True
    print("Scientific stack check failed:", str(e))

if need_science_repair:
    print("Repairing numpy/scipy/scikit-learn with NumPy-2 compatible versions...")
    pip_install([
        "--upgrade",
        "--force-reinstall",
        "numpy>=2.0,<2.2",
        "scipy>=1.13,<1.15",
        "scikit-learn>=1.6,<1.8",
    ])
    print("Repair complete. You MUST restart runtime now, then run from Cell 1 again.")
    raise SystemExit("Runtime restart required after scientific stack repair")

importlib.invalidate_caches()

# Quick sanity print so you can see what runtime actually resolved.
import torch
import numpy as np
import scipy
import sklearn
import transformers, datasets, accelerate


def safe_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not-installed"


print("torch:", torch.__version__)
print("torchvision(pkg):", safe_version("torchvision"))
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("If any import above fails, use Kaggle 'Factory reset runtime', then rerun from Cell 1.")
print("CELL 1 DONE")
```

### Cell 2 - Imports and config

```python
import os
import inspect
import gc
import json
import zipfile
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset
from huggingface_hub import login, whoami
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

TRAIN_PATH = "/kaggle/input/datasets/shefayateshamsadib/psydefdetect-kaggle-pack/psydefdetect-kaggle-pack/data/train.json"
TEST_PATH = "/kaggle/input/datasets/shefayateshamsadib/psydefdetect-kaggle-pack/psydefdetect-kaggle-pack/data/test.json"

MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 9
MAX_LEN = 320
N_SPLITS = 5

PRIMARY_METRIC = "macro_f1"   # leaderboard-first default

# Strict leaderboard gates (adjust only if competition rules differ)
MIN_OOF_MACRO_F1 = 0.300
MIN_CLASS8_RECALL = 0.05

EPOCHS = 6
LR = 1.0e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
TRAIN_BS = 8
EVAL_BS = 16
GRAD_ACC = 2
EARLY_STOP_PATIENCE = 2

# Stability-first precision mode (avoids AMP unscale FP16 runtime issues on some Kaggle stacks)
USE_FP16 = False
USE_BF16 = False

BASE_OUT = "/kaggle/working/deberta_v3_5fold"
os.makedirs(BASE_OUT, exist_ok=True)

# Run control
CLEAN_START = True
RESUME_IF_AVAILABLE = True

if CLEAN_START and os.path.exists(BASE_OUT):
    import shutil
    shutil.rmtree(BASE_OUT, ignore_errors=True)
    os.makedirs(BASE_OUT, exist_ok=True)
    print("Clean start enabled: cleared previous BASE_OUT")

# Hugging Face auth (prevents unauthenticated warning when token is available)
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        user = whoami()
        print("HF auth: OK as", user.get("name", "unknown"))
    except Exception as e:
        print("HF auth warning:", str(e))
else:
    print("HF auth: no token found in env (HF_TOKEN or HUGGINGFACE_HUB_TOKEN)")

hf_auth_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("USE_FP16:", USE_FP16)
print("USE_BF16:", USE_BF16)
print("CLEAN_START:", CLEAN_START)
print("RESUME_IF_AVAILABLE:", RESUME_IF_AVAILABLE)
print("PRIMARY_METRIC:", PRIMARY_METRIC)
print("MIN_OOF_MACRO_F1:", MIN_OOF_MACRO_F1)
print("MIN_CLASS8_RECALL:", MIN_CLASS8_RECALL)

print("CELL 2 DONE")
```

### Cell 3 - Load data and build context-aware input

```python
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
with open(TEST_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Train size:", len(train_data))
print("Test size :", len(test_data))

all_labels = [x["label"] for x in train_data]
print("Label distribution:", Counter(all_labels))


def build_input(example, max_turns=6):
    dialogue = example.get("dialogue", [])[-max_turns:]
    parts = []

    for turn in dialogue:
        speaker = str(turn.get("speaker", "unknown")).strip().lower()
        text = str(turn.get("text", "")).strip()
        parts.append(f"{speaker}: {text}")

    context = "\n".join(parts)
    target = str(example.get("current_text", "")).strip()

    return (
        "[CONTEXT]\n"
        f"{context}\n\n"
        "[TARGET]\n"
        f"{target}"
    )

train_texts_all = [build_input(x) for x in train_data]
train_labels_all = [int(x["label"]) for x in train_data]
test_texts_all = [build_input(x) for x in test_data]

print("Sample input preview:\n")
print(train_texts_all[0][:700])
print("CELL 3 DONE")
```

### Cell 4 - Tokenizer, collator, metrics, weighted trainer

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **hf_auth_kwargs)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Keep weighted CE in fp32 for numerical stability.
        weight = self.class_weights.to(device=logits.device, dtype=torch.float32)
        logits_for_loss = logits.float()
        labels = labels.long()
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits_for_loss, labels)

        return (loss, outputs) if return_outputs else loss


print("CELL 4 DONE")
```

### Cell 5 - 5-fold training + OOF + fold test probabilities

```python
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train_data), dtype=np.int64)
oof_probs = np.zeros((len(train_data), NUM_LABELS), dtype=np.float32)

test_probs_by_fold = []
fold_scores = []
fold_summaries = []
completed_folds = set()

oof_preds_path = os.path.join(BASE_OUT, "oof_preds.npy")
oof_probs_path = os.path.join(BASE_OUT, "oof_probs.npy")
fold_scores_path = os.path.join(BASE_OUT, "fold_scores.json")
fold_summaries_path = os.path.join(BASE_OUT, "fold_summaries.json")
completed_folds_path = os.path.join(BASE_OUT, "completed_folds.json")

if RESUME_IF_AVAILABLE and os.path.exists(completed_folds_path):
    try:
        if os.path.exists(oof_preds_path):
            oof_preds = np.load(oof_preds_path)
        if os.path.exists(oof_probs_path):
            oof_probs = np.load(oof_probs_path)
        if os.path.exists(fold_scores_path):
            with open(fold_scores_path, "r", encoding="utf-8") as f:
                fold_scores = json.load(f)
        if os.path.exists(fold_summaries_path):
            with open(fold_summaries_path, "r", encoding="utf-8") as f:
                fold_summaries = json.load(f)
        with open(completed_folds_path, "r", encoding="utf-8") as f:
            completed_folds = set(json.load(f))
        print("Resume state loaded. Completed folds:", sorted(completed_folds))
    except Exception as e:
        print("Resume state load failed; starting loop fresh. Reason:", str(e))
        completed_folds = set()

for completed_fold in sorted(completed_folds):
    prob_path = os.path.join(BASE_OUT, f"test_prob_fold_{completed_fold}.npy")
    if os.path.exists(prob_path):
        test_probs_by_fold.append(np.load(prob_path))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_texts_all, train_labels_all), start=1):
    if fold in completed_folds:
        print(f"Skipping fold {fold}; already completed in previous run")
        continue

    print("\n" + "=" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 70)

    X_tr = [train_texts_all[i] for i in tr_idx]
    y_tr = [train_labels_all[i] for i in tr_idx]
    X_va = [train_texts_all[i] for i in va_idx]
    y_va = [train_labels_all[i] for i in va_idx]

    # class weights from fold-train only
    classes = np.array(sorted(set(y_tr)))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y_tr))
    # Stabilize weighted CE: soften and clip extreme class weights.
    cw = np.power(cw, 0.5)
    cw = np.clip(cw, 0.7, 2.5)
    cw = cw / cw.mean()
    class_weights = torch.tensor(cw, dtype=torch.float)

    train_ds = Dataset.from_dict({"text": X_tr, "labels": y_tr})
    valid_ds = Dataset.from_dict({"text": X_va, "labels": y_va})
    test_ds = Dataset.from_dict({"text": test_texts_all})

    train_ds = train_ds.map(tokenize_batch, batched=True)
    valid_ds = valid_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Compatibility: newer transformers prefer `dtype`, older versions accept `torch_dtype`.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            dtype=torch.float32,
            **hf_auth_kwargs,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.float32,
            **hf_auth_kwargs,
        )

    fold_dir = os.path.join(BASE_OUT, f"fold_{fold}")

    # Safe fallback if Cell 2 was not re-run after edits.
    warmup_steps_value = int(globals().get("WARMUP_STEPS", 120))
    if "WARMUP_STEPS" not in globals():
        print("WARMUP_STEPS not found in current session. Using fallback warmup_steps=120")

    args_kwargs = dict(
        output_dir=fold_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps_value,
        # Disabled because some transformers+deberta combinations reload checkpoints with
        # LayerNorm key mismatches (beta/gamma vs weight/bias), causing unstable behavior.
        load_best_model_at_end=False,
        metric_for_best_model=PRIMARY_METRIC,
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,
        max_grad_norm=1.0,
        fp16=USE_FP16,
        bf16=USE_BF16,
        report_to="none",
        seed=SEED + fold,
    )

    # Compatibility: only pass `save_safetensors` when supported by installed transformers.
    # if "save_safetensors" in inspect.signature(TrainingArguments.__init__).parameters:
    #     args_kwargs["save_safetensors"] = False

    args = TrainingArguments(**args_kwargs)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[],
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed in fold {fold}: {str(e)}")
        np.save(oof_preds_path, oof_preds)
        np.save(oof_probs_path, oof_probs)
        with open(fold_scores_path, "w", encoding="utf-8") as f:
            json.dump(fold_scores, f, indent=2)
        with open(fold_summaries_path, "w", encoding="utf-8") as f:
            json.dump(fold_summaries, f, indent=2)
        with open(completed_folds_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(completed_folds)), f, indent=2)
        raise

    # validation predictions
    va_output = trainer.predict(valid_ds)
    va_logits = va_output.predictions
    va_prob = torch.softmax(torch.tensor(va_logits), dim=1).numpy()
    va_pred = np.argmax(va_prob, axis=1)

    oof_preds[va_idx] = va_pred
    oof_probs[va_idx] = va_prob

    fold_metric = float(
        compute_metrics((va_logits, np.array(y_va)))[PRIMARY_METRIC]
    )
    fold_scores.append(fold_metric)

    fold_summary = compute_metrics((va_logits, np.array(y_va)))
    fold_summary["fold"] = fold
    fold_summaries.append(fold_summary)

    print(f"Fold {fold} {PRIMARY_METRIC}: {fold_metric:.4f}")
    print(
        f"Fold {fold} macro P/R/F1: "
        f"{fold_summary['macro_precision']:.4f} / {fold_summary['macro_recall']:.4f} / {fold_summary['macro_f1']:.4f}"
    )
    print(
        f"Fold {fold} weighted P/R/F1: "
        f"{fold_summary['weighted_precision']:.4f} / {fold_summary['weighted_recall']:.4f} / {fold_summary['weighted_f1']:.4f}"
    )

    # test predictions for this fold
    te_output = trainer.predict(test_ds)
    te_logits = te_output.predictions
    te_prob = torch.softmax(torch.tensor(te_logits), dim=1).numpy()
    test_probs_by_fold.append(te_prob)
    np.save(os.path.join(BASE_OUT, f"test_prob_fold_{fold}.npy"), te_prob)

    completed_folds.add(fold)
    np.save(oof_preds_path, oof_preds)
    np.save(oof_probs_path, oof_probs)
    with open(fold_scores_path, "w", encoding="utf-8") as f:
        json.dump(fold_scores, f, indent=2)
    with open(fold_summaries_path, "w", encoding="utf-8") as f:
        json.dump(fold_summaries, f, indent=2)
    with open(completed_folds_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(completed_folds)), f, indent=2)

    # memory cleanup
    del trainer, model, train_ds, valid_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    print(f"FOLD {fold} DONE")

print("\nFinished all folds.")
print("Completed folds:", sorted(list(completed_folds)))
print("CELL 5 DONE")
```

### Cell 6 - OOF metrics, precision/recall reports, confusion matrix

```python
def summarize_oof(y_true, y_pred):
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,
    }
    return out


oof_metrics = summarize_oof(np.array(train_labels_all), oof_preds)
oof_report = classification_report(train_labels_all, oof_preds, digits=4, output_dict=True, zero_division=0)
class8_recall = float(oof_report.get("8", {}).get("recall", 0.0))

gate_pass_macro = oof_metrics["macro_f1"] >= MIN_OOF_MACRO_F1
gate_pass_class8 = class8_recall >= MIN_CLASS8_RECALL
leaderboard_gate_pass = gate_pass_macro and gate_pass_class8

oof_metrics["class8_recall"] = class8_recall
oof_metrics["leaderboard_gate_pass"] = bool(leaderboard_gate_pass)

print("\nOOF metrics:")
for k, v in oof_metrics.items():
    if isinstance(v, bool):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v:.4f}")

print("\nLeaderboard gates:")
print(f"  macro_f1 >= {MIN_OOF_MACRO_F1:.3f}: {gate_pass_macro}")
print(f"  class8_recall >= {MIN_CLASS8_RECALL:.3f}: {gate_pass_class8}")
print(f"  OVERALL: {leaderboard_gate_pass}")

print("\nPer-fold summaries:")
print(pd.DataFrame(fold_summaries))

print("\nClassification report (OOF):")
print(classification_report(train_labels_all, oof_preds, digits=4, zero_division=0))

cm = confusion_matrix(train_labels_all, oof_preds)
print("\nConfusion matrix shape:", cm.shape)
print(cm)

with open(os.path.join(BASE_OUT, "oof_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(oof_metrics, f, indent=2)

print("CELL 6 DONE")
```

### Cell 7 - Fold-weighted ensemble for test predictions

```python
# Normalize fold weights from primary metric
weights = np.array(fold_scores, dtype=np.float32)
weights = np.maximum(weights, 1e-8)
weights = weights / weights.sum()

print("Fold scores:", fold_scores)
print("Normalized fold weights:", weights)

ensemble_test_prob = np.zeros_like(test_probs_by_fold[0], dtype=np.float32)
for w, prob in zip(weights, test_probs_by_fold):
    ensemble_test_prob += w * prob

ensemble_test_pred = np.argmax(ensemble_test_prob, axis=1).astype(int)

print("Number of test predictions:", len(ensemble_test_pred))
print("First 20 predictions:", ensemble_test_pred[:20])
print("CELL 7 DONE")
```

### Cell 8 - Build submission files

```python
# IMPORTANT: keep this format only if competition requires id + label
submission = [
    {"id": ex["id"], "label": int(pred)}
    for ex, pred in zip(test_data, ensemble_test_pred)
]

if not bool(oof_metrics.get("leaderboard_gate_pass", False)):
    raise RuntimeError(
        "Submission blocked: leaderboard gate failed. "
        "Need macro_f1 and class-8 recall above thresholds."
    )

prediction_json = "/kaggle/working/prediction.json"
prediction_zip = "/kaggle/working/prediction.zip"

with open(prediction_json, "w", encoding="utf-8") as f:
    json.dump(submission, f, indent=2, ensure_ascii=False)

with zipfile.ZipFile(prediction_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(prediction_json, arcname="prediction.json")

print("Created:", prediction_json)
print("Created:", prediction_zip)
print("First 5 rows:", submission[:5])
print("Length check:", len(submission), "==", len(test_data))
print("CELL 8 DONE")
```

--------------------------------------------------

## 3) Decision gates before submit

Use OOF metrics for go or no-go.

Recommended gate:
- submit if OOF `PRIMARY_METRIC` improves your previous best by >= 0.005
- and no severe minority-class precision or recall collapse

If OOF is flat or worse:
- do not submit
- run one controlled change only

--------------------------------------------------

## 4) Controlled tuning order (one change at a time)

Try only one per run:
1. `MAX_LEN`: 320 -> 384
2. `LR`: 1.5e-5 -> 1.2e-5
3. `label_smoothing_factor`: 0.03 -> 0.05
4. `WARMUP_STEPS`: 120 -> 160

Avoid changing more than one of these in the same run.

--------------------------------------------------

## 5) Precision/recall performance targeting

If weighted F1 is okay but macro recall is low:
- keep LR stable, increase context quality, inspect confusion matrix

If precision is low and false positives are high:
- reduce class-weight aggressiveness slightly
- avoid longer training if fold metrics peak early

If both are unstable across folds:
- keep config fixed and run another seed (e.g., 52)
- compare OOF, not a single fold

--------------------------------------------------

## 6) Notes

- Warning about newly initialized classifier head is normal.
- Save only best model checkpoint per fold to reduce runtime/storage issues.
- Do not trust one fold spike; trust OOF summary and fold consistency.
- If you ever see a torch/torchvision CUDA mismatch error, start a fresh Kaggle runtime session, then run Cell 1 from this guide (do not force-upgrade torch or torchvision).
