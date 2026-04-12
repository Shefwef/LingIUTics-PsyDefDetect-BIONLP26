# evaluate.py
"""
Evaluates the fine-tuned model on the validation set.
Matches the paper's metric: macro P/R/F1 on classes 1-8, accuracy on all classes.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_MODEL  = "mistralai/Ministral-8B-Instruct-2410"
ADAPTER_DIR = "./checkpoints/ministral-8b-psydef/final"
DATA_DIR    = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LABEL_NAMES = [
    "No Defense",           # 0
    "Action",               # 1
    "Major Image-Dist.",    # 2
    "Disavowal",            # 3
    "Minor Image-Dist.",    # 4
    "Neurotic",             # 5
    "Obsessional",          # 6
    "High-Adaptive",        # 7
    "Needs More Info",      # 8
]


def load_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    print("Model loaded.")
    return model, tokenizer


def get_digit_token_ids(tokenizer):
    """Return token IDs for the characters '0' through '8'."""
    ids = {}
    for i in range(9):
        # Match training where the completion was inserted with a leading space
        # (" [/INST] {completion}"). Use the space-prefixed token so logits
        # correspond to the digit token the model learned.
        tokens = tokenizer.encode(" " + str(i), add_special_tokens=False)
        ids[i] = tokens[0]
    print(f"Digit token IDs: {ids}")
    return ids


def predict_via_logits(model, tokenizer, prompts, digit_ids, device):
    """
    Score each label by the logit of its digit token at the first
    generated position. No full generation needed — fast and exact.
    """
    # Ensure a trailing space after [/INST] so the model's next-token logits
    # correspond to the space-prefixed completion token used during training.
    texts = [f"[INST] {p} [/INST] " for p in prompts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
    # free any unused cached memory to reduce fragmentation / OOM
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # logits at the last real token position  →  what comes next?
    logits = out.logits  # (B, seq_len, vocab)
    preds = []
    for b in range(logits.shape[0]):
        last_pos = int(enc["attention_mask"][b].sum()) - 1
        last_logits = logits[b, last_pos, :]                      # (vocab,)
        scores = torch.tensor([last_logits[digit_ids[i]] for i in range(9)])
        preds.append(int(scores.argmax()))
    return preds


def evaluate(split="val", batch_size=1):
    fname = "val_processed.json" if split == "val" else "train_processed.json"
    with open(f"{DATA_DIR}/{fname}") as f:
        data = json.load(f)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    digit_ids = get_digit_token_ids(tokenizer)

    y_true, y_pred = [], []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {split}"):
        batch   = data[i : i + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels  = [int(s["label"]) for s in batch]

        preds = predict_via_logits(model, tokenizer, prompts, digit_ids, device)
        y_true.extend(labels)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Metrics (paper definition) ────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred) * 100
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(1, 9)),   # positive classes only, matching paper
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"  Results on {split.upper()} set")
    print("=" * 60)
    print(f"  Accuracy               : {acc:.2f}")
    print(f"  Macro Precision (1-8)  : {p  * 100:.2f}")
    print(f"  Macro Recall    (1-8)  : {r  * 100:.2f}")
    print(f"  Macro F1        (1-8)  : {f1 * 100:.2f}")
    print("=" * 60)

    print("\nPer-class breakdown:")
    print(classification_report(
        y_true, y_pred,
        labels=list(range(9)),
        target_names=LABEL_NAMES,
        zero_division=0,
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(9)],
                yticklabels=[f"C{i}" for i in range(9)], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix — {split}")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/cm_{split}.png", dpi=150)
    print(f"Confusion matrix → {RESULTS_DIR}/cm_{split}.png")

    # ── Save numeric results ──────────────────────────────────────────────────
    out = {
        "split": split,
        "accuracy": round(acc, 2),
        "macro_precision": round(p * 100, 2),
        "macro_recall":    round(r * 100, 2),
        "macro_f1":        round(f1 * 100, 2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    with open(f"{RESULTS_DIR}/results_{split}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved → {RESULTS_DIR}/results_{split}.json")
    return out


if __name__ == "__main__":
    evaluate("val")