"""Score all adapter checkpoints under ./checkpoints/ministral-8b-psydef
and report macro F1 (classes 1-8) on the validation set.

Usage: python score_checkpoints.py

Outputs:
 - ./results/checkpoint_scores.json
 - prints the best checkpoint by macro F1
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import precision_recall_fscore_support


BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
CKPT_ROOT = "./checkpoints/ministral-8b-psydef"
DATA_DIR = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_val_data():
    with open(f"{DATA_DIR}/val_processed.json") as f:
        data = json.load(f)
    prompts = [s["prompt"] for s in data]
    labels = [int(s["label"]) for s in data]
    return prompts, labels


def get_digit_token_ids(tokenizer):
    ids = {}
    for i in range(9):
        toks = tokenizer.encode(" " + str(i), add_special_tokens=False)
        ids[i] = toks[0]
    return ids


def predict_via_logits(model, tokenizer, prompts, digit_ids, device, batch_size=8):
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [f"[INST] {p} [/INST] " for p in batch]
        enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
        for b in range(logits.shape[0]):
            last_pos = int(enc["attention_mask"][b].sum()) - 1
            last_logits = logits[b, last_pos, :]
            scores = torch.tensor([last_logits[digit_ids[i]] for i in range(9)], device=last_logits.device)
            preds.append(int(scores.argmax()))
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return preds


def score_adapter(adapter_dir, prompts, labels, batch_size=8):
    print(f"Scoring adapter: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    device = next(model.parameters()).device
    digit_ids = get_digit_token_ids(tokenizer)
    preds = predict_via_logits(model, tokenizer, prompts, digit_ids, device, batch_size=batch_size)

    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(1, 9)), average="macro", zero_division=0
    )
    return round(p * 100, 2), round(r * 100, 2), round(f1 * 100, 2)


def main():
    prompts, labels = load_val_data()
    ckpts = [os.path.join(CKPT_ROOT, d) for d in os.listdir(CKPT_ROOT) if os.path.isdir(os.path.join(CKPT_ROOT, d))]
    results = {}
    for ckpt in sorted(ckpts):
        try:
            p, r, f1 = score_adapter(ckpt, prompts, labels, batch_size=4)
            results[os.path.basename(ckpt)] = {"precision": p, "recall": r, "macro_f1": f1}
            print(f"  -> P: {p}  R: {r}  F1: {f1}\n")
        except Exception as e:
            print(f"Failed to score {ckpt}: {e}")

    # Save results and print best
    with open(f"{RESULTS_DIR}/checkpoint_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    if results:
        best = max(results.items(), key=lambda kv: kv[1]["macro_f1"])
        print("Best checkpoint:", best[0], best[1])
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
