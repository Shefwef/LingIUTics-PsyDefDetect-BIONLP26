# predict.py
"""
Generate predictions on the test set for submission.
Output format:  predictions.json
  [ {"id": "test_00001", "label": 7}, ... ]
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL  = "mistralai/Ministral-8B-Instruct-2410"
ADAPTER_DIR = "./checkpoints/ministral-8b-psydef/final"
DATA_DIR    = "./data"
OUTPUT_FILE = "./predictions.json"
BATCH_SIZE  = 1


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def get_digit_token_ids(tokenizer):
    # Use space-prefixed digits to match training completions
    return {i: tokenizer.encode(" " + str(i), add_special_tokens=False)[0] for i in range(9)}


def predict_batch(model, tokenizer, prompts, digit_ids, device):
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

    logits = out.logits
    preds = []
    for b in range(logits.shape[0]):
        last_pos = int(enc["attention_mask"][b].sum()) - 1
        last_logits = logits[b, last_pos, :]
        scores = torch.tensor([last_logits[digit_ids[i]] for i in range(9)])
        preds.append(int(scores.argmax()))
    return preds


def predict():
    with open(f"{DATA_DIR}/test_processed.json") as f:
        test_data = json.load(f)

    print(f"Test samples: {len(test_data)}")

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    digit_ids = get_digit_token_ids(tokenizer)

    results = []
    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Predicting"):
        batch   = test_data[i : i + BATCH_SIZE]
        prompts = [s["prompt"] for s in batch]
        preds   = predict_batch(model, tokenizer, prompts, digit_ids, device)

        for s, pred in zip(batch, preds):
            results.append({"id": s["id"], "label": pred})

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved → {OUTPUT_FILE}")
    print(f"Total: {len(results)}")

    # Distribution check
    from collections import Counter
    counts = Counter(r["label"] for r in results)
    print("\nPrediction distribution:")
    for k in sorted(counts):
        print(f"  Level {k}: {counts[k]:4d}  ({counts[k]/len(results)*100:.1f}%)")


if __name__ == "__main__":
    predict()