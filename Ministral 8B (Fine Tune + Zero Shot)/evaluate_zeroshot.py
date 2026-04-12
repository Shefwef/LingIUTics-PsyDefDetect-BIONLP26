"""
Zero-shot evaluation for the PsyDefDetect task.

This script implements the exact zero-shot scoring used in the paper:
- Load the base causal LM (no adapter)
- Build the same instruction prompt as used for fine-tuning
- For each example, compute the logits for digits '0'..'8' at the next token
- Pick the argmax digit as the prediction

The loader attempts to place the model on GPU (float16/bf16) first. If that fails,
it falls back to device_map='auto' with an offload folder. Batch size is configurable.

Outputs:
- ./results/results_val_zeroshot.json
- ./results/cm_val_zeroshot.png
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


BASE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
DATA_DIR = "./data"
RESULTS_DIR = "./results"
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "./offload_zeroshot")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

LABEL_NAMES = [
    "No Defense", "Action", "Major Image-Dist.", "Disavowal",
    "Minor Image-Dist.", "Neurotic", "Obsessional", "High-Adaptive", "Needs More Info",
]


def build_prompt_from_sample(sample):
    # replicate prepare_data.build_prompt formatting
    dialogue = sample["dialogue"]
    lines = []
    for i, turn in enumerate(dialogue):
        speaker = turn["speaker"].capitalize()
        text = turn["text"].strip()
        is_last = (i == len(dialogue) - 1)
        if is_last and turn["speaker"] == "seeker":
            lines.append(f"{speaker} [TARGET]: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    # reuse system prompt from training / prepare_data
    from prepare_data import SYSTEM_PROMPT
    prompt = SYSTEM_PROMPT + "\n\nConversation:\n" + "\n".join(lines) + "\n\nDefense level of the [TARGET] utterance:"
    return prompt


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Try fast GPU-first load with float16/bf16
    # Try fast GPU-first load (place whole model on cuda:0 with reduced precision)
    if torch.cuda.is_available():
        try:
            print("Attempting GPU-first load on cuda:0 (float16/bf16)...")
            # prefer bfloat16 where supported, else float16
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    device_map={"": "cuda:0"},
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                print("Loaded model on cuda:0 with bfloat16")
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    device_map={"": "cuda:0"},
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
                print("Loaded model on cuda:0 with float16")
            model.eval()
            return model, tokenizer
        except Exception as e:
            print("GPU-first load failed, falling back to auto offload:", e)

    # Fallback 1: try 4-bit quantized load with bitsandbytes (QLoRA-style)
    try:
        print("Attempting 4-bit quantized load with bitsandbytes (nf4, float16 compute)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
        print("Loaded quantized 4-bit model via bitsandbytes")
        return model, tokenizer
    except Exception as e:
        print("4-bit quantized load failed, falling back to offload to disk:", e)

    # Fallback 2: use accelerate auto device map with offload to disk and offload_buffers
    print(f"Loading with device_map='auto', offload_folder={OFFLOAD_DIR}, offload_buffers=True")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer


def get_digit_token_ids(tokenizer):
    ids = {}
    for i in range(9):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        ids[i] = tokens[0]
    print(f"Digit token IDs: {ids}")
    return ids


def predict_via_logits(model, tokenizer, prompts, digit_ids, device):
    texts = [f"[INST] {p} [/INST]" for p in prompts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    )
    # move tensors to device (model may be sharded via accelerate)
    for k, v in enc.items():
        enc[k] = v.to(device)

    with torch.no_grad():
        out = model(**enc)
    logits = out.logits  # (B, seq_len, vocab)

    preds = []
    for b in range(logits.shape[0]):
        last_pos = int(enc["attention_mask"][b].sum()) - 1
        last_logits = logits[b, last_pos, :]
        scores = torch.tensor([last_logits[digit_ids[i]] for i in range(9)], device=last_logits.device)
        preds.append(int(scores.argmax()))

    # free cache to avoid fragmentation
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return preds


def evaluate(batch_size=None):
    with open(f"{DATA_DIR}/val_processed.json") as f:
        data = json.load(f)

    model, tokenizer = load_model_and_tokenizer()

    # determine device: if model has parameters on cuda, use that device; else cpu
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    digit_ids = get_digit_token_ids(tokenizer)

    if batch_size is None:
        batch_size = int(os.environ.get("ZEROSHOT_BATCH_SIZE", 8))

    y_true, y_pred = [], []
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating (zeroshot)"):
        batch = data[i : i + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels = [int(s["label"]) for s in batch]

        preds = predict_via_logits(model, tokenizer, prompts, digit_ids, device)
        y_true.extend(labels)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(1, 9)),
        average="macro",
        zero_division=0,
    )

    out = {
        "accuracy": round(acc, 2),
        "macro_precision": round(p * 100, 2),
        "macro_recall": round(r * 100, 2),
        "macro_f1": round(f1 * 100, 2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

    print("\nZero-shot results:")
    print(out)

    # save numeric results
    with open(f"{RESULTS_DIR}/results_val_zeroshot.json", "w") as f:
        json.dump(out, f, indent=2)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(9)],
                yticklabels=[f"C{i}" for i in range(9)], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix — Zero-shot (val)")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/cm_val_zeroshot.png", dpi=150)
    print(f"Confusion matrix → {RESULTS_DIR}/cm_val_zeroshot.png")

    return out


if __name__ == "__main__":
    # allow override
    bs = int(os.environ.get("ZEROSHOT_BATCH_SIZE", "1"))
    # set helpful allocator to reduce fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    evaluate(batch_size=bs)
"""
Zero-shot evaluation: load the base LM (no PEFT adapter) and evaluate on validation set.
Saves results to ./results/results_val_zeroshot.json and confusion matrix to ./results/cm_val_zeroshot.png
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

BASE_MODEL  = "mistralai/Ministral-8B-Instruct-2410"
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
    print("Loading base tokenizer and model (zero-shot)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Fast path: try to place model entirely on cuda:0 with float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
        )
        model.eval()
        print("Model loaded on cuda:0 (float16)")
        return model, tokenizer
    except Exception as e:
        print("GPU-first load failed, falling back to auto offload:", e)
        offload_dir = "./offload_zeroshot"
        os.makedirs(offload_dir, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            offload_folder=offload_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()
        print(f"Model loaded with auto offload (offload_folder={offload_dir})")
        return model, tokenizer


def get_digit_token_ids(tokenizer):
    ids = {}
    for i in range(9):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        ids[i] = tokens[0]
    print(f"Digit token IDs: {ids}")
    return ids


def predict_via_logits(model, tokenizer, prompts, digit_ids, device):
    texts = [f"[INST] {p} [/INST]" for p in prompts]
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

    logits = out.logits
    preds = []
    for b in range(logits.shape[0]):
        last_pos = int(enc["attention_mask"][b].sum()) - 1
        last_logits = logits[b, last_pos, :]
        scores = torch.tensor([last_logits[digit_ids[i]] for i in range(9)]).to(device)
        preds.append(int(scores.argmax()))

    # free cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return preds


def evaluate(split="val", batch_size=1):
    fname = "val_processed.json" if split == "val" else "train_processed.json"
    with open(f"{DATA_DIR}/{fname}") as f:
        data = json.load(f)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    digit_ids = get_digit_token_ids(tokenizer)

    y_true, y_pred = [], []
    for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {split} (zeroshot)"):
        batch   = data[i : i + batch_size]
        prompts = [s["prompt"] for s in batch]
        labels  = [int(s["label"]) for s in batch]

        preds = predict_via_logits(model, tokenizer, prompts, digit_ids, device)
        y_true.extend(labels)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(1, 9)),
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print(f"  Zero-shot Results on {split.upper()} set")
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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(9)],
                yticklabels=[f"C{i}" for i in range(9)], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix 1 (Zero-shot) - {split}")
    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/cm_{split}_zeroshot.png", dpi=150)
    print(f"Confusion matrix -> {RESULTS_DIR}/cm_{split}_zeroshot.png")

    out = {
        "split": split,
        "accuracy": round(acc, 2),
        "macro_precision": round(p * 100, 2),
        "macro_recall":    round(r * 100, 2),
        "macro_f1":        round(f1 * 100, 2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    with open(f"{RESULTS_DIR}/results_{split}_zeroshot.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved -> {RESULTS_DIR}/results_{split}_zeroshot.json")
    return out


if __name__ == "__main__":
    evaluate("val")
