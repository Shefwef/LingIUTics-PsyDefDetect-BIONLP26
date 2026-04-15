# Pipeline: Qwen3-8B `seedAblend_v2decode`

This document details the complete end-to-end pipeline for the `seedAblend_v2decode` model, representing the peak iteration of our system (scoring 0.3917 F1 on the leaderboard).

## 1. Data Preparation & Grouped Cross-Validation
* **Leakage Prevention:** Replaced standard K-Fold with **`StratifiedGroupKFold`** to ensure that chunks originating from the same dialogue (or related synthetic variants) were kept strictly within the same fold. This prevented the model from "memorizing" specific conversation structures.
* **Input Representation:** Utterances were tagged with `SPEAKER:` / `SEEKER:`, capped at the last 30 turns, and constrained to a maximum sequence length of 1024 tokens.

## 2. Model Architecture & Hyperparameters
* **Base Model:** `Qwen/Qwen3-8B`
* **Quantization:** 4-bit NormalFloat (NF4) double quantization via `bitsandbytes` to fit on consumer GPUs.
* **Adapter:** QLoRA targeting all attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) plus the classification `score` head.
* **Hyperparameters:** Rank $r=64$, $\alpha=128$, dropout $0.05$. 
* **Loss Function:** Implemented an **inverse-square-root class weighting** to penalize the heavily-overrepresented Level 7 class and boost minority classes.

## 3. Training Two Separate Seeds
To combat model variance on the hidden set, we trained two distinct models using the exact same grouped-CV script but with different initialization seeds:
1. **Model 1 (Anchor):** Trained with default settings. Output saved to `outputs/qwen3_8b_rrk3_gpu1_grouped_clean`.
2. **Model 2 (Seed A):** Trained utilizing `--seed 20260407`. Output saved to `outputs/qwen3_8b_rrk3_gpu1_grouped_clean_seed20260407_a`.
* **Script Used:** `scripts/train_qwen3_8b_lora_cv_grouped_clean.py`

## 4. Ensembling (Blending)
Instead of relying on a single model's hard predictions, we extracted the raw softmax probability distributions from the out-of-fold (OOF) predictions (for validation) and the test predictions.
* **Blend Weights:** We applied a weighted average combining the two models:
  * **30%** weight to Model 1 (Anchor)
  * **70%** weight to Model 2 (Seed A)

## 5. Inference & `v2decode` Post-Processing
We designed a specific post-processing algorithm termed **`v2decode`** to safely shift the probability distribution toward minority classes without destroying high-confidence majority predictions.
* **tau7 Gating:** If a sample's predicted probability for Class 7 exceeded `0.69`, the model's prediction was "locked in" as Class 7, protecting the system's precision.
* **Logit Bonus Vectors:** For samples below the `tau7` threshold, predefined bonus weights were added incrementally to minority classes (e.g., Classes 1, 6, 8) to convert ambiguous border cases into minority predictions.

## 6. Result
* **Leaderboard Score:** 0.6419 Accuracy, 0.3917 F1 (First Place at time of submission).
* **Verdict:** The multi-seed ensembling reduced output variance drastically,, while the `v2decode` thresholding maximized minority recall safely without triggering catastrophic precision collapse.