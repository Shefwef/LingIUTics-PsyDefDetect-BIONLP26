# Pipeline: Qwen3-8B (Finetuned Baseline)

This document details the pipeline of the original Qwen3-8B baseline (scoring 0.3541 F1). This was the foundational attempt at prompt-tuning the LLM before issues with synthetic data leakage and post-processing stability were fully diagnosed.

## 1. Data Preparation & Flawed Cross-Validation
* **Cross Validation:** Utilized a standard **`StratifiedKFold`**. 
* **The Flaw (Data Leakage):** Standard Stratified K-Fold balances class labels but ignores the metadata source of the rows. Because the training set contained heavily augmented and synthetically grouped dialogue turns from the same parent conversations, the standard split allowed rows from the *same* underlying conversation to exist in both the train and validation sets simultaneously. The model was largely memorizing stylistic quirks rather than generalizing.

## 2. Model Architecture
* **Base Model:** `Qwen/Qwen3-8B`
* **Quantization:** 4-bit NormalFloat (NF4).
* **Adapter:** Standard QLoRA targeting projection layers.
* **Script Used:** `scripts/train_qwen3_8b_lora_cv.py`.

## 3. Training Constraints
* Relied heavily on basic cross-entropy loss, suffering heavily from the dataset's native class imbalance. Class 7 vastly outnumbered all other classes, causing the model to default to Class 7 on ambiguous test samples.

## 4. Inference & Post-Processing
* Generated averaged predictions across the 5 leaky folds.
* **Output Path:** `outputs/qwen3_8b_rrk3_gpu1`.
* **Post-Processing:** Simple `argmax` over the output logits. No sophisticated class-protection (`tau7`) or ensembling scripts were utilized.

## 5. Result
* **Leaderboard Score:** 0.6033 Accuracy, 0.4907 Precision, 0.2858 Recall, **0.3541 F1**.
* **Verdict:** While the accuracy parameter looked deceptively high locally (due to the leakage inflating validation metrics), the low Leaderboard Recall (0.2858) demonstrated that the model failed to detect minority psychological defense mechanisms. This specific architectural failure directly necessitated the transition to the Grouped-CV pipeline in subsequent iterations.