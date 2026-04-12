# DeBERTa-v3-base 5-Fold Max Performance Guide

Goal: maximize leaderboard score while preserving strong precision and recall, using your best backbone.

This guide addresses all previously missed gaps:
- uses dialogue context plus current target text
- aligns model selection metric with your real objective
- computes macro and weighted metrics together
- reports precision and recall explicitly
- uses proper 5-fold out-of-fold workflow
- uses fold-weighted ensemble at inference
- keeps submission format with id and label

--------------------------------------------------

## 1) Choose primary optimization metric first

You must pick one primary metric for checkpoint selection and fold ranking:
- If competition uses weighted F1: select weighted_f1
- If competition uses macro F1: select macro_f1

Always log all of these:
- accuracy
- macro_f1, weighted_f1
- macro_precision, macro_recall
- weighted_precision, weighted_recall

Reason:
- one metric decides best checkpoint
- the others protect against pathological behavior

--------------------------------------------------

## 2) Input representation (highest impact)

Use the faculty-style input construction:
- recent dialogue turns (for example last 6)
- speaker tags
- explicit [CONTEXT] and [TARGET] blocks

Example format:
[CONTEXT]
speaker1: ...
speaker2: ...
...

[TARGET]
current_text

This is usually the single largest gain versus current_text-only training.

--------------------------------------------------

## 3) 5-fold training design (StratifiedKFold)

Use StratifiedKFold with shuffle=True and fixed random_state.
Recommended:
- n_splits: 5
- random_state: 42

For each fold:
1. build train and val subsets
2. compute class weights from that fold train subset only
3. train model with early stopping
4. save:
   - best checkpoint path
   - fold metrics
   - out-of-fold logits or probabilities for calibration

Never compute class weights using full dataset before splitting.

--------------------------------------------------

## 4) Recommended DeBERTa-v3-base training config

Start stable, then tune only if needed.

Suggested baseline:
- model_name: microsoft/deberta-v3-base
- max_length: 320
- train_batch_size: 8 or 12 (depending on VRAM)
- eval_batch_size: 16
- gradient_accumulation_steps: 2 (if batch is small)
- epochs: 6
- learning_rate: 1.5e-5
- weight_decay: 0.01
- warmup_ratio: 0.08
- lr_scheduler_type: cosine
- label_smoothing_factor: 0.03
- fp16: true on T4/P100
- gradient_checkpointing: true (if needed for memory)
- early_stopping_patience: 2
- save_total_limit: 1 or 2
- load_best_model_at_end: true
- metric_for_best_model: weighted_f1 or macro_f1 (decide in section 1)

Notes:
- Do not push epochs high if fold curves peak and then degrade.
- Keep one variable changed at a time.

--------------------------------------------------

## 5) Loss and imbalance strategy

Start with weighted cross-entropy (fold-train class weights).
If recall is poor on minority classes, test one alternative run:
- class-balanced focal loss (single controlled comparison)

Do not combine many imbalance tricks at once.

--------------------------------------------------

## 6) Compute metrics function

Track both macro and weighted variants for F1, precision, recall.

Expected metric dictionary keys:
- accuracy
- macro_f1
- weighted_f1
- macro_precision
- macro_recall
- weighted_precision
- weighted_recall

For best-checkpoint selection:
- set metric_for_best_model exactly to your primary metric key

--------------------------------------------------

## 7) Out-of-fold analysis (critical for reliability)

After all folds complete:
1. concatenate OOF predictions
2. compute global OOF metrics:
   - weighted_f1
   - macro_f1
   - precision and recall (both macro and weighted)
3. inspect per-class precision/recall table
4. inspect confusion matrix for major failure pairs

Use OOF metrics as your submission gate, not one lucky fold.

Submission gate suggestion:
- Submit only if OOF primary metric improves over previous best by at least 0.005
- Also require no catastrophic precision or recall collapse in minority labels

--------------------------------------------------

## 8) Fold-weighted ensemble for test inference

For each fold model:
- produce test probabilities (softmax)
- weight each fold by normalized fold validation primary metric
- combine weighted probabilities
- final label = argmax(combined_probs)

This is usually stronger than unweighted majority voting.

--------------------------------------------------

## 9) Probability calibration

Use OOF logits for temperature scaling (single temperature).
Then apply calibrated temperature to fold logits at test time before softmax ensemble.

Why:
- improves probability quality
- often stabilizes weighted-F1 behavior

--------------------------------------------------

## 10) Precision/recall performance targeting

If you want stronger precision:
- slightly reduce class-weight aggressiveness
- keep lr conservative
- avoid overtraining epochs

If you want stronger recall:
- slightly increase class-weight aggressiveness
- keep warmup and early stopping stable
- inspect missed minority classes in confusion matrix

For multiclass single-label tasks, avoid ad-hoc per-class thresholds initially.
Use better representation plus stable CV first.

--------------------------------------------------

## 11) Submission schema and packaging

Keep exact expected JSON format from the task.
If id is required, output:
[
  {"id": ..., "label": ...},
  ...
]

Then zip with prediction.json at zip root.

--------------------------------------------------

## 12) Practical high-score execution order

Phase A: Baseline CV run
1. DeBERTa-v3-base
2. Context plus target input
3. 5-fold stratified CV
4. weighted CE
5. primary metric set correctly

Phase B: One controlled improvement
Pick exactly one:
- max_length 320 to 384
- lr 1.5e-5 to 1.2e-5
- label smoothing 0.03 to 0.05

Phase C: Ensemble and calibrate
1. fold-weighted probability ensemble
2. temperature scaling with OOF logits

Phase D: Submit only with gate
- submit only if OOF primary metric and class behavior are both improved

--------------------------------------------------

## 13) Recommended next run for your current situation

Run this next:
- Backbone: microsoft/deberta-v3-base
- Input: dialogue context plus target (faculty format)
- 5-fold stratified CV
- max_length: 320
- lr: 1.5e-5
- epochs: 6
- early_stopping_patience: 2
- weighted CE from fold-train labels
- metric_for_best_model: weighted_f1 (if competition is weighted-F1)
- report all precision and recall metrics
- fold-weighted softmax ensemble for final test predictions

This is the highest-probability path to improve both score and reliability without overfitting to one split.
