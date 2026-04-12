# prepare_data.py
import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Label metadata ────────────────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    0: "Level 0 - No Defense: Purely functional utterances (greetings, phatic expressions, simple acknowledgments) with no psychological conflict engagement.",
    1: "Level 1 - Action Defenses: Distress released by acting on the environment instead of reflecting (Passive Aggression, Help-Rejecting Complaining, Acting Out).",
    2: "Level 2 - Major Image-Distorting Defenses: Reduces anxiety via all-good/all-bad distortions of self or others (Splitting, Projective Identification).",
    3: "Level 3 - Disavowal Defenses: Rejects threatening reality by denying, excusing, blaming others, or retreating into fantasy (Denial, Rationalization, Projection, Autistic Fantasy).",
    4: "Level 4 - Minor Image-Distorting Defenses: Softer distortions that temporarily inflate or deflate self-esteem (Devaluation, Idealization, Omnipotence).",
    5: "Level 5 - Neurotic Defenses: Keeps unacceptable motives out of awareness; feelings surface indirectly (Repression, Dissociation, Reaction Formation, Displacement).",
    6: "Level 6 - Obsessional Defenses: Separates feelings from events using excessive logic or symbolic acts (Isolation of Affect, Intellectualization, Undoing).",
    7: "Level 7 - High-Adaptive Defenses: Mature coping that integrates emotion and thought to channel affect constructively (Affiliation, Altruism, Anticipation, Humor, Self-Assertion, Sublimation, Suppression).",
    8: "Level 8 - Needs More Information: Utterance is too ambiguous or context is insufficient to classify."
}

SYSTEM_PROMPT = """You are an expert clinical psychologist specializing in the Defense Mechanism Rating Scales (DMRS). Your task is to analyze the final seeker utterance in a supportive conversation and classify its psychological defense level.

The defense levels are:
0 - No Defense: Purely functional utterances (greetings, phatic expressions) with no psychological conflict.
1 - Action Defenses: Distress channeled into behavior rather than reflection (Passive Aggression, Help-Rejecting Complaining, Acting Out).
2 - Major Image-Distorting: All-good/all-bad distortions of self or others (Splitting, Projective Identification).
3 - Disavowal: Rejecting threatening reality by denying, excusing, blaming, or fantasizing (Denial, Rationalization, Projection, Autistic Fantasy).
4 - Minor Image-Distorting: Softer distortions that inflate or deflate self-esteem (Devaluation, Idealization, Omnipotence).
5 - Neurotic: Keeps unacceptable motives out of awareness; feelings surface indirectly (Repression, Dissociation, Reaction Formation, Displacement).
6 - Obsessional: Separates feelings from events using excessive logic or symbolic acts (Isolation of Affect, Intellectualization, Undoing).
7 - High-Adaptive: Mature coping integrating emotion and thought constructively (Affiliation, Altruism, Anticipation, Humor, Self-Assertion, Sublimation, Suppression).
8 - Needs More Information: Utterance is too ambiguous or lacks sufficient context.

Instructions:
- Focus on the LAST seeker utterance marked as [TARGET].
- Use the full dialogue context to understand the psychological situation.
- Identify the dominant defensive function, not just surface tone.
- Respond with ONLY a single digit (0-8). Nothing else."""


def build_prompt(sample):
    """
    Build the instruction prompt from a data sample.
    The dialogue is formatted with speaker labels.
    The final seeker utterance (current_text) is marked as [TARGET].
    """
    dialogue = sample["dialogue"]
    current_text = sample["current_text"]

    # Format the dialogue context
    context_lines = []
    for i, turn in enumerate(dialogue):
        speaker = turn["speaker"].capitalize()
        text = turn["text"].strip()

        # Mark the last turn if it matches current_text (it always should)
        is_last = (i == len(dialogue) - 1)
        if is_last and turn["speaker"] == "seeker":
            context_lines.append(f"{speaker} [TARGET]: {text}")
        else:
            context_lines.append(f"{speaker}: {text}")

    context_str = "\n".join(context_lines)

    prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{context_str}\n\nDefense level of the [TARGET] utterance:"
    return prompt


def prepare(train_path="./data/train.json",
            test_path="./data/test.json",
            output_dir="./data",
            val_ratio=0.1,
            seed=42):

    # ── Load raw data ─────────────────────────────────────────────────────────
    with open(train_path) as f:
        train_raw = json.load(f)
    with open(test_path) as f:
        test_raw = json.load(f)

    print(f"Raw train samples : {len(train_raw)}")
    print(f"Raw test  samples : {len(test_raw)}")

    # ── Label distribution ────────────────────────────────────────────────────
    labels = [s["label"] for s in train_raw]
    print("\nLabel distribution in train:")
    for k, v in sorted(Counter(labels).items()):
        print(f"  Level {k}: {v:4d}  ({v/len(labels)*100:.1f}%)")

    # ── Build prompts ─────────────────────────────────────────────────────────
    train_samples = []
    for s in train_raw:
        train_samples.append({
            "id":           s["id"],
            "dialogue_id":  s["dialogue_id"],
            "prompt":       build_prompt(s),
            "label":        s["label"],
            "completion":   str(s["label"])
        })

    test_samples = []
    for s in test_raw:
        test_samples.append({
            "id":           s["id"],
            "dialogue_id":  s["dialogue_id"],
            "prompt":       build_prompt(s),
            "label":        -1,   # unknown
            "completion":   ""
        })

    # ── Stratified train/val split ────────────────────────────────────────────
    train_labels = [s["label"] for s in train_samples]
    train_split, val_split = train_test_split(
        train_samples,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed
    )

    print(f"\nSplit → train: {len(train_split)}  val: {len(val_split)}  test: {len(test_samples)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(f"{output_dir}/train_processed.json", "w") as f:
        json.dump(train_split, f, indent=2)
    with open(f"{output_dir}/val_processed.json", "w") as f:
        json.dump(val_split, f, indent=2)
    with open(f"{output_dir}/test_processed.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    print("Saved train_processed.json, val_processed.json, test_processed.json")

    # ── Verify prompt format ──────────────────────────────────────────────────
    print("\n── Sample prompt preview ──────────────────────────────────────────")
    print(train_split[0]["prompt"])
    print(f"\nLabel: {train_split[0]['label']}")


if __name__ == "__main__":
    prepare()