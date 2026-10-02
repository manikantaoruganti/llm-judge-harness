import os
import csv
import json
from dotenv import load_dotenv
from src.judge import evaluate_faithfulness

load_dotenv()

def calibrate_and_score(
    human_labels_path: str,
    golden_dataset_path: str
) -> dict:
    """
    Calibrates the LLM judge against human labels and calculates TPR/TNR.
    All judge calls are traced via LangSmith under the 'judge-calibration' project.
    """
    os.environ["LANGCHAIN_PROJECT"] = "judge-calibration"

    golden_dataset = {}
    with open(golden_dataset_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            golden_dataset[row["source_id"]] = row["input_text"]

    human_labels = []
    with open(human_labels_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            human_labels.append(row)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total_samples = 0

    print(f"Starting judge calibration for {len(human_labels)} samples...")

    for i, label_row in enumerate(human_labels):
        source_id = label_row["source_id"]
        generated_text = label_row["generated_text"]
        human_is_faithful = int(label_row["human_is_faithful"]) == 1

        source_text = golden_dataset.get(source_id)
        if not source_text:
            print(f"WARNING: Source text for source_id {source_id} not found in golden dataset. Skipping.")
            continue

        total_samples += 1
        
        # Invoke the LLM judge
        judge_result = evaluate_faithfulness(source_text, generated_text)
        judge_is_faithful = judge_result.faithful

        if human_is_faithful and judge_is_faithful:
            true_positives += 1
        elif not human_is_faithful and judge_is_faithful:
            false_positives += 1
        elif not human_is_faithful and not judge_is_faithful:
            true_negatives += 1
        elif human_is_faithful and not judge_is_faithful:
            false_negatives += 1
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{len(human_labels)} samples...")

    tpr = (
        true_positives /
        (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    tnr = (
        true_negatives /
        (true_negatives + false_positives)
        if (true_negatives + false_positives) > 0
        else 0.0
    )

    metrics = {
        "total_samples": total_samples,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4)
    }
    return metrics

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is missing or empty. Cannot run judge calibration without valid API credentials.")

    print("Running judge calibration with live LLM. This may take a while...")
    human_labels_file = "tests/fixtures/human_labels.csv"
    golden_dataset_file = "tests/fixtures/golden_dataset.csv"
    metrics_data = calibrate_and_score(human_labels_file, golden_dataset_file)
    print("Judge calibration completed.")

    output_path = "results/calibration_metrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"Calibration metrics saved to {output_path}")
    print(json.dumps(metrics_data, indent=2))
