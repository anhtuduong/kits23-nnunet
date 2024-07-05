import json
import numpy as np
from sklearn.metrics import roc_auc_score

# Given label names
label_names = {
    "4": "clear_cell_rcc",
    "5": "chromophobe",
    "6": "oncocytoma",
    "7": "papillary",
    "8": "other"
}

# Load the JSON result file
result_file = "results/Dataset080/result_histology_3dlowres.json"  # Replace with your actual file path

with open(result_file, 'r') as f:
    results = json.load(f)

# Initialize lists to store true labels and predicted logits for each class pair
true_labels = {label: [] for label in label_names}
predicted_logits = {label: [] for label in label_names}

# Process each case in metric_per_case
for case in results["metric_per_case"]:
    metrics = case["metrics"]
    for label, label_name in label_names.items():
        true_value = 1 if metrics[label]["TP"] > 0 else 0  # Binary classification
        predicted_value = sum(case["predicted_logits"])  # Sum of logits (raw outputs)
        
        true_labels[label_name].append(true_value)
        predicted_logits[label_name].append(predicted_value)

# Compute AUC for each class pair
auc_scores = {}
for label_name, true_vals in true_labels.items():
    if sum(true_vals) > 0 and sum(true_vals) < len(true_vals):  # Ensure there are both positive and negative labels
        auc = roc_auc_score(true_vals, predicted_logits[label_name])
        auc_scores[label_name] = auc

# Print or use auc_scores as needed
for label_name, auc in auc_scores.items():
    print(f"AUC for {label_name}: {auc}")
