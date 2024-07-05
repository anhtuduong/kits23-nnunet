import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Function to determine final predictions based on n_pred
def determine_final_predictions(metrics_data):
    final_predictions = {}
    
    for case in metrics_data:
        max_n_pred = 0
        predicted_label = None
        
        for label in ["4", "5", "6", "7"]:
            if label in case["metrics"]:
                if case["metrics"][label]["n_pred"] > max_n_pred:
                    max_n_pred = case["metrics"][label]["n_pred"]
                    predicted_label = label
        
        if predicted_label:
            final_predictions[case["prediction_file"]] = predicted_label
    
    return final_predictions

# Function to build confusion matrix (excluding 'other' class)
def build_confusion_matrix(metrics_data, final_predictions):
    confusion_matrix = np.zeros((4, 4), dtype=int)  # Initialize confusion matrix
    
    label_index = {"4": 0, "5": 1, "6": 2, "7": 3}  # Index mapping for labels
    
    for case in metrics_data:
        if case["prediction_file"] in final_predictions:
            predicted_label = final_predictions[case["prediction_file"]]
            ground_truth_label = None
            
            for label in ["4", "5", "6", "7"]:
                if label in case["metrics"] and case["metrics"][label]["n_ref"] != 0:
                    ground_truth_label = label
                    break
            
            if ground_truth_label and predicted_label in label_index:
                confusion_matrix[label_index[ground_truth_label]][label_index[predicted_label]] += 1
    
    return confusion_matrix

# Function to compute normalized confusion matrix (excluding 'other' class)
def compute_normalized_confusion_matrix(confusion_matrix):
    normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return np.round(normalized_confusion_matrix, decimals=2)

# Function to print confusion matrix
def print_confusion_matrix(confusion_matrix, label_names):
    labels = ["4", "5", "6", "7"]
    
    print("Confusion Matrix:")
    print("      True Labels")
    print("      4   5   6   7")
    print("-------------------")
    for i in range(len(labels)):
        print(f"Predicted {label_names[labels[i]]:12} | ", end="")
        for j in range(len(labels)):
            print(f"{confusion_matrix[i][j]:3} ", end="")
        print()

# Function to print normalized confusion matrix
def print_normalized_confusion_matrix(normalized_confusion_matrix, label_names):
    labels = ["4", "5", "6", "7"]
    
    print("Normalized Confusion Matrix:")
    print("      True Labels")
    print("      4    5    6    7")
    print("------------------------")
    for i in range(len(labels)):
        print(f"Predicted {label_names[labels[i]]:12} | ", end="")
        for j in range(len(labels)):
            print(f"{normalized_confusion_matrix[i][j]:.2f} ", end="")
        print()

# Function to save confusion matrix as image
def save_confusion_matrix_image(confusion_matrix, label_names, title, save_path):
    labels = ["4", "5", "6", "7"]
    
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="Blues")
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([label_names[label] for label in labels])
    ax.set_yticklabels([label_names[label] for label in labels])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black")
    
    ax.set_xlabel("True Labels")
    ax.set_ylabel("Predicted Labels")
    ax.set_title(title)
    fig.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

# Function to plot mean Dice and IoU scores
def plot_mean_dice_iou(data, labels, label_names, output_image):
    mean_dice_iou = data['mean']
    
    colors = 'g'  # Use a single color for both Dice and IoU

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    x = np.arange(len(labels))

    # Plot Dice values with numeric labels
    dice_values = [mean_dice_iou[label]['Dice'] for label in labels]
    rects1 = ax.bar(x, dice_values, bar_width, label='Dice', color=colors)
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{dice_values[i]:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot IoU values with numeric labels
    iou_values = [mean_dice_iou[label]['IoU'] for label in labels]
    rects2 = ax.bar(x + bar_width, iou_values, bar_width, label='IoU', color=colors, alpha=0.5)
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, f'{iou_values[i]:.2f}', ha='center', va='bottom', fontsize=8)

    # Adding labels, title and custom x-axis tick labels
    ax.set_xlabel('Labels')
    ax.set_ylabel('Metrics')
    ax.set_title('Mean Dice and IoU Scores')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels([label_names[label] for label in labels])
    ax.legend()

    # Save the plot to an image file
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    print(f"Mean Dice and IoU plot saved to {output_image}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate confusion matrices and plots for evaluation.')
    parser.add_argument('--confusion', action='store_true', help='Generate confusion matrix')

    args = parser.parse_args()

    dataset_dir = "results/Dataset081"
    json_file_path = f"{dataset_dir}/result.json"

    save_confusion_matrix_path = f"{dataset_dir}/confusion_matrix.png"
    save_normalized_confusion_matrix_path = f"{dataset_dir}/normalized_confusion_matrix.png"
    save_mean_dice_iou_path = f"{dataset_dir}/mean_dice_iou.png"

    # Label names mapping
    # label_names = {
    #     "4": "clear_cell_rcc",
    #     "5": "chromophobe",
    #     "6": "oncocytoma",
    #     "7": "papillary"
    # }

    # Labels for dataset 220
    labels = ["(1, 2, 3)", "(2, 3)", "2"]
    label_names = {
        "(1, 2, 3)": "kidney",
        "(2, 3)": "masses",
        "2": "tumor"
    }

    # Labels for dataset 81
    labels = [
        "(1, 2, 3, 4, 5, 6, 7)",
        "(2, 3, 4, 5, 6, 7)",
        "(2, 4, 5, 6, 7)",
        "4",
        "5",
        "6",
        "7"
    ]
    label_names = {
        "(1, 2, 3, 4, 5, 6, 7)": "kidney",
        "(2, 3, 4, 5, 6, 7)": "masses",
        "(2, 4, 5, 6, 7)": "tumor",
        "4": "clear_cell_rcc",
        "5": "chromophobe",
        "6": "oncocytoma",
        "7": "papillary"
    }

    # Read JSON data
    data = read_json_file(json_file_path)

    # Extract cases from metric_per_case
    metric_per_case = data["metric_per_case"]

    # Plot and save mean Dice and IoU scores
    plot_mean_dice_iou(data, labels, label_names, save_mean_dice_iou_path)

    # Generate confusion matrix if requested
    if args.confusion:
        # Determine final predictions based on n_pred
        final_predictions = determine_final_predictions(metric_per_case)

        # Build confusion matrix (excluding 'other' class)
        confusion_matrix = build_confusion_matrix(metric_per_case, final_predictions)

        # Print confusion matrix (ignoring 'other' class)
        print_confusion_matrix(confusion_matrix, label_names)

        # Compute normalized confusion matrix (excluding 'other' class)
        normalized_confusion_matrix = compute_normalized_confusion_matrix(confusion_matrix)

        # Print normalized confusion matrix (ignoring 'other' class)
        print_normalized_confusion_matrix(normalized_confusion_matrix, label_names)

        # Save confusion matrix as image
        save_confusion_matrix_image(confusion_matrix, label_names, "Confusion Matrix", save_confusion_matrix_path)

        # Save normalized confusion matrix as image
        save_confusion_matrix_image(normalized_confusion_matrix, label_names, "Confusion Matrix Normalized", save_normalized_confusion_matrix_path)
    
