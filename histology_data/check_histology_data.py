import json
import matplotlib.pyplot as plt

# Define the path to the JSON file
# json_path = "histology_data/kits.json"
# json_path = "histology_data/kits23_histology_data.json"
json_path = "histology_data/kits23_histology_data_selected.json"

# Define the path to the output text file
# output_path = "histology_data/tumor_subtype.txt"
# output_path = "histology_data/tumor_subtype_minimal.txt"
output_path = "histology_data/tumor_subtype_selected.txt"

# graph_output_path = "histology_data/tumor_subtype_distribution.png"
graph_output_path = "histology_data/tumor_subtype_distribution_selected.png"

# Initialize an empty list to store subtypes
tumor_subtypes = []

# Open and read the JSON file
with open(json_path, 'r') as f:
    # Load JSON data (assuming data is a list of dictionaries)
    data = json.load(f)

# Extract subtypes and handle potential missing values
for item in data:
    subtype = item.get("tumor_histologic_subtype")
    if subtype is not None:
        tumor_subtypes.append(subtype)

# Print the list of different tumor subtypes
count_unique = len(set(tumor_subtypes))
print(f"Number of unique tumor subtypes: {count_unique}")
# Print the list of tumor subtypes with their counts
print("--------")
for subtype in sorted(set(tumor_subtypes)):
    print(f"{subtype}: {tumor_subtypes.count(subtype)} cases")

# Save the list of tumor subtypes to a text file
with open(output_path, 'w') as f:
    for subtype in sorted(set(tumor_subtypes)):
        f.write(f"{subtype}: {tumor_subtypes.count(subtype)} cases\n")

# ------------------------------------------------------------

# Create a graph of the tumor subtypes

# Count the number of occurrences of each subtype
subtype_counts = {subtype: tumor_subtypes.count(subtype) for subtype in set(tumor_subtypes)}

# Sort the subtypes by count in descending order
sorted_subtypes = sorted(subtype_counts.items(), key=lambda x: x[1], reverse=True)

# Extract the subtypes and counts for plotting
subtypes = [subtype[0] for subtype in sorted_subtypes]
counts = [subtype[1] for subtype in sorted_subtypes]

# Create a bar plot of the tumor subtypes
plt.figure(figsize=(12, 6))
plt.bar(subtypes, counts, color='skyblue')
plt.xlabel('Tumor Subtypes')
plt.ylabel('Number of Cases')
plt.title('Distribution of Tumor Subtypes')
plt.xticks(rotation=45, ha='right')
# Display the counts on top of the bars
for i, count in enumerate(counts):
    plt.text(i, count + 1, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Save graph to image file
plt.savefig(graph_output_path)

print(f"Graph saved to: {graph_output_path}")
