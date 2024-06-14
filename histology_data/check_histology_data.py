
import json

# Define the path to the JSON file
json_path = "histology_data/kits.json"

# Define the path to the output text file
output_path = "histology_data/tumor_subtype.txt"

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

# Print the list of tumor subtypes
print("Tumor Histologic Subtypes:", tumor_subtypes)

# Save the list to a text file
with open(output_path, 'w') as f:
    # Write each subtype on a new line
    for subtype in tumor_subtypes:
        f.write(subtype + "\n")

# Print confirmation message
print("Tumor subtypes saved to:", output_path)