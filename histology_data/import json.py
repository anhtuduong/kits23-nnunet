
import json

# Define the path to the JSON file
json_path = "/Users/hoaithunguyen/Documents/FBK/Internship/Data/kits.json"

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
with open('/Users/hoaithunguyen/Documents/FBK/Internship/Data/tumor_subtype', 'w') as f:
    # Write each subtype on a new line
    for subtype in tumor_subtypes:
        f.write(subtype + "\n")

# Print confirmation message
print("Tumor subtypes saved to:", '/Users/hoaithunguyen/Documents/FBK/Internship/Data/tumor_subtype')