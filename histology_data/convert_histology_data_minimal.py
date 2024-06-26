import json

# Input and output file paths
input_json_file = 'histology_data/kits.json'
output_json_file = 'histology_data/kits23_histology_data.json'

# Function to extract required information
def extract_case_info(input_json_file, output_json_file):
    # Read the input JSON file
    with open(input_json_file, 'r') as file:
        data = json.load(file)
    
    # Extract the required information
    extracted_data = [
        {
            "case_id": case["case_id"],
            "tumor_histologic_subtype": case["tumor_histologic_subtype"]
        }
        for case in data
    ]
    
    # Save the extracted information to a new JSON file
    with open(output_json_file, 'w') as file:
        json.dump(extracted_data, file, indent=4)


# Call the function
extract_case_info(input_json_file, output_json_file)
