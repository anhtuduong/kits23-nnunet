import json

def find_min_max_sizes(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    shapes_after_crop = data["shapes_after_crop"]
    
    min_size = [float('inf'), float('inf'), float('inf')]
    max_size = [float('-inf'), float('-inf'), float('-inf')]
    
    for shape in shapes_after_crop:
        for i in range(3):
            if shape[i] < min_size[i]:
                min_size[i] = shape[i]
            if shape[i] > max_size[i]:
                max_size[i] = shape[i]
                
    return min_size, max_size

# Example usage
json_file_path = 'results/Dataset081/dataset_fingerprint.json'

min_size, max_size = find_min_max_sizes(json_file_path)
print("Min size:", min_size)
print("Max size:", max_size)
