import json

def save_json_output(detections, output_path="output/detections.json"):
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Detections saved to {output_path}")    
