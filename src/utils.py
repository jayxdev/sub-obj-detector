import cv2
import os
import numpy as np
import json

# Function to save cropped sub-object images
def save_subobject_image(frame, bbox, object_id, subobject_name, output_folder="output"):
    x1, y1, x2, y2 = bbox
    sub_object_image = frame[y1:y2, x1:x2]
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create sub-folder for the specific subobject
    subobject_folder = os.path.join(output_folder, subobject_name)
    if not os.path.exists(subobject_folder):
        os.makedirs(subobject_folder)
    
    # Save the image in the specific subobject folder
    file_name = f"{subobject_name}_{object_id}.jpg"
    output_path = os.path.join(subobject_folder, file_name)
    cv2.imwrite(output_path, sub_object_image)

def save_json_output(detections, output_path="output/detections.json"):
    # Ensure output folder exists
    
    with open(output_path, "w") as f:
        json.dump(detections, f, indent=4)
    print(f"Detections saved to {output_path}")    

