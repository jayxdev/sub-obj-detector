# Verification of predeicted heirsrchies from detection.json file

def convert_to_subobject_list(detections):
    sub_objects = {}

    # Iterate through all detections
    for frame in detections:
        for detection in frame["detections"]:
            parent_object = detection["object"]
            subobject_list = set()  # Use a set to avoid duplicates
            
            # Iterate through sub-objects
            for sub_obj in detection.get("subobjects", []):
                subobject_list.add(sub_obj["object"])
            
            # Add the sub-objects to the parent object in the dictionary
            if parent_object not in sub_objects:
                sub_objects[parent_object] = list(subobject_list)
            else:
                sub_objects[parent_object].extend(list(subobject_list))

    # Ensure unique values for each object
    for parent_object in sub_objects:
        sub_objects[parent_object] = list(set(sub_objects[parent_object]))

    return sub_objects

import json

# Load the JSON file containing the detections
detections_file = "output/detections.json"
with open(detections_file, "r") as file:
    detections = json.load(file)

print(convert_to_subobject_list(detections))   