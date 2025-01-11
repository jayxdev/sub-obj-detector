sub_objects_list={
             'Man': ['Human face', 'Footwear', 'Boy', 'Clothing', 'Wheel', 'Helmet'], 
             'Human face': ['Clothing', 'Woman', 'Human face', 'Man'], 
             'Helmet': ['Human face', 'Man', 'Helmet'], 
             'Woman': ['Clothing', 'Woman', 'Human face'], 
             'Clothing': ['Girl', 'Woman', 'Human face', 'Footwear', 'Boy', 'Clothing', 'Man', 'Helmet'], 
             'Land vehicle': ['Clothing', 'Person'], 
             'Person': ['Person', 'Human face', 'Clothing', 'Car', 'Land vehicle', 'Motorcycle', 'Bicycle', 'Man', 'Helmet'], 
             'Boy': ['Clothing', 'Human face', 'Girl'],
             'Girl': ['Clothing', 'Human face'], 
             'Car': ['Person','Vehicle registration plate'], 
             'Bus': ['Person', 'Car'], 
             'Van': ['Person'] 
          }




import json
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

if __name__ == "__main__":
    # Load the JSON file containing the detections
    detections_file = "output/detections.json"
    with open(detections_file, "r") as file:
        detections = json.load(file)
    
    print(convert_to_subobject_list(detections))   