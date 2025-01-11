import cv2
from src.object_detection import detect_objects
from src.json_output import save_json_output

# Load the video or image
video_path = "data/testall.mp4"  # Path to your sample video

detections_generator = detect_objects(video_path, show_preview=True, save_sub_objects=True)

# List to hold all detection results
all_detections = []

# Iterate through the generator and extract `frame_id` and `detections`
for detection in detections_generator:
    frame_id = detection["frame_id"]
    detections = detection["detections"]
    # images = detection["images"]
    # # Save the images
    # for image in images:
    #     image_path = image["sub_object_filename"]
    #     cv2.imshow("Detected Sub Object", image["sub_object"])y
    #     cv2.waitKey(1)  # Display the image for 1 millisecond
    #     cv2.imwrite(image_path, image["sub_object"])
    # Collect the results in a dictionary
    frame_data = {
        "frame_id": frame_id,
        "detections": detections
    }

    # Append the frame data to the list of all detections
    all_detections.append(frame_data)

# Save all the collected detections to a JSON file
output_path = "output/detections.json"
save_json_output(all_detections, output_path)



