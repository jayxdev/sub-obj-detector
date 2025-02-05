import cv2
from ultralytics import YOLO
import os
from .sub_obj_list import sub_objects_list
from PIL import Image
from .utils import save_subobject_image

# Initialize the YOLO model
def initialize_model(model_path='models/yolov8n-oiv7.pt'):
    model = YOLO(model_path)
    return model

# Function to detect objects with sub-object handling and yield results
def detect_objects(video_path, model=None, frame_skip=3, resize_frctor=2,confidence_threshold=0.3, show_preview=False, save_sub_objects=False, save_video=False, output_dir='./output/sub_objects'):
    if model is None:
        model = initialize_model()

    cap = cv2.VideoCapture(video_path)
    
    # Video output setup
    frame_width = int(300*resize_frctor)
    frame_height = inr(200*resize_frctor)
    
    # Get FPS from the video or set a default value
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    
    # Create VideoWriter to save output
    if save_video:
        out = cv2.VideoWriter('./output/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_id = 0
    last_detections = []

    if save_sub_objects and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_time = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        # Resize frame for processing
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
        sub_obj_image=[]
        if frame_id % frame_skip == 0:
            # Run YOLO detection
            results = model(resized_frame)
            inference_time = results[0].speed['inference']
            frame_detections = []
            for result in results:
                objects = []
                for box in result.boxes:
                    # Extract detection details
                    coords = box.xyxy[0].tolist()  # Bounding box coordinates
                    conf = box.conf[0].item()     # Confidence score
                    class_id = int(box.cls[0])    # Class ID
                    label = result.names[class_id]

                    if conf > confidence_threshold:
                        objects.append({
                            "object": label,
                            "confidence": conf,
                            "class_id": class_id,
                            "bbox": coords
                        })

                # Sub-object detection
                for obj in objects:
                    xmin, ymin, xmax, ymax = map(int, obj["bbox"])
                    sub_objects = []
                    for other_obj in objects:
                        if other_obj == obj:
                            continue
                        other_xmin, other_ymin, other_xmax, other_ymax = map(int, other_obj["bbox"])
                        # Check if the other object is within the bounding box of the current object
                        if xmin < other_xmin < xmax and ymin < other_ymin < ymax:
                            # Save cropped sub-object images
                            if save_sub_objects and other_obj["object"] in sub_objects_list.get(other_obj["object"],[]):
                                # Safeguard dimensions to avoid invalid cropping
                                sub_ymin = max(0, other_ymin)
                                sub_ymax = min(frame_height, other_ymax)
                                sub_xmin = max(0, other_xmin)
                                sub_xmax = min(frame_width, other_xmax)
                                save_subobject_image(resized_frame, (sub_xmin, sub_ymin, sub_xmax, sub_ymax), str(frame_id) + "_" + str(obj["class_id"]), other_obj["object"], output_dir)
                            sub_objects.append({
                                "object": other_obj["object"],
                                "class_id": other_obj["class_id"],
                                "confidence": other_obj["confidence"],
                                "bbox": other_obj["bbox"]
                            })

                    obj["subobjects"] = sub_objects
                    frame_detections.append(obj)

            last_detections = frame_detections
        else:
            frame_detections = last_detections
        
        # Draw detections on the frame
        for detection in frame_detections:
            xmin, ymin, xmax, ymax = map(int, detection["bbox"])
            label = detection["object"]
            conf = detection["confidence"]
            text = f"{label} {conf:.2f}"
            cv2.rectangle(resized_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Green color for objects
            cv2.putText(resized_frame, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Green color for objects
            fps=int(1000/inference_time)
            cv2.putText(resized_frame, f"FPS: {fps}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"Inference Time: {inference_time:.2f}ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"Green Box: Object", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(resized_frame, f"Blue Box: Sub-Object", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw sub-object bounding boxes
            for sub_object in detection["subobjects"]:
                sub_xmin, sub_ymin, sub_xmax, sub_ymax = map(int, sub_object["bbox"])
                sub_label = sub_object["object"]
                sub_text = f"{sub_label} {sub_object['confidence']:.2f}"
                cv2.rectangle(resized_frame, (sub_xmin, sub_ymin), (sub_xmax, sub_ymax), (255, 0, 0), 2) # Blue color for sub-objects
                cv2.putText(resized_frame, sub_text, (sub_xmin, sub_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Blue color for sub-objects
        
        # Write the processed frame to the output video
        if save_video:
            out.write(resized_frame)

        # Display the video frame if show_preview is True
        if show_preview:
            cv2.imshow("Object and Sub-Object Detection", resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Yield the detection results and processed frame
        yield {
            "frame_id": frame_id,
            "detections": frame_detections,
            "frame": resized_frame
        }

    cap.release()
    if save_video:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    video_path = "data/traffic.mp4"
    model = initialize_model()
    for result in detect_objects(video_path, model=model, show_preview=True, save_sub_objects=True, sub_object_class_id=1):
        print(f"Frame ID: {result['frame_id']}")
        print(f"Detections: {result['detections']}")
