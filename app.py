import streamlit as st
import time
from src.object_detection import detect_objects
import tempfile

# Shared configuration dictionary for real-time updates
shared_config = {"frame_skip": 5, "confidence_threshold": 0.5, "resize_factor": 1.5}

def update_config():
    # Streamlit sliders for real-time updates
    st.sidebar.title("Settings")
    shared_config["frame_skip"] = st.sidebar.slider("Frame Skip", 1, 30, 5)
    shared_config["confidence_threshold"] = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35)
    shared_config["resize_factor"] = st.sidebar.slider("Resize Factor", 0.5, 3.0, 1.5, 0.5)

def object_detection_stream(video_path):
    # Real-time object detection generator
    frame_skip = shared_config["frame_skip"]
    confidence_threshold = shared_config["confidence_threshold"]
    resize_factor = shared_config["resize_factor"]

    for result in detect_objects(video_path, frame_skip=frame_skip,resize_frctor=resize_factor, confidence_threshold=confidence_threshold):
        yield result

def main():
    st.title("Real-Time Object Detection")

    # Video file uploader
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    if video_file is not None:
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(video_file.read())
            tmpfile_path = tmpfile.name

        stframe = st.empty()
        object_placeholder = st.empty()

        # Update the shared configuration
        update_config()

        # Process video frames
        detections = object_detection_stream(tmpfile_path)
        for result in detections:
            frame = result["frame"]
            frame_detections = result["detections"]

            # Display the frame
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Display detected objects in a table
            detected_objects = [
                {
                    "Object": detection['object'],
                    "Confidence": f"{detection['confidence']*100:.2f}%",
                    "Subobjects": str([obj['object'] for obj in detection['subobjects']]) if detection['subobjects'] else "-",
                }
                for detection in frame_detections
                if detection['confidence'] >= shared_config["confidence_threshold"]
            ]
            object_placeholder.table(detected_objects)

            # Simulate real-time processing
            time.sleep(0.1)

if __name__ == "__main__":
    main()
