# Sub-Object Detection with YOLOv8

## Introduction
This project implements an object detection pipeline that identifies and classifies objects within videos using the YOLOv8 model. The system also handles sub-object detection, draws bounding boxes around detected objects, and saves the results as images and JSON files.

[Download Video](./output/output.mp4)

[![Video Preview](./output/output.gif)](./output/output.mp4)

## Features
- Detects multiple objects in videos
- Handles sub-object detection (objects within other objects' bounding boxes)
- Saves the detected objects and sub-objects as images
- Outputs detection results in JSON format
- Allows real-time video preview (optional)

## Requirements
- Python 3.6 or higher
- `opencv-python`
- `ultralytics` (for YOLOv8)
- `numpy`
- `json` (standard library)

You can install the necessary dependencies by running the following command:

```bash
pip install opencv-python ultralytics numpy
```

## File Structure
```
object-detection/
├── data/
│   └── testall.mp4          # Your video file(s)
├── models/                  # Models Directory
├── output/
│   ├── detections.json      # JSON file with detection results
│   └── sub_objects/         # Folder to save cropped sub-objects (optional)
│   └── output.mp4           # Processed video output
├── src/
│   ├── object_detection.py  # Object detection logic
│   ├── fine_tune_gpu        # train and fine tune model on custom dataset to add more classes
│   ├── sub_obj_list.py      # List of possible object sub-object heirarchy and function to generate it from detection.json
│   └── utils.py             # Contains function for saveing the cropped files and Logic for saving results to JSON
├── run_detection.py         # Script to run the object detection pipeline
└── app.py                   # Script to run live detection on web app ( ```bash streamlit run app.py```)
```

## Usage

### Step 1: Prepare Your Video
Place your video files in the `data` folder. For example, place your video as `data/testall.mp4`and specify your file in `run_detection.py`. 
```
video_path = "data/testall.mp4"

```

### Step 2: Run Object Detection
Run the `run_detection.py` script, which processes the video, detects objects and sub-objects, and saves the results.

```bash
python src/run_detection.py
```

The `run_detection.py` script performs the following steps:
1. Loads the video file specified in `video_path`.
2. Processes the video to detect objects using YOLOv8.
3. Detects sub-objects by checking if one object lies within the bounding box of another.
4. Draws bounding boxes around detected objects and sub-objects.
5. Saves the detection results in a JSON file (`output/detections.json`).
6. Optionally saves cropped images of sub-objects in the `output/sub_objects` directory.

### Step 3: View and Analyze Results
- **JSON output**: The results, including frame-by-frame detection details (e.g., object class, confidence, bounding box coordinates), are saved in `output/detections.json`.
- **Sub-object images**: Cropped images of detected sub-objects (if enabled) are saved in the `output/sub_objects` folder.
- **Real-time preview**: The video with detected objects is displayed in real time during processing if `show_preview=True`.


### Example Output (`detections.json`)
```json
[
  {
    "frame_id": 1,
    "detections": [
      {
        "object": "car",
        "confidence": 0.95,
        "class_id": 2,
        "bbox": [50, 100, 200, 150],
        "subobjects": [
          {
            "object": "person",
            "confidence": 0.89,
            "class_id": 0,
            "bbox": [60, 110, 100, 130]
          }
        ]
      }
    ]
  },
]
```

### Explanation of Key Sections:
- **Introduction**: Provides an overview of the project.
- **Features**: Lists key capabilities such as object detection, sub-object handling, and output saving.
- **Requirements**: Details necessary packages and dependencies.
- **File Structure**: Describes the directory structure of the project.
- **Usage**: Instructions for preparing videos, running the detection, and viewing results.
- **Example Output**: Sample of what the output JSON file looks like.
