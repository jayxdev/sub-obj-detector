from ultralytics import YOLO
if __name__ == "__main__":
    # Step 1: Define paths and number of classes
    dataset_path = "traindata"  # Replace with your dataset path
    
    
    # Save the dataset.yaml file
    yaml_file = "datasets/traindata/data.yaml"
    
    # Step 3: Load YOLOv8n model (pre-trained weights)
    model = YOLO("yolov8n.pt")  # YOLOv8n pre-trained model
    
    # Step 4: Fine-tune the model on your dataset
    model.train(data=yaml_file, epochs=50, imgsz=640, batch=16, device=0)  # Adjust epochs, batch size, etc.
    
    # Step 5: Evaluate the trained model
    results = model.val()
    print(results)  # Print validation metrics (e.g., mAP, precision, recall)
    
    # Step 6: Save the fine-tuned model
    model.save("yolov8n_finetuned.pt")
    
    # Step 7: Run inference on a test image
    test_image = "traindata/train/images/1_mp4-4_jpg.rf.3ace7efb180722835cc71a302b45f9f1.jpg"  # Replace with your image path
    results = model.predict(test_image)
    results.show()  # Display predictions on the test image
    results.save("result")  # Save the output with bounding boxes
