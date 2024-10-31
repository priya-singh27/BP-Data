from inference import get_model
import supervision as sv
import cv2
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")

def calculate_detection_percentages(image, detections):
    """
    Calculate the percentage of image area that each detection occupies.
    Returns the original detections object and a list of percentages.
    """
    image_area = image.shape[0] * image.shape[1]
    percentages = []
    
    for bbox in detections.xyxy:
        x1, y1, x2, y2 = bbox
        detection_area = (x2 - x1) * (y2 - y1)
        percentage = (detection_area / image_area) * 100
        percentages.append(percentage)
    
    return percentages

def filter_detections(detections, percentages, threshold=7.5):
    """
    Filter detections based on their percentage size.
    Returns a new detections object with only the detections above the threshold.
    """
    mask = np.array(percentages) >= threshold
    
    return sv.Detections(
        xyxy=detections.xyxy[mask],
        confidence=detections.confidence[mask] if detections.confidence is not None else None,
        class_id=detections.class_id[mask] if detections.class_id is not None else None,
        # tracker=np.array(detections.tracker)[mask] if detections.tracker is not None else None
    )

def main():
    # Load and process image
    image_file = "crop_SYS.jpg"
    image = cv2.imread(image_file)

    preprocessed_image = final_prep(image_file)
    # print(preprocessed_image.shape, len(preprocessed_image.shape))
    # Convert to 3-channel format for OCR if necessary
    final_img_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)

    
    # # Load model and run inference
    model = get_model(model_id="7-segment-display-gxhnj/2", api_key=API_KEY)
    results = model.infer(final_img_rgb, confidence=0.2)[0]
    
    # Extract class names from predictions
    predictions = results.predictions
    class_names = [pred.class_name for pred in predictions]
    
    # Create detections object with class names as tracker field
    detections = sv.Detections.from_inference(results)
    detections.tracker = class_names  # Add class names as tracker field
    
    # Calculate percentages for each detection
    percentages = calculate_detection_percentages(image, detections)
    
    # Print detection percentages
    print("\nDetection Percentages:")
    for class_name, percentage in zip(class_names, percentages):
        print(f"Class {class_name}: {percentage:.2f}% of image area")
    
    # Filter detections based on percentage threshold
    filtered_detections = filter_detections(detections, percentages)
    
    # Create annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Create annotated image with filtered detections
    annotated_image = bounding_box_annotator.annotate(scene=final_img_rgb, detections=filtered_detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=filtered_detections,
        labels=[f"{class_name} ({percentage:.1f}%)" 
                for class_name, percentage in zip(np.array(class_names)[np.array(percentages) >= 7.5], 
                                               np.array(percentages)[np.array(percentages) >= 7.5])]
    )
    
    # Display filtered annotated image
    sv.plot_image(annotated_image)

if __name__ == "__main__":
    main()