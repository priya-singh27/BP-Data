from inference import get_model
import supervision as sv
import cv2
import numpy as np

API_KEY = "UQ5F7BNjOWg13gaycmyM"

def add_margin(x1, y1, x2, y2, margin, image_shape):
    """
    Add a margin to the bounding box coordinates.
    """
    height, width, _ = image_shape
    w = x2 - x1
    h = y2 - y1
    
    # Calculate the margin size
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    # Add the margin, ensuring it stays within the image boundaries
    x1 = max(x1 - margin_w, 0)
    y1 = max(y1 - margin_h, 0)
    x2 = min(x2 + margin_w, width)
    y2 = min(y2 + margin_h, height)
    
    return x1, y1, x2, y2

def get_class_crops(image, detections, classes_of_interest, margin=0.1):
    """
    Extract crops from the image based on detected bounding boxes for specific classes.
    Returns a dictionary mapping class names to their corresponding crops.
    """
    crops = {}
    
    # Get all classes and their corresponding indices
    class_indices = {detections.class_id[i]: i for i in range(len(detections.class_id))}
    
    for class_name in classes_of_interest:
        # Find matching detection for this class
        matching_indices = [i for i in range(len(detections.class_id)) 
                          if class_name == detections.tracker[i]]
        
        if matching_indices:
            # Get the first instance of this class
            idx = matching_indices[0]
            x1, y1, x2, y2 = detections.xyxy[idx].astype(int)
            
            # Add margin to the crop
            x1, y1, x2, y2 = add_margin(x1, y1, x2, y2, margin, image.shape)
            
            # Extract the crop
            crop = image[y1:y2, x1:x2].copy()
            crops[class_name] = crop
    
    return crops

def main():
    # Load and process image
    image_file = "imgs/bpm2_hires.jpg"
    image = cv2.imread(image_file)
    
    # Load model and run inference
    model = get_model(model_id="sphygmomanometer-qcpzd/1", api_key=API_KEY)
    results = model.infer(image)[0]
    
    # Extract class names from predictions
    predictions = results.predictions
    class_names = [pred.class_name for pred in predictions]
    
    # Create detections object with class names as tracker field
    detections = sv.Detections.from_inference(results)
    detections.tracker = class_names  # Add class names as tracker field
    
    # Classes we want to extract
    classes_of_interest = ['SYS', 'DIA', 'PUL']
    
    # Get crops for each class with 10% margin
    crops = get_class_crops(image, detections, classes_of_interest, margin=0.1)
    
    # Create annotated full image
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=class_names
    )
    
    # Display original annotated image
    sv.plot_image(annotated_image)
    
    # Display and save crops
    for class_name, crop in crops.items():
        if crop is not None:
            # Save crops with margin
            cv2.imwrite(f"crop_{class_name}.jpg", crop)

if __name__ == "__main__":
    main()
