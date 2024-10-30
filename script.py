from inference import get_model
import supervision as sv
import cv2
import numpy as np

API_KEY = "xN8t7Z4ubrvi5PkiI3JI"

def get_class_crops(image, detections, classes_of_interest):
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
            # Extract the crop
            crop = image[y1:y2, x1:x2].copy()
            crops[class_name] = crop
    
    return crops

def main():
    # Load and process image
    image_file = "bpm.jpg"
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
    
    # Get crops for each class
    crops = get_class_crops(image, detections, classes_of_interest)
    
    # Create annotated full image
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_parameters=sv.TextParameters(
        font_size=40,
        text_color=(255, 255, 255),
        background_color=(0, 0, 0)
    ))
    
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=class_names
    )
    
    # Display original annotated image
    sv.plot_image(annotated_image, title="Full Detection")
    
    # Display and save crops
    for class_name, crop in crops.items():
        if crop is not None:
            sv.plot_image(crop, title=f"Crop: {class_name}")
            # Save crops
            cv2.imwrite(f"crop_{class_name}.jpg", crop)

if __name__ == "__main__":
    main()