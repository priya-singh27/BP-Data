import supervision as sv


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



def get_model_inference_results(imodel, image, confidence=0.2):
    results = imodel.infer(image, confidence = confidence)[0]

    # Extract class names from predictions
    predictions = results.predictions
    class_names = [(pred.class_name) for pred in predictions]
    print(class_names)
    
    # Create detections object with class names as tracker field
    detections = sv.Detections.from_inference(results)
    detections.tracker = class_names  # Add class names as tracker field

    return detections, class_names


def calculate_iou(box1, box2):
    """
    Calculate intersection over union between two bounding boxes.
    Boxes should be in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


def get_center_of_mass(bbox):
    """Calculate center of mass for a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def extract_measurement_values(detections):
    """
    Extract and sort digits within SYS/DIA/PUL regions, returning their values.
    """
    measurements = {}
    
    # Group all digit detections by their parent measurement type
    for measurement_type in ['SYS', 'DIA', 'PUL']:
        measurement_idx = [i for i, t in enumerate(detections.tracker) if t == measurement_type]
        if not measurement_idx:
            continue
            
        # Get the measurement bounding box
        measurement_bbox = detections.xyxy[measurement_idx[0]]
        
        # Find all digit detections that overlap with this measurement
        digit_detections = []
        for i in range(len(detections.xyxy)):
            if detections.tracker[i].isdigit():  # Check if it's a digit (0-9)
                digit_bbox = detections.xyxy[i]
                if calculate_iou(measurement_bbox, digit_bbox) > 0.1:
                    center = get_center_of_mass(digit_bbox)
                    digit_detections.append((center[0], detections.tracker[i]))
        
        # Sort digits by x-coordinate (left to right)
        digit_detections.sort(key=lambda x: x[0])
        value = ''.join(digit[1] for digit in digit_detections)
        measurements[measurement_type] = int(value) if value else None
    
    return measurements


