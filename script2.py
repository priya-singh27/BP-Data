from inference import get_model
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'#'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

API_KEY = "xN8t7Z4ubrvi5PkiI3JI"

def extract_crops_with_margin(image, predictions, classes_of_interest, margin_percent=10):
    """
    Extract crops for specified classes from the image based on predictions with added margins
    """
    crops = {}
    img_height, img_width = image.shape[:2]
    
    for pred in predictions:
        if pred.class_name in classes_of_interest:
            # Calculate original bounding box coordinates
            w = int(pred.width)
            h = int(pred.height)
            
            # Calculate margin in pixels
            margin_x = int(w * margin_percent / 100)
            margin_y = int(h * margin_percent / 100)
            
            # Calculate coordinates with margin
            x = int(pred.x - pred.width/2) - margin_x
            y = int(pred.y - pred.height/2) - margin_y
            w = w + (2 * margin_x)
            h = h + (2 * margin_y)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Extract crop
            crop = image[y:y+h, x:x+w].copy()
            crops[pred.class_name] = crop
            
    return crops

def extract_text_from_crops(crops):
    """
    Extract text from the cropped images using Tesseract OCR
    """
    extracted_text = {}
    for class_name, crop in crops.items():
        text = pytesseract.image_to_string(crop)
        extracted_text[class_name] = text.strip()
    return extracted_text

def main():
    # Load image
    image_file = "bpm.jpg"
    image = cv2.imread(image_file)
    
    # Load model and run inference
    model = get_model(model_id="sphygmomanometer-qcpzd/1", api_key=API_KEY)
    results = model.infer(image)[0]
    
    # Classes to extract
    classes_of_interest = ['SYS', 'DIA', 'PUL']
    
    # Get crops with 10% margin
    crops = extract_crops_with_margin(image, results.predictions, classes_of_interest, margin_percent=10)
    # print(crops)
    # Extract text from crops using Tesseract OCR
    extracted_text = extract_text_from_crops(crops)
    # print(f"extracted_text ${extracted_text}")
    
    # Print the extracted text
    ans={}
    for class_name, text in extracted_text.items():
        ans[class_name]=text
        # print(f"{class_name}: {text}")
    
    print(ans)

if __name__ == "__main__":
    main()


# from inference import get_model
# import cv2
# import numpy as np

# API_KEY = "xN8t7Z4ubrvi5PkiI3JI"

# def extract_crops_with_margin(image, predictions, classes_of_interest, margin_percent=10):
#     """
#     Extract crops for specified classes from the image based on predictions with added margins
#     """
#     crops = {}
#     img_height, img_width = image.shape[:2]
    
#     for pred in predictions:
#         if pred.class_name in classes_of_interest:
#             # Calculate original bounding box coordinates
#             w = int(pred.width)
#             h = int(pred.height)
            
#             # Calculate margin in pixels
#             margin_x = int(w * margin_percent / 100)
#             margin_y = int(h * margin_percent / 100)
            
#             # Calculate coordinates with margin
#             x = int(pred.x - pred.width/2) - margin_x
#             y = int(pred.y - pred.height/2) - margin_y
#             w = w + (2 * margin_x)
#             h = h + (2 * margin_y)
            
#             # Ensure coordinates are within image bounds
#             x = max(0, x)
#             y = max(0, y)
#             w = min(w, img_width - x)
#             h = min(h, img_height - y)
            
#             # Extract crop
#             crop = image[y:y+h, x:x+w].copy()
#             crops[pred.class_name] = crop
            
#     return crops

# def main():
#     # Load image
#     image_file = "bpm.jpg"
#     image = cv2.imread(image_file)
    
#     # Load model and run inference
#     model = get_model(model_id="sphygmomanometer-qcpzd/1", api_key=API_KEY)
#     results = model.infer(image)[0]
    
#     # Classes to extract
#     classes_of_interest = ['SYS', 'DIA', 'PUL']
    
#     # Get crops with 10% margin
#     crops = extract_crops_with_margin(image, results.predictions, classes_of_interest, margin_percent=10)
    
#     # Save crops
#     for class_name, crop in crops.items():
        
#         ...
#         # cv2.imwrite(f"crop_{class_name}.jpg", crop)

# if __name__ == "__main__":
#     main()