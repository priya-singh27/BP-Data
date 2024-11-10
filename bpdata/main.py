import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import os
import supervision as sv
from inference import get_model
import cv2
import numpy as np
from dotenv import load_dotenv
from .model.output_schema import BPOutputSchema  # Ensure this is correctly set up
import cv2
import numpy as np
from .utils import get_model_inference_results, extract_measurement_values , calculate_iou
load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")

MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"]

bpm_model = get_model(model_id="blood-pressure-monitor-display/1", api_key=API_KEY)#an instance of finetune yolo model 
spg_model = get_model(model_id="sphygmomanometer-qcpzd/1", api_key=API_KEY)

@app.post("/extract_ml")#/extract_llm
async def create_upload_file(req: Request, file: UploadFile) :
    content_length = req.headers.get("content-length")  # Size of file (bytes) in string
    if content_length is not None:
        content_length = int(content_length)
        
    if content_length > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File size should be less than 4 MB",
        )
        
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(ALLOWED_MIME_TYPES)}.",
        )
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert the uploaded file (binary data) into a format that OpenCV can read
    # image = cv2.imread(image)
    image = cv2.resize(image, (1000, int(image.shape[0] * 1000 / image.shape[1])))
    
    bpm_detections, bpm_class_names = get_model_inference_results(bpm_model, image.copy(), confidence=0.4)
    spg_detections, spg_class_names = get_model_inference_results(spg_model, image.copy(), confidence=0.1)


    # Create new lists for merged detections
    merged_xyxy = []
    merged_confidence = []
    merged_class_id = []
    merged_tracker = []


    # Process each BPM detection
    for i in range(len(bpm_detections.xyxy)):
        if bpm_detections.tracker[i] == '10':  # Check for class 10
            bbox1 = bpm_detections.xyxy[i]
            best_overlap = {'SYS': (0, None), 'DIA': (0, None), 'PUL': (0, None)}
            best_class = None
            
            # Find overlapping SPG detections
            for j in range(len(spg_detections.xyxy)):
                if spg_detections.tracker[j] in best_overlap:
                    bbox2 = spg_detections.xyxy[j]
                    iou = calculate_iou(bbox1, bbox2)
                    if iou > best_overlap[spg_detections.tracker[j]][0]:
                        best_overlap[spg_detections.tracker[j]] = (iou, j)
                        if iou > 0.1:  # Add threshold to ensure meaningful overlap
                            best_class = spg_detections.tracker[j]
            
            # Keep original bbox but update class if match found
            merged_xyxy.append(bpm_detections.xyxy[i])
            merged_confidence.append(bpm_detections.confidence[i])
            merged_class_id.append(bpm_detections.class_id[i])
            merged_tracker.append(best_class if best_class else bpm_detections.tracker[i])
        else:
            # Keep other BPM detections as is
            merged_xyxy.append(bpm_detections.xyxy[i])
            merged_confidence.append(bpm_detections.confidence[i])
            merged_class_id.append(bpm_detections.class_id[i])
            merged_tracker.append(bpm_detections.tracker[i])
    
    # Create new merged detections object
    merged_detections = sv.Detections(
        xyxy=np.array(merged_xyxy),
        confidence=np.array(merged_confidence),
        class_id=np.array(merged_class_id),
    )
    merged_detections.tracker = merged_tracker

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    merged_annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=merged_detections)
    merged_annotated_image = label_annotator.annotate(
        scene=merged_annotated_image,
        detections=merged_detections,
        labels=merged_detections.tracker
    )

    measurements = extract_measurement_values(merged_detections)
    print("\nExtracted measurements:")
    data={}
    for measurement_type, value in measurements.items():
        data[measurement_type]=value
        print(f"{measurement_type}: {value}")
        
    
    return data








