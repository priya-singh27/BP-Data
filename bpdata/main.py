import base64
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import os
import supervision as sv
# from inference import get_model
import cv2
import numpy as np
from dotenv import load_dotenv
from .model.output_schema import BPOutputSchema  # Ensure this is correctly set up
import cv2
import numpy as np
import requests
from .utils import get_model_inference_results, extract_measurement_values , calculate_iou
load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")
OPEN_API_KEY=os.getenv("OPENAI_API_KEY")

MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"]

# bpm_model = get_model(model_id="blood-pressure-monitor-display/1", api_key=API_KEY)#an instance of finetune yolo model 
# spg_model = get_model(model_id="sphygmomanometer-qcpzd/1", api_key=API_KEY)

# @app.post("/extract_ml")
# async def create_upload_file(req: Request, file: UploadFile) :
#     content_length = req.headers.get("content-length")  # Size of file (bytes) in string
#     if content_length is not None:
#         content_length = int(content_length)
        
#     if content_length > MAX_FILE_SIZE:
#         raise HTTPException(
#             status_code=400,
#             detail="File size should be less than 4 MB",
#         )
        
#     if file.content_type not in ALLOWED_MIME_TYPES:
#         raise HTTPException(
#             status_code=415,
#             detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(ALLOWED_MIME_TYPES)}.",
#         )
        
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Convert the uploaded file (binary data) into a format that OpenCV can read
#     # image = cv2.imread(image)
#     image = cv2.resize(image, (1000, int(image.shape[0] * 1000 / image.shape[1])))
    
#     bpm_detections, bpm_class_names = get_model_inference_results(bpm_model, image.copy(), confidence=0.4)
#     spg_detections, spg_class_names = get_model_inference_results(spg_model, image.copy(), confidence=0.1)


#     # Create new lists for merged detections
#     merged_xyxy = []
#     merged_confidence = []
#     merged_class_id = []
#     merged_tracker = []


#     # Process each BPM detection
#     for i in range(len(bpm_detections.xyxy)):
#         if bpm_detections.tracker[i] == '10':  # Check for class 10
#             bbox1 = bpm_detections.xyxy[i]
#             best_overlap = {'SYS': (0, None), 'DIA': (0, None), 'PUL': (0, None)}
#             best_class = None
            
#             # Find overlapping SPG detections
#             for j in range(len(spg_detections.xyxy)):
#                 if spg_detections.tracker[j] in best_overlap:
#                     bbox2 = spg_detections.xyxy[j]
#                     iou = calculate_iou(bbox1, bbox2)
#                     if iou > best_overlap[spg_detections.tracker[j]][0]:
#                         best_overlap[spg_detections.tracker[j]] = (iou, j)
#                         if iou > 0.1:  # Add threshold to ensure meaningful overlap
#                             best_class = spg_detections.tracker[j]
            
#             # Keep original bbox but update class if match found
#             merged_xyxy.append(bpm_detections.xyxy[i])
#             merged_confidence.append(bpm_detections.confidence[i])
#             merged_class_id.append(bpm_detections.class_id[i])
#             merged_tracker.append(best_class if best_class else bpm_detections.tracker[i])
#         else:
#             # Keep other BPM detections as is
#             merged_xyxy.append(bpm_detections.xyxy[i])
#             merged_confidence.append(bpm_detections.confidence[i])
#             merged_class_id.append(bpm_detections.class_id[i])
#             merged_tracker.append(bpm_detections.tracker[i])
    
#     # Create new merged detections object
#     merged_detections = sv.Detections(
#         xyxy=np.array(merged_xyxy),
#         confidence=np.array(merged_confidence),
#         class_id=np.array(merged_class_id),
#     )
#     merged_detections.tracker = merged_tracker

#     bounding_box_annotator = sv.BoundingBoxAnnotator()
#     label_annotator = sv.LabelAnnotator()
    
#     merged_annotated_image = bounding_box_annotator.annotate(scene=image.copy(), detections=merged_detections)
#     merged_annotated_image = label_annotator.annotate(
#         scene=merged_annotated_image,
#         detections=merged_detections,
#         labels=merged_detections.tracker
#     )

#     measurements = extract_measurement_values(merged_detections)
#     print("\nExtracted measurements:")
#     data={}
#     for measurement_type, value in measurements.items():
#         data[measurement_type]=value
#         print(f"{measurement_type}: {value}")
        
    
#     return data


def extract_json_from_markdown(text):
    """
    This function extracts markdown-encoded JSON from a given string.
    
    Args:
    - text (str): Input string containing markdown with embedded JSON.
    
    Returns:
    - dict: Parsed JSON data as a Python dictionary, or None if no valid JSON is found.
    """
    # Regular expression pattern to match markdown-formatted JSON
    json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    
    # Search for the pattern in the input text
    match = json_pattern.search(text)
    
    if match:
        # Extract the matched JSON string and load it into a Python dictionary
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error decoding JSON")
            return None
    else:
        print("No markdown JSON found.")
        return None


PROMPT = """The below is an image displaying a Digital Blood Pressure monitor, try to extract the fields SYSTOLIC, DIASTOLIC and PULSE where they represent the Systolic, Diastolic blood pressure and Pulse respectively.

Respond in a valid JSON format in the below format  

```json
{ "SYSTOLIC": <number>, "SYSTOLIC_UNIT": "<extracted unit>", "DIASTOLIC": <number>, "DIASTOLIC_UNIT": "<extracted unit>", "PULSE": <number>, "PULSE_UNIT":" <extracted unit, formatted as bpm,hz etc>"}
```
"""

@app.post("/extract_llm")
async def create_upload_file(req: Request, file: UploadFile):
    content_length = req.headers.get("content-length")
    if content_length is not None:
        content_length = int(content_length)  # Convert to integer

    # Check file size
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

    # Encode the image to Base64
    encoded_string = base64.b64encode(contents).decode("utf-8")

    # Prepare payload for OpenRouter API with the new model
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_string}"
                        },
                    },
                ],
            }
        ],
        "provider": {
            "order": [
                "DeepInfra"
            ],
            "allow_fallbacks": False
        },
        "model": "meta-llama/llama-3.2-90b-vision-instruct", 
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPEN_API_KEY}"}

    try:
        # Send request to OpenRouter API
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload      
        )
                
        response.raise_for_status()  # Raise an error for bad responses

        # Process the response as needed
        res_json=response.json()
        content=res_json["choices"][0]["message"]["content"]
        print(content)
        
        # return json.loads(content)
        extracted_json=extract_json_from_markdown(content)
        return  extracted_json # Return the JSON response from OpenRouter

    except requests.exceptions.RequestException as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))




# PROMPT = """The below is an image displaying a Digital Blood Pressure monitor, try to extract the fields SYSTOLIC, DIASTOLIC and PULSE where they represent the Systolic, Diastolic blood pressure and Pulse respectively.
 
#  Respond in a JSON format in the below format
 
#  { "SYSTOLIC": <number>, "SYSTOLIC_UNIT": <extracted unit>, "DIASTOLIC": <number>, "DIASTOLIC_UNIT": <extracted unit>, "PULSE": <number>, "PULSE_UNIT": <extracted unit, formatted as bpm,hz etc>}
#  """
 
# @app.post("/data")
# async def create_upload_file(req: Request, file: UploadFile) -> BPOutputSchema:
#         content_length = req.headers.get("content-length")
#         if content_length is not None:
#             content_length = int(content_length)  # Convert to integer
        
#         # Check file size
#         if content_length > MAX_FILE_SIZE:
#             raise HTTPException(
#                 status_code=400,
#                 detail="File size should be less than 4 MB",
#             )


#         if file.content_type not in ALLOWED_MIME_TYPES:
#             raise HTTPException(
#                 status_code=415,
#                 detail=f"Unsupported file type: {file.content_type}. Allowed types are: {', '.join(ALLOWED_MIME_TYPES)}.",
#             )

#         contents = await file.read()

#         # Encode the image to Base64
#         encoded_string = base64.b64encode(contents).decode("utf-8")


#         # Prepare payload for OpenAI API
#         payload = {
#             "model": "openai/gpt-4o-mini-2024-07-18",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": PROMPT},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{encoded_string}"
#                             },
#                         },
#                     ],
#                 }
#             ],
#             "max_tokens": 300,
#             "response_format": {"type": "json_object"},
#         }

#         headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}        

#         try:
#             # Send request to OpenAI API
#             response = requests.post(
#                 "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload      
#             )


#             response.raise_for_status()  # Raise an error for bad responses

