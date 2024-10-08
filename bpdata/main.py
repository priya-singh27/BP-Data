from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import base64
from io import BytesIO
import json
import requests
from model.output_schema import BPOutputSchema


app = FastAPI()

# OpenAI API Key
api_key = "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"

MAX_FILE_SIZE = 4 * 1024 * 1024  # 4 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"]

PROMPT = """The below is an image displaying a Digital Blood Pressure monitor, try to extract the fields SYSTOLIC, DIASTOLIC and PULSE where they represent the Systolic, Diastolic blood pressure and Pulse respectively.

Respond in a JSON format in the below format

{ "SYSTOLIC": <number>, "SYSTOLIC_UNIT": <extracted unit>, "DIASTOLIC": <number>, "DIASTOLIC_UNIT": <extracted unit>, "PULSE": <number>, "PULSE_UNIT": <extracted unit, formatted as bpm,hz etc>}
"""

@app.post("/")
async def create_upload_file(req: Request, file: UploadFile) -> BPOutputSchema:

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

    # Prepare payload for OpenAI API
    payload = {
        "model": "openai/gpt-4o-mini-2024-07-18",
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
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Send request to OpenAI API
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload
    )
    
    content = response.json()["choices"][0]['message']['content']
    extracted_data=json.loads(content)
    # Return the response from OpenAI API
    return extracted_data


