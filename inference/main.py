# main.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from inference_sdk import InferenceHTTPClient
import supervision as sv
from chord_predictor import predict_chord, get_note_name

app = FastAPI()

# Setup ONNX session
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession("model/final_model.onnx", providers=providers)

# Setup Roboflow client
ROBOFLOW_API_KEY = "3gMW5qTE5AmGJLpbANtd"
ROBOFLOW_MODEL_ID = "rechordnizer/my-first-project-9jrfe-instant-1"
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=ROBOFLOW_API_KEY)

# Preprocess image function
def preprocess_image(crop_img):
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.resize(crop_img, (640, 480))
    crop_img = crop_img.astype(np.float32) / 255.0
    crop_img = (crop_img - 0.5) / 0.5
    crop_img = np.expand_dims(crop_img, axis=0)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pil_img = Image.open(file.file).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Roboflow detection
    results = client.infer(img_bgr, model_id=ROBOFLOW_MODEL_ID)
    dets = sv.Detections.from_inference(results)
    dets = dets[dets.confidence > 0.95]

    if len(dets) == 0:
        return {"error": "No keyboard detected"}

    x1, y1, x2, y2 = dets.xyxy[0]
    h, w = img.shape[:2]
    EXTENSION_PIXELS = 50
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2) + EXTENSION_PIXELS), min(h, int(y2))
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return {"error": "Detected box is empty"}

    crop_input = preprocess_image(crop)
    output = ort_session.run(None, {"input": crop_input})[0][0]

    top_notes = np.argsort(output)[-3:][::-1]
    predicted_chord, score = predict_chord(top_notes.tolist())
    note_names = [get_note_name(idx) for idx in top_notes.tolist()]

    return {
        "top_notes": top_notes.tolist(),
        "note_names": note_names,
        "predicted_chord": predicted_chord,
        "score": float(score)
    }
