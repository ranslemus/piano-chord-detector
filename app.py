import streamlit as st
import cv2
from PIL import Image
import numpy as np
import onnxruntime as ort

from inference_sdk import InferenceHTTPClient
import supervision as sv

from chord_predictor import predict_chord, get_note_name

ROBOFLOW_API_KEY = "3gMW5qTE5AmGJLpbANtd"
ROBOFLOW_MODEL_ID = "rechordnizer/my-first-project-9jrfe-instant-1"

# Use GPU if available
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
ort_session = ort.InferenceSession("model/final_model.onnx", providers=providers)

# Preprocessing function
def preprocess_image(crop_img):
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    crop_img = cv2.resize(crop_img, (640, 480))
    crop_img = crop_img.astype(np.float32) / 255.0
    crop_img = (crop_img - 0.5) / 0.5  # normalize same as PyTorch
    crop_img = np.expand_dims(crop_img, axis=0)  # C,H,W
    crop_img = np.expand_dims(crop_img, axis=0)  # batch
    return crop_img

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

st.title("🎹 Piano Keyboard + Chord Prediction")
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    st.image(img, caption="Original Image", width=600)

    with st.spinner("Detecting keyboard..."):
        results = client.infer(img_bgr, model_id=ROBOFLOW_MODEL_ID)
        dets = sv.Detections.from_inference(results)
        dets = dets[dets.confidence > 0.95]

    if len(dets) == 0:
        st.error("No keyboard detected")
        st.stop()

    x1, y1, x2, y2 = dets.xyxy[0]

    # Draw annotations
    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()
    annotated = box_annot.annotate(img.copy(), dets)
    annotated = label_annot.annotate(annotated, dets)
    st.subheader("Detected Keyboard")
    st.image(annotated)

    # Safe cropping
    h, w = img.shape[:2]
    EXTENSION_PIXELS = 50
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2) + EXTENSION_PIXELS)
    y2 = min(h, int(y2))
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        st.error("Detected box is empty or invalid. Cannot crop image.")
        st.stop()

    st.subheader("Cropped Keyboard")
    st.image(crop)

    # Preprocess for ONNX
    crop_input = preprocess_image(crop)

    with st.spinner("Predicting chord..."):
        output = ort_session.run(None, {"input": crop_input})[0][0]

    top_notes = np.argsort(output)[-3:][::-1]
    st.success(f"Top Predicted Notes: {top_notes.tolist()}")

    predicted_chord, score = predict_chord(top_notes.tolist())
    note_indices = top_notes.tolist()
    note_names = [get_note_name(idx) for idx in note_indices]

    st.write(f"Indices: {note_indices}")
    st.write(f"Note Names: {note_names}")
    st.success(f"**Predicted Chord: {predicted_chord}** (Score: {score:.2f})")

    other_notes = np.argsort(output)[-10:][::-1]
    st.subheader("Other Top 10 Notes")
    st.write(other_notes.tolist())

    cv2.putText(
        annotated,
        f"Top notes: {top_notes.tolist()}",
        (int(x1), int(y1) - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    st.subheader("Final Annotated Image")
    st.image(annotated)
