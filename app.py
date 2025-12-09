import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from inference_sdk import InferenceHTTPClient
import supervision as sv

import torch.nn as nn
from torch import Tensor

from chord_predictor import predict_chord, get_note_name

class PianoModelBlock2D(nn.Module):
    def __init__(self, in_dim, out_dim, ksize=(3,3), stride=(1,1), drop=0.0, pad=True):
        super().__init__()
        padding = (ksize[0]//2, ksize[1]//2) if pad else (0,0)
        self.main = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=drop),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x: Tensor):
        return self.relu(self.main(x) + self.downsample(x))

class PianoModelSmall2D(nn.Module):
    def __init__(self, input_size=(480,640)):
        super().__init__()
        downscale_dim_sizes = [32,32,64,128,128,256]
        self.preprocess = nn.Sequential(
            nn.Conv2d(1, downscale_dim_sizes[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(downscale_dim_sizes[0]),
            nn.LeakyReLU(inplace=True)
        )
        self.blocks = nn.ModuleList([
            PianoModelBlock2D(downscale_dim_sizes[0], downscale_dim_sizes[1], stride=2, drop=0.2),
            PianoModelBlock2D(downscale_dim_sizes[1], downscale_dim_sizes[2], stride=2, drop=0.2),
            PianoModelBlock2D(downscale_dim_sizes[2], downscale_dim_sizes[3], stride=(2,1), drop=0.2),
            PianoModelBlock2D(downscale_dim_sizes[3], downscale_dim_sizes[4], stride=(2,1), drop=0.2),
            PianoModelBlock2D(downscale_dim_sizes[4], downscale_dim_sizes[5], stride=(2,1), drop=0.0)
        ])
        final_conv_dim = 256
        self.final_conv = nn.Conv1d(downscale_dim_sizes[-1], final_conv_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear((input_size[1]//4)*final_conv_dim, 88)

    def forward(self, x: Tensor):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        x = self.final_conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

ROBOFLOW_API_KEY = "3gMW5qTE5AmGJLpbANtd"
ROBOFLOW_MODEL_ID = "rechordnizer/my-first-project-9jrfe-instant-1"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("model/final_model.pt", map_location=device, weights_only=False)
model.eval()

# model = PianoModelSmall2D(input_size=(480, 640)).to(device)
# model.load_state_dict(torch.load("model\model_epoch4_step56000.pth", map_location=device))
# model.eval()

to_tensor = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((480,640)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

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

    box_annot = sv.BoxAnnotator()
    label_annot = sv.LabelAnnotator()
    annotated = box_annot.annotate(img.copy(), dets)
    annotated = label_annot.annotate(annotated, dets)
    st.subheader("Detected Keyboard")
    st.image(annotated)

    crop = img[int(y1):int(y2), int(x1):int(x2)]
    st.subheader("Cropped Keyboard")
    st.image(crop)

    with st.spinner("Predicting chord..."):
        crop_tensor = to_tensor(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(crop_tensor)[0].cpu().numpy()

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
        (int(x1), int(y1)-12),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0,0,255),
        9
    )
    st.subheader("Final Annotated Image")
    st.image(annotated)
