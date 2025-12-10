import streamlit as st
import numpy as np
from PIL import Image
import requests

API_URL = "https://piano-chord-detector-i4pp-h8vcyq41r-reubens-projects-5f68c943.vercel.app/api/predict"

st.title("🎹 Piano Keyboard + Chord Prediction")
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Original Image", width=600)

    import io
    img_bytes = io.BytesIO()
    pil_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    with st.spinner("Sending image to backend for prediction..."):
        files = {"file": ("image.png", img_bytes, "image/png")}
        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        st.error(f"Error from backend: {response.text}")
    else:
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            st.subheader("Prediction Results")
            st.write(f"Top Notes: {data['top_notes']}")
            st.write(f"Note Names: {data['note_names']}")
            st.success(f"Predicted Chord: {data['predicted_chord']} (Score: {data['score']:.2f})")

            # Optional: show other top 10 notes if available
            if "other_notes" in data:
                st.subheader("Other Top 10 Notes")
                st.write(data["other_notes"])
