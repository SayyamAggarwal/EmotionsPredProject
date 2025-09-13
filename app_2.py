import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from transformers import TFViTModel
from PIL import Image
import os
import datetime

# ---------------------------
# Model path and classes
# ---------------------------
MODEL_PATH = "google-finetuned_ViT-model.keras"  # Make sure this exists locally
CLASS_NAMES = ["happy", "angry", "sad"]

# ---------------------------
# Directory to save uploaded images
# ---------------------------
SAVE_DIR = "uploaded_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------
# Custom ViT layer
# ---------------------------
class ViTCLSExtractor(Layer):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", **kwargs):
        super().__init__(**kwargs)
        self.vit = TFViTModel.from_pretrained(model_name, from_pt=True, name="vit_base")

    def call(self, inputs):
        outputs = self.vit(pixel_values=inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]  # take [CLS] token
        return cls_token

# ---------------------------
# Load model
# ---------------------------
model = load_model(MODEL_PATH, custom_objects={'ViTCLSExtractor': ViTCLSExtractor})

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("😊 Emotion Detection (Happy / Angry / Sad)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Save uploaded image
    # ---------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    save_path = os.path.join(SAVE_DIR, filename)
    image.save(save_path)
    # st.success(f"Image saved at: {save_path}")

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array, verbose=0)
    pred_class = CLASS_NAMES[int(np.argmax(preds))]
    confidence = np.max(preds) * 100

    # Show prediction
    st.markdown(f"### Predicted Emotion: **{pred_class}** ({confidence:.2f}%)")

    # Emoji + message
    emojis = {"happy": "😀", "angry": "😡", "sad": "😢"}
    boosters = {
        "happy": "Keep shining! 🌞",
        "angry": "Relax, breathe, you got this 🧘",
        "sad": "Cheer up! Better days are ahead 💪"
    }
    st.markdown(f"{emojis[pred_class]} {boosters[pred_class]}")

    # Probability chart
    st.bar_chart(preds[0])
