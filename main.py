import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import io
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from googletrans import Translator

# Translator setup
translator = Translator()
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Marathi': 'mr',
    'Tamil': 'ta',
    'Telugu': 'te'
}
st.sidebar.title("Language Selector")
selected_language = st.sidebar.selectbox("Choose Language", list(LANGUAGES.keys()))
_ = lambda text: translator.translate(text, dest=LANGUAGES[selected_language]).text if selected_language != 'English' else text

# Load model
try:
    model = tf.keras.models.load_model('best_vgg19.h5', custom_objects={'preprocess_input': preprocess_input})
except Exception as e:
    st.error(_(f"Failed to load model: {str(e)}"))
    st.stop()

# UI Text
st.set_page_config(page_title=_("Tea Leaf Disease Classification"), layout="centered")
st.title(_("Tea Leaf Disease Detection using VGG19"))
st.write(_("Upload a tea leaf image to detect disease, analyze severity, view heatmaps, and receive management tips."))

uploaded_file = st.file_uploader(_("Choose an image..."), type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption=_("Uploaded Image"), use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(img_array, img_array, mask=mask)

    img_array_preprocessed = preprocess_input(img_array.copy())
    img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)

    prediction = model.predict(img_array_preprocessed)
    class_names = ['algal_spot', 'brown_blight', 'gray_blight', 'healthy', 'helopeltis', 'red_spot']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(_(f"Prediction: {predicted_class.replace('_', ' ').title()} ({confidence:.2f}% confidence)"))  
    probs = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.subheader(_("Prediction Probabilities"))
    st.bar_chart(probs)

    # Grad-CAM Heatmap
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer('block3_conv4').output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_preprocessed)
        predicted_class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, predicted_class_idx]
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10
    heatmap_resized = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap_masked = cv2.bitwise_and(np.uint8(255 * heatmap_resized), np.uint8(255 * heatmap_resized), mask=mask)
    severity = (np.sum(heatmap_masked > 0) / np.sum(mask > 0)) * 100 if np.sum(mask > 0) > 0 else 0
    st.subheader(_("Disease Severity Score"))
    st.write(_(f"Affected area: {severity:.2f}%"))

    # Overlay heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.6, heatmap_colored, 0.4, 0)
    st.image(superimposed_img, caption=_("Heatmap Overlay"), use_container_width=True)

    # Management tips
    st.subheader(_("Management Tips"))
    # Example tips – real ones can be mapped using the predicted_class
    tips = [
        _("Prune affected areas."),
        _("Avoid water logging."),
        _("Use neem-based organic sprays."),
        _("Improve soil drainage."),
        _("Ensure proper air circulation.")
    ]
    for tip in tips:
        st.markdown(f"- {tip}")
